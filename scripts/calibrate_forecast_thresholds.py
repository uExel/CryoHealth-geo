"""Walk-forward backtesting to empirically calibrate forecast.py's thresholds.

Sweeps UNCERTAINTY_CUTOFF and EXPANSION_PROB_THRESHOLD across a grid, computing
coverage, false alarm rate, miss rate, rejection rate, precision, and recall for each
combination against real historical observation data.

Usage:
    uv run python scripts/calibrate_forecast_thresholds.py \\
        --start-date 2023-01-01 \\
        --end-date 2026-01-01 \\
        --step-days 14 \\
        --prob-sweep 0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,0.99 \\
        --uncertainty-sweep 0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60 \\
        --output calibration_results.csv

Requires:
    - Real DB access (DB_HOST/DB_PORT/DB_USER/DB_PASSWORD/DB_NAME set)
    - prophet installed: uv pip install 'cryohealth-geo[forecast]'
    - pandas: installed as prophet transitive dependency

After running:
    1. Review the printed recommendation and the boundary-detection warnings.
    2. If any recommended value lands at a sweep boundary, re-run with a wider range.
    3. Update EXPANSION_PROB_THRESHOLD and UNCERTAINTY_CUTOFF in pipeline/forecast.py
       with the recommended values, adding a comment with:
         - calibration run date
         - output CSV filename
         - metric values at the chosen (uncertainty_cutoff, prob_threshold) combination
       This is required by the Issue #21 acceptance criterion.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass, field, fields
from datetime import date, timedelta

import numpy as np


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class BacktestWindow:
    """One walk-forward step: predictions vs. actuals for a lake on a given date."""
    slug: str
    as_of: date
    obs_at_t: float
    actual_t14: float | None         # None if no observation found near T+14
    p50: float | None
    p10: float | None
    p90: float | None
    expansion_probability: float | None
    relative_uncertainty: float | None
    method: str


@dataclass
class GridPoint:
    """Metrics for one (uncertainty_cutoff, prob_threshold) combination."""
    uncertainty_cutoff: float
    prob_threshold: float
    n_windows: int = 0            # total evaluable windows (actual_t14 not None)
    n_accepted: int = 0           # windows not rejected by uncertainty gate
    n_elevated: int = 0           # elevated-risk flags raised
    n_tp: int = 0                 # true positives  (elevated AND expanded)
    n_fp: int = 0                 # false positives (elevated AND NOT expanded)
    n_fn: int = 0                 # false negatives (not elevated AND expanded)
    n_tn: int = 0                 # true negatives  (not elevated AND NOT expanded)
    n_in_interval: int = 0        # actuals within [p10, p90]

    @property
    def rejection_rate(self) -> float:
        return 1.0 - (self.n_accepted / self.n_windows) if self.n_windows else float("nan")

    @property
    def empirical_coverage(self) -> float:
        return self.n_in_interval / self.n_accepted if self.n_accepted else float("nan")

    @property
    def false_alarm_rate(self) -> float:
        denom = self.n_fp + self.n_tn
        return self.n_fp / denom if denom else float("nan")

    @property
    def miss_rate(self) -> float:
        denom = self.n_fn + self.n_tp
        return self.n_fn / denom if denom else float("nan")

    @property
    def precision(self) -> float:
        denom = self.n_tp + self.n_fp
        return self.n_tp / denom if denom else float("nan")

    @property
    def recall(self) -> float:
        denom = self.n_tp + self.n_fn
        return self.n_tp / denom if denom else float("nan")

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else float("nan")


# ---------------------------------------------------------------------------
# DB access
# ---------------------------------------------------------------------------


def _fetch_all_observations(slugs: list[str]) -> dict[str, list[tuple[date, float]]]:
    """Fetch all observations for the given lake slugs from the DB."""
    from pipeline.db import connect, lake_id_for_slug  # noqa: PLC0415
    result: dict[str, list[tuple[date, float]]] = {slug: [] for slug in slugs}
    with connect() as conn:
        for slug in slugs:
            lake_id = lake_id_for_slug(conn, slug)
            if lake_id is None:
                print(f"WARNING: slug {slug!r} not found in DB — skipping", file=sys.stderr)
                continue
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT "capturedAt", "areaKm2"
                       FROM observations
                       WHERE "lakeId" = %s
                       ORDER BY "capturedAt" """,
                    (lake_id,),
                )
                result[slug] = [(row[0].date(), float(row[1])) for row in cur.fetchall()]
    return result


# ---------------------------------------------------------------------------
# Walk-forward backtesting
# ---------------------------------------------------------------------------


def _closest_within(
    observations: list[tuple[date, float]],
    target: date,
    tolerance_days: int = 15,
) -> tuple[date, float] | None:
    within = [o for o in observations if abs((o[0] - target).days) <= tolerance_days]
    return min(within, key=lambda o: abs((o[0] - target).days)) if within else None


def _run_walk_forward(
    all_obs: dict[str, list[tuple[date, float]]],
    start_date: date,
    end_date: date,
    step_days: int = 14,
) -> list[BacktestWindow]:
    """Collect BacktestWindow records across all lakes and all T steps.

    Calls compute_forecast() once per (lake, T) step. This is the slow loop —
    each call fits a Prophet model. With 6 lakes and ~50 steps, expect ~300 fits,
    each 3–8 seconds = 15–40 minutes total.
    """
    from pipeline.forecast import compute_forecast, ForecastUnavailableError  # noqa: PLC0415

    windows: list[BacktestWindow] = []
    current = start_date

    while current <= end_date:
        for slug, obs in all_obs.items():
            train_obs = [(d, a) for d, a in obs if d < current]
            if not train_obs:
                current = current + timedelta(days=step_days)
                continue

            obs_at_t_rec = max((o for o in train_obs if o[0] <= current), key=lambda o: o[0], default=None)
            if obs_at_t_rec is None:
                continue
            obs_at_t = obs_at_t_rec[1]

            try:
                result = compute_forecast(train_obs, as_of=current)
            except ForecastUnavailableError:
                print("ERROR: prophet not installed. Run: uv pip install 'cryohealth-geo[forecast]'",
                      file=sys.stderr)
                sys.exit(1)

            actual = _closest_within(obs, current + timedelta(days=14))
            actual_t14 = actual[1] if actual is not None else None

            rel_unc = None
            if result.projected_area_14d_p10 is not None and result.projected_area_14d_p50:
                rel_unc = (
                    (result.projected_area_14d_p90 or 0) - (result.projected_area_14d_p10 or 0)
                ) / result.projected_area_14d_p50

            windows.append(BacktestWindow(
                slug=slug,
                as_of=current,
                obs_at_t=obs_at_t,
                actual_t14=actual_t14,
                p50=result.projected_area_14d_p50,
                p10=result.projected_area_14d_p10,
                p90=result.projected_area_14d_p90,
                expansion_probability=result.expansion_probability,
                relative_uncertainty=rel_unc,
                method=result.method,
            ))

        current = current + timedelta(days=step_days)

    return windows


# ---------------------------------------------------------------------------
# Grid evaluation
# ---------------------------------------------------------------------------


def _evaluate_grid(
    windows: list[BacktestWindow],
    prob_sweep: list[float],
    uncertainty_sweep: list[float],
) -> list[GridPoint]:
    """Evaluate all (uncertainty_cutoff × prob_threshold) grid points."""
    grid = []
    for unc_cutoff in uncertainty_sweep:
        for prob_thresh in prob_sweep:
            gp = GridPoint(uncertainty_cutoff=unc_cutoff, prob_threshold=prob_thresh)
            for w in windows:
                if w.actual_t14 is None:
                    continue  # no ground truth — skip
                gp.n_windows += 1

                # Uncertainty gate.
                if w.relative_uncertainty is None or w.relative_uncertainty > unc_cutoff:
                    continue  # rejected
                gp.n_accepted += 1

                # Coverage.
                if w.p10 is not None and w.p90 is not None:
                    if w.p10 <= w.actual_t14 <= w.p90:
                        gp.n_in_interval += 1

                # Confusion matrix.
                actually_expanded = w.actual_t14 > w.obs_at_t
                prob = w.expansion_probability or 0.0
                predicted_elevated = prob >= prob_thresh

                if predicted_elevated and actually_expanded:
                    gp.n_tp += 1
                elif predicted_elevated and not actually_expanded:
                    gp.n_fp += 1
                elif not predicted_elevated and actually_expanded:
                    gp.n_fn += 1
                else:
                    gp.n_tn += 1

                if predicted_elevated:
                    gp.n_elevated += 1

            grid.append(gp)
    return grid


# ---------------------------------------------------------------------------
# Recommendation and boundary detection
# ---------------------------------------------------------------------------


def _recommend(grid: list[GridPoint], prob_sweep: list[float], uncertainty_sweep: list[float]) -> GridPoint | None:
    """Choose the grid point with the best F1, breaking ties by lowest false-alarm rate."""
    eligible = [gp for gp in grid if not np.isnan(gp.f1) and gp.n_accepted > 0]
    if not eligible:
        return None
    return max(eligible, key=lambda gp: (gp.f1, -gp.false_alarm_rate))


def _boundary_warnings(best: GridPoint, prob_sweep: list[float], uncertainty_sweep: list[float]) -> list[str]:
    """Emit boundary warnings if the recommended value is at the edge of the sweep."""
    warnings: list[str] = []
    if best.prob_threshold == min(prob_sweep) or best.prob_threshold == max(prob_sweep):
        warnings.append(
            f"WARNING: recommended prob_threshold={best.prob_threshold} is at the boundary "
            f"of the swept range [{min(prob_sweep)}, {max(prob_sweep)}]. "
            "Widen --prob-sweep before trusting this result."
        )
    if best.uncertainty_cutoff == min(uncertainty_sweep) or best.uncertainty_cutoff == max(uncertainty_sweep):
        warnings.append(
            f"WARNING: recommended uncertainty_cutoff={best.uncertainty_cutoff} is at the "
            f"boundary of the swept range [{min(uncertainty_sweep)}, {max(uncertainty_sweep)}]. "
            "Widen --uncertainty-sweep before trusting this result."
        )
    return warnings


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def _print_table(grid: list[GridPoint]) -> None:
    print(
        f"\n{'unc':>6}  {'prob':>6}  {'n_win':>6}  {'n_acc':>6}  "
        f"{'cov':>6}  {'far':>6}  {'miss':>6}  {'rej':>6}  "
        f"{'prec':>6}  {'rec':>6}  {'F1':>6}"
    )
    print("-" * 90)
    for gp in sorted(grid, key=lambda g: (-g.f1 if not np.isnan(g.f1) else -1)):
        print(
            f"{gp.uncertainty_cutoff:6.2f}  {gp.prob_threshold:6.2f}  "
            f"{gp.n_windows:6d}  {gp.n_accepted:6d}  "
            f"{gp.empirical_coverage:6.3f}  {gp.false_alarm_rate:6.3f}  "
            f"{gp.miss_rate:6.3f}  {gp.rejection_rate:6.3f}  "
            f"{gp.precision:6.3f}  {gp.recall:6.3f}  {gp.f1:6.3f}"
        )


def _write_csv(grid: list[GridPoint], path: str) -> None:
    metric_fields = [
        "uncertainty_cutoff", "prob_threshold", "n_windows", "n_accepted",
        "n_elevated", "n_tp", "n_fp", "n_fn", "n_tn",
        "empirical_coverage", "false_alarm_rate", "miss_rate",
        "rejection_rate", "precision", "recall", "f1",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=metric_fields)
        w.writeheader()
        for gp in grid:
            w.writerow({k: getattr(gp, k) for k in metric_fields})
    print(f"\nResults written to: {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Walk-forward backtest to calibrate forecast.py thresholds."
    )
    parser.add_argument("--start-date", required=True, help="ISO date, e.g. 2023-01-01")
    parser.add_argument("--end-date", required=True, help="ISO date, e.g. 2026-01-01")
    parser.add_argument("--step-days", type=int, default=14)
    parser.add_argument(
        "--prob-sweep",
        default="0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,0.99",
        help="Comma-separated probability thresholds to sweep.",
    )
    parser.add_argument(
        "--uncertainty-sweep",
        default="0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60",
        help="Comma-separated relative uncertainty cutoffs to sweep.",
    )
    parser.add_argument("--output", default="calibration_results.csv")
    parser.add_argument(
        "--slugs",
        default="shishper,khurdopin,badswat,passu,ghulkin,batura",
        help="Comma-separated lake slugs to include.",
    )
    args = parser.parse_args()

    start = date.fromisoformat(args.start_date)
    end = date.fromisoformat(args.end_date)
    prob_sweep = [float(x) for x in args.prob_sweep.split(",")]
    unc_sweep = [float(x) for x in args.uncertainty_sweep.split(",")]
    slugs = [s.strip() for s in args.slugs.split(",")]

    print(f"Fetching observations for {slugs} ...")
    all_obs = _fetch_all_observations(slugs)

    total_obs = sum(len(v) for v in all_obs.values())
    print(f"Loaded {total_obs} observations across {len(slugs)} lakes.")

    print(f"\nRunning walk-forward backtest: {start} → {end}, step={args.step_days} days ...")
    print("(This calls Prophet.fit() for each lake × date step — expect ~15–40 minutes)")
    windows = _run_walk_forward(all_obs, start, end, args.step_days)

    evaluable = sum(1 for w in windows if w.actual_t14 is not None)
    print(f"\n{len(windows)} windows generated, {evaluable} with ground truth (actual T+14 within 15 days).")

    grid = _evaluate_grid(windows, prob_sweep, unc_sweep)
    _print_table(grid)
    _write_csv(grid, args.output)

    best = _recommend(grid, prob_sweep, unc_sweep)
    print("\n" + "=" * 60)
    if best is None:
        print("No eligible grid point found. Check data coverage and sweep ranges.")
        return

    print("RECOMMENDED THRESHOLDS:")
    print(f"  uncertainty_cutoff   = {best.uncertainty_cutoff}")
    print(f"  prob_threshold       = {best.prob_threshold}")
    print(f"  F1={best.f1:.3f}  precision={best.precision:.3f}  recall={best.recall:.3f}")
    print(f"  false_alarm_rate={best.false_alarm_rate:.3f}  miss_rate={best.miss_rate:.3f}")
    print(f"  empirical_coverage={best.empirical_coverage:.3f}  rejection_rate={best.rejection_rate:.3f}")
    print(f"  n_windows={best.n_windows}  n_accepted={best.n_accepted}")

    for warning in _boundary_warnings(best, prob_sweep, unc_sweep):
        print(f"\n{warning}")

    print(
        "\nNext step: update EXPANSION_PROB_THRESHOLD and UNCERTAINTY_CUTOFF in "
        f"pipeline/forecast.py with the values above.\n"
        f"Add a comment with: run date, output file '{args.output}', and the metric values printed above."
    )


if __name__ == "__main__":
    main()
