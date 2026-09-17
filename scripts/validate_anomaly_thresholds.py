"""Walk-forward hold-out validation for anomaly detection thresholds (Issue #22).

Fits Isolation Forests on an early training window (2023-01-01 to 2025-07-01) and
scores a hold-out period (2025-07-01 to 2026-01-01). Reports false-positive rate (FPR)
at candidate sigma thresholds from 2σ to 5σ.

The "false positive" definition: a clean historical scene (from the hold-out period,
no known seepage event) scored as anomalous (sigma < -threshold). Since all six lakes
are in the historical baseline period (no confirmed GLOF events in 2023–2026),
every flagged scene is a false positive. The acceptance criterion is FPR < 2%.

Usage:
    uv run python scripts/validate_anomaly_thresholds.py \\
        --train-start 2023-01-01 \\
        --train-end 2025-07-01 \\
        --holdout-start 2025-07-01 \\
        --holdout-end 2026-01-01 \\
        --output validation_results.csv \\
        --sigma-range 2.0,2.5,3.0,3.5,4.0,4.5,5.0

After running:
    1. Check the boundary-detection warning: if recommended threshold is at the sweep
       boundary, widen --sigma-range before trusting the result.
    2. Update SEEPAGE_SIGMA_THRESHOLD in pipeline/anomaly.py with the recommended value.
    3. Add a provenance comment: run date, output CSV filename, FPR at chosen threshold.
    4. Do the same for SUDDEN_DRAINAGE_DELTA using the drainage FPR columns.

Requires:
    - [anomaly] extras: uv pip install 'cryohealth-geo[anomaly]'
    - SceneSource access (Planetary Computer, no auth required)
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

# Ensure repo root is on sys.path when script is invoked directly
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


@dataclass
class ValidationPoint:
    slug: str
    sigma_threshold: float
    n_holdout_scenes: int
    n_flagged_seepage: int
    n_flagged_drainage: int

    @property
    def seepage_fpr(self) -> float:
        return self.n_flagged_seepage / self.n_holdout_scenes if self.n_holdout_scenes else float("nan")

    @property
    def drainage_fpr(self) -> float:
        return self.n_flagged_drainage / self.n_holdout_scenes if self.n_holdout_scenes else float("nan")


def _check_deps() -> None:
    import importlib.util
    missing = [name for name in ["sklearn", "joblib"] if importlib.util.find_spec(name) is None]
    if missing:
        print(f"ERROR: missing [anomaly] extras: {missing}", file=sys.stderr)
        print("Install with: uv pip install 'cryohealth-geo[anomaly]'", file=sys.stderr)
        sys.exit(1)


def _validate_lake(
    slug: str,
    dam_face_bbox: tuple[float, float, float, float],
    train_start: date,
    train_end: date,
    holdout_start: date,
    holdout_end: date,
    sigma_thresholds: list[float],
    n_estimators: int = 100,
    contamination: float = 0.05,
) -> list[ValidationPoint]:
    """Fit on training window, score holdout, report per-threshold FPR."""
    import numpy as np  # noqa: PLC0415
    from sklearn.ensemble import IsolationForest  # noqa: PLC0415
    import warnings  # noqa: PLC0415
    from pipeline.anomaly import (  # noqa: PLC0415
        ANOMALY_BANDS,
        CLOUD_MAX_FRACTION,
        MIN_TRAINING_SCENES,
        SUDDEN_DRAINAGE_DELTA,
        extract_spectral_vectors,
        score_to_sigma,
    )
    from pipeline.stac_source import PlanetaryComputerSource  # noqa: PLC0415

    source = PlanetaryComputerSource()

    def _fetch_clean_vectors(start: date, end: date, max_scenes: int | None = None) -> list[tuple[date, np.ndarray]]:
        try:
            scenes = source.find_scenes_in_range(dam_face_bbox, start, end)
        except Exception as exc:  # noqa: BLE001
            print(f"  WARNING {slug}: scene fetch {start}->{end} failed: {exc}", flush=True)
            return []
        result = []
        for scene in sorted(scenes, key=lambda s: s.captured_at):
            # Step 1: Read SCL band only to filter cloudy scenes quickly
            try:
                scl_dict = source.read_bands(scene, ["SCL"], dam_face_bbox)
            except Exception:
                continue

            if "SCL" in scl_dict:
                import numpy as np  # noqa: PLC0415
                scl = scl_dict["SCL"]
                cf = float(np.isin(scl, [3, 8, 9, 10]).sum() / scl.size) if scl.size > 0 else 0.0
                if cf > CLOUD_MAX_FRACTION:
                    continue
                cloud_mask = np.isin(scl, [3, 8, 9, 10])
            else:
                cf = 0.0
                cloud_mask = None

            # Step 2: Scene is clear over dam-face AOI — read spectral bands
            try:
                bands = source.read_bands(scene, ANOMALY_BANDS, dam_face_bbox)
            except Exception:
                continue

            vecs = extract_spectral_vectors(bands, cloud_mask)
            if vecs.shape[0] == 0:
                continue
            result.append((scene.captured_at, vecs.mean(axis=0)))
            print(f"    [{slug}] Clean scene #{len(result)}: {scene.captured_at} (cloud {cf:.1%})", flush=True)
            if max_scenes and len(result) >= max_scenes:
                break
        return result

    print(f"  [{slug}] Fetching training scenes {train_start}->{train_end} ...", flush=True)
    train_scenes = _fetch_clean_vectors(train_start, train_end, max_scenes=30)
    print(f"  [{slug}] {len(train_scenes)} training scenes.", flush=True)

    if len(train_scenes) < MIN_TRAINING_SCENES:
        print(f"  SKIPPED {slug}: insufficient training scenes ({len(train_scenes)} < {MIN_TRAINING_SCENES})", flush=True)
        return []

    X_train = np.array([v for _, v in train_scenes], dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = IsolationForest(n_estimators=n_estimators, contamination=contamination, random_state=42)
        model.fit(X_train)

    train_scores = model.score_samples(X_train)
    training_mean = float(train_scores.mean())
    training_std = float(train_scores.std())

    print(f"  [{slug}] Fetching holdout scenes {holdout_start}->{holdout_end} ...", flush=True)
    holdout_scenes = _fetch_clean_vectors(holdout_start, holdout_end, max_scenes=15)
    print(f"  [{slug}] {len(holdout_scenes)} holdout scenes.", flush=True)

    if not holdout_scenes:
        return []

    points = []
    for sigma_thresh in sigma_thresholds:
        n_seepage = 0
        n_drainage = 0

        for i, (d, vec) in enumerate(holdout_scenes):
            raw_score = float(model.score_samples(vec.reshape(1, -1))[0])
            sigma = score_to_sigma(raw_score, training_mean, training_std)
            if sigma < -sigma_thresh:
                n_seepage += 1

            # Drainage check: MNDWI delta from ~30d prior.
            target_past = d - timedelta(days=30)
            past = [(pd, pv) for pd, pv in holdout_scenes[:i] if abs((pd - target_past).days) <= 15]
            if past:
                past_d, past_vec = min(past, key=lambda x: abs((x[0] - target_past).days))
                delta = float(vec[4]) - float(past_vec[4])  # MNDWI index = 4
                if delta < SUDDEN_DRAINAGE_DELTA:
                    n_drainage += 1

        points.append(ValidationPoint(
            slug=slug,
            sigma_threshold=sigma_thresh,
            n_holdout_scenes=len(holdout_scenes),
            n_flagged_seepage=n_seepage,
            n_flagged_drainage=n_drainage,
        ))
        print(
            f"    sigma={sigma_thresh:.1f}: seepage_FPR={n_seepage/len(holdout_scenes):.3f} "
            f"({n_seepage}/{len(holdout_scenes)}), "
            f"drainage_FPR={n_drainage/len(holdout_scenes):.3f} ({n_drainage}/{len(holdout_scenes)})"
        )
    return points


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hold-out validation for anomaly detection thresholds (Issue #22)."
    )
    parser.add_argument("--train-start", default="2023-01-01")
    parser.add_argument("--train-end", default="2025-07-01")
    parser.add_argument("--holdout-start", default="2025-07-01")
    parser.add_argument("--holdout-end", default="2026-01-01")
    parser.add_argument("--sigma-range", default="2.0,2.5,3.0,3.5,4.0,4.5,5.0")
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--contamination", type=float, default=0.05)
    parser.add_argument("--output", default="validation_results.csv")
    args = parser.parse_args()

    _check_deps()

    from pipeline.lakes import LAKES  # noqa: PLC0415

    train_start = date.fromisoformat(args.train_start)
    train_end = date.fromisoformat(args.train_end)
    holdout_start = date.fromisoformat(args.holdout_start)
    holdout_end = date.fromisoformat(args.holdout_end)
    sigma_thresholds = [float(x) for x in args.sigma_range.split(",")]

    applicable = {slug: lake for slug, lake in LAKES.items() if lake.dam_face_bbox_deg is not None}
    if not applicable:
        print("No lakes with dam_face_bbox_deg set. Digitize bboxes in lakes.py first.")
        sys.exit(0)

    print(f"Validating {len(applicable)} lake(s): {list(applicable)}")
    print(f"Training: {train_start} -> {train_end}")
    print(f"Holdout:  {holdout_start} -> {holdout_end}")
    print(f"Sigma thresholds: {sigma_thresholds}\n")

    all_points: list[ValidationPoint] = []
    for slug, lake in applicable.items():
        pts = _validate_lake(
            slug, lake.dam_face_bbox_deg,
            train_start, train_end,
            holdout_start, holdout_end,
            sigma_thresholds,
            args.n_estimators, args.contamination,
        )
        all_points.extend(pts)

    # Aggregate FPR across all lakes per sigma threshold.
    print("\n" + "=" * 60)
    print(f"{'sigma':>6}  {'seepage_FPR':>12}  {'drainage_FPR':>13}  {'status':>10}")
    print("-" * 50)

    recommended_sigma: float | None = None
    for thresh in sigma_thresholds:
        pts = [p for p in all_points if p.sigma_threshold == thresh]
        if not pts:
            continue
        total = sum(p.n_holdout_scenes for p in pts)
        total_seepage = sum(p.n_flagged_seepage for p in pts)
        total_drainage = sum(p.n_flagged_drainage for p in pts)
        seepage_fpr = total_seepage / total if total else float("nan")
        drainage_fpr = total_drainage / total if total else float("nan")
        meets_criterion = seepage_fpr < 0.02
        status = "PASS" if meets_criterion else "FAIL"
        print(f"{thresh:6.1f}  {seepage_fpr:12.4f}  {drainage_fpr:13.4f}  {status}")
        if meets_criterion and recommended_sigma is None:
            recommended_sigma = thresh

    print()
    if recommended_sigma is not None:
        print(f"RECOMMENDED SEEPAGE_SIGMA_THRESHOLD = {recommended_sigma}")
        # Boundary detection (same guard as calibrate_forecast_thresholds.py).
        if recommended_sigma == min(sigma_thresholds) or recommended_sigma == max(sigma_thresholds):
            print(
                f"WARNING: recommended threshold {recommended_sigma} is at the boundary of the "
                f"swept range [{min(sigma_thresholds)}, {max(sigma_thresholds)}]. "
                "Widen --sigma-range before trusting this result."
            )
        print(
            f"\nNext step: update SEEPAGE_SIGMA_THRESHOLD = {recommended_sigma} in "
            "pipeline/anomaly.py with a provenance comment:\n"
            f"  # Validated against {args.output}, run <DATE>, seepage FPR < 2% at this threshold."
        )
    else:
        print(
            "No threshold in the swept range achieves FPR < 2%. "
            "Try: wider sigma range, stricter cloud threshold, or more training data."
        )

    # Write CSV.
    with open(args.output, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["slug", "sigma_threshold", "n_holdout_scenes",
                    "n_flagged_seepage", "seepage_fpr", "n_flagged_drainage", "drainage_fpr"])
        for p in all_points:
            w.writerow([
                p.slug, p.sigma_threshold, p.n_holdout_scenes,
                p.n_flagged_seepage, round(p.seepage_fpr, 4),
                p.n_flagged_drainage, round(p.drainage_fpr, 4),
            ])
    print(f"\nResults written to: {args.output}")


if __name__ == "__main__":
    main()
