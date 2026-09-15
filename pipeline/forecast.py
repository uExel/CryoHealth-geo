"""14-day lake surface area forecast using Prophet time-series modelling.

Implements: Issue #21 (https://github.com/uExel/CryoHealth-geo/issues/21)
ADR: docs/ai/decisions/0004-forecast-separation.md

Architecture summary: this module is COMPLETELY INDEPENDENT of pipeline/hazard.py.
compute_hazard_score() is not imported here, not called here, and not modified by
this PR. The forecast is an advisory signal computed AFTER compute_hazard_score() in
hazard_batch.py, never an input to it. See ADR 0004 for the double-counting and
auditability rationale.

Prophet fit dependency:
  Prophet is an optional [forecast] extras group — install with:
    uv pip install 'cryohealth-geo[forecast]'
  Base install, NDWI, SAR, and hazard scoring are fully functional without it.
  compute_forecast() raises ForecastUnavailableError (not ImportError) when prophet
  is absent — callers catch that specific exception and log a WARNING.

Exception contract:
  - Raises ForecastUnavailableError: ONLY if 'prophet' is not installed.
    This is a deployment/environment failure, not a data failure.
  - Returns ForecastResult(method="insufficient_confidence"): for ALL other failures
    (bad data, Prophet fit error, numerical issues). Never re-raises those.

Threshold provenance:
  EXPANSION_PROB_THRESHOLD and UNCERTAINTY_CUTOFF are INITIAL CANDIDATES, not
  empirically justified values. Both must be updated from the output of
  scripts/calibrate_forecast_thresholds.py before this module is relied upon
  in production. See the comment on each constant.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import date, timedelta
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd  # only for type hints — pandas is a prophet transitive dep

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thresholds — BOTH PENDING CALIBRATION
# Run scripts/calibrate_forecast_thresholds.py against the 2023–2026 backfill,
# then update these constants with the recommended values. Include the calibration
# run date and output CSV filename in the comment (same provenance discipline as
# the SAR -14dB/-21dB thresholds and AlphaEarth training data provenance).
# ---------------------------------------------------------------------------

EXPANSION_PROB_THRESHOLD = 0.85
"""Posterior probability gate: P(Area(T+14) > current Area) >= this to flag elevated risk.

PENDING CALIBRATION — initial candidate 0.85. Justification from backtest data required
(Issue #21 acceptance criterion). Swept 0.50–0.99 in calibrate_forecast_thresholds.py.
"""

UNCERTAINTY_CUTOFF = 0.35
"""Relative uncertainty cutoff: (p90 - p10) / p50. Forecasts exceeding this are discarded
as 'insufficient_confidence' — the prediction interval is too wide to be actionable.

PENDING CALIBRATION — initial candidate 0.35. Swept 0.15–0.60 in calibrate_forecast_thresholds.py.
"""

# ---------------------------------------------------------------------------
# Data-gate thresholds (not calibration-dependent — these are engineering minimums)
# ---------------------------------------------------------------------------

MIN_OBSERVATIONS = 15
"""Minimum number of observations required to attempt a Prophet fit.
Below this: linear 30-day moving-average fallback."""

MIN_HISTORY_DAYS = 180
"""Minimum history span (days from first to latest observation) for a Prophet fit.
Below this: linear 30-day moving-average fallback. 180 days ensures at least one
seasonal half-cycle is visible, giving Prophet's yearly component something to fit."""

FORECAST_HORIZON_DAYS = 14
"""Days ahead for the forecast target. T+14 gives responders lead time before a
lake reaches a potential overtopping volume."""

PROPHET_FIT_WARN_SECONDS = 10.0
"""Wall-clock monitoring threshold for Prophet .fit() calls (time.perf_counter()).
This does NOT abort or interrupt the fit — Prophet's Stan backend is a blocking
C-extension call that cannot be cleanly interrupted from Python without a subprocess.
A fit exceeding this threshold logs a WARNING (monitoring signal) but its result
is still used. Named 'warn' not 'timeout' because it does not enforce a timeout.
"""


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ForecastUnavailableError(RuntimeError):
    """Raised ONLY when the 'prophet' package is not installed.

    This is a deployment/environment failure, not a data failure.
    Callers (pipeline/hazard_batch.py) catch this and log a WARNING;
    the hazard batch continues without a forecast for the affected lake.

    Install with: uv pip install 'cryohealth-geo[forecast]'
    """


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ForecastResult:
    """Output of compute_forecast(). Always returned; never raises on data failure.

    method values:
      "prophet"                — Prophet fit succeeded and passed all gates
      "linear_fallback"        — insufficient data for Prophet; 30d linear extrapolation
      "insufficient_confidence"— Prophet fit succeeded but uncertainty too high, OR
                                 Prophet raised an exception during fitting
      "insufficient_data"      — fewer than MIN_OBSERVATIONS or less than MIN_HISTORY_DAYS
    """
    method: str
    expansion_probability: float | None   # P(Area(T+14) > current_area_km2)
    projected_area_14d_p10: float | None  # km² — 10th percentile
    projected_area_14d_p50: float | None  # km² — 50th percentile (median)
    projected_area_14d_p90: float | None  # km² — 90th percentile
    current_area_km2: float | None        # latest observed area, km²
    forecast_date: str                    # ISO date of the T+14 target
    as_of: str                            # ISO date of the latest observation used
    observation_count: int                # number of observations fed to the model
    history_days: int                     # span from first to latest observation
    elevated_risk: bool = False           # True iff method="prophet" AND
                                          # expansion_probability >= EXPANSION_PROB_THRESHOLD
    note: str = ""                        # human-readable summary of gate outcome


# ---------------------------------------------------------------------------
# Pure math helpers (no I/O — testable with synthetic data)
# ---------------------------------------------------------------------------


def _check_data_gate(
    observations: list[tuple[date, float]],
) -> tuple[int, int]:
    """Return (observation_count, history_days) for the gate check.
    Raises nothing — just returns the counts; caller decides on fallback."""
    n = len(observations)
    if n < 2:
        return n, 0
    dates = [o[0] for o in observations]
    span = (max(dates) - min(dates)).days
    return n, span


def _linear_fallback(
    observations: list[tuple[date, float]],
    as_of: date,
    horizon: int = FORECAST_HORIZON_DAYS,
) -> ForecastResult:
    """30-day linear moving-average extrapolation for sparse-data lakes.

    Computes the slope of the last 30 days of observations (or all available if fewer)
    and extrapolates forward by `horizon` days. Uncertainty bounds are set to ±20%
    of the projected value as a fixed-width interval (not a real statistical bound —
    explicitly documented as such in the note field).
    """
    n, span = _check_data_gate(observations)
    latest_date, latest_area = max(observations, key=lambda o: o[0])
    forecast_date = as_of + timedelta(days=horizon)

    # Use last 30 days of observations for slope estimation.
    cutoff = latest_date - timedelta(days=30)
    recent = [(d, a) for d, a in observations if d >= cutoff]
    if len(recent) < 2:
        recent = sorted(observations, key=lambda o: o[0])[-2:]

    # Linear slope: Δarea / Δdays
    dates_numeric = [(d - recent[0][0]).days for d, _ in recent]
    areas = [a for _, a in recent]
    if len(set(dates_numeric)) < 2 or max(dates_numeric) == 0:
        slope = 0.0
    else:
        slope = float(np.polyfit(dates_numeric, areas, 1)[0])

    proj = max(0.0, latest_area + slope * horizon)
    margin = proj * 0.20  # ±20% fixed-width interval — not a statistical bound

    return ForecastResult(
        method="linear_fallback",
        expansion_probability=float(np.clip(0.5 + slope * 10, 0.0, 1.0)),  # heuristic
        projected_area_14d_p10=max(0.0, proj - margin),
        projected_area_14d_p50=proj,
        projected_area_14d_p90=proj + margin,
        current_area_km2=latest_area,
        forecast_date=forecast_date.isoformat(),
        as_of=latest_date.isoformat(),
        observation_count=n,
        history_days=span,
        note=(
            f"Linear 30-day moving-average fallback (<{MIN_OBSERVATIONS} observations "
            f"or <{MIN_HISTORY_DAYS} days history). Bounds are ±20% of projection, "
            "not a statistical confidence interval."
        ),
    )


def _preprocess(observations: list[tuple[date, float]]) -> "pd.DataFrame":
    """Convert observations to Prophet's (ds, y) DataFrame.

    - Clips area to ≥ 0 (negative values are sensor artefacts)
    - Deduplicates dates: keeps the last value per date (most recent reprocessing)
    - Sorts by date ascending (Prophet requirement)
    """
    import pandas as pd  # noqa: PLC0415 — prophet transitive dep, import here

    deduped: dict[date, float] = {}
    for d, area in observations:
        deduped[d] = max(0.0, area)  # last write wins; clip to 0

    return pd.DataFrame(
        [{"ds": pd.Timestamp(d), "y": a} for d, a in sorted(deduped.items())]
    )


def _fit_and_sample(
    df: "pd.DataFrame",
    horizon: int = FORECAST_HORIZON_DAYS,
    n_samples: int = 1000,
) -> dict:
    """Fit a Prophet model and draw posterior samples at T+horizon.

    Prophet model configuration rationale (Karakoram glacial lakes):
      - yearly_seasonality=True: strong annual cycle (monsoon June–Sept, freeze Oct–Mar)
      - weekly_seasonality=False: satellite revisit ≠ weekly; enabling this would fit noise
      - daily_seasonality=False: no sub-daily variation in lake area measurements
      - changepoint_prior_scale=0.05: conservative; lake area is a slow signal with
        sparse observations — a higher prior over-fits to cloud-gap artefacts
      - uncertainty_samples=1000: sufficient for stable P(expansion) estimate

    Returns:
        dict with keys 'p10', 'p50', 'p90' (floats) and 'samples' (1D np.ndarray
        of length n_samples, clipped to ≥0).
    """
    from prophet import Prophet  # noqa: PLC0415 — optional dep

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.05,
        uncertainty_samples=n_samples,
    )

    # Suppress Prophet's verbose Stan output.
    import logging as _logging  # noqa: PLC0415
    _logging.getLogger("prophet").setLevel(_logging.WARNING)
    _logging.getLogger("cmdstanpy").setLevel(_logging.WARNING)

    t0 = time.perf_counter()
    model.fit(df)
    elapsed = time.perf_counter() - t0

    if elapsed > PROPHET_FIT_WARN_SECONDS:
        logger.warning(
            "Prophet .fit() took %.1fs (warn threshold %.0fs). "
            "Fit result still used — this is a monitoring signal, not an abort. "
            "If consistently slow, consider reducing observation lookback or "
            "upgrading to a faster Stan backend.",
            elapsed, PROPHET_FIT_WARN_SECONDS,
        )

    # Build a single future period at T+horizon.
    import pandas as pd  # noqa: PLC0415
    future = pd.DataFrame({"ds": [df["ds"].max() + pd.Timedelta(days=horizon)]})
    forecast = model.predictive_samples(future)

    # predictive_samples returns {'yhat': np.ndarray of shape (1, n_samples)}.
    samples = np.clip(forecast["yhat"][0], 0, None)
    return {
        "p10": float(np.percentile(samples, 10)),
        "p50": float(np.percentile(samples, 50)),
        "p90": float(np.percentile(samples, 90)),
        "samples": samples,
    }


def _expansion_probability(samples: np.ndarray, current_area: float) -> float:
    """P(Area(T+14) > current_area) across the posterior sample array."""
    if len(samples) == 0:
        return 0.0
    return float((samples > current_area).mean())


def _relative_uncertainty(p10: float, p50: float, p90: float) -> float:
    """(p90 - p10) / p50 — relative width of the prediction interval.
    Returns inf when p50 == 0 to guarantee discard on degenerate forecasts."""
    if p50 <= 0:
        return float("inf")
    return (p90 - p10) / p50


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def compute_forecast(
    observations: list[tuple[date, float]],
    as_of: date | None = None,
) -> ForecastResult:
    """Compute a 14-day lake surface area forecast.

    Args:
        observations: List of (date, area_km2) tuples from the DB. Order does not matter.
        as_of: Reference date (latest observation must be on or before this). Defaults to
               the date of the latest observation in the list.

    Returns:
        ForecastResult with method indicating which path ran.

    Raises:
        ForecastUnavailableError: ONLY if the 'prophet' package is not installed.
            Install with: uv pip install 'cryohealth-geo[forecast]'

    All other errors (bad data, Prophet fit failure, numerical issues) are caught
    internally and returned as ForecastResult(method="insufficient_confidence").
    """
    # Dependency check first — fail fast with a clear error, before any data work.
    import importlib.util  # noqa: PLC0415
    if importlib.util.find_spec("prophet") is None:
        raise ForecastUnavailableError(
            "prophet is not installed. "
            "Install with: uv pip install 'cryohealth-geo[forecast]'"
        )

    if not observations:
        return ForecastResult(
            method="insufficient_data",
            expansion_probability=None,
            projected_area_14d_p10=None,
            projected_area_14d_p50=None,
            projected_area_14d_p90=None,
            current_area_km2=None,
            forecast_date=(date.today() + timedelta(days=FORECAST_HORIZON_DAYS)).isoformat(),
            as_of=date.today().isoformat(),
            observation_count=0,
            history_days=0,
            note="No observations available.",
        )

    obs_sorted = sorted(observations, key=lambda o: o[0])
    latest_date, latest_area = obs_sorted[-1]
    effective_as_of = as_of or latest_date
    forecast_date = effective_as_of + timedelta(days=FORECAST_HORIZON_DAYS)

    n, span = _check_data_gate(observations)

    if n < MIN_OBSERVATIONS or span < MIN_HISTORY_DAYS:
        logger.info(
            "compute_forecast: insufficient data gate triggered "
            "(n=%d < %d obs or span=%d < %d days) — linear fallback",
            n, MIN_OBSERVATIONS, span, MIN_HISTORY_DAYS,
        )
        return _linear_fallback(observations, effective_as_of)

    # Prophet path — all errors caught and returned as insufficient_confidence.
    try:
        df = _preprocess(observations)
        result = _fit_and_sample(df, horizon=FORECAST_HORIZON_DAYS)
        p10, p50, p90 = result["p10"], result["p50"], result["p90"]
        samples = result["samples"]

        rel_uncertainty = _relative_uncertainty(p10, p50, p90)
        if rel_uncertainty > UNCERTAINTY_CUTOFF:
            logger.info(
                "compute_forecast: uncertainty %.2f exceeds cutoff %.2f — discarding forecast",
                rel_uncertainty, UNCERTAINTY_CUTOFF,
            )
            return ForecastResult(
                method="insufficient_confidence",
                expansion_probability=None,
                projected_area_14d_p10=p10,
                projected_area_14d_p50=p50,
                projected_area_14d_p90=p90,
                current_area_km2=latest_area,
                forecast_date=forecast_date.isoformat(),
                as_of=latest_date.isoformat(),
                observation_count=n,
                history_days=span,
                note=(
                    f"Forecast discarded: relative uncertainty "
                    f"{rel_uncertainty:.2f} > cutoff {UNCERTAINTY_CUTOFF}. "
                    "UNCERTAINTY_CUTOFF is pending calibration — see Issue #21 AC "
                    "and scripts/calibrate_forecast_thresholds.py."
                ),
            )

        expansion_prob = _expansion_probability(samples, latest_area)
        elevated = expansion_prob >= EXPANSION_PROB_THRESHOLD

        logger.info(
            "compute_forecast: prophet ok — p50=%.4f expansion_prob=%.3f "
            "elevated=%s uncertainty=%.2f",
            p50, expansion_prob, elevated, rel_uncertainty,
        )

        return ForecastResult(
            method="prophet",
            expansion_probability=expansion_prob,
            projected_area_14d_p10=p10,
            projected_area_14d_p50=p50,
            projected_area_14d_p90=p90,
            current_area_km2=latest_area,
            forecast_date=forecast_date.isoformat(),
            as_of=latest_date.isoformat(),
            observation_count=n,
            history_days=span,
            elevated_risk=elevated,
            note=(
                f"Prophet fit: p50={p50:.4f} km², "
                f"expansion_prob={expansion_prob:.3f}, "
                f"elevated_risk={elevated} "
                f"(threshold {EXPANSION_PROB_THRESHOLD} — PENDING CALIBRATION). "
                f"Uncertainty (p90-p10)/p50={rel_uncertainty:.2f} "
                f"(cutoff {UNCERTAINTY_CUTOFF} — PENDING CALIBRATION)."
            ),
        )

    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Prophet fit failed for lake — returning insufficient_confidence: %s", exc
        )
        return ForecastResult(
            method="insufficient_confidence",
            expansion_probability=None,
            projected_area_14d_p10=None,
            projected_area_14d_p50=None,
            projected_area_14d_p90=None,
            current_area_km2=latest_area,
            forecast_date=forecast_date.isoformat(),
            as_of=latest_date.isoformat(),
            observation_count=n,
            history_days=span,
            note=f"Prophet fit raised an exception: {exc!r}",
        )


__all__ = [
    "ForecastResult",
    "ForecastUnavailableError",
    "compute_forecast",
]
