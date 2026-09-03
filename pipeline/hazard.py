"""Composite hazard index (PRD §8). Pure calculation functions here take already-fetched
data and contain no I/O — same split as ndwi.py — so the methodology can be unit-tested
without a database, DEM, or network. See docs/HAZARD_METHODOLOGY.md for the weights and
thresholds below, written in prose, and kept in lockstep with this file: change one,
change the other.

Every score's `components` dict is deliberately verbose — raw inputs, normalized risk
values, weights, and thresholds all included — so a stored score is reproducible from
what's in the database alone (PRD §9 R3), without needing this file's source at all.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta

from pipeline.meteo import MeteoInputs

logger = logging.getLogger(__name__)

METHODOLOGY_VERSION = "1.1"

# Must sum to 1.0 — enforced below. Proportional trimming from v1.0 (all 6 original
# weights scaled by 0.90) makes room for the 0.10 thermal melting component (Issue #16),
# preserving the relative ordering of all existing risk signals.
WEIGHTS: dict[str, float] = {
    "area_growth": 0.27,
    "historical_glof": 0.135,
    "dam_type": 0.18,
    "glacier_contact": 0.09,
    "slope": 0.09,
    "seasonal_anomaly": 0.135,
    "thermal": 0.10,
}
assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9, "WEIGHTS must sum to 1.0"

# (score >= threshold) -> tier, checked highest-first. Initial, documented starting
# points for a prototype methodology (docs/HAZARD_METHODOLOGY.md) — not yet calibrated
# against real GLOF event outcomes, since none of the 6 monitored lakes has had one in
# the observed period. Revisit once there's a labeled event to calibrate against.
TIER_THRESHOLDS: list[tuple[float, str]] = [
    (0.65, "critical"),
    (0.45, "high"),
    (0.25, "watch"),
    (0.0, "normal"),
]

DAM_TYPE_RISK: dict[str, float] = {"moraine": 1.0, "ice": 0.8, "unknown": 0.5, "bedrock": 0.2}

# All valid dam type values — anything else is a data quality issue and will be logged.
KNOWN_DAM_TYPES: frozenset[str] = frozenset(DAM_TYPE_RISK)


def _dam_type_risk(dam_type: str) -> float:
    """Returns the risk value for dam_type, logging a WARNING if it is not a known
    type so data quality problems (e.g. capitalisation typos) surface in logs rather
    than disappearing silently into the 'unknown' fallback."""
    if dam_type not in KNOWN_DAM_TYPES:
        logger.warning(
            "Unrecognised damType=%r — scoring as 'unknown' (%.1f). "
            "Known types: %s",
            dam_type,
            DAM_TYPE_RISK["unknown"],
            sorted(KNOWN_DAM_TYPES),
        )
    return DAM_TYPE_RISK.get(dam_type, DAM_TYPE_RISK["unknown"])

# Growth of this fraction (50%) or more over the window is treated as maximum risk for
# that component; growth is clamped to [0, GROWTH_SATURATION] before normalizing to
# [0, 1]. Only growth is treated as risk-additive — a shrinking lake isn't scored as
# safer, but isn't scored as more hazardous either under this composite. Rapid shrink can
# mean an active drainage event already underway, which is a different, more urgent
# signal than this precursor index is designed to catch; flagged as a known limitation,
# not solved here.
GROWTH_SATURATION = 0.5
SEASONAL_SATURATION = 0.3
OBSERVATION_TOLERANCE_DAYS = 15

# Saturation thresholds for thermal melting risk (Issue #16).
# 3-day max-temp anomaly: +4°C above 14-year ERA5 seasonal mean reaches maximum risk.
T_SAT = 4.0
# 14-day cumulative rainfall: 80 mm (heavy monsoon convective rainfall) saturates risk.
P_SAT = 80.0
# 0°C isotherm: altitudes from 3500 m (valley glaciers) to 5500 m (glacier accumulation zones).
FL_BASE = 3500.0
FL_SAT = 5500.0

THERMAL_SUB_WEIGHTS: dict[str, float] = {
    "temp_anomaly": 0.5,
    "precip": 0.3,
    "freezing_level": 0.2,
}
assert abs(sum(THERMAL_SUB_WEIGHTS.values()) - 1.0) < 1e-9, "THERMAL_SUB_WEIGHTS must sum to 1.0"


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def thermal_risk(meteo: MeteoInputs | None) -> float:
    """Normalized thermal melting risk in [0, 1].

    Combines 3-day temperature anomaly, 14-day cumulative rainfall,
    and 0°C isotherm altitude.

    Returns 0.0 if meteo is None (graceful degradation — missing weather
    contributes zero risk, never fabricates or fails).
    """
    if meteo is None:
        return 0.0

    t_risk = _clamp01(max(meteo.max_temp_anomaly_3d_c, 0.0) / T_SAT)
    p_risk = _clamp01(max(meteo.precip_14d_mm, 0.0) / P_SAT)
    fl_risk = _clamp01((meteo.freezing_level_m - FL_BASE) / (FL_SAT - FL_BASE))

    return _clamp01(
        THERMAL_SUB_WEIGHTS["temp_anomaly"] * t_risk
        + THERMAL_SUB_WEIGHTS["precip"] * p_risk
        + THERMAL_SUB_WEIGHTS["freezing_level"] * fl_risk
    )


def _latest_on_or_before(observations: list[tuple[date, float]], as_of: date) -> tuple[date, float] | None:
    candidates = [o for o in observations if o[0] <= as_of]
    return max(candidates, key=lambda o: o[0]) if candidates else None


def _closest_within(
    observations: list[tuple[date, float]], target: date, tolerance_days: int
) -> tuple[date, float] | None:
    within = [o for o in observations if abs((o[0] - target).days) <= tolerance_days]
    return min(within, key=lambda o: abs((o[0] - target).days)) if within else None


def growth_rate(
    observations: list[tuple[date, float]],
    as_of: date,
    window_days: int,
    tolerance_days: int = OBSERVATION_TOLERANCE_DAYS,
) -> float | None:
    """Fractional area change from the observation closest to (as_of - window_days) to
    the latest observation on/before as_of. None if either endpoint isn't available
    within tolerance — this composite never fabricates a signal from missing data."""
    latest = _latest_on_or_before(observations, as_of)
    if latest is None:
        return None
    past = _closest_within(observations, as_of - timedelta(days=window_days), tolerance_days)
    if past is None or past[1] == 0:
        return None
    return (latest[1] - past[1]) / past[1]


def seasonal_anomaly(observations: list[tuple[date, float]], as_of: date) -> float | None:
    """Fractional deviation of the latest area from the mean area observed in the same
    calendar month in other years. None without at least one other year's data for that
    month — a single-year record has no seasonal baseline to compare against."""
    latest = _latest_on_or_before(observations, as_of)
    if latest is None:
        return None
    same_month = [area for d, area in observations if d.month == latest[0].month and d.year != latest[0].year]
    if not same_month:
        return None
    baseline = sum(same_month) / len(same_month)
    if baseline == 0:
        return None
    return (latest[1] - baseline) / baseline


def tier_for_score(score: float) -> str:
    for threshold, tier in TIER_THRESHOLDS:
        if score >= threshold:
            return tier
    return "normal"  # unreachable: TIER_THRESHOLDS always has a 0.0 floor


@dataclass(frozen=True)
class LakeStaticInputs:
    dam_type: str
    glacier_contact: bool
    historical_glof: bool


@dataclass(frozen=True)
class HazardResult:
    score: float
    tier: str
    components: dict


def compute_hazard_score(
    observations: list[tuple[date, float]],
    static: LakeStaticInputs,
    slope_deg: float,
    as_of: date,
    exposure: dict | None = None,
    meteo: MeteoInputs | None = None,
) -> HazardResult:
    growth_30d = growth_rate(observations, as_of, 30)
    growth_90d = growth_rate(observations, as_of, 90)
    anomaly = seasonal_anomaly(observations, as_of)

    # Missing signals contribute zero risk rather than being excluded from the weighted
    # sum — excluding them would silently redistribute their weight onto whatever data
    # happens to be available, which is worse than a documented, visible zero.
    risks = {
        "area_growth": _clamp01(
            0.5 * (max(growth_30d, 0.0) if growth_30d is not None else 0.0) / GROWTH_SATURATION
            + 0.5 * (max(growth_90d, 0.0) if growth_90d is not None else 0.0) / GROWTH_SATURATION
        ),
        "seasonal_anomaly": _clamp01(
            (max(anomaly, 0.0) if anomaly is not None else 0.0) / SEASONAL_SATURATION
        ),
        "dam_type": _dam_type_risk(static.dam_type),
        "glacier_contact": 1.0 if static.glacier_contact else 0.3,
        "slope": _clamp01(slope_deg / 40.0),
        "historical_glof": 1.0 if static.historical_glof else 0.0,
        "thermal": thermal_risk(meteo),
    }
    score = sum(WEIGHTS[k] * risks[k] for k in WEIGHTS)
    tier = tier_for_score(score)

    components = {
        "methodology_version": METHODOLOGY_VERSION,
        "as_of": as_of.isoformat(),
        "weights": WEIGHTS,
        "tier_thresholds": TIER_THRESHOLDS,
        "raw_inputs": {
            "growth_30d_pct": growth_30d,
            "growth_90d_pct": growth_90d,
            "seasonal_anomaly_pct": anomaly,
            "dam_type": static.dam_type,
            "glacier_contact": static.glacier_contact,
            "slope_deg": slope_deg,
            "historical_glof": static.historical_glof,
            "meteo": {
                "max_temp_anomaly_3d_c": meteo.max_temp_anomaly_3d_c,
                "precip_14d_mm": meteo.precip_14d_mm,
                "freezing_level_m": meteo.freezing_level_m,
                "source": meteo.source,
                "fetched_at": meteo.fetched_at,
            }
            if meteo is not None
            else None,
        },
        "normalized_risks": risks,
        "exposure": exposure,
    }
    return HazardResult(score=score, tier=tier, components=components)
