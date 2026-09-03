"""Open-Meteo weather data fetch for hazard scoring thermal component (Issue #16).

Architecture decisions:
- Q1 Endpoints: Open-Meteo forecast (with past_days) for as_of within last 90 days;
     archive/ERA5 for older dates. The 90-day cutoff avoids the ~5-day ERA5 processing
     lag that would cause silent MeteoUnavailableErrors for dates 3–8 days old.
     The 14-year seasonal baseline ALWAYS uses archive regardless of as_of — mixing
     endpoints for the baseline vs. the current reading would produce an inconsistent
     anomaly methodology day-to-day.
- Q2 Cache: in-memory dict, 6-hour TTL. Key includes endpoint type so a boundary date
     near the 90-day cutoff cannot return a stale value from the wrong endpoint.
- Q3 Additive sub-score with proportional trimming in hazard.py.

This module is pure-I/O: all weather risk normalization (thermal_risk, T_SAT, etc.)
lives in hazard.py, keeping the clean separation of I/O and compute.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Endpoint configuration (Q1)
# ---------------------------------------------------------------------------

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/era5"

# as_of dates within this many days of today use the forecast endpoint;
# older dates use the archive/ERA5 endpoint.
FORECAST_CUTOFF_DAYS = 90

# Number of past years used to build the seasonal temperature baseline.
BASELINE_YEARS = 14

# ---------------------------------------------------------------------------
# In-memory cache (Q2)
# ---------------------------------------------------------------------------

_CACHE_TTL_SECONDS = 6 * 3600  # 6-hour TTL per spec

# Two separate caches: one for full MeteoInputs, one for the ERA5 baseline mean
# (which is keyed by (lat, lon, month) and shared across as_of dates in the
# same month, avoiding redundant 14-year archive downloads).
_inputs_cache: dict[tuple, tuple[datetime, MeteoInputs]] = {}
_baseline_cache: dict[tuple, tuple[datetime, float]] = {}


def clear_meteo_cache() -> None:
    """Clear in-memory caches. Intended for testing."""
    _inputs_cache.clear()
    _baseline_cache.clear()


def _cache_get_inputs(key: tuple) -> MeteoInputs | None:
    entry = _inputs_cache.get(key)
    if entry is None:
        return None
    cached_at, value = entry
    if (datetime.now(UTC) - cached_at).total_seconds() > _CACHE_TTL_SECONDS:
        del _inputs_cache[key]
        return None
    return value


def _cache_set_inputs(key: tuple, value: MeteoInputs) -> None:
    _inputs_cache[key] = (datetime.now(UTC), value)


def _cache_get_baseline(key: tuple) -> float | None:
    entry = _baseline_cache.get(key)
    if entry is None:
        return None
    cached_at, value = entry
    if (datetime.now(UTC) - cached_at).total_seconds() > _CACHE_TTL_SECONDS:
        del _baseline_cache[key]
        return None
    return value


def _cache_set_baseline(key: tuple, value: float) -> None:
    _baseline_cache[key] = (datetime.now(UTC), value)


# ---------------------------------------------------------------------------
# Public exception
# ---------------------------------------------------------------------------


class MeteoUnavailableError(RuntimeError):
    """Raised when the Open-Meteo API is unreachable, returns a non-200 status,
    or the response is malformed/missing expected fields.

    Caught *narrowly* in hazard_batch.py so the thermal component gracefully
    degrades to 0.0 without failing the batch run for other failure modes."""


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MeteoInputs:
    """Preprocessed weather metrics for one lake on one date.

    All values are computed from Open-Meteo data and stored verbatim in
    HazardResult.components so a score can be reproduced from the database
    alone (PRD §9 R3).
    """

    max_temp_anomaly_3d_c: float
    """3-day mean daily-max temperature deviation above the 14-year ERA5
    same-calendar-month baseline (°C). Positive -> warmer than seasonal norm."""

    precip_14d_mm: float
    """14-day cumulative precipitation ending on as_of (mm)."""

    freezing_level_m: float
    """Mean 0°C isotherm altitude over the 3-day window (m above sea level).
    Derived from Open-Meteo's freezinglevel_height hourly variable where
    available; falls back to a temperature-based lapse-rate estimate otherwise."""

    source: str
    """'open-meteo-forecast' or 'open-meteo-archive'."""

    fetched_at: str
    """ISO 8601 UTC timestamp of the API call — stored for provenance."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _endpoint_type(as_of: date) -> str:
    """Select endpoint per Q1 decision: forecast for <=90 days old, archive otherwise."""
    return "forecast" if (date.today() - as_of).days <= FORECAST_CUTOFF_DAYS else "archive"


def _get_json(url: str, params: dict, timeout: int) -> dict:
    """HTTP GET -> parsed JSON. Raises MeteoUnavailableError on any failure."""
    try:
        resp = requests.get(url, params=params, timeout=timeout)
    except requests.RequestException as exc:
        raise MeteoUnavailableError(f"Open-Meteo request failed: {exc}") from exc
    if resp.status_code != 200:
        raise MeteoUnavailableError(
            f"Open-Meteo returned HTTP {resp.status_code}: {resp.text[:200]}"
        )
    try:
        return resp.json()
    except ValueError as exc:
        raise MeteoUnavailableError(f"Open-Meteo response is not valid JSON: {exc}") from exc


def _extract_daily(data: dict, key: str) -> list[float]:
    """Pull a named daily variable from the Open-Meteo response; raise on missing key."""
    try:
        raw = data["daily"][key]
    except KeyError as exc:
        raise MeteoUnavailableError(
            f"Missing field 'daily.{key}' in Open-Meteo response"
        ) from exc
    return [float(v) for v in raw if v is not None]


def _extract_hourly(data: dict, key: str) -> list[float]:
    """Pull a named hourly variable; returns [] (not an error) if absent — freezing level
    is best-effort and falls back to the lapse-rate estimate when missing."""
    try:
        raw = data["hourly"][key]
    except KeyError:
        return []
    return [float(v) for v in raw if v is not None]


def _lapse_rate_freezing_level(temp_2m_c: float, base_altitude_m: float = 3200.0) -> float:
    """Estimate 0°C isotherm altitude using standard environmental lapse rate
    (6.5 °C / km). Used as fallback when freezinglevel_height is missing."""
    return base_altitude_m + max(temp_2m_c, 0.0) / 6.5 * 1000.0


def _fetch_current_window(
    lat: float, lon: float, as_of: date, endpoint_type: str, timeout: int
) -> tuple[float, float, float]:
    """Fetch temperature, precip, and freezing level for the 14-day window ending on as_of.

    Returns:
        (mean_max_temp_3d_c, precip_14d_mm, mean_freezing_level_m)
    """
    start_date = as_of - timedelta(days=13)  # 14 days inclusive

    base_params: dict = {
        "latitude": lat,
        "longitude": lon,
        "daily": "temperature_2m_max,precipitation_sum",
        "hourly": "freezinglevel_height",
        "timezone": "UTC",
    }

    if endpoint_type == "forecast":
        url = FORECAST_URL
        past_days = max((date.today() - start_date).days + 1, 1)
        params = {
            **base_params,
            "past_days": past_days,
            "start_date": start_date.isoformat(),
            "end_date": as_of.isoformat(),
        }
    else:
        url = ARCHIVE_URL
        params = {
            **base_params,
            "start_date": start_date.isoformat(),
            "end_date": as_of.isoformat(),
        }

    data = _get_json(url, params, timeout)

    max_temps = _extract_daily(data, "temperature_2m_max")
    precip = _extract_daily(data, "precipitation_sum")
    fl_hourly = _extract_hourly(data, "freezinglevel_height")

    if not max_temps:
        raise MeteoUnavailableError(
            "Open-Meteo returned no temperature_2m_max data for the requested window"
        )

    # 3-day mean max temp (last 3 days of the window)
    temps_3d = max_temps[-3:] if len(max_temps) >= 3 else max_temps
    mean_max_temp_3d = sum(temps_3d) / len(temps_3d)

    # 14-day cumulative precip
    precip_14d = sum(precip)

    # Freezing level: hourly mean if available, else lapse-rate estimate
    if fl_hourly:
        mean_fl = sum(fl_hourly) / len(fl_hourly)
    else:
        logger.debug(
            "freezinglevel_height not in response for %s %s — using lapse-rate estimate",
            lat,
            lon,
        )
        mean_fl = _lapse_rate_freezing_level(mean_max_temp_3d)

    return mean_max_temp_3d, precip_14d, mean_fl


def _fetch_baseline_temp(lat: float, lon: float, month: int, timeout: int) -> float:
    """14-year ERA5 mean daily-max temperature for the given calendar month.

    Always uses the archive endpoint (Q1). Cached by (lat, lon, month) since
    the baseline is the same for all as_of dates within the same month.
    """
    cache_key = ("baseline", lat, lon, month)
    cached = _cache_get_baseline(cache_key)
    if cached is not None:
        return cached

    today = date.today()
    end_year = today.year - 1
    start_year = end_year - BASELINE_YEARS + 1

    data = _get_json(
        ARCHIVE_URL,
        {
            "latitude": lat,
            "longitude": lon,
            "daily": "temperature_2m_max",
            "start_date": f"{start_year}-01-01",
            "end_date": f"{end_year}-12-31",
            "timezone": "UTC",
        },
        timeout,
    )

    try:
        times = data["daily"]["time"]
        temps_raw = data["daily"]["temperature_2m_max"]
    except KeyError as exc:
        raise MeteoUnavailableError(
            f"Missing field in ERA5 baseline response: {exc}"
        ) from exc

    month_str = f"-{month:02d}-"
    same_month = [
        float(t) for d, t in zip(times, temps_raw, strict=False)
        if month_str in d and t is not None
    ]

    if not same_month:
        raise MeteoUnavailableError(
            f"ERA5 baseline returned no data for month {month:02d} "
            f"in years {start_year}–{end_year}"
        )

    baseline = sum(same_month) / len(same_month)
    _cache_set_baseline(cache_key, baseline)
    return baseline


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def fetch_meteo_inputs(
    lat: float, lon: float, as_of: date, *, timeout: int = 10
) -> MeteoInputs:
    """Fetch Open-Meteo weather inputs for hazard thermal scoring.

    Args:
        lat:     Lake latitude (WGS84 decimal degrees).
        lon:     Lake longitude (WGS84 decimal degrees).
        as_of:   Date for which to compute the weather window (typically date.today()
                 for live monitoring; a past date for backfill).
        timeout: Per-request HTTP timeout in seconds.

    Returns:
        MeteoInputs containing the 3-day max-temp anomaly (relative to the
        14-year ERA5 same-month baseline), 14-day cumulative precipitation,
        and mean freezing-level altitude.

    Raises:
        MeteoUnavailableError: On any network failure, HTTP error, or malformed
            response.
    """
    endpoint_type = _endpoint_type(as_of)
    cache_key = (lat, lon, as_of, endpoint_type)

    cached = _cache_get_inputs(cache_key)
    if cached is not None:
        logger.debug("meteo cache hit for (%.4f, %.4f, %s, %s)", lat, lon, as_of, endpoint_type)
        return cached

    mean_max_temp_3d, precip_14d, freezing_level = _fetch_current_window(
        lat, lon, as_of, endpoint_type, timeout
    )
    baseline_temp = _fetch_baseline_temp(lat, lon, as_of.month, timeout)
    anomaly = mean_max_temp_3d - baseline_temp

    result = MeteoInputs(
        max_temp_anomaly_3d_c=round(anomaly, 3),
        precip_14d_mm=round(precip_14d, 2),
        freezing_level_m=round(freezing_level, 1),
        source=f"open-meteo-{endpoint_type}",
        fetched_at=datetime.now(UTC).isoformat(),
    )
    _cache_set_inputs(cache_key, result)
    logger.debug(
        "meteo fetched for (%.4f, %.4f, %s): anomaly=%.2f°C precip=%.1fmm fl=%.0fm",
        lat,
        lon,
        as_of,
        anomaly,
        precip_14d,
        freezing_level,
    )
    return result


__all__ = [
    "MeteoInputs",
    "MeteoUnavailableError",
    "clear_meteo_cache",
    "fetch_meteo_inputs",
]
