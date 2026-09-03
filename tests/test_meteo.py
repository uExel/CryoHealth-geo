"""Tests for pipeline/meteo.py and thermal hazard scoring (Issue #16).

All network calls are mocked with unittest.mock; no real network I/O is performed.
Tests verify:
- Proportional weight rebalance in v1.1 sums to 1.0.
- thermal_risk normalization: saturation, sub-threshold, intermediate, and None fallback.
- compute_hazard_score dynamic score increase under heatwave anomaly.
- fetch_meteo_inputs: forecast endpoint for <=90d, archive for >90d, baseline always archive.
- In-memory 6-hour caching.
- Error handling & graceful degradation: timeout, 5xx, invalid JSON, missing keys -> MeteoUnavailableError.
"""

from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import pytest
import requests

from pipeline.hazard import (
    WEIGHTS,
    LakeStaticInputs,
    compute_hazard_score,
    thermal_risk,
)
from pipeline.meteo import (
    ARCHIVE_URL,
    FORECAST_URL,
    MeteoInputs,
    MeteoUnavailableError,
    clear_meteo_cache,
    fetch_meteo_inputs,
)


@pytest.fixture(autouse=True)
def _reset_cache():
    """Clear in-memory meteo cache before each test."""
    clear_meteo_cache()
    yield
    clear_meteo_cache()


# ---------------------------------------------------------------------------
# Weight & methodology tests
# ---------------------------------------------------------------------------


def test_weights_v1_1_sum_to_one():
    """WEIGHTS must sum to 1.0 and include the 0.10 thermal component."""
    assert sum(WEIGHTS.values()) == pytest.approx(1.0)
    assert WEIGHTS["thermal"] == pytest.approx(0.10)
    assert WEIGHTS["area_growth"] == pytest.approx(0.27)
    assert WEIGHTS["dam_type"] == pytest.approx(0.18)


# ---------------------------------------------------------------------------
# Pure thermal_risk calculation tests
# ---------------------------------------------------------------------------


def test_thermal_risk_none_meteo_returns_zero():
    """Graceful degradation: None meteo returns 0.0 risk."""
    assert thermal_risk(None) == 0.0


def test_thermal_risk_all_saturation_returns_one():
    """Max temp anomaly >= 4°C, precip >= 80mm, freezing level >= 5500m -> 1.0."""
    meteo = MeteoInputs(
        max_temp_anomaly_3d_c=4.5,
        precip_14d_mm=95.0,
        freezing_level_m=5600.0,
        source="test",
        fetched_at="2026-09-03T00:00:00Z",
    )
    assert thermal_risk(meteo) == pytest.approx(1.0)


def test_thermal_risk_sub_threshold_returns_zero():
    """Below seasonal mean (anomaly <= 0), no precip, low freezing level (<= 3500m) -> 0.0."""
    meteo = MeteoInputs(
        max_temp_anomaly_3d_c=-2.0,
        precip_14d_mm=0.0,
        freezing_level_m=3200.0,
        source="test",
        fetched_at="2026-09-03T00:00:00Z",
    )
    assert thermal_risk(meteo) == pytest.approx(0.0)


def test_thermal_risk_halfway_inputs_return_half():
    """Anomaly = +2°C (0.5), precip = 40mm (0.5), freezing level = 4500m (0.5) -> 0.5."""
    meteo = MeteoInputs(
        max_temp_anomaly_3d_c=2.0,
        precip_14d_mm=40.0,
        freezing_level_m=4500.0,
        source="test",
        fetched_at="2026-09-03T00:00:00Z",
    )
    assert thermal_risk(meteo) == pytest.approx(0.5)


def test_compute_hazard_score_heatwave_raises_score():
    """Acceptance criterion: severe temperature anomaly (+4°C) dynamically raises hazard score."""
    obs = [(date(2026, 6, 1), 1.0), (date(2026, 7, 1), 1.2)]
    static = LakeStaticInputs(dam_type="moraine", glacier_contact=True, historical_glof=False)

    baseline_result = compute_hazard_score(obs, static, slope_deg=20.0, as_of=date(2026, 7, 1))

    heatwave_meteo = MeteoInputs(
        max_temp_anomaly_3d_c=4.5,  # > +4°C heatwave
        precip_14d_mm=20.0,
        freezing_level_m=4800.0,
        source="test",
        fetched_at="2026-09-03T00:00:00Z",
    )
    heatwave_result = compute_hazard_score(
        obs, static, slope_deg=20.0, as_of=date(2026, 7, 1), meteo=heatwave_meteo
    )

    assert heatwave_result.score > baseline_result.score
    assert heatwave_result.components["normalized_risks"]["thermal"] > 0.0
    assert heatwave_result.components["raw_inputs"]["meteo"]["max_temp_anomaly_3d_c"] == 4.5


# ---------------------------------------------------------------------------
# fetch_meteo_inputs mocked HTTP tests
# ---------------------------------------------------------------------------


def _mock_window_payload(max_temp=25.0, precip=30.0, fl=4600.0):
    """Synthetic response payload for window query."""
    return {
        "daily": {
            "time": ["2026-08-19", "2026-08-20", "2026-08-21"],
            "temperature_2m_max": [max_temp, max_temp + 1.0, max_temp - 0.5],
            "precipitation_sum": [precip / 3.0, precip / 3.0, precip / 3.0],
        },
        "hourly": {
            "time": ["2026-08-21T00:00", "2026-08-21T01:00"],
            "freezinglevel_height": [fl, fl + 10.0],
        },
    }


def _mock_baseline_payload(mean_temp=20.0):
    """Synthetic ERA5 baseline response payload covering all 12 months."""
    times = []
    temps = []
    for m in range(1, 13):
        times.extend([f"2020-{m:02d}-01", f"2020-{m:02d}-15", f"2021-{m:02d}-01"])
        temps.extend([mean_temp, mean_temp + 1.0, mean_temp - 1.0])
    return {
        "daily": {
            "time": times,
            "temperature_2m_max": temps,
        }
    }



@patch("pipeline.meteo.requests.get")
def test_fetch_meteo_inputs_recent_date_uses_forecast_endpoint(mock_get):
    """Dates within 90 days use the forecast endpoint for the window query."""
    def side_effect(url, params, **kwargs):
        resp = MagicMock()
        resp.status_code = 200
        if url == FORECAST_URL:
            resp.json.return_value = _mock_window_payload(max_temp=25.0)
        else:
            resp.json.return_value = _mock_baseline_payload(mean_temp=21.0)
        return resp

    mock_get.side_effect = side_effect

    today = date.today()
    inputs = fetch_meteo_inputs(lat=36.4, lon=74.61, as_of=today)

    assert inputs.source == "open-meteo-forecast"
    assert inputs.max_temp_anomaly_3d_c > 0  # 25+ vs 21
    # Check that FORECAST_URL was called for the window
    assert any(c.args[0] == FORECAST_URL for c in mock_get.call_args_list)
    # Check that ARCHIVE_URL was called for the baseline
    assert any(c.args[0] == ARCHIVE_URL for c in mock_get.call_args_list)


@patch("pipeline.meteo.requests.get")
def test_fetch_meteo_inputs_old_date_uses_archive_endpoint(mock_get):
    """Dates older than 90 days use the archive endpoint for both window and baseline."""
    def side_effect(url, params, **kwargs):
        resp = MagicMock()
        resp.status_code = 200
        if "precipitation_sum" in params.get("daily", ""):
            resp.json.return_value = _mock_window_payload(max_temp=22.0)
        else:
            resp.json.return_value = _mock_baseline_payload(mean_temp=20.0)
        return resp


    mock_get.side_effect = side_effect

    old_date = date.today() - timedelta(days=120)
    inputs = fetch_meteo_inputs(lat=36.4, lon=74.61, as_of=old_date)

    assert inputs.source == "open-meteo-archive"
    # All calls should be to ARCHIVE_URL
    assert all(c.args[0] == ARCHIVE_URL for c in mock_get.call_args_list)


@patch("pipeline.meteo.requests.get")
def test_fetch_meteo_inputs_caching_prevents_duplicate_calls(mock_get):
    """Calling fetch_meteo_inputs twice with same parameters hits the cache."""
    resp_window = MagicMock()
    resp_window.status_code = 200
    resp_window.json.return_value = _mock_window_payload()

    resp_base = MagicMock()
    resp_base.status_code = 200
    resp_base.json.return_value = _mock_baseline_payload()

    mock_get.side_effect = [resp_window, resp_base]

    today = date.today()
    res1 = fetch_meteo_inputs(lat=36.4, lon=74.61, as_of=today)
    res2 = fetch_meteo_inputs(lat=36.4, lon=74.61, as_of=today)

    assert res1 == res2
    # Only 2 network calls were made (window + baseline), not 4
    assert mock_get.call_count == 2


@patch("pipeline.meteo.requests.get")
def test_fetch_meteo_inputs_timeout_raises_meteo_unavailable(mock_get):
    """Network timeout raises MeteoUnavailableError."""
    mock_get.side_effect = requests.Timeout("Connection timed out")

    with pytest.raises(MeteoUnavailableError, match="request failed"):
        fetch_meteo_inputs(lat=36.4, lon=74.61, as_of=date.today())


@patch("pipeline.meteo.requests.get")
def test_fetch_meteo_inputs_http_503_raises_meteo_unavailable(mock_get):
    """HTTP 5xx response raises MeteoUnavailableError."""
    resp = MagicMock()
    resp.status_code = 503
    resp.text = "Service Unavailable"
    mock_get.return_value = resp

    with pytest.raises(MeteoUnavailableError, match="returned HTTP 503"):
        fetch_meteo_inputs(lat=36.4, lon=74.61, as_of=date.today())


@patch("pipeline.meteo.requests.get")
def test_fetch_meteo_inputs_invalid_json_raises_meteo_unavailable(mock_get):
    """Invalid JSON response raises MeteoUnavailableError."""
    resp = MagicMock()
    resp.status_code = 200
    resp.json.side_effect = ValueError("Expecting value")
    mock_get.return_value = resp

    with pytest.raises(MeteoUnavailableError, match="not valid JSON"):
        fetch_meteo_inputs(lat=36.4, lon=74.61, as_of=date.today())


@patch("pipeline.meteo.requests.get")
def test_fetch_meteo_inputs_missing_key_raises_meteo_unavailable(mock_get):
    """Missing expected keys in payload raises MeteoUnavailableError."""
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"daily": {}}  # empty daily
    mock_get.return_value = resp

    with pytest.raises(MeteoUnavailableError, match="Missing field"):
        fetch_meteo_inputs(lat=36.4, lon=74.61, as_of=date.today())


@patch("pipeline.meteo.requests.get")
def test_lapse_rate_fallback_when_freezing_level_absent(mock_get):
    """When hourly freezinglevel_height is missing, lapse-rate estimate is used."""
    def side_effect(url, params, **kwargs):
        resp = MagicMock()
        resp.status_code = 200
        if url == FORECAST_URL:
            # daily present, hourly absent
            resp.json.return_value = {
                "daily": {
                    "time": ["2026-08-21"],
                    "temperature_2m_max": [15.0],
                    "precipitation_sum": [10.0],
                },
            }
        else:
            resp.json.return_value = _mock_baseline_payload(mean_temp=12.0)
        return resp

    mock_get.side_effect = side_effect

    inputs = fetch_meteo_inputs(lat=36.4, lon=74.61, as_of=date.today())
    # Lapse rate: 3200 + (15 / 6.5) * 1000 = ~5507.7 m
    assert inputs.freezing_level_m > 3200.0
    assert inputs.freezing_level_m == pytest.approx(3200.0 + (15.0 / 6.5) * 1000.0, abs=1.0)
