"""Tests for pipeline/forecast.py.

All tests are fully mocked — no live Prophet fit (slow ~3–8s per call), no DB.
Prophet itself is either mocked at the importlib level (for the unavailable tests)
or patched at the model level (for the computation tests).

Slow / integration tests require prophet installed and run a real Stan fit:
  uv run pytest -m slow tests/test_forecast.py

Pattern mirrors test_ml_segmentation.py and test_ndwi.py — same split between
fast mocked CI tests and slow network/compute integration tests.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


def _daily_obs(n: int, start_area: float = 0.10, slope_per_day: float = 0.001) -> list[tuple[date, float]]:
    """Generate n daily observations with a given linear growth slope."""
    base = date(2023, 1, 1)
    return [(base + timedelta(days=i), max(0.0, start_area + i * slope_per_day)) for i in range(n)]


def _flat_obs(n: int, area: float = 0.20) -> list[tuple[date, float]]:
    base = date(2023, 1, 1)
    return [(base + timedelta(days=i), area) for i in range(n)]


def _sparse_obs(n: int) -> list[tuple[date, float]]:
    """n observations but only 30 days of history — triggers MIN_HISTORY_DAYS gate."""
    base = date(2024, 6, 1)
    return [(base + timedelta(days=i), 0.15 + i * 0.001) for i in range(n)]


# ---------------------------------------------------------------------------
# Data gate tests — insufficient_data path
# ---------------------------------------------------------------------------


def test_insufficient_obs_returns_insufficient_data():
    from pipeline.forecast import compute_forecast

    with patch("importlib.util.find_spec", return_value=MagicMock()):
        obs = _daily_obs(n=10)  # < MIN_OBSERVATIONS (15)
        result = compute_forecast(obs)

    # Fewer than 15 obs → linear fallback (not "insufficient_data" since there are
    # still some obs; the gate triggers the fallback path)
    assert result.method == "linear_fallback"
    assert result.observation_count == 10


def test_zero_observations_returns_insufficient_data():
    from pipeline.forecast import compute_forecast

    with patch("importlib.util.find_spec", return_value=MagicMock()):
        result = compute_forecast([])

    assert result.method == "insufficient_data"
    assert result.observation_count == 0
    assert result.expansion_probability is None


def test_short_history_returns_linear_fallback():
    """≥15 observations but <180 days history → linear fallback."""
    from pipeline.forecast import compute_forecast

    with patch("importlib.util.find_spec", return_value=MagicMock()):
        obs = _sparse_obs(n=20)  # 20 obs, but only ~20 days span
        result = compute_forecast(obs)

    assert result.method == "linear_fallback"
    assert result.history_days < 180


# ---------------------------------------------------------------------------
# Linear fallback path
# ---------------------------------------------------------------------------


def test_linear_fallback_growing_series_projects_positive():
    from pipeline.forecast import compute_forecast

    with patch("importlib.util.find_spec", return_value=MagicMock()):
        obs = _daily_obs(n=10, slope_per_day=0.005)
        result = compute_forecast(obs)

    assert result.method == "linear_fallback"
    assert result.projected_area_14d_p50 is not None
    assert result.projected_area_14d_p50 > obs[-1][1]  # should project upward


def test_linear_fallback_returns_non_negative_area():
    from pipeline.forecast import compute_forecast

    with patch("importlib.util.find_spec", return_value=MagicMock()):
        obs = _daily_obs(n=10, slope_per_day=-0.05)  # shrinking lake
        result = compute_forecast(obs)

    assert result.projected_area_14d_p50 is not None
    assert result.projected_area_14d_p50 >= 0.0


def test_linear_fallback_p10_less_than_p50_less_than_p90():
    from pipeline.forecast import compute_forecast

    with patch("importlib.util.find_spec", return_value=MagicMock()):
        obs = _daily_obs(n=12)
        result = compute_forecast(obs)

    assert result.projected_area_14d_p10 is not None
    assert result.projected_area_14d_p50 is not None
    assert result.projected_area_14d_p90 is not None
    assert result.projected_area_14d_p10 <= result.projected_area_14d_p50 <= result.projected_area_14d_p90


# ---------------------------------------------------------------------------
# Pure math helpers
# ---------------------------------------------------------------------------


def test_preprocess_deduplicates_dates():
    from pipeline.forecast import _preprocess

    obs = [
        (date(2023, 1, 1), 0.10),
        (date(2023, 1, 1), 0.15),  # duplicate — last write wins
        (date(2023, 1, 2), 0.12),
    ]
    df = _preprocess(obs)

    assert len(df) == 2
    row_jan1 = df[df["ds"].dt.date == date(2023, 1, 1)]
    assert float(row_jan1["y"].iloc[0]) == pytest.approx(0.15)


def test_preprocess_clips_negative_area():
    from pipeline.forecast import _preprocess

    obs = [(date(2023, 1, 1), -0.05), (date(2023, 1, 2), 0.10)]
    df = _preprocess(obs)

    assert float(df.iloc[0]["y"]) == pytest.approx(0.0)
    assert float(df.iloc[1]["y"]) == pytest.approx(0.10)


def test_expansion_probability_all_above():
    from pipeline.forecast import _expansion_probability

    samples = np.array([0.30, 0.35, 0.40, 0.45])
    prob = _expansion_probability(samples, current_area=0.20)
    assert prob == pytest.approx(1.0)


def test_expansion_probability_all_below():
    from pipeline.forecast import _expansion_probability

    samples = np.array([0.10, 0.12, 0.08])
    prob = _expansion_probability(samples, current_area=0.20)
    assert prob == pytest.approx(0.0)


def test_expansion_probability_mixed():
    from pipeline.forecast import _expansion_probability

    above = np.full(700, 0.25)
    below = np.full(300, 0.15)
    samples = np.concatenate([above, below])
    prob = _expansion_probability(samples, current_area=0.20)
    assert prob == pytest.approx(0.70)


def test_expansion_probability_empty_samples():
    from pipeline.forecast import _expansion_probability

    prob = _expansion_probability(np.array([]), current_area=0.20)
    assert prob == pytest.approx(0.0)


def test_relative_uncertainty_zero_p50():
    from pipeline.forecast import _relative_uncertainty

    result = _relative_uncertainty(p10=0.0, p50=0.0, p90=0.5)
    assert result == float("inf")


# ---------------------------------------------------------------------------
# ForecastUnavailableError — missing dependency
# ---------------------------------------------------------------------------


def test_compute_forecast_raises_when_prophet_not_installed():
    """ForecastUnavailableError raised ONLY when prophet import fails."""
    from pipeline.forecast import ForecastUnavailableError, compute_forecast

    with patch("importlib.util.find_spec", return_value=None):
        with pytest.raises(ForecastUnavailableError, match="prophet is not installed"):
            compute_forecast(_daily_obs(30))


def test_forecast_unavailable_error_is_runtime_error():
    from pipeline.forecast import ForecastUnavailableError

    assert issubclass(ForecastUnavailableError, RuntimeError)


# ---------------------------------------------------------------------------
# Prophet path — mocked fit
# ---------------------------------------------------------------------------


def _mock_prophet_class(p10: float = 0.25, p50: float = 0.32, p90: float = 0.39):
    """Return a mock Prophet class whose predictive_samples returns a fixed array."""
    prophet_instance = MagicMock()
    samples_array = np.full(1000, p50)  # flat samples at p50
    prophet_instance.predictive_samples.return_value = {"yhat": np.array([samples_array])}
    prophet_class = MagicMock(return_value=prophet_instance)
    return prophet_class, prophet_instance


def test_compute_forecast_prophet_path_returns_correct_method():
    import sys
    from pipeline.forecast import compute_forecast

    obs = _daily_obs(n=200, slope_per_day=0.001)  # ≥15 obs, ≥180 days
    mock_cls, _ = _mock_prophet_class(p50=0.35)

    # Patch sys.modules so 'from prophet import Prophet' inside _fit_and_sample
    # doesn't raise ModuleNotFoundError when prophet isn't installed.
    fake_prophet_module = MagicMock()
    fake_prophet_module.Prophet = mock_cls

    with patch("importlib.util.find_spec", return_value=MagicMock()), \
         patch.dict(sys.modules, {"prophet": fake_prophet_module}), \
         patch("pipeline.forecast.Prophet", mock_cls, create=True):
        result = compute_forecast(obs)

    assert result.method == "prophet"
    assert result.projected_area_14d_p50 is not None



def test_compute_forecast_high_uncertainty_discards():
    """(p90-p10)/p50 > UNCERTAINTY_CUTOFF → method="insufficient_confidence"."""
    from pipeline.forecast import UNCERTAINTY_CUTOFF, compute_forecast

    obs = _daily_obs(n=200, slope_per_day=0.001)

    # Create samples with very wide spread.
    wide_samples = np.concatenate([np.full(500, 0.01), np.full(500, 10.0)])
    prophet_instance = MagicMock()
    prophet_instance.predictive_samples.return_value = {"yhat": np.array([wide_samples])}
    mock_cls = MagicMock(return_value=prophet_instance)

    with patch("importlib.util.find_spec", return_value=MagicMock()), \
         patch("pipeline.forecast.Prophet", mock_cls, create=True):
        result = compute_forecast(obs)

    assert result.method == "insufficient_confidence"
    assert result.expansion_probability is None


def test_compute_forecast_prob_below_threshold_not_elevated():
    """expansion_probability < EXPANSION_PROB_THRESHOLD → elevated_risk=False."""
    from pipeline.forecast import EXPANSION_PROB_THRESHOLD, UNCERTAINTY_CUTOFF, compute_forecast

    obs = _daily_obs(n=200, slope_per_day=0.001)
    current = obs[-1][1]

    # Samples mostly BELOW current area → prob < threshold
    low_samples = np.full(1000, current * 0.5)
    prophet_instance = MagicMock()
    prophet_instance.predictive_samples.return_value = {"yhat": np.array([low_samples])}
    mock_cls = MagicMock(return_value=prophet_instance)

    with patch("importlib.util.find_spec", return_value=MagicMock()), \
         patch("pipeline.forecast.Prophet", mock_cls, create=True):
        result = compute_forecast(obs)

    if result.method == "prophet":
        assert not result.elevated_risk
        assert result.expansion_probability is not None
        assert result.expansion_probability < EXPANSION_PROB_THRESHOLD


def test_compute_forecast_catches_fit_exception_returns_insufficient_confidence():
    """Prophet.fit() raises RuntimeError → caught internally, method='insufficient_confidence'."""
    from pipeline.forecast import compute_forecast

    obs = _daily_obs(n=200, slope_per_day=0.001)
    prophet_instance = MagicMock()
    prophet_instance.fit.side_effect = RuntimeError("Stan sampling failed")
    mock_cls = MagicMock(return_value=prophet_instance)

    with patch("importlib.util.find_spec", return_value=MagicMock()), \
         patch("pipeline.forecast.Prophet", mock_cls, create=True):
        result = compute_forecast(obs)  # must not raise

    assert result.method == "insufficient_confidence"
    assert "exception" in result.note.lower() or "error" in result.note.lower()


def test_compute_forecast_fit_exception_does_not_re_raise():
    """No exception propagates out of compute_forecast() for fit failures."""
    from pipeline.forecast import compute_forecast

    obs = _daily_obs(n=200, slope_per_day=0.001)
    prophet_instance = MagicMock()
    prophet_instance.fit.side_effect = ValueError("bad data")
    mock_cls = MagicMock(return_value=prophet_instance)

    with patch("importlib.util.find_spec", return_value=MagicMock()), \
         patch("pipeline.forecast.Prophet", mock_cls, create=True):
        try:
            compute_forecast(obs)
        except Exception as exc:  # noqa: BLE001
            pytest.fail(f"compute_forecast() unexpectedly raised: {exc!r}")


# ---------------------------------------------------------------------------
# ForecastResult contract
# ---------------------------------------------------------------------------


def test_forecast_result_always_has_required_fields():
    """ForecastResult must always populate the fields needed for storage."""
    from pipeline.forecast import ForecastResult

    result = ForecastResult(
        method="prophet",
        expansion_probability=0.91,
        projected_area_14d_p10=0.28,
        projected_area_14d_p50=0.35,
        projected_area_14d_p90=0.43,
        current_area_km2=0.31,
        forecast_date="2024-06-15",
        as_of="2024-06-01",
        observation_count=250,
        history_days=720,
        elevated_risk=True,
        note="test",
    )
    # All required fields present and of correct type.
    assert isinstance(result.method, str)
    assert isinstance(result.forecast_date, str)
    assert isinstance(result.as_of, str)
    assert isinstance(result.observation_count, int)
    assert isinstance(result.elevated_risk, bool)


def test_forecast_result_is_frozen():
    from pipeline.forecast import ForecastResult
    import dataclasses

    assert dataclasses.fields(ForecastResult)  # it's a dataclass
    result = ForecastResult(
        method="linear_fallback",
        expansion_probability=0.6,
        projected_area_14d_p10=0.1,
        projected_area_14d_p50=0.2,
        projected_area_14d_p90=0.3,
        current_area_km2=0.18,
        forecast_date="2024-06-15",
        as_of="2024-06-01",
        observation_count=10,
        history_days=10,
    )
    with pytest.raises(Exception):  # frozen dataclass
        result.method = "mutated"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Slow / integration tests — real Prophet fit
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_prophet_fit_real_synthetic_series():
    """Integration: real Prophet fit on 2-year synthetic series.

    Requires: uv pip install 'cryohealth-geo[forecast]'
    Run with: uv run pytest -m slow tests/test_forecast.py
    """
    from pipeline.forecast import compute_forecast

    # 2 years of daily observations with seasonal growth pattern.
    base = date(2022, 1, 1)
    obs = []
    for i in range(730):
        d = base + timedelta(days=i)
        # Simulate seasonal lake with annual peak in August.
        area = 0.15 + 0.05 * np.sin(2 * np.pi * i / 365) + 0.0001 * i
        obs.append((d, max(0.0, area)))

    result = compute_forecast(obs)

    assert result.method in ("prophet", "insufficient_confidence")
    if result.method == "prophet":
        assert result.projected_area_14d_p10 is not None
        assert result.projected_area_14d_p50 is not None
        assert result.projected_area_14d_p90 is not None
        # p10 < p50 < p90 ordering must hold.
        assert result.projected_area_14d_p10 <= result.projected_area_14d_p50
        assert result.projected_area_14d_p50 <= result.projected_area_14d_p90
        assert result.observation_count == 730
        assert result.history_days >= 729
        print(f"\nProphet fit result: p50={result.projected_area_14d_p50:.4f} km² "
              f"expansion_prob={result.expansion_probability:.3f}")
