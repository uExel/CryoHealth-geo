"""Tests for pipeline/anomaly.py (Issue #22).

All tests are fast and mocked — no live SceneSource calls, no sklearn fits on real data.
The @pytest.mark.slow test (bottom) fits a real IsolationForest on synthetic data and
exercises the full compute_anomaly() path with a mock SceneSource.

Run fast tests only:
    uv run pytest tests/test_anomaly.py -v -m "not slow"

Run all including slow:
    uv run pytest tests/test_anomaly.py -v
"""

from __future__ import annotations

import sys
from dataclasses import fields
from datetime import date, timedelta
from types import ModuleType
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Module import guard: anomaly.py must import even without sklearn installed.
# (Same pattern as test_forecast.py which patches sys.modules for prophet.)
# ---------------------------------------------------------------------------

# anomaly.py uses importlib.util.find_spec at function call time, not at module
# import time, so we don't need to patch sys.modules at import. However, we do
# need to ensure sklearn is not required at collection time. The module only does
# `import numpy as np` at module level (always available) — no sklearn import.
from pipeline.anomaly import (
    AnomalyResult,
    AnomalyUnavailableError,
    CLOUD_MAX_FRACTION,
    MIN_SCENES_FOR_INFERENCE,
    MIN_TRAINING_SCENES,
    SEEPAGE_SIGMA_THRESHOLD,
    SUDDEN_DRAINAGE_DELTA,
    compute_mndwi,
    compute_ndwi,
    extract_spectral_vectors,
    score_to_sigma,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bands(h: int = 10, w: int = 10, value: float = 500.0) -> dict[str, np.ndarray]:
    """Uniform test band arrays with a nonzero water-like B03 > B08 (positive NDWI)."""
    # B03 (Green) = 600 so NDWI > 0, MNDWI > 0 (water-like)
    bands = {
        "B03": np.full((h, w), 600.0, dtype=np.float32),
        "B04": np.full((h, w), value, dtype=np.float32),
        "B05": np.full((h, w), value, dtype=np.float32),
        "B08": np.full((h, w), value, dtype=np.float32),   # B03 > B08 → NDWI > 0
        "B11": np.full((h, w), value, dtype=np.float32),   # B03 > B11 → MNDWI > 0
        "B12": np.full((h, w), value, dtype=np.float32),
    }
    return bands


def _make_scene_source(scenes: list[tuple[date, dict[str, np.ndarray]]]) -> MagicMock:
    """Return a mock SceneSource whose read_bands() returns the given per-scene bands."""
    from pipeline.stac_source import SceneRef

    mock_scenes = []
    for captured, bands in scenes:
        ref = SceneRef(
            scene_id=f"scene-{captured.isoformat()}",
            captured_at=captured,
            cloud_cover_pct=0.0,
            assets={},
        )
        mock_scenes.append((ref, bands))

    source = MagicMock()
    source.find_scenes_in_range.return_value = [r for r, _ in mock_scenes]
    source.read_bands.side_effect = lambda scene, bands, bbox: next(
        b for r, b in mock_scenes if r.scene_id == scene.scene_id
    )
    return source


# ---------------------------------------------------------------------------
# compute_ndwi
# ---------------------------------------------------------------------------

class TestComputeNdwi:
    def test_pure_water_positive(self):
        """B03 >> B08 → NDWI strongly positive (water)."""
        b03 = np.array([[1000.0, 800.0]], dtype=np.float32)
        b08 = np.array([[100.0, 50.0]], dtype=np.float32)
        result = compute_ndwi(b03, b08)
        assert (result > 0).all()

    def test_bare_soil_negative(self):
        """B03 < B08 → NDWI negative (non-water)."""
        b03 = np.array([[200.0]], dtype=np.float32)
        b08 = np.array([[800.0]], dtype=np.float32)
        result = compute_ndwi(b03, b08)
        assert result[0, 0] < 0

    def test_safe_divide_zero_denominator(self):
        """Both bands = 0 → denominator = 0 → NDWI = 0.0, not NaN or inf."""
        b03 = np.zeros((3, 3), dtype=np.float32)
        b08 = np.zeros((3, 3), dtype=np.float32)
        result = compute_ndwi(b03, b08)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))
        assert (result == 0.0).all()

    def test_returns_float32(self):
        b03 = np.ones((4, 4), dtype=np.float64)
        b08 = np.ones((4, 4), dtype=np.float64)
        result = compute_ndwi(b03, b08)
        assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# compute_mndwi
# ---------------------------------------------------------------------------

class TestComputeMndwi:
    def test_swir_dominant_negative(self):
        """B11 >> B03 → MNDWI strongly negative (dry moraine / bare rock)."""
        b03 = np.array([[100.0]], dtype=np.float32)
        b11 = np.array([[900.0]], dtype=np.float32)
        result = compute_mndwi(b03, b11)
        assert result[0, 0] < 0

    def test_water_positive(self):
        """B03 > B11 → MNDWI positive (moist surface)."""
        b03 = np.array([[800.0]], dtype=np.float32)
        b11 = np.array([[100.0]], dtype=np.float32)
        result = compute_mndwi(b03, b11)
        assert result[0, 0] > 0

    def test_safe_divide_zero_denominator(self):
        b03 = np.zeros((2, 2), dtype=np.float32)
        b11 = np.zeros((2, 2), dtype=np.float32)
        result = compute_mndwi(b03, b11)
        assert not np.any(np.isnan(result))
        assert (result == 0.0).all()


# ---------------------------------------------------------------------------
# extract_spectral_vectors
# ---------------------------------------------------------------------------

class TestExtractSpectralVectors:
    def test_shape_no_mask(self):
        """(H, W, 6) → flatten → (H*W, 6)."""
        bands = _make_bands(h=5, w=8)
        result = extract_spectral_vectors(bands)
        assert result.shape == (40, 6)

    def test_excludes_cloud_pixels(self):
        """Cloud-masked pixels are removed from the output."""
        bands = _make_bands(h=4, w=4)  # 16 pixels
        cloud_mask = np.zeros((4, 4), dtype=bool)
        cloud_mask[0, :] = True  # mask 4 pixels in row 0
        result = extract_spectral_vectors(bands, cloud_mask)
        assert result.shape == (12, 6)

    def test_all_cloud_returns_empty(self):
        """100% cloud mask → empty (0, 6) array, no raise."""
        bands = _make_bands(h=3, w=3)
        cloud_mask = np.ones((3, 3), dtype=bool)
        result = extract_spectral_vectors(bands, cloud_mask)
        assert result.shape == (0, 6)

    def test_missing_band_returns_empty(self):
        """Missing band → empty (0, 6) array with WARNING, no raise."""
        bands = _make_bands()
        del bands["B11"]
        result = extract_spectral_vectors(bands)
        assert result.shape == (0, 6)

    def test_band_shape_mismatch_returns_empty(self):
        """Mismatched band shapes → empty (0, 6) array, no raise."""
        bands = _make_bands(h=5, w=5)
        bands["B11"] = np.ones((3, 3), dtype=np.float32)  # different shape
        result = extract_spectral_vectors(bands)
        assert result.shape == (0, 6)

    def test_returns_float32(self):
        result = extract_spectral_vectors(_make_bands())
        assert result.dtype == np.float32

    def test_column_order_swir1_first(self):
        """Column 0 should be SWIR1 (B11); column 4 should be MNDWI."""
        bands = {
            "B03": np.full((2, 2), 600.0, dtype=np.float32),
            "B04": np.full((2, 2), 100.0, dtype=np.float32),
            "B05": np.full((2, 2), 200.0, dtype=np.float32),
            "B08": np.full((2, 2), 300.0, dtype=np.float32),
            "B11": np.full((2, 2), 400.0, dtype=np.float32),  # SWIR1 → col 0
            "B12": np.full((2, 2), 500.0, dtype=np.float32),
        }
        result = extract_spectral_vectors(bands)
        assert result.shape[1] == 6
        # Col 0 = B11 (SWIR1)
        np.testing.assert_allclose(result[:, 0], 400.0, rtol=1e-3)
        # Col 4 = MNDWI = (B03 - B11) / (B03 + B11) = (600 - 400) / (600 + 400) = 0.2
        np.testing.assert_allclose(result[:, 4], 0.2, rtol=1e-3)


# ---------------------------------------------------------------------------
# score_to_sigma
# ---------------------------------------------------------------------------

class TestScoreToSigma:
    def test_at_mean_gives_zero(self):
        assert score_to_sigma(0.5, training_mean=0.5, training_std=0.1) == pytest.approx(0.0)

    def test_below_mean_gives_negative_sigma(self):
        """Score below training mean = anomalous = negative sigma."""
        sigma = score_to_sigma(0.3, training_mean=0.5, training_std=0.1)
        assert sigma == pytest.approx(-2.0)

    def test_above_mean_gives_positive_sigma(self):
        sigma = score_to_sigma(0.7, training_mean=0.5, training_std=0.1)
        assert sigma == pytest.approx(2.0)

    def test_zero_std_returns_zero(self):
        """Degenerate training set (all scores identical) → 0.0, no ZeroDivisionError."""
        assert score_to_sigma(0.5, training_mean=0.5, training_std=0.0) == 0.0


# ---------------------------------------------------------------------------
# load_anomaly_model
# ---------------------------------------------------------------------------

class TestLoadAnomalyModel:
    def test_raises_when_sklearn_not_installed(self, tmp_path):
        from pipeline.anomaly import load_anomaly_model
        with patch("importlib.util.find_spec", return_value=None):
            with pytest.raises(AnomalyUnavailableError, match="scikit-learn"):
                load_anomaly_model(str(tmp_path), "passu")

    def test_raises_when_joblib_missing(self, tmp_path):
        """Model .joblib file missing → AnomalyUnavailableError."""
        from pipeline.anomaly import load_anomaly_model
        # sklearn present, but no .joblib file
        with pytest.raises(AnomalyUnavailableError, match="not found"):
            load_anomaly_model(str(tmp_path), "passu")

    def test_raises_when_sidecar_missing(self, tmp_path):
        """Model .joblib present but sidecar .json missing → AnomalyUnavailableError."""
        import joblib
        from pipeline.anomaly import load_anomaly_model
        model_path = tmp_path / "passu_isolation_forest.joblib"
        joblib.dump(object(), model_path)
        # No sidecar .json
        with pytest.raises(AnomalyUnavailableError, match="sidecar"):
            load_anomaly_model(str(tmp_path), "passu")


# ---------------------------------------------------------------------------
# compute_anomaly — gates
# ---------------------------------------------------------------------------

class TestComputeAnomalyGates:
    """Test applicability gates — no model loading required."""

    def test_no_dam_face_bbox(self):
        """None bbox → method='dam_face_not_digitized'."""
        from pipeline.anomaly import compute_anomaly
        source = MagicMock()
        result = compute_anomaly(source, None, "moraine", "/model", "passu")
        assert result.method == "dam_face_not_digitized"
        assert result.seepage_flag is False
        assert result.drainage_flag is False

    def test_ice_dam_not_applicable(self):
        from pipeline.anomaly import compute_anomaly
        source = MagicMock()
        bbox = (74.0, 36.0, 74.1, 36.1)
        result = compute_anomaly(source, bbox, "ice", "/model", "shishper")
        assert result.method == "not_applicable_ice_dam"
        assert result.seepage_flag is False

    def test_bedrock_dam_not_applicable(self):
        from pipeline.anomaly import compute_anomaly
        source = MagicMock()
        bbox = (74.0, 36.0, 74.1, 36.1)
        result = compute_anomaly(source, bbox, "bedrock", "/model", "passu")
        assert result.method == "not_applicable_bedrock_dam"

    def test_unknown_dam_type_still_applies_if(self):
        """dam_type='unknown' → IF applied (with WARNING), not skipped."""
        from pipeline.anomaly import compute_anomaly, load_anomaly_model

        bbox = (74.0, 36.0, 74.1, 36.1)
        # Patch load_anomaly_model to raise AnomalyUnavailableError so we
        # confirm it was called (got past the dam-type gate) before failing.
        with patch("pipeline.anomaly.load_anomaly_model", side_effect=AnomalyUnavailableError("no model")):
            with pytest.raises(AnomalyUnavailableError):
                compute_anomaly(MagicMock(), bbox, "unknown", "/model", "badswat")

    def test_model_unavailable_raises(self):
        from pipeline.anomaly import compute_anomaly
        bbox = (74.0, 36.0, 74.1, 36.1)
        # load_anomaly_model will raise because sklearn is present but .joblib is missing
        with patch("pipeline.anomaly.load_anomaly_model", side_effect=AnomalyUnavailableError("missing")):
            with pytest.raises(AnomalyUnavailableError):
                compute_anomaly(MagicMock(), bbox, "moraine", "/no/such/dir", "passu")

    def test_insufficient_scenes_returns_result(self, tmp_path):
        """< MIN_SCENES_FOR_INFERENCE cloud-free scenes → method='insufficient_scenes'."""
        from pipeline.anomaly import compute_anomaly
        import json, joblib

        # Write a minimal model + sidecar.
        from sklearn.ensemble import IsolationForest
        X = np.random.default_rng(0).random((25, 6)).astype(np.float32)
        model = IsolationForest(n_estimators=10).fit(X)
        joblib.dump(model, tmp_path / "passu_isolation_forest.joblib")
        scores = model.score_samples(X)
        (tmp_path / "passu_isolation_forest.json").write_text(json.dumps({
            "training_mean_score": float(scores.mean()),
            "training_std_score": float(scores.std()),
        }))

        # SceneSource returns only 1 clean scene (< MIN_SCENES_FOR_INFERENCE=3).
        today = date.today()
        bands = _make_bands()
        # Add clear SCL (all zeros = vegetation, not cloud).
        bands["SCL"] = np.zeros((10, 10), dtype=np.int32)
        source = _make_scene_source([(today, bands)])

        result = compute_anomaly(source, (74.0, 36.0, 74.1, 36.1), "moraine", str(tmp_path), "passu")
        assert result.method == "insufficient_scenes"
        assert result.seepage_flag is False


# ---------------------------------------------------------------------------
# compute_anomaly — IF scoring path
# ---------------------------------------------------------------------------

class TestComputeAnomalyScoring:
    """Test the IF scoring path. Uses a real sklearn fit on synthetic data."""

    @pytest.fixture
    def model_dir(self, tmp_path):
        """Create a fitted IsolationForest + sidecar for slug 'passu'."""
        from sklearn.ensemble import IsolationForest
        import json, joblib

        rng = np.random.default_rng(42)
        # Match the spectral vector scale of _make_clean_bands:
        # [SWIR1(B11)=400, SWIR2(B12)=500, RedEdge(B05)=500, NDWI=0.09, MNDWI=0.2, Red(B04)=500]
        loc = np.array([400.0, 500.0, 500.0, 0.09, 0.20, 500.0], dtype=np.float32)
        scale = np.array([15.0, 15.0, 15.0, 0.02, 0.02, 15.0], dtype=np.float32)
        X_normal = rng.normal(loc=loc, scale=scale, size=(30, 6)).astype(np.float32)
        model = IsolationForest(n_estimators=50, contamination=0.05, random_state=42)
        model.fit(X_normal)
        scores = model.score_samples(X_normal)

        joblib.dump(model, tmp_path / "passu_isolation_forest.joblib")
        (tmp_path / "passu_isolation_forest.json").write_text(json.dumps({
            "training_mean_score": float(scores.mean()),
            "training_std_score": float(scores.std()),
        }))
        return tmp_path

    def _make_clean_bands(self, mndwi_value: float = 0.2) -> dict[str, np.ndarray]:
        """Bands with controllable MNDWI. MNDWI = (B03 - B11) / (B03 + B11)."""
        # Solving for B11 given B03=600 and desired MNDWI:
        # mndwi = (600 - B11) / (600 + B11) → B11 = 600*(1-mndwi)/(1+mndwi)
        b11 = 600.0 * (1.0 - mndwi_value) / (1.0 + mndwi_value)
        bands = {
            "B03": np.full((10, 10), 600.0, dtype=np.float32),
            "B04": np.full((10, 10), 500.0, dtype=np.float32),
            "B05": np.full((10, 10), 500.0, dtype=np.float32),
            "B08": np.full((10, 10), 500.0, dtype=np.float32),
            "B11": np.full((10, 10), b11, dtype=np.float32),
            "B12": np.full((10, 10), 500.0, dtype=np.float32),
            "SCL": np.zeros((10, 10), dtype=np.int32),  # all clear
        }
        return bands

    def test_normal_scene_not_flagged(self, model_dir):
        """Scene with normal spectral signature → seepage_flag=False."""
        from pipeline.anomaly import compute_anomaly

        today = date.today()
        # Build MIN_SCENES_FOR_INFERENCE normal scenes.
        scenes = [(today - timedelta(days=i), self._make_clean_bands(0.2))
                  for i in range(MIN_SCENES_FOR_INFERENCE - 1, -1, -1)]
        source = _make_scene_source(scenes)

        result = compute_anomaly(source, (74.0, 36.0, 74.1, 36.1), "moraine", str(model_dir), "passu")
        assert result.method == "isolation_forest"
        # A normal scene close to the training distribution should not exceed 3σ.
        assert result.seepage_flag is False

    def test_anomalous_scene_flagged(self, model_dir):
        """Scene with extreme spectral values (far from training distribution) → seepage_flag=True."""
        from sklearn.ensemble import IsolationForest
        import json, joblib

        # Fit a model on a very tight cluster; any outlier will be far from it.
        rng = np.random.default_rng(99)
        X_tight = rng.normal(loc=0.5, scale=0.001, size=(30, 6)).astype(np.float32)
        model = IsolationForest(n_estimators=50, contamination=0.05, random_state=42)
        model.fit(X_tight)
        scores = model.score_samples(X_tight)
        joblib.dump(model, model_dir / "passu_isolation_forest.joblib")
        (model_dir / "passu_isolation_forest.json").write_text(json.dumps({
            "training_mean_score": float(scores.mean()),
            "training_std_score": float(scores.std()),
        }))

        from pipeline.anomaly import compute_anomaly
        today = date.today()

        # Outlier scene: extreme values (far outside the tight training cluster).
        def _outlier_bands():
            b = self._make_clean_bands(0.2)
            # SWIR1 (B11) massively elevated → spectral anomaly.
            b["B11"] = np.full((10, 10), 9999.0, dtype=np.float32)
            return b

        scenes = [(today - timedelta(days=i), _outlier_bands())
                  for i in range(MIN_SCENES_FOR_INFERENCE - 1, -1, -1)]
        source = _make_scene_source(scenes)

        result = compute_anomaly(source, (74.0, 36.0, 74.1, 36.1), "moraine", str(model_dir), "passu")
        assert result.method == "isolation_forest"
        assert result.seepage_flag is True
        assert result.seepage_score_sigma is not None
        assert result.seepage_score_sigma < -SEEPAGE_SIGMA_THRESHOLD

    def test_sudden_drainage_flagged(self, model_dir):
        """MNDWI drop > SUDDEN_DRAINAGE_DELTA between past and latest scene → drainage_flag=True."""
        from pipeline.anomaly import compute_anomaly

        today = date.today()
        past = today - timedelta(days=31)

        # Past scene: high MNDWI (wet moraine)
        past_bands = self._make_clean_bands(mndwi_value=0.4)
        # Latest scene: low MNDWI (dry — sudden drainage)
        latest_bands = self._make_clean_bands(mndwi_value=0.4 + SUDDEN_DRAINAGE_DELTA - 0.05)

        # Need MIN_SCENES_FOR_INFERENCE total scenes.
        scenes = [
            (past, past_bands),
            (today - timedelta(days=15), self._make_clean_bands(0.35)),
            (today, latest_bands),
        ]
        source = _make_scene_source(scenes)

        result = compute_anomaly(source, (74.0, 36.0, 74.1, 36.1), "moraine", str(model_dir), "passu")
        assert result.drainage_flag is True
        assert result.drainage_delta_mndwi is not None
        assert result.drainage_delta_mndwi < SUDDEN_DRAINAGE_DELTA

    def test_drainage_not_flagged_small_drop(self, model_dir):
        """MNDWI drop smaller than threshold → drainage_flag=False."""
        from pipeline.anomaly import compute_anomaly

        today = date.today()
        past = today - timedelta(days=31)

        # Small MNDWI change — well within normal variation.
        scenes = [
            (past, self._make_clean_bands(mndwi_value=0.3)),
            (today - timedelta(days=15), self._make_clean_bands(0.28)),
            (today, self._make_clean_bands(mndwi_value=0.27)),  # Δ = -0.03 < 0.15
        ]
        source = _make_scene_source(scenes)

        result = compute_anomaly(source, (74.0, 36.0, 74.1, 36.1), "moraine", str(model_dir), "passu")
        assert result.drainage_flag is False

    def test_anomaly_result_is_frozen(self, model_dir):
        """AnomalyResult must be frozen (immutable)."""
        result = AnomalyResult(
            method="test",
            seepage_score_sigma=None,
            drainage_delta_mndwi=None,
            seepage_flag=False,
            drainage_flag=False,
            scenes_used=0,
            latest_scene_date=None,
        )
        with pytest.raises(Exception):  # FrozenInstanceError
            result.seepage_flag = True  # type: ignore[misc]

    def test_scene_fetch_failure_returns_insufficient(self, model_dir):
        """Scene fetch exception → method='insufficient_scenes', no raise."""
        from pipeline.anomaly import compute_anomaly

        source = MagicMock()
        source.find_scenes_in_range.side_effect = RuntimeError("network down")

        result = compute_anomaly(source, (74.0, 36.0, 74.1, 36.1), "moraine", str(model_dir), "passu")
        assert result.method == "insufficient_scenes"
        assert result.seepage_flag is False


# ---------------------------------------------------------------------------
# Slow integration test — real IF fit on synthetic 30-scene history
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_full_compute_anomaly_synthetic_history(tmp_path):
    """Fit a real IsolationForest on 30 synthetic scenes and verify the full
    compute_anomaly() path works end-to-end with a mock SceneSource."""
    from sklearn.ensemble import IsolationForest
    import json, joblib
    from pipeline.anomaly import compute_anomaly

    rng = np.random.default_rng(0)

    # Build 30 "historical" (normal) scenes for training.
    X_train = rng.normal(loc=[400, 500, 300, 0.15, 0.20, 250], scale=20, size=(30, 6)).astype(np.float32)
    model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    model.fit(X_train)
    train_scores = model.score_samples(X_train)

    joblib.dump(model, tmp_path / "passu_isolation_forest.joblib")
    (tmp_path / "passu_isolation_forest.json").write_text(json.dumps({
        "training_mean_score": float(train_scores.mean()),
        "training_std_score": float(train_scores.std()),
    }))

    # Build MIN_SCENES_FOR_INFERENCE "current" scenes near the training mean → normal.
    today = date.today()

    def _bands_from_vector(vec: list[float]) -> dict[str, np.ndarray]:
        """Reconstruct approximate bands from a 6-dim vector [SWIR1,SWIR2,RE,NDWI,MNDWI,Red]."""
        b11, b12, b05, ndwi, mndwi, b04 = vec
        # Approximate B03 and B08 from NDWI = (B03-B08)/(B03+B08).
        # If ndwi > 0: B03 > B08. Use B03=600 as reference, solve B08.
        b03 = 600.0
        denom = 1.0 - ndwi
        b08 = b03 * (1.0 - ndwi) / (1.0 + ndwi) if abs(1 + ndwi) > 1e-6 else b03
        return {
            "B03": np.full((5, 5), b03, dtype=np.float32),
            "B04": np.full((5, 5), b04, dtype=np.float32),
            "B05": np.full((5, 5), b05, dtype=np.float32),
            "B08": np.full((5, 5), b08, dtype=np.float32),
            "B11": np.full((5, 5), b11, dtype=np.float32),
            "B12": np.full((5, 5), b12, dtype=np.float32),
            "SCL": np.zeros((5, 5), dtype=np.int32),
        }

    scenes = [
        (today - timedelta(days=i), _bands_from_vector(X_train[i].tolist()))
        for i in range(MIN_SCENES_FOR_INFERENCE - 1, -1, -1)
    ]
    source = _make_scene_source(scenes)

    result = compute_anomaly(source, (74.0, 36.0, 74.1, 36.1), "moraine", str(tmp_path), "passu")

    assert result.method == "isolation_forest"
    assert result.scenes_used == MIN_SCENES_FOR_INFERENCE
    assert result.seepage_score_sigma is not None
    assert result.latest_scene_date == today.isoformat()
    # Normal scene from training distribution should not exceed the threshold.
    assert result.seepage_flag is False
