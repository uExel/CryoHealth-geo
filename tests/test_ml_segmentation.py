"""Tests for pipeline/alphaearth_source.py, pipeline/ml_segmentation.py, pipeline/geom.py.

All tests are fully mocked — no live GEE calls, no real ONNX weights, no CUDA.
Follows the same pattern as test_ndwi.py and test_sar_source.py.

Slow/integration tests requiring real credentials are marked @pytest.mark.slow
and excluded from CI: uv run pytest -m "not slow"
Run integration tests with: uv run pytest -m slow
"""

from __future__ import annotations

import os
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_embeddings(h: int = 32, w: int = 32) -> np.ndarray:
    """Synthetic 64-band AlphaEarth embedding patch."""
    rng = np.random.default_rng(42)
    return rng.uniform(-2.0, 2.0, (64, h, w)).astype(np.float32)


def _make_mock_onnx_session(logits: np.ndarray) -> MagicMock:
    session = MagicMock()
    inp = MagicMock()
    inp.name = "input"
    session.get_inputs.return_value = [inp]
    session.run.return_value = [logits]
    return session


# ===========================================================================
# alphaearth_source — is_available()
# ===========================================================================


def test_ae_is_available_false_when_earthengine_api_not_installed():
    import sys
    modules = {k: v for k, v in sys.modules.items() if k != "ee"}
    with patch.dict(sys.modules, {"ee": None}):
        from importlib import reload
        import pipeline.alphaearth_source as ae_mod
        # Patch importlib so that 'import ee' raises ImportError inside is_available
        with patch("builtins.__import__", side_effect=lambda name, *a, **kw: (_ for _ in ()).throw(ImportError()) if name == "ee" else __import__(name, *a, **kw)):
            result = ae_mod.is_available()
    assert result is False


def test_ae_is_available_false_when_no_credentials_and_no_adc(tmp_path):
    """is_available() is False when service account vars unset and ADC file missing."""
    from pipeline.alphaearth_source import is_available

    env = {k: v for k, v in os.environ.items()
           if k not in ("EE_SERVICE_ACCOUNT", "EE_PRIVATE_KEY_JSON")}
    with patch.dict(os.environ, env, clear=True):
        with patch.dict("sys.modules", {"ee": MagicMock()}):
            with patch("pathlib.Path.exists", return_value=False):
                result = is_available()
    assert result is False


def test_ae_is_available_true_with_service_account_vars(tmp_path):
    """is_available() is True when service account env vars are set."""
    key = tmp_path / "key.json"
    key.write_text("{}")
    from pipeline.alphaearth_source import is_available

    with patch.dict(os.environ, {
        "EE_SERVICE_ACCOUNT": "sa@proj.iam.gserviceaccount.com",
        "EE_PRIVATE_KEY_JSON": str(key),
    }):
        with patch.dict("sys.modules", {"ee": MagicMock()}):
            result = is_available()
    assert result is True


# ===========================================================================
# alphaearth_source — fetch_embeddings()
# ===========================================================================


def test_fetch_embeddings_raises_on_invalid_year():
    from pipeline.alphaearth_source import fetch_embeddings

    bbox = (74.60, 36.39, 74.62, 36.41)
    with pytest.raises(ValueError, match="2017–2024"):
        fetch_embeddings(bbox, year=2010)

    with pytest.raises(ValueError, match="2017–2024"):
        fetch_embeddings(bbox, year=2030)


def test_fetch_embeddings_returns_64_band_array():
    """fetch_embeddings returns (64, H, W) float32 when GEE returns valid data."""
    from pipeline.alphaearth_source import clear_embedding_cache, fetch_embeddings

    clear_embedding_cache()
    bbox = (74.60, 36.39, 74.62, 36.41)
    year = 2023

    # Mock the earthengine-api and the sampleRectangle call.
    mock_ee = MagicMock()
    # sampleRectangle returns one 32×32 band worth of data for each of 64 bands.
    band_data = [[float(i) for i in range(32)] for _ in range(32)]
    mock_ee.ServiceAccountCredentials.return_value = MagicMock()

    mock_sample = MagicMock()
    mock_sample.get.return_value.getInfo.return_value = band_data
    mock_image = MagicMock()
    mock_image.sampleRectangle.return_value = mock_sample
    mock_collection = MagicMock()
    mock_collection.filterDate.return_value.filterBounds.return_value.mosaic.return_value.select.return_value = mock_image
    mock_ee.ImageCollection.return_value = mock_collection
    mock_ee.Geometry.Rectangle.return_value = MagicMock()
    mock_ee.data._credentials = None  # trigger initialisation

    with patch.dict(os.environ, {
        "EE_SERVICE_ACCOUNT": "sa@proj.iam.gserviceaccount.com",
        "EE_PRIVATE_KEY_JSON": "/fake/key.json",
    }):
        with patch.dict("sys.modules", {"ee": mock_ee}):
            with patch("pipeline.alphaearth_source._authenticate_ee"):
                result = fetch_embeddings(bbox, year)

    assert result.shape == (64, 32, 32)
    assert result.dtype == np.float32


def test_fetch_embeddings_cache_hit_skips_second_gee_call():
    """Second call with same (bbox, year) returns cached array without calling GEE."""
    from pipeline.alphaearth_source import clear_embedding_cache, _cache_set

    clear_embedding_cache()
    bbox = (74.60, 36.39, 74.62, 36.41)
    year = 2022

    # Pre-populate cache
    fake_embeddings = np.ones((64, 16, 16), dtype=np.float32)
    _cache_set((bbox, year), fake_embeddings)

    mock_ee = MagicMock()
    with patch.dict("sys.modules", {"ee": mock_ee}):
        from pipeline.alphaearth_source import fetch_embeddings
        result = fetch_embeddings(bbox, year)

    # GEE should NOT have been called — cache hit
    mock_ee.ImageCollection.assert_not_called()
    np.testing.assert_array_equal(result, fake_embeddings)


def test_fetch_embeddings_cache_miss_on_different_year():
    """Different year = cache miss = GEE fetch attempted."""
    from pipeline.alphaearth_source import clear_embedding_cache, _cache_set

    clear_embedding_cache()
    bbox = (74.60, 36.39, 74.62, 36.41)
    _cache_set((bbox, 2022), np.ones((64, 8, 8), dtype=np.float32))

    # Requesting year=2023 (not in cache) should raise EmbeddingUnavailableError
    # because no real GEE is available in this test.
    from pipeline.alphaearth_source import EmbeddingUnavailableError, fetch_embeddings

    with pytest.raises(EmbeddingUnavailableError):
        fetch_embeddings(bbox, year=2023)


def test_fetch_embeddings_raises_on_timeout():
    """EmbeddingUnavailableError raised when GEE fetch exceeds timeout."""
    from pipeline.alphaearth_source import (
        EmbeddingUnavailableError,
        clear_embedding_cache,
        fetch_embeddings,
    )

    clear_embedding_cache()
    bbox = (74.60, 36.39, 74.62, 36.41)

    def slow_getinfo():
        time.sleep(0.2)
        return [[0.0] * 8 for _ in range(8)]

    mock_sample = MagicMock()
    mock_sample.get.return_value.getInfo.side_effect = slow_getinfo
    mock_image = MagicMock()
    mock_image.sampleRectangle.return_value = mock_sample
    mock_collection = MagicMock()
    mock_collection.filterDate.return_value.filterBounds.return_value.mosaic.return_value.select.return_value = mock_image
    mock_ee = MagicMock()
    mock_ee.ImageCollection.return_value = mock_collection
    mock_ee.Geometry.Rectangle.return_value = MagicMock()

    with patch.dict("sys.modules", {"ee": mock_ee}):
        with patch("pipeline.alphaearth_source._authenticate_ee"):
            with pytest.raises(EmbeddingUnavailableError, match="timed out"):
                fetch_embeddings(bbox, year=2023, timeout_s=0.05)


def test_clear_embedding_cache_forces_refetch():
    """clear_embedding_cache() causes the next call to miss the cache."""
    from pipeline.alphaearth_source import (
        EmbeddingUnavailableError,
        clear_embedding_cache,
        fetch_embeddings,
        _cache_set,
    )

    bbox = (74.60, 36.39, 74.62, 36.41)
    year = 2021
    _cache_set((bbox, year), np.ones((64, 4, 4), dtype=np.float32))
    clear_embedding_cache()

    # After clearing, no GEE is mocked, so it should fail rather than return cached data.
    with pytest.raises(EmbeddingUnavailableError):
        fetch_embeddings(bbox, year)


# ===========================================================================
# ml_segmentation — is_available()
# ===========================================================================


def test_ml_is_available_false_when_model_path_not_set():
    from pipeline.ml_segmentation import is_available

    env = {k: v for k, v in os.environ.items() if k != "ML_MODEL_PATH"}
    with patch.dict(os.environ, env, clear=True):
        assert is_available() is False


def test_ml_is_available_false_when_file_missing(tmp_path):
    from pipeline.ml_segmentation import is_available

    with patch.dict(os.environ, {"ML_MODEL_PATH": str(tmp_path / "ghost.onnx")}):
        with patch.dict("sys.modules", {"onnxruntime": MagicMock()}):
            assert is_available() is False


def test_ml_is_available_true_when_file_exists_and_ort_installed(tmp_path):
    from pipeline.ml_segmentation import is_available

    dummy = tmp_path / "adapter.onnx"
    dummy.write_bytes(b"fake")
    with patch.dict(os.environ, {"ML_MODEL_PATH": str(dummy)}):
        with patch.dict("sys.modules", {"onnxruntime": MagicMock()}):
            assert is_available() is True


# ===========================================================================
# ml_segmentation — run_inference()
# ===========================================================================


def test_run_inference_raises_on_wrong_embedding_shape():
    from pipeline.ml_segmentation import run_inference

    with pytest.raises(ValueError, match="shape \\(64, H, W\\)"):
        run_inference(np.ones((32, 32, 32), dtype=np.float32))

    with pytest.raises(ValueError, match="shape \\(64, H, W\\)"):
        run_inference(np.ones((10, 32, 32), dtype=np.float32))


def test_run_inference_returns_bool_mask_all_water(tmp_path):
    """Positive logits → sigmoid > 0.5 → all water."""
    from pipeline.ml_segmentation import reset_session, run_inference

    h, w = 32, 32
    embeddings = _make_embeddings(h, w)
    logits = np.ones((1, 1, h, w), dtype=np.float32) * 5.0
    session = _make_mock_onnx_session(logits)

    dummy = tmp_path / "adapter.onnx"
    dummy.write_bytes(b"fake")

    reset_session()
    with patch.dict(os.environ, {"ML_MODEL_PATH": str(dummy)}):
        with patch("pipeline.ml_segmentation._load_session", return_value=session):
            mask = run_inference(embeddings)

    assert mask.shape == (h, w)
    assert mask.dtype == bool
    assert mask.all()


def test_run_inference_returns_bool_mask_no_water(tmp_path):
    """Strongly negative logits → sigmoid < 0.5 → no water pixels."""
    from pipeline.ml_segmentation import reset_session, run_inference

    h, w = 16, 16
    embeddings = _make_embeddings(h, w)
    logits = np.ones((1, 1, h, w), dtype=np.float32) * -10.0
    session = _make_mock_onnx_session(logits)

    dummy = tmp_path / "adapter.onnx"
    dummy.write_bytes(b"fake")

    reset_session()
    with patch.dict(os.environ, {"ML_MODEL_PATH": str(dummy)}):
        with patch("pipeline.ml_segmentation._load_session", return_value=session):
            mask = run_inference(embeddings)

    assert not mask.any()


def test_run_inference_handles_3d_logit_output(tmp_path):
    """Model returning (1, H, W) logits (no channel dim) is handled correctly."""
    from pipeline.ml_segmentation import reset_session, run_inference

    h, w = 8, 8
    embeddings = _make_embeddings(h, w)
    logits = np.ones((1, h, w), dtype=np.float32) * 3.0
    session = _make_mock_onnx_session(logits)

    dummy = tmp_path / "adapter.onnx"
    dummy.write_bytes(b"fake")

    reset_session()
    with patch.dict(os.environ, {"ML_MODEL_PATH": str(dummy)}):
        with patch("pipeline.ml_segmentation._load_session", return_value=session):
            mask = run_inference(embeddings)

    assert mask.shape == (h, w)
    assert mask.dtype == bool


def test_run_inference_raises_unavailable_when_path_not_set():
    from pipeline.ml_segmentation import InferenceUnavailableError, reset_session, run_inference

    env = {k: v for k, v in os.environ.items() if k != "ML_MODEL_PATH"}
    reset_session()
    with patch.dict(os.environ, env, clear=True):
        with pytest.raises(InferenceUnavailableError):
            run_inference(_make_embeddings())


def test_run_inference_raises_unavailable_on_onnx_error(tmp_path):
    from pipeline.ml_segmentation import InferenceUnavailableError, reset_session, run_inference

    session = MagicMock()
    inp = MagicMock()
    inp.name = "input"
    session.get_inputs.return_value = [inp]
    session.run.side_effect = RuntimeError("ONNX exploded")

    dummy = tmp_path / "adapter.onnx"
    dummy.write_bytes(b"fake")

    reset_session()
    with patch.dict(os.environ, {"ML_MODEL_PATH": str(dummy)}):
        with patch("pipeline.ml_segmentation._load_session", return_value=session):
            with pytest.raises(InferenceUnavailableError, match="inference failed"):
                run_inference(_make_embeddings())


def test_run_inference_raises_on_timeout(tmp_path):
    from pipeline.ml_segmentation import InferenceUnavailableError, reset_session, run_inference

    def slow_run(*a, **kw):
        time.sleep(0.15)
        return [np.ones((1, 1, 32, 32), dtype=np.float32)]

    session = MagicMock()
    inp = MagicMock()
    inp.name = "input"
    session.get_inputs.return_value = [inp]
    session.run.side_effect = slow_run

    dummy = tmp_path / "adapter.onnx"
    dummy.write_bytes(b"fake")

    reset_session()
    with patch.dict(os.environ, {"ML_MODEL_PATH": str(dummy)}):
        with patch("pipeline.ml_segmentation._load_session", return_value=session):
            with pytest.raises(InferenceUnavailableError, match="timeout"):
                run_inference(_make_embeddings(), timeout_s=0.05)


# ===========================================================================
# water_area_km2_from_mask()
# ===========================================================================


def test_water_area_km2_empty_mask():
    from pipeline.ml_segmentation import water_area_km2_from_mask
    assert water_area_km2_from_mask(np.zeros((10, 10), dtype=bool)) == pytest.approx(0.0)


def test_water_area_km2_all_water():
    from pipeline.ml_segmentation import water_area_km2_from_mask
    # 100 pixels × 100 m² = 10,000 m² = 0.01 km²
    assert water_area_km2_from_mask(np.ones((10, 10), dtype=bool)) == pytest.approx(0.01)


def test_water_area_km2_partial():
    from pipeline.ml_segmentation import water_area_km2_from_mask
    mask = np.zeros((4, 4), dtype=bool)
    mask[0, 0] = True
    mask[2, 2] = True  # 2 pixels
    assert water_area_km2_from_mask(mask) == pytest.approx(2 * 100 / 1_000_000)


# ===========================================================================
# batch.py integration — fallback to NDWI when ML unavailable
# ===========================================================================


def test_batch_ndwi_fallback_when_alphaearth_unavailable():
    """_process_scene returns NDWI source_id when AlphaEarth is not available."""
    from pipeline.batch import _NDWI_SOURCE_ID, _process_scene

    rng = np.random.default_rng(1)
    h, w = 16, 16
    mock_source = MagicMock()
    mock_source.read_bands.return_value = {
        "B03": rng.uniform(100, 300, (h, w)).astype(np.float32),
        "B08": rng.uniform(50, 150, (h, w)).astype(np.float32),
        "SCL": np.full((h, w), 6, dtype=np.uint8),  # all usable
    }
    mock_scene = MagicMock()
    mock_scene.scene_id = "test-scene-001"
    mock_scene.captured_at.year = 2024

    env = {k: v for k, v in os.environ.items() if k != "ML_MODEL_PATH"}
    with patch.dict(os.environ, env, clear=True):
        with patch("pipeline.batch.ae_is_available", return_value=False):
            result = _process_scene(mock_source, mock_scene, (74.60, 36.39, 74.62, 36.41))

    assert result is not None
    area_km2, clouds, source_id = result
    assert source_id == _NDWI_SOURCE_ID
    assert area_km2 >= 0.0


def test_batch_ml_source_when_alphaearth_and_adapter_available(tmp_path):
    """_process_scene returns AlphaEarth source_id when both paths are available."""
    from pipeline.batch import _ML_SEG_SOURCE_ID, _process_scene

    rng = np.random.default_rng(2)
    h, w = 16, 16
    mock_source = MagicMock()
    mock_source.read_bands.return_value = {
        "B03": rng.uniform(100, 300, (h, w)).astype(np.float32),
        "B08": rng.uniform(50, 150, (h, w)).astype(np.float32),
        "SCL": np.full((h, w), 6, dtype=np.uint8),
    }
    mock_scene = MagicMock()
    mock_scene.scene_id = "test-scene-ml-001"
    mock_scene.captured_at.year = 2024

    water_mask = np.zeros((h, w), dtype=bool)
    water_mask[4:12, 4:12] = True  # 64 water pixels

    dummy = tmp_path / "adapter.onnx"
    dummy.write_bytes(b"fake")

    with patch.dict(os.environ, {"ML_MODEL_PATH": str(dummy)}):
        with patch("pipeline.batch.ae_is_available", return_value=True):
            with patch("pipeline.batch.ml_is_available", return_value=True):
                with patch("pipeline.batch.ae_fetch_embeddings",
                           return_value=_make_embeddings(h, w)):
                    with patch("pipeline.batch.ml_run_inference",
                               return_value=water_mask):
                        result = _process_scene(
                            mock_source, mock_scene, (74.60, 36.39, 74.62, 36.41)
                        )

    assert result is not None
    area_km2, clouds, source_id = result
    assert source_id == _ML_SEG_SOURCE_ID
    assert area_km2 > 0.0


def test_batch_falls_back_to_ndwi_on_embedding_error():
    """_process_scene falls back to NDWI when EmbeddingUnavailableError is raised."""
    from pipeline.alphaearth_source import EmbeddingUnavailableError
    from pipeline.batch import _NDWI_SOURCE_ID, _process_scene

    rng = np.random.default_rng(3)
    h, w = 16, 16
    mock_source = MagicMock()
    mock_source.read_bands.return_value = {
        "B03": rng.uniform(100, 300, (h, w)).astype(np.float32),
        "B08": rng.uniform(50, 150, (h, w)).astype(np.float32),
        "SCL": np.full((h, w), 6, dtype=np.uint8),
    }
    mock_scene = MagicMock()
    mock_scene.scene_id = "test-scene-err-001"
    mock_scene.captured_at.year = 2024

    with patch("pipeline.batch.ae_is_available", return_value=True):
        with patch("pipeline.batch.ml_is_available", return_value=True):
            with patch("pipeline.batch.ae_fetch_embeddings",
                       side_effect=EmbeddingUnavailableError("GEE timeout")):
                result = _process_scene(
                    mock_source, mock_scene, (74.60, 36.39, 74.62, 36.41)
                )

    assert result is not None
    _, _, source_id = result
    assert source_id == _NDWI_SOURCE_ID


# ===========================================================================
# geom.py — smooth_water_mask
# ===========================================================================


def test_smooth_water_mask_empty_input_no_crash():
    from pipeline.geom import smooth_water_mask
    mask = np.zeros((16, 16), dtype=bool)
    result = smooth_water_mask(mask)
    assert not result.any()


def test_smooth_water_mask_invalid_kernel_raises():
    from pipeline.geom import smooth_water_mask
    with pytest.raises(ValueError, match="odd integer"):
        smooth_water_mask(np.ones((8, 8), dtype=bool), kernel_size=2)


def test_smooth_water_mask_removes_isolated_pixel():
    pytest.importorskip("scipy")
    from pipeline.geom import smooth_water_mask
    mask = np.zeros((20, 20), dtype=bool)
    mask[10, 10] = True   # isolated — should be removed by opening
    mask[2:8, 2:8] = True  # solid block — should survive
    result = smooth_water_mask(mask, kernel_size=3)
    assert not result[10, 10]
    assert result[4, 4]


def test_smooth_water_mask_fills_interior_hole():
    pytest.importorskip("scipy")
    from pipeline.geom import smooth_water_mask
    mask = np.ones((20, 20), dtype=bool)
    mask[10, 10] = False  # single interior hole
    result = smooth_water_mask(mask, kernel_size=3)
    assert result[10, 10]  # hole filled by closing


# ===========================================================================
# Slow / integration tests (require real GEE credentials + ONNX weights)
# ===========================================================================


@pytest.mark.slow
def test_real_gee_fetch_returns_64_band_array():
    """Integration: real GEE call returns (64, H, W) float32.
    Requires: EE_SERVICE_ACCOUNT + EE_PRIVATE_KEY_JSON set in env.
    Run with: uv run pytest -m slow tests/test_ml_segmentation.py
    """
    from pipeline.alphaearth_source import clear_embedding_cache, fetch_embeddings
    from pipeline.lakes import LAKES

    clear_embedding_cache()
    bbox = LAKES["shishper"].bbox()
    embeddings = fetch_embeddings(bbox, year=2023)

    assert embeddings.ndim == 3
    assert embeddings.shape[0] == 64
    assert embeddings.dtype == np.float32
    print(f"\nShishper 2023 embeddings shape: {embeddings.shape}")


@pytest.mark.slow
def test_adapter_inference_under_2_5_seconds():
    """Integration: real adapter inference < 2.5 s on CPU.
    Requires: ML_MODEL_PATH set to real .onnx weights (Issue #19).
    """
    from pipeline.ml_segmentation import reset_session, run_inference

    embeddings = _make_embeddings(h=512, w=512)  # realistic AOI size

    reset_session()
    t0 = time.monotonic()
    mask = run_inference(embeddings)
    elapsed = time.monotonic() - t0

    assert elapsed < 2.5, f"Adapter inference took {elapsed:.2f} s — exceeds 2.5 s target"
    assert mask.shape == (512, 512)
    assert mask.dtype == bool
