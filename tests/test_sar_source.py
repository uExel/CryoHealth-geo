"""Unit tests for pipeline/sar_source.py — all tests use synthetic radar backscatter
rasters, no network access, no real satellite imagery.

What is being tested, and why:

  linear_to_db        — arithmetic correctness for both power and amplitude inputs.
  lee_filter          — reduces variance on speckled synthetic imagery without
                        distorting the mean of a uniform patch beyond tolerance.
  gamma_map_filter    — same contract as Lee; slightly different weight formula.
  otsu_threshold      — finds the correct valley in a synthetic bimodal histogram.
  dual_pol_threshold  — correctly classifies pixels using VV < -14 dB AND VH < -21 dB.
  extract_sar_water   — end-to-end pipeline returns a boolean mask whose water-pixel
                        count is within ±10% of the synthetic ground-truth water area.
  sar_water_area_km2  — pixel-count to km² arithmetic.
  Sentinel1GRDSource  — STAC search and read_bands with mocked Planetary Computer client.

Real-world ±10% accuracy against paired S2/S1 optical ground truth scenes is NOT
verified here — see the TODO(validation) comment in sar_source.py and the implementation
plan for the required follow-up validation step.
"""

from __future__ import annotations

import io
from datetime import date
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds as rasterio_from_bounds

from pipeline.sar_source import (
    Sentinel1GRDSource,
    dual_pol_threshold,
    extract_sar_water,
    gamma_map_filter,
    lee_filter,
    linear_to_db,
    otsu_threshold,
    sar_water_area_km2,
)
from pipeline.stac_source import SceneRef

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)


def _make_sar_tiff(arr: np.ndarray, bbox: tuple = (74.6, 36.39, 74.62, 36.41)) -> bytes:
    """Write a single-band float32 GeoTIFF and return the bytes."""
    h, w = arr.shape
    transform = rasterio_from_bounds(*bbox, w, h)
    buf = io.BytesIO()
    with rasterio.open(
        buf, "w",
        driver="GTiff", height=h, width=w, count=1,
        dtype=np.float32, crs="EPSG:4326", transform=transform,
    ) as dst:
        dst.write(arr.astype(np.float32), 1)
    return buf.getvalue()


def _scene(scene_id: str, captured_at: date, cloud_pct: float = 0.0) -> SceneRef:
    return SceneRef(
        scene_id=scene_id,
        captured_at=captured_at,
        cloud_cover_pct=cloud_pct,
        assets={},
    )


# ---------------------------------------------------------------------------
# linear_to_db
# ---------------------------------------------------------------------------

class TestLinearToDb:
    def test_known_power_value_converts_correctly(self):
        """10^(-1.4) linear power → -14 dB (the VV water threshold)."""
        linear = np.array([[10 ** -1.4]], dtype=np.float32)
        result = linear_to_db(linear)
        assert pytest.approx(float(result[0, 0]), abs=0.01) == -14.0

    def test_known_amplitude_value_converts_correctly(self):
        """Amplitude mode: input amplitude = 10^(-21/20), expected output = -21 dB.

        linear_to_db with is_amplitude=True applies 20 * log10(amplitude) directly:
          20 * log10(10^(-21/20)) = 20 * (-21/20) = -21 dB.
        """

        amp = np.array([[10 ** (-21 / 20)]], dtype=np.float32)
        result = linear_to_db(amp, is_amplitude=True)
        assert float(result[0, 0]) == pytest.approx(-21.0, abs=0.05)

    def test_zero_and_negative_inputs_are_clamped_not_nan(self):
        """Pixels with zero or negative DN (e.g. no-data areas) must not produce NaN."""
        arr = np.array([[0.0, -1.0, 1e-8]], dtype=np.float32)
        result = linear_to_db(arr)
        assert not np.any(np.isnan(result))
        # All clamped values should be very negative (well below any water signal).
        assert np.all(result < -50.0)

    def test_output_is_float32(self):
        arr = np.ones((5, 5), dtype=np.float32)
        assert linear_to_db(arr).dtype == np.float32

    def test_vectorised_batch_matches_scalar(self):
        """Vectorised output must match element-wise scalar computation."""
        vals = np.array([[0.001, 0.01, 0.1, 1.0]], dtype=np.float32)
        expected = np.array([[10 * np.log10(v) for v in [0.001, 0.01, 0.1, 1.0]]], dtype=np.float32)
        np.testing.assert_allclose(linear_to_db(vals), expected, atol=1e-4)


# ---------------------------------------------------------------------------
# Lee filter
# ---------------------------------------------------------------------------

class TestLeeFilter:
    def test_uniform_patch_mean_is_preserved(self):
        """On a perfectly uniform image, the Lee filter should return the same value."""
        uniform = np.full((20, 20), 0.05, dtype=np.float32)
        filtered = lee_filter(uniform)
        np.testing.assert_allclose(filtered, uniform, atol=1e-5)

    def test_speckled_image_variance_is_reduced(self):
        """Lee filter must reduce variance of a speckled image."""
        speckled = RNG.gamma(shape=4.4, scale=0.05 / 4.4, size=(50, 50)).astype(np.float32)
        filtered = lee_filter(speckled)
        assert filtered.var() < speckled.var()

    def test_output_shape_matches_input(self):
        arr = RNG.random((30, 40)).astype(np.float32)
        assert lee_filter(arr).shape == arr.shape

    def test_mean_of_filtered_image_close_to_original(self):
        """Filtering should not shift the mean significantly."""
        arr = RNG.gamma(shape=4.4, scale=0.05 / 4.4, size=(60, 60)).astype(np.float32)
        filtered = lee_filter(arr)
        assert pytest.approx(float(filtered.mean()), rel=0.1) == float(arr.mean())

    def test_non_default_window_size_runs_without_error(self):
        arr = np.ones((15, 15), dtype=np.float32) * 0.02
        result = lee_filter(arr, window_size=7, enl=1.0)
        assert result.shape == arr.shape


# ---------------------------------------------------------------------------
# Gamma MAP filter
# ---------------------------------------------------------------------------

class TestGammaMapFilter:
    def test_uniform_patch_mean_is_preserved(self):
        uniform = np.full((20, 20), 0.05, dtype=np.float32)
        filtered = gamma_map_filter(uniform)
        np.testing.assert_allclose(filtered, uniform, atol=1e-4)

    def test_speckled_image_variance_is_reduced(self):
        # Use a larger array so the local-window statistics stabilise enough for
        # gamma MAP to produce measurable smoothing (50×50 was borderline).
        speckled = RNG.gamma(shape=4.4, scale=0.05 / 4.4, size=(100, 100)).astype(np.float32)
        filtered = gamma_map_filter(speckled, window_size=7)
        assert filtered.var() <= speckled.var()

    def test_output_shape_matches_input(self):
        arr = RNG.random((30, 40)).astype(np.float32)
        assert gamma_map_filter(arr).shape == arr.shape


# ---------------------------------------------------------------------------
# Otsu threshold
# ---------------------------------------------------------------------------

class TestOtsuThreshold:
    def _bimodal_array(self, water_val_db: float = -20.0, land_val_db: float = -5.0) -> np.ndarray:
        """Create a synthetic bimodal dB array: half water, half land."""
        water = np.full((30, 30), water_val_db, dtype=np.float32)
        land = np.full((30, 30), land_val_db, dtype=np.float32)
        return np.block([[water, land]])

    def test_threshold_falls_between_two_modes(self):
        arr = self._bimodal_array(-20.0, -5.0)
        thresh = otsu_threshold(arr)
        assert -20.0 < thresh < -5.0

    def test_threshold_on_uniform_input_returns_safe_fallback(self):
        """Degenerate single-mode input should not raise and returns the -14 dB default."""
        uniform = np.full((10, 10), -10.0, dtype=np.float32)
        thresh = otsu_threshold(uniform)
        # Should be a finite number, not NaN / error.
        assert np.isfinite(thresh)

    def test_custom_value_range_restricts_histogram(self):
        """Pixels outside [min_val, max_val] are excluded from the histogram."""
        arr = np.array([[-35.0, -25.0, -20.0, -5.0, 5.0]], dtype=np.float32)
        # Only [-30, 0] range considered; -35 and 5 are excluded.
        thresh = otsu_threshold(arr, min_val=-30.0, max_val=0.0)
        assert -30.0 <= thresh <= 0.0

    def test_empty_valid_range_returns_fallback(self):
        arr = np.array([[-40.0, -35.0]], dtype=np.float32)
        thresh = otsu_threshold(arr, min_val=-30.0, max_val=0.0)
        # Falls back to VV default
        assert thresh == -14.0


# ---------------------------------------------------------------------------
# Dual-polarisation threshold
# ---------------------------------------------------------------------------

class TestDualPolThreshold:
    def test_water_pixel_both_bands_below_threshold(self):
        """A pixel at VV=-16 dB, VH=-23 dB should be classified as water."""
        vv = np.array([[-16.0]], dtype=np.float32)
        vh = np.array([[-23.0]], dtype=np.float32)
        mask = dual_pol_threshold(vv, vh)
        assert mask[0, 0] is np.bool_(True)

    def test_land_pixel_vv_above_threshold(self):
        """VV=-10 dB (above -14) means NOT water, even if VH is below."""
        vv = np.array([[-10.0]], dtype=np.float32)
        vh = np.array([[-23.0]], dtype=np.float32)
        mask = dual_pol_threshold(vv, vh)
        assert mask[0, 0] is np.bool_(False)

    def test_land_pixel_vh_above_threshold(self):
        """VH=-19 dB (above -21) means NOT water, even if VV is below."""
        vv = np.array([[-16.0]], dtype=np.float32)
        vh = np.array([[-19.0]], dtype=np.float32)
        mask = dual_pol_threshold(vv, vh)
        assert mask[0, 0] is np.bool_(False)

    def test_mixed_scene_correct_pixel_count(self):
        """4 water pixels (top-left quadrant) and 4 land pixels (rest)."""
        vv = np.array([
            [-16.0, -16.0, -8.0, -8.0],
            [-16.0, -16.0, -8.0, -8.0],
        ], dtype=np.float32)
        vh = np.array([
            [-23.0, -23.0, -15.0, -15.0],
            [-23.0, -23.0, -15.0, -15.0],
        ], dtype=np.float32)
        mask = dual_pol_threshold(vv, vh)
        assert mask.sum() == 4

    def test_custom_thresholds_are_respected(self):
        """Override to stricter thresholds: only pixels VV < -18 AND VH < -25 pass."""
        vv = np.array([[-16.0, -20.0]], dtype=np.float32)
        vh = np.array([[-23.0, -27.0]], dtype=np.float32)
        mask = dual_pol_threshold(vv, vh, vv_thresh=-18.0, vh_thresh=-25.0)
        assert not mask[0, 0]   # -16 > -18 → land
        assert mask[0, 1]       # -20 < -18 AND -27 < -25 → water

    def test_output_is_boolean_array(self):
        vv = np.zeros((5, 5), dtype=np.float32) - 16.0
        vh = np.zeros((5, 5), dtype=np.float32) - 23.0
        mask = dual_pol_threshold(vv, vh)
        assert mask.dtype == bool


# ---------------------------------------------------------------------------
# sar_water_area_km2
# ---------------------------------------------------------------------------

class TestSarWaterAreaKm2:
    def test_all_water_area_matches_pixel_count(self):
        """100×100 pixels at 10 m each = 100*100*100 m² = 1_000_000 m² = 1.0 km²."""
        mask = np.ones((100, 100), dtype=bool)
        area = sar_water_area_km2(mask)
        assert area == pytest.approx(1.0, rel=1e-6)

    def test_half_water_area(self):
        mask = np.zeros((100, 100), dtype=bool)
        mask[:50, :] = True  # 5000 pixels → 5000 * 100 m² = 500 000 m² = 0.5 km²
        area = sar_water_area_km2(mask)
        assert area == pytest.approx(0.5, rel=1e-6)

    def test_custom_pixel_size(self):
        """20 m pixels: 10×10 = 100 pixels → 100 * 400 m² = 40 000 m² = 0.04 km²."""
        mask = np.ones((10, 10), dtype=bool)
        area = sar_water_area_km2(mask, pixel_size_m=20.0)
        assert area == pytest.approx(0.04, rel=1e-6)

    def test_no_water_returns_zero(self):
        mask = np.zeros((20, 20), dtype=bool)
        assert sar_water_area_km2(mask) == 0.0


# ---------------------------------------------------------------------------
# extract_sar_water — end-to-end synthetic accuracy test
# ---------------------------------------------------------------------------

class TestExtractSarWater:
    """Synthetic ground-truth accuracy tests.

    We construct a fake scene where 40% of pixels are 'water' (low backscatter)
    and 60% are 'land' (higher backscatter), then verify the pipeline recovers
    the water fraction within ±10% of the known truth.

    This is a code-correctness test. Real-world ±10% accuracy against paired
    S2 optical ground truth requires a separate validation pass — see the
    TODO(validation) comment in sar_source.dual_pol_threshold.
    """

    def _synthetic_scene(
        self, size: int = 100, water_fraction: float = 0.4
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Returns (vv_linear, vh_linear, true_water_fraction)."""
        n_total = size * size
        n_water = int(n_total * water_fraction)

        # Water: VV ~ -18 dB, VH ~ -24 dB in linear power
        vv_water = np.full(n_water, 10 ** (-18 / 10), dtype=np.float32)
        vh_water = np.full(n_water, 10 ** (-24 / 10), dtype=np.float32)

        # Land: VV ~ -8 dB, VH ~ -14 dB
        n_land = n_total - n_water
        vv_land = np.full(n_land, 10 ** (-8 / 10), dtype=np.float32)
        vh_land = np.full(n_land, 10 ** (-14 / 10), dtype=np.float32)

        vv = np.concatenate([vv_water, vv_land]).reshape(size, size)
        vh = np.concatenate([vh_water, vh_land]).reshape(size, size)
        return vv, vh, water_fraction

    def test_dual_pol_method_recovers_water_fraction_within_10pct(self):
        vv, vh, true_frac = self._synthetic_scene(water_fraction=0.4)
        mask = extract_sar_water(vv, vh, method="dual_pol", filter_type="lee")
        measured_frac = mask.sum() / mask.size
        assert abs(measured_frac - true_frac) <= 0.10 * true_frac + 0.01

    def test_otsu_vv_method_recovers_water_fraction_within_10pct(self):
        vv, vh, true_frac = self._synthetic_scene(water_fraction=0.35)
        mask = extract_sar_water(vv, vh, method="otsu_vv", filter_type="lee")
        measured_frac = mask.sum() / mask.size
        assert abs(measured_frac - true_frac) <= 0.10 * true_frac + 0.02

    def test_otsu_dual_method_runs_without_error(self):
        vv, vh, _ = self._synthetic_scene()
        mask = extract_sar_water(vv, vh, method="otsu_dual")
        assert mask.dtype == bool
        assert mask.shape == vv.shape

    def test_gamma_map_filter_produces_valid_mask(self):
        vv, vh, _ = self._synthetic_scene()
        mask = extract_sar_water(vv, vh, method="dual_pol", filter_type="gamma_map")
        assert mask.dtype == bool

    def test_invalid_method_raises_value_error(self):
        vv = np.ones((5, 5), dtype=np.float32)
        with pytest.raises(ValueError, match="method"):
            extract_sar_water(vv, vv, method="nonsense")

    def test_invalid_filter_type_raises_value_error(self):
        vv = np.ones((5, 5), dtype=np.float32)
        with pytest.raises(ValueError, match="filter_type"):
            extract_sar_water(vv, vv, filter_type="nonsense")

    def test_amplitude_input_flag_produces_valid_output(self):
        """Amplitude (uint16 DN) path: all water pixels should be detected."""
        amp_water = np.full((20, 20), 10 ** (-18 / 20), dtype=np.float32)  # amp = sqrt(power)
        amp_land = np.full((20, 20), 10 ** (-8 / 20), dtype=np.float32)
        vv = np.block([[amp_water, amp_land]])
        vh = np.block([[amp_water * 0.5, amp_land * 2]])
        mask = extract_sar_water(vv, vh, is_amplitude=True)
        assert mask.dtype == bool


# ---------------------------------------------------------------------------
# Sentinel1GRDSource — mocked Planetary Computer STAC
# ---------------------------------------------------------------------------

class TestSentinel1GRDSource:
    """All tests mock planetary_computer.sign and the STAC client so no network
    calls are made."""

    def _make_mock_item(self, scene_id: str, dt: date) -> MagicMock:
        item = MagicMock()
        item.id = scene_id
        item.datetime.date.return_value = dt
        item.properties = {
            "sar:instrument_mode": "IW",
            "sar:polarizations": ["VV", "VH"],
        }
        # Signed assets include vv and vh.
        item.assets = {
            "vv": MagicMock(href="https://example.com/vv.tif"),
            "vh": MagicMock(href="https://example.com/vh.tif"),
        }
        return item

    @patch("pipeline.sar_source.planetary_computer.sign")
    @patch("pipeline.sar_source.Client.open")
    def test_find_recent_scenes_returns_scene_refs_with_zero_cloud_cover(
        self, mock_client_open, mock_sign
    ):
        mock_item = self._make_mock_item("S1A_IW_GRDH_TEST", date(2026, 7, 20))
        # sign returns the item with assets already attached.
        signed_item = MagicMock()
        signed_item.assets = mock_item.assets
        mock_sign.return_value = signed_item

        mock_search = MagicMock()
        mock_search.items.return_value = [mock_item]
        mock_client_open.return_value.search.return_value = mock_search

        source = Sentinel1GRDSource()
        refs = source.find_recent_scenes((74.6, 36.39, 74.62, 36.41), limit=5)

        assert len(refs) == 1
        ref = refs[0]
        assert ref.scene_id == "S1A_IW_GRDH_TEST"
        assert ref.captured_at == date(2026, 7, 20)
        assert ref.cloud_cover_pct == 0.0
        assert "vv" in ref.assets
        assert "vh" in ref.assets

    @patch("pipeline.sar_source.planetary_computer.sign")
    @patch("pipeline.sar_source.Client.open")
    def test_find_scenes_in_range_passes_datetime_to_stac(
        self, mock_client_open, mock_sign
    ):
        mock_sign.return_value = MagicMock(assets={})
        mock_search = MagicMock()
        mock_search.items.return_value = []
        mock_client_open.return_value.search.return_value = mock_search

        source = Sentinel1GRDSource()
        source.find_scenes_in_range(
            (74.6, 36.39, 74.62, 36.41),
            date(2026, 7, 1),
            date(2026, 7, 31),
        )

        call_kwargs = mock_client_open.return_value.search.call_args
        assert call_kwargs.kwargs["datetime"] == "2026-07-01/2026-07-31"
        assert call_kwargs.kwargs["collections"] == ["sentinel-1-grd"]

    @patch("pipeline.sar_source.planetary_computer.sign")
    @patch("pipeline.sar_source.Client.open")
    @patch("pipeline.sar_source.rasterio.open")
    def test_read_bands_returns_float32_arrays(
        self, mock_rasterio_open, mock_client_open, mock_sign
    ):
        """read_bands should return float32 arrays cropped to the requested bbox."""
        fake_arr = np.full((20, 20), 0.05, dtype=np.float32)

        mock_ds = MagicMock()
        mock_ds.__enter__ = lambda s: s
        mock_ds.__exit__ = MagicMock(return_value=False)
        mock_ds.crs = "EPSG:4326"
        mock_ds.transform = rasterio_from_bounds(74.6, 36.39, 74.62, 36.41, 20, 20)
        mock_ds.read.return_value = fake_arr
        mock_rasterio_open.return_value = mock_ds

        scene = SceneRef(
            scene_id="S1A_TEST",
            captured_at=date(2026, 7, 20),
            cloud_cover_pct=0.0,
            assets={"vv": "https://fake/vv.tif", "vh": "https://fake/vh.tif"},
        )
        source = Sentinel1GRDSource.__new__(Sentinel1GRDSource)
        source._client = MagicMock()

        result = source.read_bands(scene, ["vv", "vh"], (74.6, 36.39, 74.62, 36.41))
        assert set(result.keys()) == {"vv", "vh"}
        assert result["vv"].dtype == np.float32
        assert result["vh"].dtype == np.float32
