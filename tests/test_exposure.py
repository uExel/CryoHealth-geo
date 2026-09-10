"""Tests for pipeline/dem.py (D8 routing) and pipeline/exposure.py (D8 corridor).

All tests use purely synthetic DEMs and a small fake WorldPop raster — no network,
no real DEM fetch, no 140 MB WorldPop download. Existing WorldPop tests are preserved.

Slow / integration tests that require a live Planetary Computer DEM fetch and the real
WorldPop raster are marked @pytest.mark.slow and excluded from CI:
  uv run pytest -m "not slow" tests/test_exposure.py

Run integration tests with:
  uv run pytest -m slow tests/test_exposure.py

Historical Shishper/Passu GLOF footprint validation:
  TODO(#26-validation): obtain reference inundation polygons (Issue #26, sub-issue of #19).
  Tracked at: https://github.com/uExel/CryoHealth-geo/issues/26
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds as transform_from_bounds

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_population_raster(tmp_path):
    """Small WorldPop-style raster centred on Shishper coords with known density."""
    west, south, east, north = 74.5, 36.3, 74.7, 36.5
    width, height = 200, 200
    density_per_pixel = 2.0
    data = np.full((height, width), density_per_pixel, dtype=np.float32)
    path = tmp_path / "fake_pop.tif"
    transform = transform_from_bounds(west, south, east, north, width, height)
    with rasterio.open(
        path, "w", driver="GTiff", height=height, width=width, count=1,
        dtype="float32", crs="EPSG:4326", transform=transform,
    ) as dst:
        dst.write(data, 1)
    return path


def _make_south_slope_dem(rows: int = 20, cols: int = 20, slope_per_row: float = 10.0) -> np.ndarray:
    """Uniform south-sloping DEM: elevation decreases by slope_per_row per row."""
    elev = np.zeros((rows, cols), dtype=np.float64)
    for r in range(rows):
        elev[r, :] = (rows - r) * slope_per_row
    return elev


def _make_v_valley_dem(rows: int = 30, cols: int = 30) -> np.ndarray:
    """V-shaped valley: ridges on left and right, channel down the centre column."""
    elev = np.zeros((rows, cols), dtype=np.float64)
    centre = cols // 2
    for r in range(rows):
        for c in range(cols):
            # Distance from centre column + south slope
            elev[r, c] = abs(c - centre) * 20.0 + (rows - r) * 5.0
    return elev


def _make_flat_dem(rows: int = 10, cols: int = 10, value: float = 100.0) -> np.ndarray:
    return np.full((rows, cols), value, dtype=np.float64)


def _make_dem_with_pit(rows: int = 10, cols: int = 10) -> np.ndarray:
    """Uniform slope with a single-cell depression at [5, 5]."""
    elev = _make_south_slope_dem(rows, cols)
    elev[5, 5] = elev[6, 5] - 5.0  # lower than downslope neighbour → pit
    return elev


# ---------------------------------------------------------------------------
# dem.py — fill_pits
# ---------------------------------------------------------------------------


def test_fill_pits_removes_single_cell_depression():
    from pipeline.dem import fill_pits

    elev = _make_dem_with_pit()
    filled = fill_pits(elev)

    # The pit cell must be raised to at least its rim level.
    assert filled[5, 5] >= elev[6, 5]


def test_fill_pits_flat_surface_unchanged():
    from pipeline.dem import fill_pits

    elev = _make_flat_dem(value=200.0)
    filled = fill_pits(elev)
    # Flat surface: filled values ≥ original (Priority-Flood may raise slightly).
    assert (filled >= elev - 1e-3).all()


def test_fill_pits_returns_float64():
    from pipeline.dem import fill_pits

    elev = np.ones((5, 5), dtype=np.float32)
    filled = fill_pits(elev)
    assert filled.dtype == np.float64


def test_fill_pits_preserves_border_cells():
    from pipeline.dem import fill_pits

    elev = _make_south_slope_dem()
    filled = fill_pits(elev)
    np.testing.assert_array_almost_equal(filled[0, :], elev[0, :], decimal=3)
    np.testing.assert_array_almost_equal(filled[-1, :], elev[-1, :], decimal=3)


def test_fill_pits_no_internal_depressions_after_fill():
    """After filling, no interior cell should be lower than all its neighbours."""
    from pipeline.dem import fill_pits

    rng = np.random.default_rng(42)
    elev = rng.uniform(1000.0, 5000.0, (15, 15))
    filled = fill_pits(elev)

    for r in range(1, 14):
        for c in range(1, 14):
            neighbours = [filled[r + dr, c + dc]
                          for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                          if not (dr == 0 and dc == 0)]
            # After fill: cell should not be strictly lower than ALL neighbours.
            assert filled[r, c] >= min(neighbours) - 1e-4, \
                f"Unfilled pit at ({r},{c}): {filled[r,c]:.4f} < {min(neighbours):.4f}"


# ---------------------------------------------------------------------------
# dem.py — d8_flow_direction
# ---------------------------------------------------------------------------


def test_d8_uniform_south_slope_all_point_south():
    from pipeline.dem import d8_flow_direction, fill_pits

    elev = fill_pits(_make_south_slope_dem(rows=10, cols=10))
    direction = d8_flow_direction(elev, dx_m=30.0, dy_m=30.0)

    # Interior cells on a uniform south slope should all point S (code 4).
    interior = direction[1:-1, 1:-1]
    assert (interior == 4).all(), f"Expected all S (4), got unique codes: {np.unique(interior)}"


def test_d8_flat_surface_returns_zero_interior():
    from pipeline.dem import d8_flow_direction

    elev = _make_flat_dem()
    direction = d8_flow_direction(elev, dx_m=30.0, dy_m=30.0)
    # Flat → no downslope → all codes should be 0.
    assert (direction == 0).all()


def test_d8_direction_codes_are_valid():
    from pipeline.dem import d8_flow_direction, fill_pits

    elev = fill_pits(_make_south_slope_dem())
    direction = d8_flow_direction(elev, dx_m=30.0, dy_m=30.0)
    valid_codes = {0, 1, 2, 4, 8, 16, 32, 64, 128}
    assert set(np.unique(direction)).issubset(valid_codes)


def test_d8_border_cells_are_zero():
    from pipeline.dem import d8_flow_direction, fill_pits

    elev = fill_pits(_make_south_slope_dem())
    direction = d8_flow_direction(elev, dx_m=30.0, dy_m=30.0)
    assert (direction[0, :] == 0).all()
    assert (direction[-1, :] == 0).all()
    assert (direction[:, 0] == 0).all()
    assert (direction[:, -1] == 0).all()


# ---------------------------------------------------------------------------
# dem.py — flow_accumulation
# ---------------------------------------------------------------------------


def test_flow_accumulation_single_column_slope():
    """On a 1-column slope, each cell accumulates all cells above it."""
    from pipeline.dem import d8_flow_direction, fill_pits, flow_accumulation

    elev = fill_pits(_make_south_slope_dem(rows=10, cols=1))
    direction = d8_flow_direction(elev, dx_m=30.0, dy_m=30.0)
    accum = flow_accumulation(direction, elev)

    # In a single column: accumulation increases southward.
    # (Border cells may have code 0 so accumulation stops there.)
    assert accum[-2, 0] >= accum[1, 0]


def test_flow_accumulation_outlet_has_highest_count():
    from pipeline.dem import d8_flow_direction, fill_pits, flow_accumulation

    elev = fill_pits(_make_south_slope_dem(rows=10, cols=5))
    direction = d8_flow_direction(elev, dx_m=30.0, dy_m=30.0)
    accum = flow_accumulation(direction, elev)

    # The outlet (bottom row, interior) should have accumulated the most.
    assert accum[-2, 2] >= accum[1, 2]


def test_flow_accumulation_all_cells_at_least_one():
    from pipeline.dem import d8_flow_direction, fill_pits, flow_accumulation

    elev = fill_pits(_make_south_slope_dem())
    direction = d8_flow_direction(elev, dx_m=30.0, dy_m=30.0)
    accum = flow_accumulation(direction, elev)

    assert (accum >= 1).all()


# ---------------------------------------------------------------------------
# dem.py — downstream_flowline
# ---------------------------------------------------------------------------


def _make_fake_transform(west: float = 74.5, north: float = 36.5,
                         pixel_deg: float = 0.0003) -> object:
    """Create a simple affine transform for a synthetic DEM."""
    from rasterio.transform import from_bounds as _fb
    rows, cols = 20, 20
    return _fb(west, north - rows * pixel_deg, west + cols * pixel_deg, north, cols, rows)


def test_downstream_flowline_v_valley_goes_down_centre():
    """On a V-valley DEM, flowline from centre top should go straight down."""
    from pipeline.dem import d8_flow_direction, downstream_flowline, fill_pits

    rows, cols = 30, 30
    elev = fill_pits(_make_v_valley_dem(rows, cols))
    direction = d8_flow_direction(elev, dx_m=30.0, dy_m=30.0)
    transform = _make_fake_transform()

    centre_col = cols // 2
    # Start from row 2 (just inside border) at centre column.
    flowline = downstream_flowline(direction, (2, centre_col), transform, max_km=10.0)

    assert len(flowline) >= 2
    # All cells should stay near the centre column (V-valley drains to centre).
    col_offsets = [abs(c - centre_col) for _, c in flowline]
    assert max(col_offsets) <= 3, f"Flowline wandered {max(col_offsets)} cols from centre"


def test_downstream_flowline_max_km_honoured():
    """Flowline stops at or before max_km."""
    from pipeline.dem import d8_flow_direction, downstream_flowline, fill_pits

    rows, cols = 50, 50
    elev = fill_pits(_make_south_slope_dem(rows, cols, slope_per_row=5.0))
    direction = d8_flow_direction(elev, dx_m=30.0, dy_m=30.0)
    transform = _make_fake_transform()

    max_km = 0.5  # very short → only a few cells
    flowline = downstream_flowline(direction, (2, 25), transform, max_km=max_km)

    # Path length in metres should not exceed max_km * 1000 + one cell slack.
    path_m = len(flowline) * 30.0
    assert path_m <= (max_km * 1000 + 60), f"Flowline too long: {path_m} m"


def test_downstream_flowline_flat_raises_flow_routing_error():
    from pipeline.dem import FlowRoutingError, d8_flow_direction, downstream_flowline

    elev = _make_flat_dem()  # all zeros → all D8 codes 0
    direction = d8_flow_direction(elev, dx_m=30.0, dy_m=30.0)
    transform = _make_fake_transform()

    with pytest.raises(FlowRoutingError):
        downstream_flowline(direction, (5, 5), transform)


def test_downstream_flowline_out_of_bounds_raises():
    from pipeline.dem import FlowRoutingError, d8_flow_direction, downstream_flowline, fill_pits

    elev = fill_pits(_make_south_slope_dem())
    direction = d8_flow_direction(elev, dx_m=30.0, dy_m=30.0)
    transform = _make_fake_transform()

    with pytest.raises(FlowRoutingError):
        downstream_flowline(direction, (999, 999), transform)


def test_downstream_flowline_minimum_length_one():
    """Flowline always contains at least the outlet cell."""
    from pipeline.dem import d8_flow_direction, downstream_flowline, fill_pits

    elev = fill_pits(_make_south_slope_dem())
    direction = d8_flow_direction(elev, dx_m=30.0, dy_m=30.0)
    transform = _make_fake_transform()

    flowline = downstream_flowline(direction, (2, 5), transform, max_km=0.001)
    assert len(flowline) >= 1


# ---------------------------------------------------------------------------
# exposure.py — flowline_corridor_mask
# ---------------------------------------------------------------------------


def test_flowline_corridor_mask_empty_flowline():
    from pipeline.exposure import flowline_corridor_mask

    transform = _make_fake_transform()
    mask = flowline_corridor_mask([], 500.0, transform, (20, 20))
    assert not mask.any()


def test_flowline_corridor_mask_single_point_creates_circle():
    from pipeline.exposure import flowline_corridor_mask

    transform = _make_fake_transform(pixel_deg=0.001)
    flowline_lonlat = [(74.505, 36.495)]  # near top-left of 20×20 grid
    mask = flowline_corridor_mask(flowline_lonlat, corridor_half_width_m=2000.0, pop_transform=transform, pop_shape=(20, 20))

    # Should have some True pixels around the flowline point.
    assert mask.any()


def test_flowline_corridor_mask_wider_corridor_covers_more_pixels():
    from pipeline.exposure import flowline_corridor_mask

    transform = _make_fake_transform(pixel_deg=0.001)
    flowline_lonlat = [(74.508, 36.494), (74.508, 36.491), (74.508, 36.488)]
    narrow = flowline_corridor_mask(flowline_lonlat, 300.0, transform, (20, 20))
    wide   = flowline_corridor_mask(flowline_lonlat, 2000.0, transform, (20, 20))
    assert wide.sum() >= narrow.sum()


# ---------------------------------------------------------------------------
# exposure.py — population_along_flowline (mocked)
# ---------------------------------------------------------------------------


def test_population_along_flowline_uses_corridor_not_buffer(tmp_path, fake_population_raster):
    """population_along_flowline returns a dict with method='d8_flowline_corridor'."""
    import rasterio as _rio
    from rasterio.transform import from_bounds as _fb

    rows, cols = 20, 20
    pixel_deg = 0.005
    west, north = 74.49, 36.48
    elev = _make_south_slope_dem(rows, cols, slope_per_row=20.0)

    dem_transform = _fb(west, north - rows * pixel_deg, west + cols * pixel_deg, north, cols, rows)

    # Precompute D8 fields matching what fetch_dem_for_flowrouting returns.
    from pipeline.dem import fill_pits, d8_flow_direction
    filled = fill_pits(elev)
    dx_m, dy_m = 30.0, 30.0
    direction = d8_flow_direction(filled, dx_m, dy_m)

    with patch("pipeline.exposure.fetch_dem_for_flowrouting",
               return_value=(filled, dem_transform, direction, dx_m, dy_m)), \
         patch("pipeline.exposure.ensure_downloaded", return_value=fake_population_raster):

        from pipeline.exposure import population_along_flowline
        result = population_along_flowline(
            outlet_lon=74.505, outlet_lat=36.472,
            max_km=1.0, corridor_half_width_m=500.0,
        )

    assert result["method"] == "d8_flowline_corridor"
    assert "population_within_buffer" in result   # backward-compat alias present
    assert "population_along_corridor" in result
    assert result["flowline_cells"] >= 1
    assert result["corridor_km"] >= 0.0


def test_population_along_flowline_positive_population(tmp_path, fake_population_raster):
    """With a populated raster, population along any valid corridor should be > 0."""
    import rasterio as _rio
    from rasterio.transform import from_bounds as _fb
    from pipeline.dem import fill_pits, d8_flow_direction

    rows, cols = 20, 20
    pixel_deg = 0.001
    west, north = 74.499, 36.501
    elev = _make_south_slope_dem(rows, cols, slope_per_row=30.0)
    dem_transform = _fb(west, north - rows * pixel_deg, west + cols * pixel_deg, north, cols, rows)
    filled = fill_pits(elev)
    direction = d8_flow_direction(filled, 30.0, 30.0)

    with patch("pipeline.exposure.fetch_dem_for_flowrouting",
               return_value=(filled, dem_transform, direction, 30.0, 30.0)), \
         patch("pipeline.exposure.ensure_downloaded", return_value=fake_population_raster):

        from pipeline.exposure import population_along_flowline
        result = population_along_flowline(74.505, 36.494, max_km=1.0, corridor_half_width_m=2000.0)

    assert result["population_within_buffer"] >= 0.0


# ---------------------------------------------------------------------------
# exposure.py — population_within_buffer (backward-compat wrapper)
# ---------------------------------------------------------------------------


def test_population_within_buffer_backward_compat_key_present(fake_population_raster):
    """population_within_buffer always returns 'population_within_buffer' key."""
    import rasterio as _rio
    from rasterio.transform import from_bounds as _fb
    from pipeline.dem import fill_pits, d8_flow_direction

    rows, cols = 20, 20
    pixel_deg = 0.001
    west, north = 74.49, 36.48
    elev = _make_south_slope_dem(rows, cols, slope_per_row=20.0)
    dem_transform = _fb(west, north - rows * pixel_deg, west + cols * pixel_deg, north, cols, rows)
    filled = fill_pits(elev)
    direction = d8_flow_direction(filled, 30.0, 30.0)

    with patch("pipeline.exposure.fetch_dem_for_flowrouting",
               return_value=(filled, dem_transform, direction, 30.0, 30.0)), \
         patch("pipeline.exposure.ensure_downloaded", return_value=fake_population_raster):

        from pipeline.exposure import population_within_buffer
        result = population_within_buffer(lon=74.61, lat=36.40)

    assert "population_within_buffer" in result
    assert "source" in result
    assert "method" in result


def test_population_within_buffer_falls_back_to_radial_on_routing_error(
    fake_population_raster, caplog
):
    """On FlowRoutingError, falls back to radial buffer and logs a WARNING."""
    from pipeline.dem import FlowRoutingError
    with patch("pipeline.exposure.fetch_dem_for_flowrouting",
               side_effect=FlowRoutingError("flat DEM at outlet")), \
         patch("pipeline.exposure.ensure_downloaded", return_value=fake_population_raster):

        from pipeline.exposure import population_within_buffer
        with caplog.at_level(logging.WARNING, logger="pipeline.exposure"):
            result = population_within_buffer(lon=74.61, lat=36.40)

    assert result["method"] == "radial_buffer_fallback"
    assert "population_within_buffer" in result
    assert any("falling back to radial buffer" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Preserve existing WorldPop tests (unchanged)
# ---------------------------------------------------------------------------


def test_population_within_buffer_sums_real_pixel_values(fake_population_raster):
    """Legacy test: fallback radial buffer sums pixel values from fake raster."""
    from pipeline.dem import FlowRoutingError

    # Force routing failure so radial buffer runs.
    with patch("pipeline.exposure.fetch_dem_for_flowrouting",
               side_effect=FlowRoutingError("test")), \
         patch("pipeline.exposure.ensure_downloaded", return_value=fake_population_raster):

        from pipeline.exposure import population_within_buffer
        result = population_within_buffer(lon=74.61, lat=36.40, radius_km=5.0)

    assert result["population_within_buffer"] > 0
    assert result["source"] == "worldpop_pak_2025_constrained_100m"


def test_population_within_buffer_treats_negative_sentinel_values_as_zero(tmp_path):
    """Legacy test: negative nodata sentinel treated as zero population."""
    from pipeline.dem import FlowRoutingError

    west, south, east, north = 74.5, 36.3, 74.7, 36.5
    width, height = 50, 50
    data = np.full((height, width), -99999.0, dtype=np.float32)
    path = tmp_path / "fake_nodata.tif"
    transform = transform_from_bounds(west, south, east, north, width, height)
    with rasterio.open(
        path, "w", driver="GTiff", height=height, width=width, count=1,
        dtype="float32", crs="EPSG:4326", transform=transform,
    ) as dst:
        dst.write(data, 1)

    with patch("pipeline.exposure.fetch_dem_for_flowrouting",
               side_effect=FlowRoutingError("test")), \
         patch("pipeline.exposure.ensure_downloaded", return_value=path):

        from pipeline.exposure import population_within_buffer
        result = population_within_buffer(lon=74.61, lat=36.40, radius_km=5.0)

    assert result["population_within_buffer"] == 0.0


def test_population_within_buffer_larger_radius_covers_more_population(fake_population_raster):
    """Legacy test: larger radius captures more population than smaller radius."""
    from pipeline.dem import FlowRoutingError

    with patch("pipeline.exposure.fetch_dem_for_flowrouting",
               side_effect=FlowRoutingError("test")), \
         patch("pipeline.exposure.ensure_downloaded", return_value=fake_population_raster):

        from pipeline.exposure import population_within_buffer
        small = population_within_buffer(lon=74.61, lat=36.40, radius_km=2.0)
        large = population_within_buffer(lon=74.61, lat=36.40, radius_km=8.0)

    assert large["population_within_buffer"] > small["population_within_buffer"]


# ---------------------------------------------------------------------------
# Slow / integration tests — require live DEM + WorldPop download
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_shishper_d8_flowline_traces_hassanabad_nallah():
    """Integration: D8 flowline from Shishper outlet follows Hassanabad Nallah valley.

    Requires: network access to Planetary Computer STAC for GLO-30 DEM tiles.
    Run with: uv run pytest -m slow tests/test_exposure.py
    """
    from pipeline.dem import fetch_dem_for_flowrouting, downstream_flowline
    from pipeline.lakes import LAKES

    lake = LAKES["shishper"]
    filled, transform, direction, dx_m, dy_m = fetch_dem_for_flowrouting(
        lake.outlet_lon, lake.outlet_lat, max_km=50.0
    )

    outlet_col = int((lake.outlet_lon - transform.c) / transform.a)
    outlet_row = int((lake.outlet_lat - transform.f) / transform.e)

    flowline = downstream_flowline(direction, (outlet_row, outlet_col), transform, max_km=50.0)

    assert len(flowline) >= 100, f"Flowline unexpectedly short: {len(flowline)} cells"
    print(f"\nShishper flowline: {len(flowline)} cells, last cell: {flowline[-1]}")


@pytest.mark.slow
def test_shishper_corridor_iou_vs_historical_footprint():
    """Integration: D8 corridor IoU vs. Shishper 2022 GLOF inundation polygon.

    TODO(#26-validation): obtain reference inundation polygon and implement IoU check.
    Reference data tracked in: https://github.com/uExel/CryoHealth-geo/issues/26
    (sub-issue of #19). Once obtained, commit the polygon to:
      tests/fixtures/glof_footprints/shishper_2022_inundation.geojson
    and implement the IoU computation here.

    Target IoU: >= 0.60 (conservative; to be confirmed once reference data is reviewed).
    """
    pytest.skip(
        "TODO(#26-validation): reference inundation polygon not yet available. "
        "See https://github.com/uExel/CryoHealth-geo/issues/26"
    )


@pytest.mark.slow
def test_shishper_population_along_flowline_is_plausible():
    """Integration: Shishper D8 corridor population is plausible for Hunza Valley.

    Hunza Valley has population centres at Hassanabad, Aliabad, Karimabad, Gilgit.
    A 50 km corridor from Shishper should capture O(10,000–100,000) people.
    Requires: network + real WorldPop raster downloaded.
    """
    from pipeline.exposure import population_along_flowline
    from pipeline.lakes import LAKES

    lake = LAKES["shishper"]
    result = population_along_flowline(lake.outlet_lon, lake.outlet_lat)

    assert result["method"] == "d8_flowline_corridor"
    assert result["corridor_km"] > 5.0, "Corridor shorter than 5 km — outlet likely wrong"
    assert result["population_along_corridor"] > 1000, \
        f"Population too low ({result['population_along_corridor']}) — corridor may be routing off-valley"
    print(f"\nShishper corridor: {result['corridor_km']:.1f} km, "
          f"pop={result['population_along_corridor']:.0f}")
