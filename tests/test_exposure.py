"""exposure.py tests run against a small synthetic local GeoTIFF standing in for the
real WorldPop national raster — no network, no 140MB download. ensure_downloaded() is
monkeypatched to point at the fixture file instead. The real download-once-and-cache
behavior and the real "server ignores Range requests" finding were verified live
against the actual WorldPop endpoint before this file existed (see
docs/ai/HANDOFF.md)."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds as transform_from_bounds

from pipeline.exposure import population_within_buffer


@pytest.fixture
def fake_population_raster(tmp_path):
    # Centered on (74.61, 36.40) — shishper's coordinates — with a known, uniform
    # population density so the expected sum is computable by hand.
    west, south, east, north = 74.5, 36.3, 74.7, 36.5
    width, height = 200, 200
    density_per_pixel = 2.0  # constant "people per pixel"
    data = np.full((height, width), density_per_pixel, dtype=np.float32)

    path = tmp_path / "fake_pop.tif"
    transform = transform_from_bounds(west, south, east, north, width, height)
    with rasterio.open(
        path, "w", driver="GTiff", height=height, width=width, count=1,
        dtype="float32", crs="EPSG:4326", transform=transform,
    ) as dst:
        dst.write(data, 1)
    return path


def test_population_within_buffer_sums_real_pixel_values(fake_population_raster):
    with patch("pipeline.exposure.ensure_downloaded", return_value=fake_population_raster):
        result = population_within_buffer(lon=74.61, lat=36.40, radius_km=5.0)

    assert result["population_within_buffer"] > 0
    assert result["buffer_km"] == 5.0
    assert result["source"] == "worldpop_pak_2025_constrained_100m"


def test_population_within_buffer_treats_negative_sentinel_values_as_zero(tmp_path):
    west, south, east, north = 74.5, 36.3, 74.7, 36.5
    width, height = 50, 50
    data = np.full((height, width), -99999.0, dtype=np.float32)  # all nodata

    path = tmp_path / "fake_nodata.tif"
    transform = transform_from_bounds(west, south, east, north, width, height)
    with rasterio.open(
        path, "w", driver="GTiff", height=height, width=width, count=1,
        dtype="float32", crs="EPSG:4326", transform=transform,
    ) as dst:
        dst.write(data, 1)

    with patch("pipeline.exposure.ensure_downloaded", return_value=path):
        result = population_within_buffer(lon=74.61, lat=36.40, radius_km=5.0)

    assert result["population_within_buffer"] == 0.0


def test_population_within_buffer_larger_radius_covers_more_population(fake_population_raster):
    with patch("pipeline.exposure.ensure_downloaded", return_value=fake_population_raster):
        small = population_within_buffer(lon=74.61, lat=36.40, radius_km=2.0)
        large = population_within_buffer(lon=74.61, lat=36.40, radius_km=8.0)

    assert large["population_within_buffer"] > small["population_within_buffer"]
