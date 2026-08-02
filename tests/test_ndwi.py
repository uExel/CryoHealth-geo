from __future__ import annotations

import numpy as np
import pytest

from pipeline.ndwi import cloud_fraction, cloud_mask, compute_ndwi, water_area_km2


def test_compute_ndwi_pure_water_is_strongly_positive():
    # Water absorbs NIR far more than green — a real water pixel looks like this.
    green = np.array([[100.0]])
    nir = np.array([[20.0]])
    ndwi = compute_ndwi(green, nir)
    assert ndwi[0, 0] == pytest.approx((100 - 20) / (100 + 20))
    assert ndwi[0, 0] > 0


def test_compute_ndwi_bare_ground_is_negative_or_near_zero():
    # Bare rock/scree reflects NIR strongly relative to green.
    green = np.array([[80.0]])
    nir = np.array([[150.0]])
    ndwi = compute_ndwi(green, nir)
    assert ndwi[0, 0] < 0


def test_compute_ndwi_handles_zero_denominator_without_dividing_by_zero():
    green = np.array([[0.0]])
    nir = np.array([[0.0]])
    ndwi = compute_ndwi(green, nir)
    assert ndwi[0, 0] == 0.0


def test_cloud_mask_excludes_only_cloud_shadow_and_cirrus_not_snow_ice():
    # SCL: 3=shadow, 4=veg, 6=water, 8/9=cloud, 10=cirrus, 11=snow/ice.
    scl = np.array([3, 4, 6, 8, 9, 10, 11])
    usable = cloud_mask(scl)
    assert list(usable) == [False, True, True, False, False, False, True]


def test_water_area_km2_counts_only_ndwi_positive_and_usable_pixels():
    # 2x2 grid, 10m pixels -> each pixel is 100 m^2 = 0.0001 km^2.
    ndwi = np.array([[0.3, 0.3], [-0.1, 0.3]])
    usable = np.array([[True, False], [True, True]])  # top-right is cloud
    area = water_area_km2(ndwi, usable)
    # (0,0) is water+usable; (0,1) is water but cloud-masked out; (1,0) isn't water;
    # (1,1) is water+usable -> 2 qualifying pixels.
    assert area == pytest.approx(2 * 100 / 1_000_000)


def test_water_area_km2_without_a_mask_counts_all_ndwi_positive_pixels():
    ndwi = np.array([[0.3, 0.3], [-0.1, 0.3]])
    area = water_area_km2(ndwi)
    assert area == pytest.approx(3 * 100 / 1_000_000)


def test_cloud_fraction_reports_the_obscured_share_of_the_aoi():
    usable = np.array([True, True, False, False])
    assert cloud_fraction(usable) == pytest.approx(0.5)


def test_cloud_fraction_of_empty_aoi_is_fully_unusable_not_a_crash():
    assert cloud_fraction(np.array([])) == 1.0
