"""Regression test for a real bug found running the PoC live against Planetary
Computer: independently-windowed bands at different resolutions don't always come back
the exact same pixel dimensions, even nominally covering the same AOI."""

from __future__ import annotations

import numpy as np

from pipeline.poc import _crop_to_common_shape, _upsample_scl_to_10m


def test_upsample_scl_to_10m_doubles_each_dimension():
    scl = np.array([[4, 6], [8, 11]])
    up = _upsample_scl_to_10m(scl)
    assert up.shape == (4, 4)
    assert up[0, 0] == up[0, 1] == up[1, 0] == up[1, 1] == 4


def test_crop_to_common_shape_handles_mismatched_bands():
    # Exactly the shape mismatch observed live: B03 one row taller than 2x SCL.
    green = np.zeros((223, 180))
    nir = np.zeros((223, 180))
    scl_upsampled = np.zeros((222, 180))
    g, n, s = _crop_to_common_shape(green, nir, scl_upsampled)
    assert g.shape == n.shape == s.shape == (222, 180)


def test_crop_to_common_shape_is_a_noop_when_already_aligned():
    a = np.zeros((10, 10))
    b = np.zeros((10, 10))
    ra, rb = _crop_to_common_shape(a, b)
    assert ra.shape == rb.shape == (10, 10)
