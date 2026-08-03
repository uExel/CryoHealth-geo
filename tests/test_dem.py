"""pipeline/dem.py's slope math is pure once given an elevation array + pixel size —
tested directly here with synthetic terrain, no STAC/network dependency. The live
GLO-30 STAC search/read path was verified by hand against real lakes (see
docs/ai/HANDOFF.md) before this file existed."""

from __future__ import annotations

import math

import numpy as np
import pytest

from pipeline.dem import _slope_from_elevation


def test_flat_terrain_has_zero_slope():
    elevation = np.full((10, 10), 3000.0)
    assert _slope_from_elevation(elevation, dx_m=30.0, dy_m=30.0) == pytest.approx(0.0)


def test_known_uniform_gradient_matches_the_analytical_slope():
    # A plane rising 30m per 30m pixel along x is a true 45-degree slope everywhere
    # except at the array edges (np.gradient uses one-sided differences there, which
    # still land on the same 45 degrees for a perfectly uniform plane).
    rows, cols = 10, 10
    elevation = np.tile(np.arange(cols, dtype=np.float64) * 30.0, (rows, 1))
    slope = _slope_from_elevation(elevation, dx_m=30.0, dy_m=30.0)
    assert slope == pytest.approx(45.0, abs=0.5)


def test_gentler_gradient_gives_a_proportionally_smaller_slope():
    # Half the rise over the same run -> arctan(0.5) in degrees.
    rows, cols = 10, 10
    elevation = np.tile(np.arange(cols, dtype=np.float64) * 15.0, (rows, 1))
    slope = _slope_from_elevation(elevation, dx_m=30.0, dy_m=30.0)
    assert slope == pytest.approx(math.degrees(math.atan(0.5)), abs=0.5)
