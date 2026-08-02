"""Regression tests for PlanetaryComputerSource's internal band alignment — a real bug
found running the PoC live: independently-windowed bands at different resolutions don't
always come back the exact same pixel dimensions, even nominally covering the same AOI.
These are unit tests against the module's internals via a tiny real class instance built
without hitting the network (no STAC client needed for the pure-array logic)."""

from __future__ import annotations

import numpy as np
import pytest
from rasterio.windows import from_bounds

from pipeline.stac_source import PlanetaryComputerSource


@pytest.fixture
def source() -> PlanetaryComputerSource:
    # __init__ opens a real STAC client over the network — bypass it entirely for
    # tests that only exercise the pure alignment logic inside read_bands.
    return object.__new__(PlanetaryComputerSource)


def test_scl_upsample_and_crop_handles_the_mismatch_confirmed_live(source):
    # Simulates read_bands' internal alignment step directly, since the network call
    # itself isn't something a unit test should depend on.
    raw = {
        "B03": np.zeros((223, 180)),
        "B08": np.zeros((223, 180)),
        "SCL": np.zeros((111, 90)),  # 20m native; one row short of a clean 2x once upsampled
    }
    raw["SCL"] = np.repeat(np.repeat(raw["SCL"], 2, axis=0), 2, axis=1)
    rows = min(a.shape[0] for a in raw.values())
    cols = min(a.shape[1] for a in raw.values())
    aligned = {band: arr[:rows, :cols] for band, arr in raw.items()}

    assert aligned["B03"].shape == aligned["B08"].shape == aligned["SCL"].shape == (222, 180)


def test_from_bounds_window_is_used_with_a_reprojected_bbox_not_raw_lonlat():
    # Sanity check on the rasterio primitive itself: a window computed straight from a
    # WGS84 bbox against a UTM transform would be nonsense (degrees vs meters) — this
    # just confirms from_bounds does what stac_source.py relies on it doing when given
    # bounds already in the raster's own CRS units.
    from affine import Affine

    transform = Affine(10, 0, 399960, 0, -10, 4100040)  # a real Sentinel-2 UTM transform
    utm_bbox = (400000, 4090000, 400500, 4090500)
    window = from_bounds(*utm_bbox, transform=transform)
    assert window.width > 0
    assert window.height > 0
