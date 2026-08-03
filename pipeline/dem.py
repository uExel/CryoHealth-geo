"""Mean terrain slope around a lake AOI, from the Copernicus GLO-30 DEM (PRD §8's named
source). Read via Microsoft Planetary Computer's STAC catalog — anonymous, no account
needed, same pattern as PlanetaryComputerSource in stac_source.py. Static input (terrain
doesn't change day to day), so results are cached per-process rather than re-fetched on
every hazard run.
"""

from __future__ import annotations

import math
from functools import cache

import numpy as np
import planetary_computer
import rasterio
from pystac_client import Client
from rasterio.merge import merge as rasterio_merge
from rasterio.windows import from_bounds

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "cop-dem-glo-30"

# Same constant/approach as cdse_source.py: 1 degree of longitude shrinks with
# cos(latitude); latitude itself doesn't.
METERS_PER_DEGREE_LAT = 111_320


@cache
def mean_slope_degrees(bbox: tuple[float, float, float, float]) -> float:
    """bbox is (west, south, east, north) in WGS84. Returns the mean slope, in degrees,
    of the terrain within that box — steeper surrounding terrain means more energy and
    higher downstream hazard once/if a dam breaches (PRD §8)."""
    west, south, east, north = bbox
    client = Client.open(STAC_URL)
    search = client.search(collections=[COLLECTION], bbox=bbox)
    items = [planetary_computer.sign(item) for item in search.items()]
    if not items:
        raise RuntimeError(f"No {COLLECTION} tile covers bbox {bbox}")

    if len(items) == 1:
        with rasterio.open(items[0].assets["data"].href) as src:
            window = from_bounds(west, south, east, north, transform=src.transform)
            elevation = src.read(1, window=window)
            transform = src.window_transform(window)
    else:
        # AOI straddles a DEM tile boundary — mosaic before windowing.
        sources = [rasterio.open(item.assets["data"].href) for item in items]
        try:
            mosaic, transform = rasterio_merge(sources, bounds=(west, south, east, north))
            elevation = mosaic[0]
        finally:
            for src in sources:
                src.close()

    lat_center = (south + north) / 2
    pixel_size_x_deg = abs(transform.a)
    pixel_size_y_deg = abs(transform.e)
    dx_m = pixel_size_x_deg * METERS_PER_DEGREE_LAT * math.cos(math.radians(lat_center))
    dy_m = pixel_size_y_deg * METERS_PER_DEGREE_LAT

    return _slope_from_elevation(elevation, dx_m, dy_m)


def _slope_from_elevation(elevation: np.ndarray, dx_m: float, dy_m: float) -> float:
    """Pure math, no I/O — kept separate so it's testable with a synthetic array. dx_m/
    dy_m are the real-world pixel size (meters) along each axis."""
    grad_y, grad_x = np.gradient(elevation.astype(np.float64), dy_m, dx_m)
    slope_rad = np.arctan(np.sqrt(grad_x**2 + grad_y**2))
    return float(np.degrees(np.mean(slope_rad)))
