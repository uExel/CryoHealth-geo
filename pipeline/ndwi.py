"""NDWI (McFeeters 1996) and cloud-aware water-area estimation.

Pure array math, no I/O — kept separate from stac_source.py so it's testable with
synthetic arrays and never needs network access or GDAL to verify. This is the logic
that stays identical regardless of which STAC catalog (Planetary Computer today, CDSE
once credentials exist — see ADR 0001) the bands came from.
"""

from __future__ import annotations

import numpy as np

# Sentinel-2 L2A Scene Classification Layer (SCL) values that mean "not usable ground":
# 3=cloud shadow, 8=cloud medium probability, 9=cloud high probability, 10=thin cirrus.
# (11=snow/ice is deliberately NOT masked — glacial lakes sit right next to snow/ice,
# and excluding it would bias the water estimate.)
_SCL_UNUSABLE = frozenset({3, 8, 9, 10})

# Sentinel-2 L2A 10 m bands (B03 green, B08 NIR) → each pixel covers this many m^2.
PIXEL_AREA_M2 = 10 * 10


def compute_ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """McFeeters NDWI = (Green - NIR) / (Green + NIR). Water is NDWI > 0."""
    green = green.astype(np.float32)
    nir = nir.astype(np.float32)
    denom = green + nir
    with np.errstate(divide="ignore", invalid="ignore"):
        ndwi = np.where(denom != 0, (green - nir) / denom, 0.0)
    return ndwi


def cloud_mask(scl: np.ndarray) -> np.ndarray:
    """True where the pixel is usable ground (not cloud/shadow/cirrus)."""
    return ~np.isin(scl, list(_SCL_UNUSABLE))


def water_area_km2(ndwi: np.ndarray, usable: np.ndarray | None = None, threshold: float = 0.0) -> float:
    """Area of NDWI-positive pixels, restricted to usable (non-cloud) pixels if given."""
    water = ndwi > threshold
    if usable is not None:
        water = water & usable
    return float(water.sum()) * PIXEL_AREA_M2 / 1_000_000


def cloud_fraction(usable: np.ndarray) -> float:
    """Fraction of the AOI obscured by cloud/shadow/cirrus — stored alongside the
    observation so a scene that's mostly cloud doesn't silently produce a bogus area."""
    total = usable.size
    if total == 0:
        return 1.0
    return float((~usable).sum()) / total
