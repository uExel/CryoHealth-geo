"""Geometry helpers: binary mask smoothing and GeoJSON vectorization.

Kept separate from ndwi.py so that ndwi.py remains 100% pure numpy with no
GDAL/rasterio dependency (see ADR 0002 and the ndwi.py module docstring).

Shared by both the NDWI path and the AlphaEarth/ML segmentation path.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

try:
    import rasterio.features  # type: ignore[import-untyped]
    _RASTERIO_AVAILABLE = True
except ImportError:
    _RASTERIO_AVAILABLE = False


# ---------------------------------------------------------------------------
# Mask smoothing
# ---------------------------------------------------------------------------


def smooth_water_mask(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Morphological open-then-close to remove speckle from a binary water mask.

    Open (erode → dilate) removes isolated noise pixels.
    Close (dilate → erode) fills small holes inside the water body.

    Args:
        mask: Boolean (H × W) array. True = water pixel.
        kernel_size: Side length of the square structuring element (default 3).
                     Must be a positive odd integer.

    Returns:
        Smoothed boolean mask, same shape as input.
    """
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError(
            f"kernel_size must be a positive odd integer, got {kernel_size}"
        )

    if not mask.any():
        return mask.copy()

    try:
        from scipy.ndimage import binary_closing, binary_opening  # noqa: PLC0415

        struct = np.ones((kernel_size, kernel_size), dtype=bool)
        smoothed = binary_opening(mask, structure=struct)
        smoothed = binary_closing(smoothed, structure=struct)
        return smoothed.astype(bool)
    except ImportError:
        logger.warning(
            "scipy not installed — smooth_water_mask returning unsmoothed mask. "
            "Install scipy with: uv pip install 'cryohealth-geo[ml]'"
        )
        return mask.copy()


# ---------------------------------------------------------------------------
# GeoJSON vectorization
# ---------------------------------------------------------------------------


def vectorize_water_mask(
    mask: np.ndarray,
    transform: object,  # affine.Affine
    crs: str = "EPSG:4326",
    min_area_pixels: int = 10,
) -> list[dict]:
    """Convert a binary water mask to a list of GeoJSON Feature dicts.

    Each contiguous water region becomes one Polygon feature. Regions
    with fewer than ``min_area_pixels`` are dropped to remove noise.

    Args:
        mask: Boolean (H × W) array. True = water pixel.
        transform: An ``affine.Affine`` geo-transform mapping pixel coords to CRS coords.
        crs: CRS string stored in each feature's properties (default EPSG:4326).
        min_area_pixels: Polygons smaller than this are dropped.

    Returns:
        List of GeoJSON Feature dicts. Empty list if no water pixels.

    Raises:
        RuntimeError: if rasterio is not installed.
    """
    if not _RASTERIO_AVAILABLE:
        raise RuntimeError(
            "rasterio is not installed — vectorize_water_mask requires rasterio."
        )

    if not mask.any():
        return []

    water_int = mask.astype(np.uint8)
    features = []

    for geom, value in rasterio.features.shapes(
        water_int, mask=water_int, transform=transform
    ):
        if value != 1:
            continue

        coords = geom["coordinates"][0]
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        pixel_w = abs(transform.a)
        pixel_h = abs(transform.e)
        approx_pixels = (
            ((max(xs) - min(xs)) / pixel_w) * ((max(ys) - min(ys)) / pixel_h)
        )

        if approx_pixels < min_area_pixels:
            continue

        features.append(
            {
                "type": "Feature",
                "geometry": geom,
                "properties": {
                    "water": True,
                    "crs": crs,
                    "approx_area_pixels": int(approx_pixels),
                },
            }
        )

    return features


__all__ = ["smooth_water_mask", "vectorize_water_mask"]
