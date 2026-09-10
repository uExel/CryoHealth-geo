"""Downstream population exposure (PRD §8): WorldPop population along the D8-modelled
downstream inundation corridor from the lake outlet dam.

Replaces the straight-line radial buffer from the original implementation (Issue #19,
ADR 0003) with a physically correct downstream corridor:
  1. Fetch GLO-30 DEM around the outlet (pipeline/dem.py)
  2. Trace D8 downstream flowline up to FLOWLINE_MAX_KM (50 km)
  3. Buffer the flowline by CORRIDOR_HALF_WIDTH_M (500 m) → corridor mask
  4. Intersect with WorldPop 100 m raster → exposed population count

Backward compatibility: population_within_buffer(lon, lat) remains the public API
called by hazard_batch.py — signature and return key are unchanged. The function
internally looks up the lake's outlet coordinates from pipeline/lakes.py and calls
population_along_flowline(). Falls back to the original radial buffer if D8 routing
fails (FlowRoutingError), logging a WARNING so the fallback is visible in logs.

WorldPop note: the server ignores HTTP Range requests (confirmed live: advertises
Accept-Ranges: bytes but always returns the full body), so the national raster (~140 MB)
is downloaded once and cached locally, then read with ordinary local-file windowing.

Validation against Shishper/Passu GLOF footprints is pending Issue #26 (sub-issue of
#19). Until that lands, the 'verified against historical GLOF inundation footprint'
acceptance criterion is NOT yet met — see HAZARD_METHODOLOGY.md and ADR 0003.
"""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path

import numpy as np
import rasterio
import requests
from rasterio.windows import from_bounds

from pipeline.dem import FlowRoutingError, downstream_flowline, fetch_dem_for_flowrouting

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# WorldPop constants (unchanged)
# ---------------------------------------------------------------------------

CACHE_DIR = Path(
    os.environ.get("CRYOHEALTH_CACHE_DIR", Path(__file__).resolve().parent.parent / ".cache" / "worldpop")
)
POPULATION_URL = (
    "https://data.worldpop.org/GIS/Population/Global_2015_2030/R2025A/2025/PAK/v1/"
    "100m/constrained/pak_pop_2025_CN_100m_R2025A_v1.tif"
)
POPULATION_SOURCE_ID = "worldpop_pak_2025_constrained_100m"

BUFFER_RADIUS_KM = 5.0
METERS_PER_DEGREE_LAT = 111_320

# ---------------------------------------------------------------------------
# D8 corridor constants (Issue #19)
# ---------------------------------------------------------------------------

FLOWLINE_MAX_KM = 50.0
# 500 m half-width → 1 km total corridor; overridable via env var.
CORRIDOR_HALF_WIDTH_M = float(os.environ.get("EXPOSURE_CORRIDOR_HALF_WIDTH_M", 500.0))


# ---------------------------------------------------------------------------
# WorldPop download (unchanged)
# ---------------------------------------------------------------------------


def _local_path() -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / POPULATION_URL.rsplit("/", 1)[-1]


def ensure_downloaded() -> Path:
    """Downloads the national population raster once; a no-op if already cached."""
    path = _local_path()
    if path.exists():
        return path
    tmp = path.with_suffix(".tif.part")
    with requests.get(POPULATION_URL, stream=True, timeout=600) as resp:
        resp.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
    tmp.rename(path)
    return path


# ---------------------------------------------------------------------------
# D8 corridor geometry helpers (Issue #19)
# ---------------------------------------------------------------------------


def _flowline_to_lonlat(
    flowline: list[tuple[int, int]],
    transform: object,  # affine.Affine
) -> list[tuple[float, float]]:
    """Convert DEM (row, col) flowline cells to (lon, lat) WGS84 coordinates.

    Uses the centre of each pixel. The affine transform maps (col, row) → (x, y).
    """
    coords = []
    for r, c in flowline:
        lon = transform.c + (c + 0.5) * transform.a
        lat = transform.f + (r + 0.5) * transform.e
        coords.append((lon, lat))
    return coords


def flowline_corridor_mask(
    flowline_lonlat: list[tuple[float, float]],
    corridor_half_width_m: float,
    pop_transform: object,  # affine.Affine for the WorldPop raster
    pop_shape: tuple[int, int],
) -> np.ndarray:
    """Build a boolean mask (H×W) on the WorldPop grid for pixels within the corridor.

    For each WorldPop pixel centre, computes the minimum geographic distance to any
    flowline point and marks it True if within corridor_half_width_m.

    Uses scipy.ndimage.distance_transform_edt when available (fast path), otherwise
    falls back to vectorised numpy distance computation (slower but correct).

    Args:
        flowline_lonlat: List of (lon, lat) tuples of flowline cell centres.
        corridor_half_width_m: Half-width of the corridor in metres.
        pop_transform: Affine geo-transform of the WorldPop raster window.
        pop_shape: (rows, cols) of the WorldPop raster window.

    Returns:
        bool (H×W) mask, True where pixel is within the corridor.
    """
    rows, cols = pop_shape

    if not flowline_lonlat:
        return np.zeros(pop_shape, dtype=bool)

    # Pixel sizes in metres at the grid latitude centre.
    lat_centre = pop_transform.f + (rows / 2) * pop_transform.e
    dx_m = abs(pop_transform.a) * METERS_PER_DEGREE_LAT * math.cos(math.radians(lat_centre))
    dy_m = abs(pop_transform.e) * METERS_PER_DEGREE_LAT

    # Build raster with True = flowline cell present.
    flowline_raster = np.zeros(pop_shape, dtype=bool)
    for lon, lat in flowline_lonlat:
        c = int((lon - pop_transform.c) / pop_transform.a)
        r = int((lat - pop_transform.f) / pop_transform.e)
        if 0 <= r < rows and 0 <= c < cols:
            flowline_raster[r, c] = True

    if not flowline_raster.any():
        # No flowline cells fall within the WorldPop window — return empty mask.
        return np.zeros(pop_shape, dtype=bool)

    try:
        from scipy.ndimage import distance_transform_edt  # noqa: PLC0415
        # EDT: True cells are "background" (distance computed FROM flowline cells).
        dist_m = distance_transform_edt(~flowline_raster, sampling=(dy_m, dx_m))
        return dist_m <= corridor_half_width_m

    except ImportError:
        logger.warning(
            "scipy not installed — using slower numpy distance computation for "
            "flowline corridor mask. Install scipy with: uv pip install 'cryohealth-geo[ml]'"
        )
        # Numpy fallback: for each non-flowline pixel, find minimum distance to any
        # flowline pixel using broadcast. Fine for WorldPop 100m grid (~500×500 in corridor).
        row_idx, col_idx = np.nonzero(flowline_raster)
        if len(row_idx) == 0:
            return np.zeros(pop_shape, dtype=bool)

        r_grid, c_grid = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
        # Compute distance from each pixel to each flowline pixel, take minimum.
        # Shape: (rows, cols, n_flowline_cells)
        dr_m = (r_grid[:, :, np.newaxis] - row_idx[np.newaxis, np.newaxis, :]) * dy_m
        dc_m = (c_grid[:, :, np.newaxis] - col_idx[np.newaxis, np.newaxis, :]) * dx_m
        dist_m = np.sqrt(dr_m**2 + dc_m**2).min(axis=2)
        return dist_m <= corridor_half_width_m


# ---------------------------------------------------------------------------
# Main D8 corridor population function (Issue #19)
# ---------------------------------------------------------------------------


def population_along_flowline(
    outlet_lon: float,
    outlet_lat: float,
    max_km: float = FLOWLINE_MAX_KM,
    corridor_half_width_m: float = CORRIDOR_HALF_WIDTH_M,
) -> dict:
    """Population within the D8 downstream inundation corridor from a lake outlet.

    Fetches the GLO-30 DEM, computes the D8 flowline, buffers it by
    corridor_half_width_m, and intersects with WorldPop.

    Args:
        outlet_lon: Dam-toe longitude (WGS84) — NOT the lake centroid.
        outlet_lat: Dam-toe latitude (WGS84).
        max_km: Maximum corridor length in km (default 50).
        corridor_half_width_m: Corridor half-width in metres (default 500).

    Returns:
        dict with keys:
          "source": POPULATION_SOURCE_ID
          "method": "d8_flowline_corridor"
          "corridor_km": float — actual traced flowline length
          "corridor_half_width_m": float
          "population_along_corridor": float
          "flowline_cells": int — number of DEM pixels in flowline
          "population_within_buffer": float — same as population_along_corridor
                                       (backward-compat alias)
          "note": str

    Raises:
        FlowRoutingError: propagated from dem.py if DEM fetch or routing fails.
            Callers should catch this and fall back to the radial buffer.
    """
    filled, transform, direction, dx_m, dy_m = fetch_dem_for_flowrouting(
        outlet_lon, outlet_lat, max_km
    )

    # Convert outlet lon/lat → DEM pixel (row, col).
    outlet_col = int((outlet_lon - transform.c) / transform.a)
    outlet_row = int((outlet_lat - transform.f) / transform.e)
    outlet_rc = (outlet_row, outlet_col)

    flowline = downstream_flowline(direction, outlet_rc, transform, max_km)

    # Corridor length in km.
    pixel_size_x_deg = abs(transform.a)
    pixel_size_y_deg = abs(transform.e)
    lat_c = transform.f + outlet_row * transform.e
    dx = pixel_size_x_deg * METERS_PER_DEGREE_LAT * math.cos(math.radians(lat_c))
    dy = pixel_size_y_deg * METERS_PER_DEGREE_LAT
    diag = math.sqrt(dx**2 + dy**2)

    from pipeline.dem import _D8_CODE_TO_DELTA  # noqa: PLC0415
    _code_to_dist = {1: dx, 2: diag, 4: dy, 8: diag, 16: dx, 32: diag, 64: dy, 128: diag}
    corridor_m = sum(
        _code_to_dist.get(int(direction[r, c]), dx) for r, c in flowline[:-1]
    )
    corridor_km = corridor_m / 1000.0

    flowline_lonlat = _flowline_to_lonlat(flowline, transform)

    # Bounding box of corridor + corridor_half_width_m margin.
    lons = [p[0] for p in flowline_lonlat]
    lats = [p[1] for p in flowline_lonlat]
    lat_mid = (min(lats) + max(lats)) / 2
    lon_margin = corridor_half_width_m / (METERS_PER_DEGREE_LAT * math.cos(math.radians(lat_mid)))
    lat_margin = corridor_half_width_m / METERS_PER_DEGREE_LAT
    bbox = (
        min(lons) - lon_margin,
        min(lats) - lat_margin,
        max(lons) + lon_margin,
        max(lats) + lat_margin,
    )

    pop_path = ensure_downloaded()
    with rasterio.open(pop_path) as src:
        window = from_bounds(*bbox, transform=src.transform)
        pop_data = src.read(1, window=window, boundless=True, fill_value=0)
        pop_transform = src.window_transform(window)

    corridor_mask = flowline_corridor_mask(
        flowline_lonlat, corridor_half_width_m, pop_transform, pop_data.shape
    )
    pop_clipped = np.clip(pop_data, 0, None)
    total_pop = float((pop_clipped * corridor_mask).sum())

    logger.info(
        "D8 corridor exposure: outlet=(%.4f,%.4f) flowline=%d cells (%.1f km) "
        "corridor=%.0fm pop=%.0f",
        outlet_lon, outlet_lat, len(flowline), corridor_km, corridor_half_width_m, total_pop,
    )

    return {
        "source": POPULATION_SOURCE_ID,
        "method": "d8_flowline_corridor",
        "corridor_km": round(corridor_km, 2),
        "corridor_half_width_m": corridor_half_width_m,
        "population_along_corridor": round(total_pop, 1),
        "population_within_buffer": round(total_pop, 1),  # backward-compat alias
        "flowline_cells": len(flowline),
        "note": (
            f"D8 downstream corridor from lake outlet dam, GLO-30 DEM, "
            f"{max_km:.0f} km max, {corridor_half_width_m:.0f} m half-width. "
            f"Validation vs. GLOF footprints pending Issue #26."
        ),
    }


# ---------------------------------------------------------------------------
# Backward-compatible public API (unchanged caller signature)
# ---------------------------------------------------------------------------


def _find_lake_by_centroid(lon: float, lat: float):
    """Return the LakeAoi whose centroid is closest to (lon, lat)."""
    from pipeline.lakes import LAKES  # noqa: PLC0415 — avoid circular at module load
    min_sq = float("inf")
    closest = None
    for lake in LAKES.values():
        sq = (lake.lon - lon) ** 2 + (lake.lat - lat) ** 2
        if sq < min_sq:
            min_sq = sq
            closest = lake
    return closest


def population_within_buffer(lon: float, lat: float, radius_km: float = BUFFER_RADIUS_KM) -> dict:
    """Downstream population exposure. Backward-compatible public API.

    Internally calls population_along_flowline() using the lake's outlet coordinates
    (looked up by centroid proximity from pipeline/lakes.py).

    Falls back to the original straight-line square buffer on FlowRoutingError,
    logging a WARNING. The 'population_within_buffer' key is always present in the
    returned dict regardless of which path ran. A 'method' field records the path.

    Args:
        lon: Lake centroid longitude — used to look up the matching LakeAoi.
        lat: Lake centroid latitude.
        radius_km: Fallback buffer radius in km (used only when routing fails).

    Returns:
        dict always containing:
          "source": POPULATION_SOURCE_ID
          "population_within_buffer": float — exposed population count
          "method": "d8_flowline_corridor" | "radial_buffer_fallback"
    """
    lake = _find_lake_by_centroid(lon, lat)
    if lake is not None:
        try:
            return population_along_flowline(lake.outlet_lon, lake.outlet_lat)
        except FlowRoutingError as exc:
            logger.warning(
                "D8 flow routing failed for lake near (%.4f, %.4f) — "
                "falling back to radial buffer: %s",
                lon, lat, exc,
            )

    # Original radial-buffer fallback (kept for safety).
    return _radial_buffer_fallback(lon, lat, radius_km)


def _radial_buffer_fallback(lon: float, lat: float, radius_km: float) -> dict:
    """Original straight-line square buffer implementation (fallback only)."""
    path = ensure_downloaded()
    lat_deg = radius_km * 1000 / METERS_PER_DEGREE_LAT
    lon_deg = radius_km * 1000 / (METERS_PER_DEGREE_LAT * math.cos(math.radians(lat)))
    bbox = (lon - lon_deg, lat - lat_deg, lon + lon_deg, lat + lat_deg)

    with rasterio.open(path) as src:
        window = from_bounds(*bbox, transform=src.transform)
        data = src.read(1, window=window, boundless=True, fill_value=0)

    total_population = float(np.clip(data, 0, None).sum())

    return {
        "source": POPULATION_SOURCE_ID,
        "method": "radial_buffer_fallback",
        "buffer_km": radius_km,
        "buffer_shape": "square",
        "population_within_buffer": round(total_population, 1),
        "note": (
            "Straight-line square buffer fallback — D8 routing failed. "
            "Not flow-path routed. See HAZARD_METHODOLOGY.md and ADR 0003."
        ),
    }
