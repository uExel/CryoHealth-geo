"""DEM utilities: terrain slope and D8 flow routing from the Copernicus GLO-30 DEM.

Read via Microsoft Planetary Computer's STAC catalog — anonymous, no account needed,
same pattern as PlanetaryComputerSource in stac_source.py.

Two separate capabilities, both using the same DEM source:

1. mean_slope_degrees(bbox) — mean terrain slope for the hazard composite score.
   Static input; cached per-process via @functools.cache.

2. D8 flow routing (Issue #19, ADR 0003) — fetch a larger DEM extent around the
   lake outlet dam, fill pits, compute flow direction, trace the downstream flowline.
   Used by pipeline/exposure.py to replace the radial population buffer with a
   physically correct downstream inundation corridor.

All routing math functions (fill_pits, d8_flow_direction, flow_accumulation,
downstream_flowline) are pure NumPy with no I/O — same split as ndwi.py — so they
are testable with synthetic DEMs without any network access.
"""

from __future__ import annotations

import heapq
import math
from functools import cache
from typing import TYPE_CHECKING

import numpy as np
import planetary_computer
import rasterio
from pystac_client import Client
from rasterio.merge import merge as rasterio_merge
from rasterio.windows import from_bounds

if TYPE_CHECKING:
    import affine  # only for type hints

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "cop-dem-glo-30"

# Same constant as cdse_source.py and exposure.py.
METERS_PER_DEGREE_LAT = 111_320

# D8 direction codes (ESRI standard) → (Δrow, Δcol)
_D8_CODE_TO_DELTA: dict[int, tuple[int, int]] = {
    1:   (0,  1),   # E
    2:   (1,  1),   # SE
    4:   (1,  0),   # S
    8:   (1, -1),   # SW
    16:  (0, -1),   # W
    32:  (-1, -1),  # NW
    64:  (-1,  0),  # N
    128: (-1,  1),  # NE
}


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class FlowRoutingError(RuntimeError):
    """Raised when D8 routing cannot complete.

    Possible causes: flat/nearly-flat DEM with no downslope neighbours from the
    outlet cell, outlet coordinates outside the DEM extent, or no GLO-30 tiles
    covering the requested area.
    Callers (pipeline/exposure.py) catch this and fall back to the radial buffer.
    """


# ---------------------------------------------------------------------------
# Slope (pre-existing, unchanged)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# D8 flow routing — pure math (no I/O, testable with synthetic arrays)
# ---------------------------------------------------------------------------


def fill_pits(elevation: np.ndarray) -> np.ndarray:
    """Remove depressions from an elevation array using the Priority-Flood algorithm.

    Implements Wang & Liu (2006): process cells in priority order (lowest elevation
    first from the boundary inward), raising any cell that is lower than its
    already-processed neighbour. O(n log n) where n is the number of cells.

    Args:
        elevation: 2D float array. May be any numeric dtype; returned as float64.

    Returns:
        Depression-free elevation array, float64, same shape as input.
        Border cells are preserved. Interior pits are raised to the level of their
        lowest rim neighbour + a negligible epsilon (1e-5 m) so D8 can always
        drain them without creating new flat areas.
    """
    elev = elevation.astype(np.float64)
    rows, cols = elev.shape
    filled = np.full_like(elev, np.inf)
    in_queue = np.zeros((rows, cols), dtype=bool)

    # Min-heap: (elevation, row, col)
    heap: list[tuple[float, int, int]] = []

    # Seed with all border cells at their real elevation.
    for r in range(rows):
        for c in [0, cols - 1]:
            if not in_queue[r, c]:
                heapq.heappush(heap, (elev[r, c], r, c))
                filled[r, c] = elev[r, c]
                in_queue[r, c] = True
    for c in range(cols):
        for r in [0, rows - 1]:
            if not in_queue[r, c]:
                heapq.heappush(heap, (elev[r, c], r, c))
                filled[r, c] = elev[r, c]
                in_queue[r, c] = True

    while heap:
        z, r, c = heapq.heappop(heap)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1),
                       (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols) or in_queue[nr, nc]:
                continue
            in_queue[nr, nc] = True
            # Raise neighbour so it is at least as high as the current cell.
            # The 1e-5 m increment ensures D8 can always find a downslope direction.
            filled[nr, nc] = max(elev[nr, nc], filled[r, c] + 1e-5)
            heapq.heappush(heap, (filled[nr, nc], nr, nc))

    return filled


def d8_flow_direction(elevation: np.ndarray, dx_m: float, dy_m: float) -> np.ndarray:
    """Compute D8 steepest-descent flow direction for each cell.

    For each cell, finds the 8-connected neighbour with the steepest downslope
    gradient and assigns the corresponding ESRI D8 code:
      E=1, SE=2, S=4, SW=8, W=16, NW=32, N=64, NE=128, flat/border=0.

    Diagonal distances use sqrt(dx²+dy²). Vectorized with np.pad — no Python cell loop.

    Args:
        elevation: 2D float array (should be pit-filled first).
        dx_m: Pixel width in metres (E–W direction).
        dy_m: Pixel height in metres (N–S direction).

    Returns:
        uint8 array of D8 direction codes, same shape as elevation.
        Border cells and flat cells (no downslope neighbour) are coded 0.
    """
    rows, cols = elevation.shape
    diag_m = math.sqrt(dx_m**2 + dy_m**2)

    # (Δrow, Δcol, D8-code, distance)
    neighbours = [
        (0,   1,   1,   dx_m),
        (1,   1,   2,   diag_m),
        (1,   0,   4,   dy_m),
        (1,  -1,   8,   diag_m),
        (0,  -1,   16,  dx_m),
        (-1, -1,   32,  diag_m),
        (-1,  0,   64,  dy_m),
        (-1,  1,   128, diag_m),
    ]

    # Pad elevation by 1 cell on all sides (edge-fill) so shifted views stay in bounds.
    padded = np.pad(elevation.astype(np.float64), 1, mode="edge")

    max_slope = np.full((rows, cols), -np.inf)
    direction = np.zeros((rows, cols), dtype=np.int32)

    for dr, dc, code, dist in neighbours:
        # Slice the padded array to get neighbour values aligned with the original grid.
        r0 = 1 + dr
        c0 = 1 + dc
        neighbour = padded[r0: r0 + rows, c0: c0 + cols]
        slope = (elevation - neighbour) / dist
        mask = slope > max_slope
        max_slope = np.where(mask, slope, max_slope)
        direction = np.where(mask, code, direction)

    # Borders and flat cells → no valid direction.
    direction[0, :]  = 0
    direction[-1, :] = 0
    direction[:, 0]  = 0
    direction[:, -1] = 0
    direction[max_slope <= 0] = 0

    return direction.astype(np.uint8)


def flow_accumulation(direction: np.ndarray, elevation: np.ndarray) -> np.ndarray:
    """Compute upstream cell count (flow accumulation) for each cell.

    Processes cells from highest to lowest elevation, adding each cell's
    accumulated count to its D8 receiver. This is equivalent to a topological
    sort by elevation for a pit-free DEM.

    Args:
        direction: uint8 D8 direction array (from d8_flow_direction).
        elevation: float64 elevation array (used only for sort order).

    Returns:
        int32 array of upstream cell counts including the cell itself.
        Outlet cells (no receiver or border) retain their own accumulated value.
    """
    rows, cols = direction.shape
    accum = np.ones(rows * cols, dtype=np.int32)
    dir_flat = direction.ravel().astype(np.int32)
    # Process highest-elevation cells first so each cell's tally is complete
    # before it is added to its receiver.
    order = np.argsort(elevation.ravel())[::-1]

    for flat_idx in order:
        code = int(dir_flat[flat_idx])
        if code not in _D8_CODE_TO_DELTA:
            continue
        r, c = divmod(int(flat_idx), cols)
        dr, dc = _D8_CODE_TO_DELTA[code]
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            accum[nr * cols + nc] += accum[flat_idx]

    return accum.reshape(rows, cols)


def downstream_flowline(
    direction: np.ndarray,
    outlet_rc: tuple[int, int],
    transform: object,  # affine.Affine
    max_km: float = 50.0,
) -> list[tuple[int, int]]:
    """Trace the D8 downstream flowline from an outlet cell up to max_km.

    Walks the D8 direction grid from outlet_rc, stepping to the next receiver
    cell at each iteration. Stops when:
    - The next cell would be outside the DEM extent.
    - The current cell has no valid direction (code 0 — flat or border).
    - The cumulative path length exceeds max_km.
    - A cell is revisited (prevents infinite loops on flat areas).

    Args:
        direction: uint8 D8 direction array.
        outlet_rc: (row, col) of the lake outlet / dam-toe cell in the DEM grid.
        transform: affine.Affine geo-transform for computing step distances.
        max_km: Maximum flowline length in kilometres (default 50).

    Returns:
        List of (row, col) tuples starting at outlet_rc and ending at the last
        reachable cell within max_km. Minimum length 1 (just the outlet cell).

    Raises:
        FlowRoutingError: if outlet_rc is outside the DEM bounds, or if the
            outlet cell itself has direction code 0 (flat DEM at outlet).
    """
    rows, cols = direction.shape
    r0, c0 = outlet_rc

    if not (0 <= r0 < rows and 0 <= c0 < cols):
        raise FlowRoutingError(
            f"Outlet cell {outlet_rc} is outside DEM extent ({rows}×{cols}). "
            "Check outlet_lon/outlet_lat in lakes.py and the DEM fetch bbox."
        )

    if direction[r0, c0] == 0:
        raise FlowRoutingError(
            f"Outlet cell {outlet_rc} has D8 direction code 0 (flat or border). "
            "The DEM may be too flat at the outlet, or the outlet coordinate is on "
            "a ridge rather than the dam toe. Check outlet_lon/outlet_lat in lakes.py."
        )

    # Pixel dimensions in metres.
    pixel_size_x_deg = abs(transform.a)
    pixel_size_y_deg = abs(transform.e)
    lat_centre = transform.f + (r0 + 0.5) * transform.e
    dx_m = pixel_size_x_deg * METERS_PER_DEGREE_LAT * math.cos(math.radians(lat_centre))
    dy_m = pixel_size_y_deg * METERS_PER_DEGREE_LAT
    diag_m = math.sqrt(dx_m**2 + dy_m**2)

    dist_per_code: dict[int, float] = {
        1: dx_m, 2: diag_m, 4: dy_m,  8: diag_m,
        16: dx_m, 32: diag_m, 64: dy_m, 128: diag_m,
    }

    max_m = max_km * 1000.0
    flowline: list[tuple[int, int]] = [(r0, c0)]
    visited: set[tuple[int, int]] = {(r0, c0)}
    total_dist = 0.0
    r, c = r0, c0

    while True:
        code = int(direction[r, c])
        if code not in _D8_CODE_TO_DELTA:
            break  # flat or border cell — end of traceable flowline
        dr, dc = _D8_CODE_TO_DELTA[code]
        nr, nc = r + dr, c + dc
        if not (0 <= nr < rows and 0 <= nc < cols):
            break
        if (nr, nc) in visited:
            break  # cycle in flat area — stop
        step = dist_per_code[code]
        total_dist += step
        if total_dist > max_m:
            break
        flowline.append((nr, nc))
        visited.add((nr, nc))
        r, c = nr, nc

    return flowline


# ---------------------------------------------------------------------------
# D8 flow routing — I/O (fetches DEM, calls pure-math functions above)
# ---------------------------------------------------------------------------


@cache
def fetch_dem_for_flowrouting(
    outlet_lon: float,
    outlet_lat: float,
    max_km: float = 50.0,
) -> tuple[np.ndarray, object]:
    """Fetch, mosaic, and pit-fill a GLO-30 DEM covering the flow-routing corridor.

    Fetches a bounding box centred on the outlet point, expanded by (max_km + 5 km)
    in every direction so the full 50 km downstream corridor is within the DEM extent.
    Result is cached by (outlet_lon, outlet_lat, max_km) — terrain is static.

    Uses the same Planetary Computer STAC path as mean_slope_degrees() — same source,
    no new credentials required.

    Args:
        outlet_lon: Dam-toe / outlet longitude (WGS84).
        outlet_lat: Dam-toe / outlet latitude (WGS84).
        max_km: Corridor length in km (default 50). Used to size the fetch bbox.

    Returns:
        (filled_elevation: np.ndarray float64, transform: affine.Affine)

    Raises:
        FlowRoutingError: if no GLO-30 tile covers the requested bbox.
    """
    margin_km = max_km + 5.0  # 5 km margin around the corridor
    lat_margin = margin_km * 1000 / METERS_PER_DEGREE_LAT
    lon_margin = margin_km * 1000 / (
        METERS_PER_DEGREE_LAT * math.cos(math.radians(outlet_lat))
    )

    west  = outlet_lon - lon_margin
    east  = outlet_lon + lon_margin
    south = outlet_lat - lat_margin
    north = outlet_lat + lat_margin

    client = Client.open(STAC_URL)
    search = client.search(collections=[COLLECTION], bbox=(west, south, east, north))
    items = [planetary_computer.sign(item) for item in search.items()]

    if not items:
        raise FlowRoutingError(
            f"No {COLLECTION} DEM tile covers outlet ({outlet_lon:.4f}, {outlet_lat:.4f}). "
            "Check outlet coordinates in pipeline/lakes.py."
        )

    if len(items) == 1:
        with rasterio.open(items[0].assets["data"].href) as src:
            window = from_bounds(west, south, east, north, transform=src.transform)
            elevation = src.read(1, window=window).astype(np.float64)
            transform = src.window_transform(window)
    else:
        sources = [rasterio.open(item.assets["data"].href) for item in items]
        try:
            mosaic, transform = rasterio_merge(
                sources, bounds=(west, south, east, north)
            )
            elevation = mosaic[0].astype(np.float64)
        finally:
            for src in sources:
                src.close()

    lat_center = (south + north) / 2
    pixel_size_x_deg = abs(transform.a)
    pixel_size_y_deg = abs(transform.e)
    dx_m = pixel_size_x_deg * METERS_PER_DEGREE_LAT * math.cos(math.radians(lat_center))
    dy_m = pixel_size_y_deg * METERS_PER_DEGREE_LAT

    filled = fill_pits(elevation)
    direction = d8_flow_direction(filled, dx_m, dy_m)

    return filled, transform, direction, dx_m, dy_m


__all__ = [
    "FlowRoutingError",
    "d8_flow_direction",
    "downstream_flowline",
    "fetch_dem_for_flowrouting",
    "fill_pits",
    "flow_accumulation",
    "mean_slope_degrees",
]
