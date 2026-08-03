"""Downstream population exposure (PRD §8): WorldPop population within a fixed-radius
square buffer around each lake, as a documented simplification of the full flow-path-
buffer model PRD §8 describes (true flow-routing needs flow accumulation from the DEM —
a separate, larger scope item, not required for the tier itself). Real gridded
population data (WorldPop, Pakistan, 2025, 100m, constrained) — never synthetic.
Explicitly does NOT affect the hazard tier; stored in HazardScore.components for
prioritization display only (PRD §8 is explicit on this point).

WorldPop's server ignores HTTP Range requests (confirmed live: it advertises
`Accept-Ranges: bytes` but always returns the full body regardless of the Range header
sent), so windowed/streamed reads aren't possible — the national raster (~140MB) is
downloaded once and cached locally, then read with ordinary local-file windowing.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import rasterio
import requests
from rasterio.windows import from_bounds

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


def population_within_buffer(lon: float, lat: float, radius_km: float = BUFFER_RADIUS_KM) -> dict:
    """Real population count within a straight-line, square radius_km buffer around
    (lon, lat) — a simplification of PRD §8's flow-path-buffer model (see module
    docstring). Negative pixel values (WorldPop's nodata sentinel in this product) are
    treated as zero population, never as negative population."""
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
        "buffer_km": radius_km,
        "buffer_shape": "square",
        "population_within_buffer": round(total_population, 1),
        "note": "Straight-line square buffer, not flow-path routed — see PRD §8 and module docstring.",
    }
