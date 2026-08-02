"""Where scenes come from, kept behind one small interface for exactly one reason: this
session has no Copernicus Data Space Ecosystem (CDSE) credentials and won't create an
account to get them (see ADR 0001). Planetary Computer's STAC API is genuinely
anonymous, so it's what proves the NDWI pipeline against real Sentinel-2 imagery today.
Swapping in CDSE once real credentials exist means adding a CdseSource class here and
changing one line in poc.py — not touching ndwi.py or lakes.py at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Protocol

import numpy as np
import planetary_computer
import rasterio
from pystac_client import Client
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds


@dataclass(frozen=True)
class SceneRef:
    scene_id: str
    captured_at: date
    cloud_cover_pct: float
    # STAC asset href per band, already access-signed if the source needs that.
    assets: dict[str, str]


class SceneSource(Protocol):
    def find_recent_scenes(self, bbox: tuple[float, float, float, float], limit: int) -> list[SceneRef]: ...

    def read_bands(
        self, scene: SceneRef, bands: list[str], bbox: tuple[float, float, float, float]
    ) -> dict[str, np.ndarray]: ...


class PlanetaryComputerSource:
    """Anonymous, no account needed — see ADR 0001 for why this is the PoC source and
    not the production choice."""

    STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
    COLLECTION = "sentinel-2-l2a"

    def __init__(self) -> None:
        self._client = Client.open(self.STAC_URL)

    def find_recent_scenes(self, bbox: tuple[float, float, float, float], limit: int = 12) -> list[SceneRef]:
        """Newest first, up to `limit` candidates. Scene-level cloud_cover% (STAC
        metadata) is an average over the whole tile, not the tiny AOI — a scene can
        read 58% cloud overall while the AOI itself is 100% obscured (confirmed live
        against Shishper). Returning several candidates lets the caller pick the first
        one that's actually usable *at the AOI*, which is what R2's staleness rule
        needs: check recent scenes until one works, not just the literal latest."""
        search = self._client.search(
            collections=[self.COLLECTION],
            bbox=bbox,
            sortby=[{"field": "properties.datetime", "direction": "desc"}],
            max_items=limit,
            query={"eo:cloud_cover": {"lt": 80}},
        )
        return [
            SceneRef(
                scene_id=item.id,
                captured_at=item.datetime.date(),
                cloud_cover_pct=item.properties.get("eo:cloud_cover", 100.0),
                assets={k: a.href for k, a in planetary_computer.sign(item).assets.items()},
            )
            for item in search.items()
        ]

    def read_bands(
        self, scene: SceneRef, bands: list[str], bbox: tuple[float, float, float, float]
    ) -> dict[str, np.ndarray]:
        """bbox is WGS84 (lon/lat) — Sentinel-2 COGs are served in their native UTM
        zone, so it's reprojected per band before windowing. Getting this wrong doesn't
        error, it silently reads the wrong window — worth the explicit comment."""
        out: dict[str, np.ndarray] = {}
        for band in bands:
            href = scene.assets[band]
            with rasterio.open(href) as src:
                utm_bbox = transform_bounds("EPSG:4326", src.crs, *bbox)
                window = from_bounds(*utm_bbox, transform=src.transform)
                out[band] = src.read(1, window=window)
        return out
