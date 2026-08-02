"""Where scenes come from, kept behind one small interface. Every SceneSource.read_bands
implementation returns bands already pixel-aligned on one grid — any resampling or
windowing quirk a particular source needs is handled inside that source, never leaked
to the caller. That contract is what let CdseSource (pipeline/cdse_source.py) get built
later without poc.py needing to know which source has which quirks.
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

    def find_scenes_in_range(
        self, bbox: tuple[float, float, float, float], start: date, end: date
    ) -> list[SceneRef]: ...

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
        return self._search(bbox, max_items=limit)

    def find_scenes_in_range(
        self, bbox: tuple[float, float, float, float], start: date, end: date
    ) -> list[SceneRef]:
        """Every candidate in [start, end], not just the newest usable one — backfill
        wants a full time series, so every scene gets evaluated (and either written as
        an observation or skipped for being too cloudy), not just the latest state."""
        return self._search(bbox, max_items=None, datetime_range=f"{start.isoformat()}/{end.isoformat()}")

    def _search(
        self,
        bbox: tuple[float, float, float, float],
        max_items: int | None,
        datetime_range: str | None = None,
    ) -> list[SceneRef]:
        search = self._client.search(
            collections=[self.COLLECTION],
            bbox=bbox,
            datetime=datetime_range,
            sortby=[{"field": "properties.datetime", "direction": "desc"}],
            max_items=max_items,
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
        error, it silently reads the wrong window — worth the explicit comment.

        SCL ships at 20m/pixel vs B03/B08's 10m, and independently-windowed bands at
        different resolutions don't always come back exactly proportional in pixel
        count either (confirmed live: B03 was one row taller than 2x SCL for a real
        scene) — both handled here so callers never see misaligned arrays."""
        raw: dict[str, np.ndarray] = {}
        for band in bands:
            href = scene.assets[band]
            with rasterio.open(href) as src:
                utm_bbox = transform_bounds("EPSG:4326", src.crs, *bbox)
                window = from_bounds(*utm_bbox, transform=src.transform)
                raw[band] = src.read(1, window=window)

        if "SCL" in raw:
            raw["SCL"] = np.repeat(np.repeat(raw["SCL"], 2, axis=0), 2, axis=1)

        rows = min(a.shape[0] for a in raw.values())
        cols = min(a.shape[1] for a in raw.values())
        return {band: arr[:rows, :cols] for band, arr in raw.items()}
