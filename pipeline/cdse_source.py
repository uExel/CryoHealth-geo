"""Production scene source: Copernicus Data Space Ecosystem (ADR 0001's decision).

Two CDSE services, verified live against a real account before this was written:
- STAC search (stac.dataspace.copernicus.eu) — unauthenticated, candidate discovery,
  same shape as PlanetaryComputerSource's search.
- Sentinel Hub Process API (sh.dataspace.copernicus.eu/process/v1) — OAuth2
  client_credentials, band retrieval. This is deliberately NOT "download the raw
  Sentinel-2 asset and window it" (CDSE serves those as s3://eodata/... URIs needing
  separate S3 credentials this client doesn't have — confirmed live, 401 "Token
  audience not allowed"). Process API instead computes the crop+reproject server-side:
  request a bbox in EPSG:4326 at a fixed width/height, get back all requested bands
  already aligned on one pixel grid. That's not a shortcut — it structurally eliminates
  the two bugs task #3 found in the manual-windowing approach (UTM reprojection,
  10m/20m resolution misalignment), rather than needing the same fixes ported here.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass

import numpy as np
import requests
from pystac_client import Client

from pipeline.stac_source import SceneRef

TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
STAC_URL = "https://stac.dataspace.copernicus.eu/v1"
PROCESS_URL = "https://sh.dataspace.copernicus.eu/process/v1"
COLLECTION = "sentinel-2-l2a"

# ~10m/pixel target resolution. 1 degree of longitude shrinks with latitude
# (cos(lat)); latitude itself doesn't. Computed per-AOI in read_bands, not hardcoded,
# since lake AOIs sit across a ~1.2 degree latitude spread (Ghizer to Hunza).
METERS_PER_DEGREE_LAT = 111_320


class CdseAuthError(RuntimeError):
    """Raised with the real API error message, never swallowed into a generic 401."""


@dataclass
class _Token:
    value: str
    expires_at: float


class CdseSource:
    def __init__(self, client_id: str | None = None, client_secret: str | None = None) -> None:
        self._client_id = client_id or os.environ.get("CDSE_CLIENT_ID")
        self._client_secret = client_secret or os.environ.get("CDSE_CLIENT_SECRET")
        if not self._client_id or not self._client_secret:
            raise CdseAuthError(
                "CDSE_CLIENT_ID and CDSE_CLIENT_SECRET must be set (env vars or constructor args) "
                "to use CdseSource. Create a free account + OAuth client at dataspace.copernicus.eu."
            )
        self._stac = Client.open(STAC_URL)
        self._token: _Token | None = None

    def _access_token(self) -> str:
        # 60s margin so a token doesn't expire mid-request.
        if self._token is None or time.time() > self._token.expires_at - 60:
            resp = requests.post(
                TOKEN_URL,
                data={
                    "grant_type": "client_credentials",
                    "client_id": self._client_id,
                    "client_secret": self._client_secret,
                },
                timeout=30,
            )
            if resp.status_code != 200:
                raise CdseAuthError(f"CDSE token request failed ({resp.status_code}): {resp.text}")
            body = resp.json()
            self._token = _Token(value=body["access_token"], expires_at=time.time() + body["expires_in"])
        return self._token.value

    def find_recent_scenes(self, bbox: tuple[float, float, float, float], limit: int = 12) -> list[SceneRef]:
        search = self._stac.search(
            collections=[COLLECTION],
            bbox=bbox,
            sortby=[{"field": "properties.datetime", "direction": "desc"}],
            max_items=limit,
            query={"eo:cloud_cover": {"lt": 80}},
        )
        # No per-band hrefs needed — Process API takes the scene's date, not asset URLs.
        return [
            SceneRef(
                scene_id=item.id,
                captured_at=item.datetime.date(),
                cloud_cover_pct=item.properties.get("eo:cloud_cover", 100.0),
                assets={},
            )
            for item in search.items()
        ]

    def read_bands(
        self, scene: SceneRef, bands: list[str], bbox: tuple[float, float, float, float]
    ) -> dict[str, np.ndarray]:
        west, south, east, north = bbox
        lat_center = (south + north) / 2
        width_m = (east - west) * METERS_PER_DEGREE_LAT * math.cos(math.radians(lat_center))
        height_m = (north - south) * METERS_PER_DEGREE_LAT
        width_px = max(1, round(width_m / 10))
        height_px = max(1, round(height_m / 10))

        band_lines = ",".join(f'"{b}"' for b in bands)
        # UINT16 output: reflectance bands *10000 (matches raw Sentinel-2 L2A DN scale,
        # keeps 4 significant digits); SCL is already a small integer, output as-is.
        return_lines = "\n".join(
            f"  out[{i}] = s.{b} === undefined ? 0 : "
            f"(s.{b} <= 1 ? Math.round(s.{b} * 10000) : Math.round(s.{b}));"
            for i, b in enumerate(bands)
        )
        evalscript = f"""//VERSION=3
function setup() {{
  return {{ input: [{band_lines}], output: {{ bands: {len(bands)}, sampleType: "UINT16" }} }};
}}
function evaluatePixel(s) {{
  let out = new Array({len(bands)});
{return_lines}
  return out;
}}"""

        day = scene.captured_at.isoformat()
        request = {
            "input": {
                "bounds": {
                    "properties": {"crs": "http://www.opengis.net/def/crs/OGC/1.3/CRS84"},
                    "bbox": [west, south, east, north],
                },
                "data": [
                    {
                        "type": COLLECTION,
                        "dataFilter": {"timeRange": {"from": f"{day}T00:00:00Z", "to": f"{day}T23:59:59Z"}},
                    }
                ],
            },
            "output": {
                "width": width_px,
                "height": height_px,
                "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}],
            },
            "evalscript": evalscript,
        }

        resp = requests.post(
            PROCESS_URL,
            headers={"Authorization": f"Bearer {self._access_token()}"},
            json=request,
            timeout=60,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"CDSE Process API request failed ({resp.status_code}): {resp.text}")

        return _read_multiband_tiff(resp.content, bands)


def _read_multiband_tiff(tiff_bytes: bytes, bands: list[str]) -> dict[str, np.ndarray]:
    import io

    import rasterio

    with rasterio.open(io.BytesIO(tiff_bytes)) as src:
        return {band: src.read(i + 1) for i, band in enumerate(bands)}
