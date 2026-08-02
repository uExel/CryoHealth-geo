# PLAN — CryoHealth-geo#6: wire CdseSource
Goal: #1 (G1 milestone) · Task: #6 · Loop budget: 3 · Rollback: revert PR, no schema touched

Verified live before writing the class (curl, throwaway, not committed):
- Token endpoint https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token,
  client_credentials grant — works with the real client, confirmed
- CDSE STAC search (stac.dataspace.copernicus.eu/v1/search) — unauthenticated, same
  shape as Planetary Computer's, confirmed real Shishper-area results
- Raw OData/S3 asset download needs a DIFFERENT audience the sh- client doesn't have
  (confirmed: 401 "Token audience not allowed") — not the path
- Sentinel Hub Process API (sh.dataspace.copernicus.eu/process/v1) IS the right path
  for this client: POST an evalscript + bbox + date range, get back a GeoTIFF already
  reprojected to EPSG:4326 and cropped to the exact AOI — confirmed with a real
  B03/B08/SCL request over Shishper, valid raster returned

Steps:
1. pipeline/cdse_source.py: CdseSource(SceneSource) — find_recent_scenes via STAC
   (mirrors PlanetaryComputerSource), read_bands via Process API (server-side
   crop+reproject means no UTM/resolution-alignment code needed here at all)
   — verify: uv run pytest (mocked HTTP)
2. poc.py: --source {planetary-computer,cdse} flag, default unchanged (no
   credentials required by default — cdse only activates when asked + configured)
   — verify: uv run python -m pipeline.poc --lake shishper --source cdse against
   real CDSE, live
3. .env.example documenting CDSE_CLIENT_ID/CDSE_CLIENT_SECRET (no real values)

Assumptions: Process API's per-request evalscript output (fixed width/height, single
CRS) is the production-shape read path — reusing it for a future scheduled job means
the same class, not a rewrite. GATE: N/A, direct instruction + all endpoints
verified live before code was written.
