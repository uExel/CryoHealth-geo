# HANDOFF — CryoHealth-geo — 2026-08-02 16:20 PKT
Session: cdse-source  Model: fable-5  Branch: feat/6-cdse-source  Goal: #1  Task: #6

## State
CdseSource built and live-verified against Shaan's real CDSE account. Production EO
platform (ADR 0001's decision) is now actually wired, not just chosen. Cross-validated
against PlanetaryComputerSource on the same real scene: 0.6291 km^2 (CDSE) vs
0.6258 km^2 (Planetary Computer) — 0.5% apart, consistent with two independent access
paths on the same underlying data.

Real integration finding: CDSE's STAC asset hrefs are s3://eodata/... URIs needing
separate S3 credentials the OAuth client doesn't have (confirmed live: 401 "Token
audience not allowed" against the OData download API). Used the Sentinel Hub Process
API instead — it crops+reprojects server-side to the exact AOI, which structurally
eliminates both alignment bugs task #3 had to fix for manual windowing (not by porting
the fixes, by not needing them).

Refactored the SceneSource contract while here: read_bands must return pre-aligned
bands regardless of source. Moved PlanetaryComputerSource's UTM-reprojection/SCL-
upsample quirks fully inside its own read_bands so poc.py stays source-agnostic —
otherwise CdseSource's already-aligned output would have been wrongly re-upsampled by
logic that belonged to a different source's quirk.

## Done this session
- pipeline/cdse_source.py: CdseSource(SceneSource), STAC discovery + Process API bands
- poc.py: --source {planetary-computer,cdse} flag
- stac_source.py: alignment logic moved inside PlanetaryComputerSource.read_bands
- 15/15 tests passing (5 new, CDSE tests fully mocked — no live creds needed in CI)
- .env.example; real credentials in local .env only, confirmed gitignored throughout
- ADR 0001 updated with the production-wiring confirmation

## Not done / deferred
- Scheduled job, DB writes into CryoHealth-api's schema, backfill to 2023 — separate
  future tasks, this was "make CDSE actually work," not "build the production service"

## Next action
Open PR for feat/6-cdse-source -> main.

## Open questions for a human
- none blocking

## Failed approaches (do not retry)
- Reading CDSE Sentinel-2 bands via the STAC item's raw s3://eodata/... href with the
  OAuth2 client_credentials bearer token — 401 "Token audience not allowed", needs
  separate S3 credentials this client type doesn't have. Sentinel Hub Process API is
  the correct path for a sh-prefixed OAuth client.

## Loops run
- lint fix loop: 1/3 (line length x3), passed

## Files touched
pipeline/cdse_source.py (new), pipeline/poc.py, pipeline/stac_source.py, pyproject.toml,
tests/test_cdse_source.py (new), tests/test_stac_source.py (new, replaces test_poc.py),
.env.example (new), docs/ai/decisions/0001-eo-platform.md, CLAUDE.md

## Verification status
tests: 15/15 passing  review: n/a (no auth/PII surface — CDSE creds are env-var only,
gitignored, never logged)  qa: n/a
Live-verified against real production CDSE: real bearer token, real STAC search, real
Process API band retrieval, real NDWI result, cross-checked against a second
independent implementation on the same scene.

## Resume with
/uexel:orient   (then: scheduled FastAPI service + DB writes, or another G1/G2 task)
