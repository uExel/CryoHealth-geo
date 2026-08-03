# CryoHealth-geo

Sentinel-2 EO service: NDWI water-extent monitoring and hazard scoring for GLOF early warning. Owns geospatial computation only — NestJS (CryoHealth-api) owns product logic/policy; the two share one PostGIS database. PRD + architecture: uExel/cryo-harness and CryoHealth-api/ARCHITECTURE.md.

## Working here
- Start sessions with /uexel:orient, end with /uexel:handoff. Pipeline:
  orient → plan → gate → build → verify → report → handoff (docs: uExel/cryo-harness).
- Task state lives in GitHub issues — labels + milestones, no boards.
- Shared AI working files: docs/ai/ (PLAN, TODO, HANDOFF, LEARNINGS, sessions, decisions).

## Map
<!-- One line per top-level folder whose purpose a newcomer can't infer from its name.
     Delete rows that are obvious — every line here loads in every session. -->
- pipeline/ — NDWI/cloud-mask/area logic (ndwi.py, pure math, no I/O); scene sources
  behind one interface (stac_source.py's SceneSource) — planetary-computer (default,
  no credentials) and cdse (production, needs CDSE_CLIENT_ID/CDSE_CLIENT_SECRET).
  hazard.py (pure math, weights/thresholds in docs/HAZARD_METHODOLOGY.md) computes the
  composite score; dem.py (GLO-30 slope) and exposure.py (WorldPop population) feed it;
  hazard_client.py reports to CryoHealth-api's /alerts/hazard-scores (needs
  CRYOHEALTH_API_KEY, matching that repo's GEO_SERVICE_API_KEY).
- tests/ — run with `uv run pytest`; CDSE/exposure/hazard_client tests are fully mocked,
  no live network/creds
- docs/HAZARD_METHODOLOGY.md — the hazard index's weights, thresholds, normalization;
  kept in lockstep with pipeline/hazard.py (same METHODOLOGY_VERSION)

## Gotchas
- Every SceneSource.read_bands must return bands already pixel-aligned on one grid —
  that contract lives at the source, never leaked to poc.py. PlanetaryComputerSource
  needs internal UTM reprojection + SCL 2x-upsample to honor it (Sentinel-2 COGs are
  UTM, not lon/lat, and SCL ships at 20m vs B03/B08's 10m); CdseSource gets this for
  free since Process API crops+reprojects server-side.
- CDSE bands are NOT reachable via the STAC item's raw `s3://eodata/...` href with just
  an OAuth2 client_credentials token — confirmed live, 401 "Token audience not
  allowed" (that needs separate S3 credentials). Use the Sentinel Hub Process API
  instead (see cdse_source.py + ADR 0001's update).
- CDSE credentials are env vars only (CDSE_CLIENT_ID/CDSE_CLIENT_SECRET via .env,
  gitignored) — never hardcode, never commit real values, .env.example has the shape.
- WorldPop's server advertises `Accept-Ranges: bytes` but ignores actual Range headers
  and always returns the full ~140MB national raster (confirmed live) — exposure.py
  downloads it once to .cache/worldpop/ (gitignored) rather than attempting a windowed
  remote read.

## gstack (REQUIRED — global install)

**Before doing ANY work, verify gstack is installed:**

```bash
test -d ~/.claude/skills/gstack/bin && echo "GSTACK_OK" || echo "GSTACK_MISSING"
```

If GSTACK_MISSING: STOP. Do not proceed. Tell the user:

> gstack is required for all AI-assisted work in this repo.
> Install it:
> ```bash
> git clone --depth 1 https://github.com/garrytan/gstack.git ~/.claude/skills/gstack
> cd ~/.claude/skills/gstack && ./setup --team
> ```
> Then restart your AI coding tool.

Do not skip skills, ignore gstack errors, or work around missing gstack.

Using gstack skills: After install, skills like /qa, /ship, /review, /investigate,
and /browse are available. Use /browse for all web browsing.
Use ~/.claude/skills/gstack/... for gstack file paths (the global path).

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
