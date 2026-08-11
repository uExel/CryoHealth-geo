# CryoHealth-geo

Sentinel-2 EO service: NDWI water-extent monitoring and hazard scoring for GLOF early warning. Owns geospatial computation only — NestJS (CryoHealth-api) owns product logic/policy; the two share one PostGIS database. PRD + architecture: uExel/cryo-harness and CryoHealth-api/ARCHITECTURE.md.

## Working here

- Start sessions with /uexel:orient, end with /uexel:handoff. Pipeline:
  orient → plan → gate → build → verify → report → handoff (docs: uExel/cryo-harness).
- Task state lives in GitHub issues — labels + milestones, no boards.
- Shared AI working files: docs/ai/ (PLAN, TODO, HANDOFF, LEARNINGS, sessions, decisions).

## Commands (`uv`)

```bash
uv sync
uv run uvicorn pipeline.service:app --reload   # GET /health, POST /run, POST /run-hazard
uv run pytest                                  # uv run pytest tests/test_x.py::test_name for one test
uv run ruff check . && uv run ruff format .
```

Network-touching modules (`stac_source.py`, `cdse_source.py`, `hazard_client.py`) are fully
mocked in tests — no live network calls in the test suite.

## Map

- pipeline/ — ndwi.py (pure math); scene sources behind SceneSource (stac_source.py:
  planetary-computer default, cdse production); hazard.py (pure math, see
  docs/HAZARD_METHODOLOGY.md) + dem.py/exposure.py feed it; hazard_client.py reports to
  CryoHealth-api (needs CRYOHEALTH_API_KEY = that repo's GEO_SERVICE_API_KEY)
- tests/ — `uv run pytest`; network-touching modules are fully mocked
- docs/HAZARD_METHODOLOGY.md — weights/thresholds, kept in lockstep with hazard.py

## Gotchas

- Every SceneSource.read_bands returns pre-aligned bands (UTM reprojection + SCL
  2x-upsample handled per-source, never leaked to callers)
- CDSE bands need the Sentinel Hub Process API, not raw `s3://eodata/...` (401 live —
  needs separate S3 creds). Credentials are env-only, gitignored .env
- WorldPop's server ignores Range headers despite advertising support — exposure.py
  downloads the ~140MB raster once to .cache/worldpop/ (gitignored)

## gstack (REQUIRED — global install)

**Before doing ANY work, verify gstack is installed:**

```bash
test -d ~/.claude/skills/gstack/bin && echo "GSTACK_OK" || echo "GSTACK_MISSING"
```

If GSTACK_MISSING: STOP. Do not proceed. Tell the user:

> gstack is required for all AI-assisted work in this repo.
> Install it:
>
> ```bash
> git clone --depth 1 https://github.com/garrytan/gstack.git ~/.claude/skills/gstack
> cd ~/.claude/skills/gstack && ./setup --team
> ```
>
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
