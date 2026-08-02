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
- pipeline/ — NDWI/cloud-mask/area logic (ndwi.py, pure math, no I/O) and the scene
  source abstraction (stac_source.py) that keeps STAC-catalog choice swappable
- tests/ — run with `uv run pytest`

## Gotchas
- No CDSE (Copernicus Data Space Ecosystem) credentials exist in this project yet —
  the PoC pipeline reads from Microsoft Planetary Computer's anonymous STAC API
  instead. See docs/ai/decisions/0001-eo-platform.md before assuming CDSE is wired up.
- Sentinel-2 COGs are in UTM, not lon/lat — bbox reprojection in stac_source.py is not
  optional, dropping it silently reads the wrong window instead of erroring.
- SCL (cloud mask) ships at 20m/pixel vs B03/B08's 10m — must be upsampled 2x before
  use as a mask; see poc.py's `_upsample_scl_to_10m`.

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
