# HANDOFF — CryoHealth-geo — 2026-08-02 15:05 PKT
Session: eo-platform-spike  Model: fable-5  Branch: feat/3-eo-platform-spike  Goal: #1  Task: #3

## State
Spike A done: ADR 0001 picks Copernicus Data Space Ecosystem for production (CDSE's
transparent free 10k-credit/month quota vs GEE's opaque commercial pricing + likely
noncommercial-tier ineligibility for funded work). Working proof built and run live
twice, reproducibly: real NDWI/cloud-mask/water-area pipeline against real, current
Sentinel-2 L2A imagery via Microsoft Planetary Computer's anonymous STAC API (zero
credentials — no GEE/CDSE account was created this session, on principle). Shishper:
0.6258 km^2 from a 2026-07-20 scene, 0% AOI cloud fraction, after correctly skipping a
100%-clouded 2026-07-28 candidate first — this multi-scene fallback IS the R2 staleness
rule's actual behavior, not just the spike's proof mechanism.

Two real bugs found and fixed by actually running this against live data (not just
unit tests): (1) Sentinel-2 COGs are in UTM, a lon/lat bbox needs reprojecting before
windowing or it silently reads the wrong pixels; (2) independently-windowed bands at
different resolutions (10m vs 20m) don't always come back exactly 2x related in pixel
count — B03 was one row taller than 2x SCL for a real scene. Both are now handled and
covered by regression tests, not just patched.

## Done this session
- pyproject.toml (uv-managed), pipeline/ package, tests/ (11 passing), ruff clean
- ADR 0001 (EO platform decision + honest scope note on the proof's data source)
- CI workflow (build didn't exist before this session)

## Not done / deferred
- CDSE wiring for production — needs Shaan (or whoever) to create the free CDSE
  account; a CdseSource class behind the same SceneSource interface is the remaining
  work, not a rewrite
- The scheduled FastAPI service, DB writes, backfill to 2023 — separate future tasks
  per the six-week plan, not this spike

## Next action
Open PR for feat/3-eo-platform-spike -> main.

## Open questions for a human
- Create a free CDSE account (dataspace.copernicus.eu) when ready to build production
  wiring — blocking: no (not needed for this task, needed for the next EO task)

## Failed approaches (do not retry)
- Trusting scene-level `eo:cloud_cover` STAC metadata as "the AOI is usable" — it's a
  whole-tile average; a scene can read 58% cloud overall while a specific small AOI is
  100% obscured. Always compute AOI-level cloud_fraction from the actual SCL pixels.
- Assuming two independently-windowed bands at different native resolutions will have
  exactly proportional pixel dimensions — crop to common shape defensively instead.

## Loops run
- live-verification fix loop: 2/3 (UTM reprojection, resolution-mismatch crop), passed
- lint fix loop: 1/3 (ambiguous var name, line length), passed

## Files touched
pyproject.toml, pipeline/**, tests/**, docs/ai/decisions/0001-eo-platform.md,
.github/workflows/ci.yml, CLAUDE.md, .gitignore

## Verification status
tests: 11/11 passing  review: n/a (no auth/data-integrity surface beyond what's already
in the ADR's own honesty framing)  qa: n/a
Live-verified twice, reproducibly, against real current satellite imagery — not mocked,
not simulated.

## Resume with
/uexel:orient   (then: either CDSE account setup + CdseSource, or move to a different
G1/G2 task while that's pending)
