# PLAN — CryoHealth-geo#3: Spike A, EO platform choice + PoC
Goal: #1 (G1 milestone) · Task: #3 · Loop budget: 3 · Rollback: revert PR, no schema/DB touched

Scope note (decided before writing code, documented in ADR 0001): the DoD's "working
proof on the chosen platform" assumes credentials this session doesn't have and won't
create (no GEE/CDSE account creation). Proof instead validates the same NDWI pipeline
against real Sentinel-2 imagery via Planetary Computer's anonymous STAC API — a
different, zero-auth data source, with the scene-source behind an interface so swapping
in CDSE later (once Shaan creates the free account) is additive, not a rewrite.

Steps:
1. ADR 0001: GEE vs CDSE comparison + decision (CDSE — transparent free quota vs GEE's
   opaque commercial pricing and noncommercial-tier ineligibility for funded work)
2. pipeline/ndwi.py: pure NDWI + cloud-mask + area math, tested with synthetic arrays
   (no network) — verify: uv run pytest
3. pipeline/lakes.py: the same 6 cited lakes as CryoHealth-api's seed (ported, not
   re-derived, so the two services agree on where a lake is)
4. pipeline/stac_source.py + poc.py: real STAC search + band read against Planetary
   Computer, UTM reprojection (Sentinel-2 COGs aren't in lon/lat), SCL 20m->10m
   alignment — verify: uv run python -m pipeline.poc --lake shishper produces a real
   area value against live imagery

GATE: scope decision (proof source != chosen production platform) stated up front in
ADR 0001, not discovered mid-build or silently smoothed over.
