# PLAN — CryoHealth-geo#11: hazard index
Goal: #2 (G2 milestone) · Task: #11 · Loop budget: 3 · Rollback: revert PR, no schema owned here
Depends on: CryoHealth-api#12 (service API key auth for /alerts/hazard-scores) — built first

Explicitly out of scope: full hydrological flow-path modeling for exposure (a real
fixed-radius WorldPop buffer is used instead, documented as a simplification —
see docs/HAZARD_METHODOLOGY.md). Exposure never affects the tier (PRD §8).

Steps:
1. pipeline/dem.py: mean_slope_degrees(bbox) via Copernicus GLO-30 DEM (Planetary
   Computer STAC, anonymous). Pure slope math split into a testable _slope_from_elevation
   helper, matching ndwi.py's pure-math pattern.
   — verify: uv run pytest (synthetic terrain) + live read against real lake AOIs
2. pipeline/exposure.py: population_within_buffer(lon, lat, radius_km) via real WorldPop
   Pakistan population raster. WorldPop's server ignores HTTP Range requests (confirmed
   live) — downloads and caches the ~140MB national raster once, reads windows locally.
   — verify: uv run pytest (synthetic raster) + live read against the real cached file
3. pipeline/hazard.py: pure composite-index math (growth rate, seasonal anomaly, dam
   type, glacier contact, slope, historical GLOF -> weighted score -> tier). No I/O.
   — verify: uv run pytest against hand-computed expected values
4. pipeline/hazard_client.py: POST /alerts/hazard-scores via the new GEO_SERVICE_API_KEY
   service auth (CryoHealth-api#12) — this service has no user to log in as.
5. pipeline/hazard_batch.py: orchestrator — DB reads (Observations + Lake static
   fields) + dem.py + exposure.py + hazard.py + hazard_client.py, one pass per lake,
   per-lake exception isolation matching batch.py's pattern.
6. Wire into service.py: daily job runs the observation batch then the hazard pass, in
   that order; add POST /run-hazard for a manual trigger independent of /run.
7. docs/HAZARD_METHODOLOGY.md: the weights, thresholds, and normalization for every
   component, kept in lockstep with hazard.py — R3's "methodology doc in repo" criterion.

Verification: live-verify against real Observation data (the 2023-2026 backfilled
dataset, CryoHealth-geo#8) + real DEM + real WorldPop + a real running CryoHealth-api
instance — confirm a real tier transition creates a real Alert, confirm components are
reproducible by hand from docs/HAZARD_METHODOLOGY.md, clean up any test artifacts
created during verification (same discipline as CryoHealth-api#12's live check).
