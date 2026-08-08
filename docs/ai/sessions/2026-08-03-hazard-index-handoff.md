# HANDOFF — CryoHealth-geo — 2026-08-03 13:27 PKT
Session: hazard-index  Model: sonnet-5  Branch: feat/11-hazard-index  Goal: #2  Task: #11

## State
Composite hazard index built end-to-end and live-verified against real infrastructure:
real Observation history (the 2023-2026 backfilled dataset, CryoHealth-geo#8), real
Copernicus GLO-30 DEM slope, real WorldPop population exposure, and a real running
CryoHealth-api instance (CryoHealth-api#12's new service API key auth). All 6 lakes now
have a real hazard score, tier, and active alert in the database — this is genuine first
output, not test data, and was left in place rather than cleaned up.

**Result for all 6 lakes** (first real run, `as_of` = today):

| Lake | Score | Tier | Alert created |
|---|---|---|---|
| badswat | 0.9567 | critical | yes |
| khurdopin | 0.5596 | high | yes |
| shishper | 0.4913 | high | yes |
| batura | 0.3636 | watch | yes |
| ghulkin | 0.3157 | watch | yes |
| passu | 0.2815 | watch | yes |

Every lake transitioned from the default `normal` tier, so every lake correctly created
exactly one active `Alert` — confirmed directly via `psql` (`Lake.currentTier` and
`alerts` both checked), not just via the CLI's own reported counts.

**Two real bugs found and fixed during live verification** (both logged in LEARNINGS.md):
1. `lake_id_for_slug` returns a `uuid.UUID` (psycopg auto-parses `uuid` columns) — fine
   as a SQL bind param, but `json.dumps`/httpx has no encoder for it, so posting it
   straight into the hazard-score HTTP payload raised `TypeError`. Fixed with `str()` at
   the DB→HTTP boundary in `hazard_batch.py`.
2. `_observations()`'s query had no `ORDER BY`. `hazard.py`'s closest-observation lookup
   (`min()`/`max()` over rows) ties when two observations are equidistant from a target
   date — which one wins depended on Postgres's unspecified return order. Confirmed
   live: this landed on a genuinely noisy khurdopin reading (0.0026 km², vs. its
   same-distance neighbor at 0.1327 km²) and produced a spurious 1900% "growth" figure,
   pushing khurdopin from `high` to a wrong `critical`. Fixed by adding
   `ORDER BY "capturedAt"` — ties now deterministically resolve to the earlier date.
   Documented as a known limitation category in docs/HAZARD_METHODOLOGY.md (small,
   sub-pixel lakes have real day-to-day NDWI noise; the 50% growth-saturation clamp
   bounds any single outlier's contribution, but doesn't eliminate the underlying noise).

Exposure sanity-check: real WorldPop population within a 5km buffer ranged from ~2 to 40
people for the remote high-altitude lakes, but ~2963 for Ghulkin — which sits near
settled Karakoram Highway villages (Hussaini/Gulmit), a plausible, explainable pattern,
not an anomaly.

## Done this session
- CryoHealth-api#12 (separate, merged first): `@AllowServiceKey()` — service API key
  auth for `POST /alerts/hazard-scores`, since this service has no user to log in as
- pipeline/dem.py (new): mean_slope_degrees via Copernicus GLO-30 DEM (Planetary
  Computer, anonymous); pure slope math split into _slope_from_elevation for testability
- pipeline/exposure.py (new): population_within_buffer via real WorldPop Pakistan
  population raster. Found live: WorldPop's server advertises `Accept-Ranges: bytes` but
  ignores actual Range headers — downloads and caches the ~140MB national raster once
  (.cache/worldpop/, gitignored) instead of attempting a windowed remote read
- pipeline/hazard.py (new): pure composite-index math — growth rate (30d/90d), seasonal
  anomaly, dam type, glacier contact, slope, historical GLOF → weighted score → tier
- pipeline/hazard_client.py (new): POST /alerts/hazard-scores via the new service key
- pipeline/hazard_batch.py (new): orchestrator tying DB + dem.py + exposure.py +
  hazard.py + hazard_client.py together, per-lake exception isolation (batch.py's
  pattern)
- service.py: daily job now runs the observation batch then the hazard pass in
  sequence; added POST /run-hazard for a manual trigger
- docs/HAZARD_METHODOLOGY.md (new): every weight, threshold, and normalization,
  explicit known-limitations section — R3's "methodology doc in repo" criterion
- tests/test_dem.py, test_exposure.py, test_hazard.py, test_hazard_client.py,
  test_hazard_batch.py (new): 56/56 tests passing, mocked/synthetic, no live
  DB/network/DEM/WorldPop dependency
- Live verification against real infrastructure end to end (see State above)

## Not done / deferred
- Real hydrological flow-path modeling for exposure — a fixed-radius WorldPop buffer is
  used instead, explicitly documented as a simplification (docs/HAZARD_METHODOLOGY.md)
- Threshold calibration against a real GLOF event — none of the 6 lakes has had one in
  the observed period; thresholds are a documented starting point, not final
- Auth on POST /run and /run-hazard — same documented gap as task #8, meant for a
  private network

## Next action
Task #11 is done pending PR merge. After merge: consider a periodic re-run cadence
decision (currently wired into the same daily job as the observation batch) and whether
the initial tier/alert state for all 6 lakes should prompt any immediate human review
given badswat landed at `critical` on the very first real run.

## Open questions for a human
- badswat's first real hazard score is `critical` (0.9567) — moraine dam, glacier
  contact, historical GLOF record, plus real recent growth. Worth a human look at
  whether this matches on-the-ground expectations, since it's the system's first live
  output at the highest tier.

## Failed approaches (do not retry)
- Passing a DB-sourced `uuid.UUID` straight into an HTTP JSON payload — cast to `str()`
  at that boundary (see LEARNINGS.md)
- An unordered SQL query feeding closest-date tie-breaking logic — always ORDER BY
  (see LEARNINGS.md)
- Windowed/streamed reads against WorldPop's raster server — it ignores Range headers
  despite advertising support; download once, read locally

## Loops run
- lint fix loop: 1 pass (functools.cache rewrite, UP017 x2, line-length x5, all
  auto-fixed or manually wrapped), passed clean

## Files touched
pipeline/dem.py (new), pipeline/exposure.py (new), pipeline/hazard.py (new),
pipeline/hazard_client.py (new), pipeline/hazard_batch.py (new), pipeline/service.py,
docs/HAZARD_METHODOLOGY.md (new), docs/ai/PLAN.md, docs/ai/LEARNINGS.md,
docs/ai/HANDOFF.md, CLAUDE.md, .env.example, .gitignore, tests/test_dem.py (new),
tests/test_exposure.py (new), tests/test_hazard.py (new), tests/test_hazard_client.py
(new), tests/test_hazard_batch.py (new)

(Separate repo, merged first: CryoHealth-api's allow-service-key.decorator.ts,
jwt-auth.guard.ts, jwt-auth.guard.spec.ts, validation.schema.ts, alerts.controller.ts,
record-hazard-score.dto.ts, .env.example — see that repo's own HANDOFF.md.)

## Verification status
tests: 56/56 passing  lint: clean  review: n/a (no new inbound auth surface; outbound
API key sent via env var only, gitignored .env, matches CryoHealth-api's
GEO_SERVICE_API_KEY)  qa: n/a
Live-verified: real GLO-30 DEM slope reads for all 6 lakes, real WorldPop population
buffer reads for all 6 lakes, real Observation-history-driven growth/anomaly
calculations, real POST to a real running CryoHealth-api instance, real tier
transitions, real Alert rows created and confirmed via psql — two real bugs found and
fixed in the process (see above).

## Resume with
/uexel:orient — task #11 ready for PR once reviewed.

## Addendum — 2026-08-03 (harness maintenance)
cryo-harness renamed to uxl-harness across the org (github.com/uExel/uxl-harness); this repo's .claude/settings.json marketplace pointer updated to match.
