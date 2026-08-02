# PLAN — CryoHealth-geo#8: scheduled service
Goal: #1 (G1 milestone) · Task: #8 · Loop budget: 3 · Rollback: revert PR, no schema owned here
Depends on: CryoHealth-api#10 (Observation idempotency constraint) — built first, this branch

Explicitly out of scope (stated up front): hazard index computation (§8), calling
/alerts/hazard-scores. This writes Observation rows + Lake.stale flags only — R2, not R3.

Steps:
1. pipeline/db.py: psycopg-based writer. upsert_observation (ON CONFLICT on the new
   unique index — DO NOTHING, since a scene's computed area for a fixed date/source is
   deterministic, no reason to overwrite), set_lake_stale (based on most recent
   Observation.capturedAt vs now, >60 days -> stale=true)
   — verify: uv run pytest (mocked) + live insert against real local Postgres
2. stac_source.py / cdse_source.py: find_scenes_in_range(bbox, start, end) — ALL usable
   candidates in a window (backfill needs every observation, not just "stop at first
   usable" like find_recent_scenes)
3. pipeline/batch.py: one pass over every seeded lake — latest-scene mode (daily
   scheduled use) and range mode (backfill use), same underlying per-scene logic
4. pipeline/service.py: FastAPI app, GET /health, POST /run (manual trigger),
   APScheduler running batch.run_latest() daily
5. pipeline/backfill.py: CLI entrypoint for the range mode

Verification: live-verify with a SMALL bounded run (few lakes, ~last 30 days) proving
real DB writes, idempotent re-run (0 new rows second time), staleness flag computed
correctly. The FULL 2023-to-now backfill across all lakes is NOT run as part of this
task without asking first — real CDSE credit/time cost, a separate deliberate action.
