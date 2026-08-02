# HANDOFF — CryoHealth-geo — 2026-08-02 20:10 PKT
Session: scheduled-service  Model: sonnet-5  Branch: feat/8-scheduled-service  Goal: #1  Task: #8

## State
Scheduled service built end-to-end and live-verified against the real local Postgres
and real CDSE. All 5 plan steps done: db.py, find_scenes_in_range on both sources,
batch.py (run_latest/run_backfill sharing per-scene logic), service.py (FastAPI +
in-process APScheduler), backfill.py (CLI). Writes Observation rows + Lake.stale only —
no hazard index, no alerts-engine call, as scoped.

Real bug found and fixed during live verification: `ON CONFLICT ON CONSTRAINT
idx_observation_dedupe` failed with "constraint ... does not exist" — TypeORM's
`@Index(..., { unique: true })` creates a plain unique index, not a table constraint,
and Postgres's ON CONFLICT ON CONSTRAINT form only resolves against real constraints.
Fixed by targeting the column list directly (`ON CONFLICT ("lakeId", "capturedAt",
source) DO NOTHING`), which works against any unique index. Logged in LEARNINGS.md.

Small bounded live verification (shishper, last 30 days, real CDSE): first run wrote
4 new rows with real distinct areaKm2/cloudFraction values, correct source tag
(sentinel2-cdse), stale=false; immediate re-run wrote 0 new rows (idempotent);
confirmed directly in Postgres via psql, not just via the CLI's own reported counts.

service.py verified separately: starts clean, APScheduler registers the daily job on
startup, GET /health reports scheduler_running=true.

Per explicit instruction this session, the full 2023-to-present backfill across all six
lakes was then started (`pipeline/backfill.py --start 2023-01-01`, all lakes, cdse
source) — this is the action docs/ai/PLAN.md flagged as a deliberate, separately-
confirmed step given real CDSE quota/wall-clock cost. Launched as a tracked background
job; not yet complete as of this handoff. Check job output before treating the full
historical dataset as populated — this handoff covers the code and small-scale
verification, not the full run's outcome.

## Done this session
- pipeline/db.py (new): psycopg writer — upsert_observation, refresh_staleness,
  lake_id_for_slug
- find_scenes_in_range added to both PlanetaryComputerSource and CdseSource (shared
  _search helper, range vs latest-N modes)
- pipeline/batch.py (new): run_latest (daily, newest usable scene, stop on first hit)
  and run_backfill (full time series in a window), per-lake exception isolation so one
  lake's failure doesn't sink the batch
- pipeline/service.py (new): FastAPI app, GET /health, POST /run, APScheduler daily job
- pipeline/backfill.py (new): CLI entrypoint for range backfills
- tests/test_db.py, tests/test_batch.py (new): mocked, no live DB/network — 30/30
  tests passing, ruff clean
- Live verification against real Postgres + real CDSE (see State above)
- Full 2023-to-now backfill started (background, all 6 lakes) — outcome pending

## Not done / deferred
- Hazard index computation, alerts-engine integration — explicitly out of scope (R3,
  separate future task)
- Auth on POST /run — documented gap, meant for a private network not public exposure
- Production deployment (persistent process/VPS) — this session verified the service
  starts and schedules correctly locally; actually running it 24/7 is infra work per
  the PRD's six-week plan, not part of this task

## Next action
Confirm the background 2023 backfill completed cleanly (check job output / query
Postgres for per-lake row counts and any lakes with errors), then open PR for
feat/8-scheduled-service -> main.

## Open questions for a human
- none blocking

## Failed approaches (do not retry)
- `ON CONFLICT ON CONSTRAINT idx_observation_dedupe` — fails against a TypeORM plain
  unique index. Use the column-list ON CONFLICT form instead (see LEARNINGS.md).

## Loops run
- lint fix loop: 1 pass (line length x8, UP017 x3, unused import x1), all auto-fixed
  or manually wrapped, passed clean

## Files touched
pipeline/db.py (new), pipeline/batch.py (new), pipeline/service.py (new),
pipeline/backfill.py (new), pipeline/cdse_source.py (find_scenes_in_range,
line-length wrap), pipeline/stac_source.py (find_scenes_in_range, line-length wrap),
tests/test_db.py (new), tests/test_batch.py (new), docs/ai/LEARNINGS.md,
docs/ai/HANDOFF.md

## Verification status
tests: 30/30 passing  lint: clean  review: n/a (no new auth/PII surface; DB creds via
env only, matches CryoHealth-api's existing docker-compose Postgres, gitignored .env
throughout)  qa: n/a
Live-verified: real Postgres writes (psql-confirmed), idempotent re-run, staleness
computation, real FastAPI service startup + scheduler registration, real CDSE Process
API calls end to end.

## Resume with
Check background backfill job output, then open PR for feat/8-scheduled-service.
