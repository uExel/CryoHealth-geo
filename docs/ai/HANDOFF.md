# HANDOFF — CryoHealth-geo — 2026-08-09 01:22 PKT

Session: hetzner-tunnel-deploy Model: claude-sonnet-5 Branch: main Goal: none Task: none (ad hoc, cross-repo deploy infra)

## State

`geo` is deployed and running on the Hetzner box (`ubuntu-4gb-hel1-1`, 204.168.190.206),
container stable (no crash loop). CD pipeline (`deploy.yml`, workflow_run on green `ci`)
confirmed working end-to-end (`deploy #3`). GHCR pull auth fixed (user supplied a
`read:packages` PAT, `docker login`'d as `deploy` on the server). One real bug caught and
fixed on first actual container run — see Done this session.
**`CDSE_CLIENT_ID`/`CDSE_CLIENT_SECRET` on the server are still `REPLACE_ME`
placeholders** — the daily job will run (container is up, no auth is on `/run`/`/run-hazard`
so it'll boot fine) but will fall back to Planetary Computer rather than the real
Copernicus production source until real credentials are provided. No public/tunnel route
exists for this service — it has no inbound surface, only outbound calls.

## Done this session

- Dockerfile (new): python:3.12-slim + uv, rasterio wheels bundle GDAL so no system
  packages needed for GDAL itself — but the slim base is missing `libexpat1`, which
  rasterio's wheel dynamically links regardless. **First real container run
  crash-looped** (`ImportError: libexpat.so.1: cannot open shared object file`) — never
  caught before because this had never actually run in a container. Fixed by installing
  `libexpat1` via apt in the Dockerfile; verified locally (image builds, `import
rasterio` succeeds) before pushing (af566f8)
- .dockerignore (new)
- deploy.yml: build+push to GHCR, SSH deploy, gated on ci
- GitHub Actions secrets set: DEPLOY_HOST, DEPLOY_USER, DEPLOY_SSH_KEY
- Server: GHCR login configured, `geo` container stable and running

## Not done / deferred

- Real Copernicus CDSE credentials not yet provided — service falls back to Planetary
  Computer (non-production source per service.py's own documented behavior) until they
  are. See docker-compose.prod.yml / cryohealth-infra/README.md for where to put them.
- No live verification yet that the daily observation+hazard job actually completes a
  full pass against production data (container is up and would run on its own schedule,
  but no one has watched a full cycle complete since this deploy)

## Next action

Once real CDSE credentials are available: put them in `/opt/cryohealth/.env` on the
server (see cryohealth-infra/README.md), then `docker compose -f
docker-compose.prod.yml up -d geo` to restart with the real source. Otherwise: watch the
next scheduled daily run (or `POST /run` manually against the container, private network
only) to confirm output end to end against production infra for the first time.

## Open questions for a human

- Real Copernicus CDSE credentials for production Sentinel-2 — blocking: no for now (geo
  runs fine on the fallback source), yes for genuinely production-quality hazard data

## Failed approaches (do not retry)

- GitHub Actions "Re-run failed jobs" from the `...`/`Re-run jobs` menu without confirming
  the modal that appears does nothing silently — always screenshot after clicking to
  confirm the "Re-run jobs" dialog actually appeared and was confirmed
- Making the `cryohealth-api`/`cryohealth-geo` GHCR packages public instead of using a PAT
  — blocked by uExel org policy ("Setting is disabled by organization administrators" on
  the package visibility toggle)
- `python:3.12-slim` + rasterio's manylinux wheel without any system packages — the wheel
  bundles GDAL but still needs `libexpat1` from the OS; always test-boot a freshly built
  image locally (`docker run --rm --entrypoint sh <image> -c 'python -c "import
rasterio"'`) before assuming a slim base is sufficient

## Loops run

- none (ad hoc, no /uexel:plan → /uexel:build loop)

## Files touched

Dockerfile (new), .dockerignore (new), .github/workflows/deploy.yml (new)

## Verification status

tests: unaffected (no code changes, only packaging/CD) deploy: **live** — container
running stably on the server, no crash loop; production Sentinel-2 source not yet
confirmed live end-to-end (see Not done)

## Resume with

/uexel:orient
