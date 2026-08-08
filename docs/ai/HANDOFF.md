# HANDOFF — CryoHealth-geo — 2026-08-09 00:48 PKT
Session: hetzner-tunnel-deploy  Model: claude-sonnet-5  Branch: main  Goal: none  Task: none (ad hoc, cross-repo deploy infra)

## State
Docker image + CD pipeline built (`Dockerfile`, `.dockerignore`, `deploy.yml` gated on
green `ci`). GitHub Actions secrets set (`DEPLOY_HOST`/`DEPLOY_USER`/`DEPLOY_SSH_KEY`).
Latest `deploy` run built and pushed the image to GHCR successfully but failed at the SSH
step: the Hetzner box (`ubuntu-4gb-hel1-1`, 204.168.190.206) can't pull the private GHCR
image (`docker pull` → `unauthorized`) — no `docker login ghcr.io` credential on the box
yet. Asked the user for a GitHub PAT (`read:packages`) to fix this; awaiting reply.
Separately: the `geo` container itself has never been started on the server — its
`CDSE_CLIENT_ID`/`CDSE_CLIENT_SECRET` in the server's `.env` are still `REPLACE_ME`
placeholders (real Copernicus credentials not yet available), and the docker-compose
`:?required` guard on those vars means `geo` won't even parse without *some* non-empty
value there.

## Done this session
- Dockerfile (new): python:3.12-slim + uv, rasterio wheels bundle GDAL so no system
  packages needed (matches ci.yml's plain `uv sync`)
- .dockerignore (new)
- deploy.yml: build+push to GHCR, SSH deploy, gated on ci
- GitHub Actions secrets set: DEPLOY_HOST, DEPLOY_USER, DEPLOY_SSH_KEY

## Not done / deferred
- `geo` has never actually run on the Hetzner box — blocked on both the GHCR pull PAT and
  real CDSE credentials (see Open questions)
- No live verification of the daily observation+hazard job running from the deployed
  container (only ever run locally/dev, per the prior hazard-index session's handoff)

## Next action
Once the user supplies a GHCR `read:packages` PAT and real CDSE credentials: `docker
login ghcr.io` on the box, put the real `CDSE_CLIENT_ID`/`CDSE_CLIENT_SECRET` in
`/opt/cryohealth/.env` (see cryohealth-infra/README.md), then start `geo` for the first
time: `docker compose -f docker-compose.prod.yml up -d geo`.

## Open questions for a human
- GHCR pull PAT for the deploy box — blocking: yes (blocks the image pull)
- Real Copernicus CDSE credentials for production Sentinel-2 — blocking: yes (geo won't
  fetch real imagery without them; falls back to Planetary Computer per service.py's
  documented behavior, which the team already decided isn't acceptable for production)

## Failed approaches (do not retry)
- GitHub Actions "Re-run failed jobs" from the `...`/`Re-run jobs` menu without confirming
  the modal that appears does nothing silently — always screenshot after clicking to
  confirm the "Re-run jobs" dialog actually appeared and was confirmed
- Making the `cryohealth-api`/`cryohealth-geo` GHCR packages public instead of using a PAT
  — blocked by uExel org policy ("Setting is disabled by organization administrators" on
  the package visibility toggle)

## Loops run
- none (ad hoc, no /uexel:plan → /uexel:build loop)

## Files touched
Dockerfile (new), .dockerignore (new), .github/workflows/deploy.yml (new)

## Verification status
tests: unaffected (no code changes, only packaging/CD)  deploy: not yet live on the
server (blocked on GHCR PAT + CDSE credentials)

## Resume with
/uexel:orient   (then: check with user whether the GHCR PAT and CDSE credentials were
provided, docker login on the server, start the geo container for the first time)
