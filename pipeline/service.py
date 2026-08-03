"""uvicorn pipeline.service:app

GET  /health        — liveness
POST /run            — trigger one observation batch pass now (same logic the daily
                        scheduler runs)
POST /run-hazard     — trigger one hazard-scoring pass now, independent of /run

Known gap, stated plainly: neither POST route has auth. This is meant to run on a
private network (a VPS, per the PRD's six-week plan), not exposed publicly — real
service-to-service auth for *inbound* requests to this service is a documented gap.
(Outbound calls this service makes — CryoHealth-api's /alerts/hazard-scores — do have
service-to-service auth: CryoHealth-api#12.)

The scheduler runs in-process (APScheduler) rather than relying on an external cron
hitting this service, so "the service is up" and "the schedule runs" are the same
guarantee — no separate scheduler infra to keep in sync with this one. The daily job
runs the observation batch and the hazard pass back to back, in that order, so hazard
scoring always sees that run's freshest observation (if any landed).
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from dataclasses import asdict

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from fastapi import FastAPI

from pipeline.batch import run_latest
from pipeline.hazard_batch import run_hazard_pass
from pipeline.stac_source import PlanetaryComputerSource

logger = logging.getLogger(__name__)

RUN_INTERVAL_HOURS = 24


def _make_source():
    if os.environ.get("CDSE_CLIENT_ID"):
        from pipeline.cdse_source import CdseSource

        return CdseSource()
    logger.warning("CDSE_CLIENT_ID not set — falling back to Planetary Computer (not the production source)")
    return PlanetaryComputerSource()


def _run_observations() -> None:
    source = _make_source()
    results = run_latest(source)
    for r in results:
        logger.info(
            "batch: %s written=%d checked=%d stale=%s error=%s",
            r.slug, r.observations_written, r.scenes_checked, r.stale, r.error,
        )


def _run_hazard() -> None:
    results = run_hazard_pass()
    for r in results:
        logger.info(
            "hazard: %s score=%s tier=%s alert_created=%s error=%s",
            r.slug, r.score, r.tier, r.alert_created, r.error,
        )


def _daily_job() -> None:
    _run_observations()
    _run_hazard()


scheduler = AsyncIOScheduler()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    scheduler.add_job(
        _daily_job, IntervalTrigger(hours=RUN_INTERVAL_HOURS), id="daily-job", replace_existing=True
    )
    scheduler.start()
    yield
    scheduler.shutdown(wait=False)


app = FastAPI(title="CryoHealth-geo", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "scheduler_running": scheduler.running}


@app.post("/run")
def run_now() -> dict:
    source = _make_source()
    results = run_latest(source)
    return {"results": [asdict(r) for r in results]}


@app.post("/run-hazard")
def run_hazard_now() -> dict:
    results = run_hazard_pass()
    return {"results": [asdict(r) for r in results]}
