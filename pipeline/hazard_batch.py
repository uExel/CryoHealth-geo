"""One pass computing + reporting hazard scores for every seeded lake — mirrors batch.py's
shape for Observations. Reads Observations + Lake static fields from Postgres directly
(read-only; this service still owns no migrations), computes slope from the DEM and
exposure from WorldPop, then reports the score to CryoHealth-api over HTTP (see
hazard_client.py's docstring for why this write path differs from Observations').
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from uuid import uuid4

import dataclasses
import os
import psycopg

from pipeline.anomaly import AnomalyUnavailableError, compute_anomaly
from pipeline.db import connect, lake_id_for_slug
from pipeline.dem import mean_slope_degrees
from pipeline.exposure import population_within_buffer
from pipeline.forecast import ForecastUnavailableError, compute_forecast
from pipeline.hazard import LakeStaticInputs, compute_hazard_score
from pipeline.hazard_client import report_hazard_score
from pipeline.lakes import LAKES
from pipeline.meteo import MeteoInputs, MeteoUnavailableError, fetch_meteo_inputs
from pipeline.stac_source import PlanetaryComputerSource

logger = logging.getLogger(__name__)

# Covers all signals used in hazard.py:
#   90-day growth window + 15-day tolerance  = ~105 days
#   Seasonal anomaly needs one prior calendar-year month = ~395 days
#   Safety margin -> 548 days total (≈18 months)
OBSERVATION_LOOKBACK_DAYS = 548

# Directory containing per-lake Isolation Forest .joblib + .json sidecar files.
# Set ANOMALY_MODEL_DIR env var to override (e.g. in docker-compose).
# None disables anomaly detection for the entire batch run with a WARNING.
ANOMALY_MODEL_DIR: str | None = os.environ.get("ANOMALY_MODEL_DIR")


@dataclass
class HazardRunResult:
    slug: str
    score: float | None = None
    tier: str | None = None
    alert_created: bool = False
    error: str | None = None
    computed_at: str = ""  # ISO 8601 UTC; non-empty only on success
    # Advisory signals (Issue #21, #22) — separate from the hazard score.
    # None when the relevant extra is not installed or model is unavailable.
    forecast: dict | None = None
    anomaly: dict | None = None


def _lake_static_inputs(conn: psycopg.Connection, lake_id: str) -> LakeStaticInputs | None:
    with conn.cursor() as cur:
        cur.execute(
            """SELECT "damType", "glacierContact", "historicalGlof" FROM lakes WHERE id = %s""",
            (lake_id,),
        )
        row = cur.fetchone()
    if row is None:
        return None
    return LakeStaticInputs(dam_type=row[0], glacier_contact=row[1], historical_glof=row[2])


def _observations(conn: psycopg.Connection, lake_id: str, as_of: date) -> list[tuple[date, float]]:
    # ORDER BY matters: hazard.py's tie-breaking (equidistant candidate dates on either
    # side of a target) depends on iteration order via min()/max() — an unordered query
    # made that non-deterministic and, confirmed live, could land on a noisy outlier
    # reading over its more representative same-distance neighbor.
    #
    # Cutoff is anchored to as_of (not date.today()) so historical/backfill runs
    # (e.g. run_hazard_pass(as_of=date(2024, 1, 1))) look back from the correct point.
    cutoff = as_of - timedelta(days=OBSERVATION_LOOKBACK_DAYS)
    with conn.cursor() as cur:
        cur.execute(
            """SELECT "capturedAt", "areaKm2"
               FROM observations
               WHERE "lakeId" = %s AND "capturedAt" >= %s
               ORDER BY "capturedAt" """,
            (lake_id, cutoff),
        )
        return [(row[0].date(), float(row[1])) for row in cur.fetchall()]


def _read_db_inputs(
    conn: psycopg.Connection,
    slugs: list[str],
    as_of: date,
) -> dict[str, tuple[str, LakeStaticInputs, list[tuple[date, float]]] | str]:
    """Phase 1: read every lake's DB data into memory and return it.

    Returns slug -> (lake_id, static, observations) on success,
    or slug -> error_message (str) when the lake is missing or invalid.

    Keeping all DB reads here means the connection is released before any
    slow HTTP work (DEM tile fetch, WorldPop raster download, API POST)
    begins in Phase 2, preventing idle-connection timeouts.
    """
    db_inputs: dict[str, tuple[str, LakeStaticInputs, list[tuple[date, float]]] | str] = {}
    for slug in slugs:
        lake_id = lake_id_for_slug(conn, slug)
        if lake_id is None:
            logger.warning("slug %r not found in DB — skipping", slug)
            db_inputs[slug] = "lake not seeded in DB"
            continue

        static = _lake_static_inputs(conn, lake_id)
        if static is None:
            logger.warning(
                "slug %r (lake_id=%s) has NULL static fields — skipping", slug, lake_id
            )
            db_inputs[slug] = (
                "lake row found but static fields (damType/glacierContact/historicalGlof) are NULL"
            )
            continue

        observations = _observations(conn, lake_id, as_of)
        db_inputs[slug] = (lake_id, static, observations)

    return db_inputs


def run_hazard_pass(lake_slugs: list[str] | None = None, as_of: date | None = None) -> list[HazardRunResult]:
    as_of = as_of or date.today()
    run_id = f"hazard-{uuid4()}"
    slugs = lake_slugs or list(LAKES)
    results: list[HazardRunResult] = []

    # Phase 1: DB reads — connection opened and closed here. -----------------
    # All lake data is loaded into memory before any network I/O begins so
    # slow HTTP calls (DEM, WorldPop, API POST) never hold a live Postgres
    # connection open. Idle-connection timeouts on the first WorldPop run
    # (~140 MB raster download) were the original motivation for this split.
    with connect() as conn:
        db_inputs = _read_db_inputs(conn, slugs, as_of)
    # DB connection is CLOSED here — everything below is pure compute + HTTP.

    # Phase 2: compute scores + report — no open DB connection. ---------------
    for slug, data in db_inputs.items():
        lake = LAKES[slug]
        try:
            if isinstance(data, str):
                results.append(HazardRunResult(slug, error=data))
                continue

            lake_id, static, observations = data
            slope = mean_slope_degrees(lake.bbox())
            exposure = population_within_buffer(lake.lon, lake.lat)

            meteo: MeteoInputs | None = None
            try:
                meteo = fetch_meteo_inputs(lake.lat, lake.lon, as_of)
            except MeteoUnavailableError:
                logger.warning(
                    "Weather API unavailable for %s — thermal component will be 0.0", slug
                )

            result = compute_hazard_score(
                observations, static, slope, as_of, exposure, meteo=meteo
            )

            # --- Advisory signals (Issues #21, #22): forecast + anomaly ----------
            # Neither modifies compute_hazard_score() — both are separate signals.
            # Neither triggers alerts directly (ADR 0004, ADR 0005).
            forecast_dict: dict | None = None
            try:
                forecast_result = compute_forecast(observations, as_of=as_of)
                forecast_dict = dataclasses.asdict(forecast_result)
            except ForecastUnavailableError as exc:
                logger.warning("Forecast unavailable for %s: %s", slug, exc)

            anomaly_dict: dict | None = None
            if ANOMALY_MODEL_DIR is not None:
                try:
                    source = PlanetaryComputerSource()
                    anomaly_result = compute_anomaly(
                        source=source,
                        dam_face_bbox=lake.dam_face_bbox(),
                        dam_type=static.dam_type,
                        model_dir=ANOMALY_MODEL_DIR,
                        slug=slug,
                        as_of=as_of,
                    )
                    anomaly_dict = dataclasses.asdict(anomaly_result)
                except AnomalyUnavailableError as exc:
                    logger.warning("Anomaly detection unavailable for %s: %s", slug, exc)
            else:
                logger.debug(
                    "ANOMALY_MODEL_DIR not set — anomaly detection skipped for %s. "
                    "Set env var ANOMALY_MODEL_DIR after running fit_anomaly_model.py.",
                    slug,
                )

            # Merge advisory signals into components alongside the hazard score.
            # components["forecast"] and components["anomaly"] are explicitly labeled;
            # they are never blended into the composite score (ADR 0004, ADR 0005).
            components_with_advisories = {
                **result.components,
                "forecast": forecast_dict,
                "anomaly": anomaly_dict,
            }

            response = report_hazard_score(
                str(lake_id), run_id, result.score, result.tier, components_with_advisories
            )
            results.append(
                HazardRunResult(
                    slug,
                    score=result.score,
                    tier=result.tier,
                    alert_created=response.get("alert") is not None,
                    computed_at=datetime.now(UTC).isoformat(),
                    forecast=forecast_dict,
                    anomaly=anomaly_dict,
                )
            )
        except Exception as exc:  # noqa: BLE001 — one lake's failure must not sink the pass
            logger.exception("run_hazard_pass failed for %s", slug)
            results.append(HazardRunResult(slug, error=str(exc)))

    return results


__all__ = ["HazardRunResult", "run_hazard_pass"]

