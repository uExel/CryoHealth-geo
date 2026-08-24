"""One pass computing + reporting hazard scores for every seeded lake — mirrors batch.py's
shape for Observations. Reads Observations + Lake static fields from Postgres directly
(read-only; this service still owns no migrations), computes slope from the DEM and
exposure from WorldPop, then reports the score to CryoHealth-api over HTTP (see
hazard_client.py's docstring for why this write path differs from Observations').
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from uuid import uuid4

import psycopg

from pipeline.db import connect, lake_id_for_slug
from pipeline.dem import mean_slope_degrees
from pipeline.exposure import population_within_buffer
from pipeline.hazard import LakeStaticInputs, compute_hazard_score
from pipeline.hazard_client import report_hazard_score
from pipeline.lakes import LAKES

logger = logging.getLogger(__name__)


@dataclass
class HazardRunResult:
    slug: str
    score: float | None = None
    tier: str | None = None
    alert_created: bool = False
    error: str | None = None


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


def _observations(conn: psycopg.Connection, lake_id: str) -> list[tuple[date, float]]:
    # ORDER BY matters: hazard.py's tie-breaking (equidistant candidate dates on either
    # side of a target) depends on iteration order via min()/max() — an unordered query
    # made that non-deterministic and, confirmed live, could land on a noisy outlier
    # reading over its more representative same-distance neighbor.
    with conn.cursor() as cur:
        cur.execute(
            """SELECT "capturedAt", "areaKm2" FROM observations WHERE "lakeId" = %s ORDER BY "capturedAt" """,
            (lake_id,),
        )
        return [(row[0].date(), float(row[1])) for row in cur.fetchall()]


def run_hazard_pass(lake_slugs: list[str] | None = None, as_of: date | None = None) -> list[HazardRunResult]:
    as_of = as_of or date.today()
    run_id = f"hazard-{uuid4()}"
    results: list[HazardRunResult] = []

    with connect() as conn:
        for slug in lake_slugs or list(LAKES):
            lake = LAKES[slug]
            try:
                lake_id = lake_id_for_slug(conn, slug)
                if lake_id is None:
                    results.append(HazardRunResult(slug, error="lake not seeded in DB"))
                    continue

                static = _lake_static_inputs(conn, lake_id)
                if static is None:
                    results.append(HazardRunResult(slug,
                        error="lake row found but static fields (damType/glacierContact/historicalGlof) are NULL"))
                    continue

                observations = _observations(conn, lake_id)
                slope = mean_slope_degrees(lake.bbox())
                exposure = population_within_buffer(lake.lon, lake.lat)

                result = compute_hazard_score(observations, static, slope, as_of, exposure)
                response = report_hazard_score(
                    str(lake_id), run_id, result.score, result.tier, result.components
                )
                results.append(
                    HazardRunResult(
                        slug,
                        score=result.score,
                        tier=result.tier,
                        alert_created=response.get("alert") is not None,
                    )
                )
            except Exception as exc:  # noqa: BLE001 — one lake's failure must not sink the pass
                logger.exception("run_hazard_pass failed for %s", slug)
                results.append(HazardRunResult(slug, error=str(exc)))

    return results


__all__ = ["HazardRunResult", "run_hazard_pass"]
