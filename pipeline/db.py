"""Writes into CryoHealth-api's schema. This service owns no migrations — CryoHealth-api
does (ARCHITECTURE.md: "migrations are the only schema authority"); this file writes to
tables that repo defines, nothing more.

Connection config mirrors CryoHealth-api's own .env.example naming (DB_HOST/DB_PORT/...)
so the same local docker-compose Postgres works for both services without translation.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta

import psycopg

STALE_AFTER_DAYS = 60


def _dsn() -> str:
    host = os.environ.get("DB_HOST", "localhost")
    port = os.environ.get("DB_PORT", "5433")
    user = os.environ.get("DB_USER", "cryohealth")
    password = os.environ.get("DB_PASSWORD", "cryohealth-dev")
    name = os.environ.get("DB_NAME", "cryohealth")
    return f"host={host} port={port} user={user} password={password} dbname={name}"


@contextmanager
def connect():
    with psycopg.connect(_dsn()) as conn:
        yield conn


@dataclass(frozen=True)
class ObservationWrite:
    lake_id: str
    captured_at: date
    area_km2: float
    cloud_fraction: float
    scene_id: str
    run_id: str
    source: str = "sentinel2-cdse"


def upsert_observation(conn: psycopg.Connection, obs: ObservationWrite) -> bool:
    """Idempotent against idx_observation_dedupe (lakeId, capturedAt, source) —
    CryoHealth-api#10. DO NOTHING on conflict: a scene's computed area for a fixed
    date+source is deterministic, there's no reason a re-run should overwrite it.
    Returns True if a new row was actually inserted, False if it already existed.

    Targets the conflicting columns directly rather than `ON CONFLICT ON CONSTRAINT
    idx_observation_dedupe` — TypeORM's `@Index(..., { unique: true })` creates a plain
    unique index, not a table constraint, and Postgres only resolves ON CONFLICT ON
    CONSTRAINT against an actual constraint catalog entry (confirmed live: raised
    "constraint ... does not exist" against a real unique index of the same name).
    The column-list form works against any unique index regardless of that distinction."""
    # capturedAt is normalized to midnight UTC — see the comment on the Observation
    # entity in CryoHealth-api; that convention is what makes the unique index mean
    # "one observation per lake per day per source."
    captured_at_utc = datetime.combine(obs.captured_at, datetime.min.time(), tzinfo=UTC)
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO observations
                (id, "lakeId", "capturedAt", source, "areaKm2", "cloudFraction", "sceneId", "runId")
            VALUES (gen_random_uuid(), %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT ("lakeId", "capturedAt", source) DO NOTHING
            RETURNING id
            """,
            (
                obs.lake_id, captured_at_utc, obs.source, obs.area_km2,
                obs.cloud_fraction, obs.scene_id, obs.run_id,
            ),
        )
        inserted = cur.fetchone() is not None
    conn.commit()
    return inserted


def refresh_staleness(conn: psycopg.Connection, lake_id: str) -> bool:
    """>= STALE_AFTER_DAYS since the most recent observation -> stale=true; a fresher
    one clears it. Returns the new stale value. A lake with zero observations ever is
    stale by definition (never silently treated as fresh)."""
    with conn.cursor() as cur:
        cur.execute(
            """SELECT MAX("capturedAt") FROM observations WHERE "lakeId" = %s""",
            (lake_id,),
        )
        row = cur.fetchone()
        last_captured = row[0] if row else None

        if last_captured is None:
            is_stale = True
        else:
            age = datetime.now(UTC) - last_captured
            is_stale = age > timedelta(days=STALE_AFTER_DAYS)

        cur.execute(
            """UPDATE lakes SET stale = %s WHERE id = %s""",
            (is_stale, lake_id),
        )
    conn.commit()
    return is_stale


def lake_id_for_slug(conn: psycopg.Connection, slug: str) -> str | None:
    with conn.cursor() as cur:
        cur.execute("""SELECT id FROM lakes WHERE slug = %s""", (slug,))
        row = cur.fetchone()
        return row[0] if row else None
