"""db.py tests run against a mocked psycopg connection — no real Postgres. The real
schema/constraint behavior (idx_observation_dedupe, ON CONFLICT DO NOTHING) was
verified live against a running Postgres before this file existed (attempted a real
duplicate INSERT, confirmed rejection) — see CryoHealth-api#10. These tests guard the
query shape and Python-side logic going forward."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from unittest.mock import MagicMock

from pipeline.db import ObservationWrite, lake_id_for_slug, refresh_staleness, upsert_observation


def _mock_conn(fetchone_return=None):
    conn = MagicMock()
    cursor = MagicMock()
    cursor.fetchone.return_value = fetchone_return
    conn.cursor.return_value.__enter__.return_value = cursor
    return conn, cursor


def test_upsert_observation_returns_true_when_a_new_row_is_inserted():
    conn, cursor = _mock_conn(fetchone_return=("some-uuid",))
    obs = ObservationWrite(
        lake_id="lake-1",
        captured_at=date(2026, 7, 20),
        area_km2=1.23,
        cloud_fraction=0.1,
        scene_id="S2A_TEST",
        run_id="run-1",
    )

    inserted = upsert_observation(conn, obs)

    assert inserted is True
    conn.commit.assert_called_once()
    args = cursor.execute.call_args[0][1]
    # captured_at is normalized to midnight UTC before hitting the query.
    assert args[1] == datetime(2026, 7, 20, tzinfo=UTC)


def test_upsert_observation_returns_false_on_conflict_do_nothing():
    conn, _cursor = _mock_conn(fetchone_return=None)
    obs = ObservationWrite(
        lake_id="lake-1",
        captured_at=date(2026, 7, 20),
        area_km2=1.23,
        cloud_fraction=0.1,
        scene_id="S2A_TEST",
        run_id="run-1",
    )

    inserted = upsert_observation(conn, obs)

    assert inserted is False
    conn.commit.assert_called_once()


def test_refresh_staleness_is_stale_when_lake_has_no_observations():
    conn, _cursor = _mock_conn(fetchone_return=(None,))

    stale = refresh_staleness(conn, "lake-1")

    assert stale is True
    conn.commit.assert_called_once()


def test_refresh_staleness_is_fresh_within_the_window():
    recent = datetime.now(UTC) - timedelta(days=1)
    conn, _cursor = _mock_conn(fetchone_return=(recent,))

    stale = refresh_staleness(conn, "lake-1")

    assert stale is False


def test_refresh_staleness_is_stale_past_the_window():
    old = datetime.now(UTC) - timedelta(days=61)
    conn, _cursor = _mock_conn(fetchone_return=(old,))

    stale = refresh_staleness(conn, "lake-1")

    assert stale is True


def test_lake_id_for_slug_returns_none_when_not_found():
    conn, _cursor = _mock_conn(fetchone_return=None)

    result = lake_id_for_slug(conn, "unknown-slug")

    assert result is None


def test_lake_id_for_slug_returns_the_id():
    conn, _cursor = _mock_conn(fetchone_return=("lake-uuid",))

    result = lake_id_for_slug(conn, "shishper")

    assert result == "lake-uuid"
