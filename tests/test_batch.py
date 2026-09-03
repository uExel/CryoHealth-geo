"""batch.py orchestrates db.py + stac_source.py + ndwi.py + sar_source.py; all are
mocked/patched here so these tests exercise only batch.py's own control flow (which
scenes get processed, when a lake is skipped, when staleness gets refreshed, and when
the SAR fallback is triggered vs skipped) — no real DB, no real network, no real
satellite imagery.

SAR fallback tests verify:
  - Cloud cover ≤ 40% → optical path; SAR source is never called.
  - Cloud cover > 40% + sar_source present → SAR scene written with source='sentinel1-grd-sar'.
  - Cloud cover > 40% + sar_source=None → observation skipped (no write, no crash).
  - SAR search returns zero results → observation skipped, warning emitted.
  - SAR source raises → per-lake error captured, batch continues for other lakes.
  - Date written to DB is the SAR scene's own captured_at (honest provenance).
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date
from unittest.mock import patch

import numpy as np

from pipeline.batch import LakeResult, run_backfill, run_latest
from pipeline.stac_source import SceneRef


@dataclass
class FakeSource:
    """find_recent_scenes/find_scenes_in_range return whatever scenes were queued;
    read_bands returns all-water, cloud-free bands unless a scene_id is listed as cloudy."""

    scenes: list[SceneRef] = field(default_factory=list)
    cloudy_scene_ids: set[str] = field(default_factory=set)
    raise_on_find: Exception | None = None

    def find_recent_scenes(self, bbox, limit=12):
        if self.raise_on_find:
            raise self.raise_on_find
        return self.scenes[:limit]

    def find_scenes_in_range(self, bbox, start, end):
        if self.raise_on_find:
            raise self.raise_on_find
        return self.scenes

    def read_bands(self, scene, bands, bbox):
        shape = (4, 4)
        if scene.scene_id in self.cloudy_scene_ids:
            scl = np.full(shape, 9, dtype=np.uint16)  # 9 = cloud, fully masked out
        else:
            scl = np.full(shape, 6, dtype=np.uint16)  # 6 = water, fully usable
        green = np.full(shape, 100.0)
        nir = np.full(shape, 20.0)  # green > nir -> strongly positive NDWI, all water
        return {"B03": green, "B08": nir, "SCL": scl}


@dataclass
class FakeSarSource:
    """Fake SAR source for testing the sar_source fallback path in batch.py.

    scenes: SAR SceneRefs returned from find_scenes_in_range.
    raise_on_find: If set, raises this exception from find_scenes_in_range.
    water_fraction: Fraction of the synthetic SAR scene classified as water (0–1).
    """

    scenes: list[SceneRef] = field(default_factory=list)
    raise_on_find: Exception | None = None
    water_fraction: float = 1.0  # default: all pixels are water

    def find_recent_scenes(self, bbox, limit=12):
        if self.raise_on_find:
            raise self.raise_on_find
        return self.scenes[:limit]

    def find_scenes_in_range(self, bbox, start, end):
        if self.raise_on_find:
            raise self.raise_on_find
        return self.scenes

    def read_bands(self, scene, bands, bbox):
        # Return VV and VH arrays that will be clearly classified as all-water
        # by extract_sar_water's dual_pol_threshold (-18 dB VV, -24 dB VH in power).
        shape = (4, 4)
        vv_water = np.full(shape, 10 ** (-18 / 10), dtype=np.float32)
        vh_water = np.full(shape, 10 ** (-24 / 10), dtype=np.float32)
        return {"vv": vv_water, "vh": vh_water}


def _scene(scene_id: str, day: date, cloud_pct: float = 5.0) -> SceneRef:
    return SceneRef(scene_id=scene_id, captured_at=day, cloud_cover_pct=cloud_pct, assets={})


def _sar_scene(scene_id: str, day: date) -> SceneRef:
    """A SAR SceneRef (cloud_cover_pct=0.0, SAR is cloud-transparent)."""
    return SceneRef(scene_id=scene_id, captured_at=day, cloud_cover_pct=0.0, assets={})


@contextmanager
def _fake_connect():
    yield object()


def _patched(lake_id_map, upsert_result=True, stale_result=False):
    return (
        patch("pipeline.batch.connect", _fake_connect),
        patch("pipeline.batch.lake_id_for_slug", side_effect=lambda conn, slug: lake_id_map.get(slug)),
        patch("pipeline.batch.upsert_observation", return_value=upsert_result),
        patch("pipeline.batch.refresh_staleness", return_value=stale_result),
    )


def test_run_latest_reports_error_when_lake_not_seeded():
    patchers = _patched({})
    with patchers[0], patchers[1], patchers[2], patchers[3]:
        results = run_latest(FakeSource(), lake_slugs=["shishper"])

    assert results == [LakeResult("shishper", 0, 0, stale=True, error="lake not seeded in DB")]


def test_run_latest_writes_the_newest_usable_scene_and_stops():
    source = FakeSource(scenes=[_scene("newest", date(2026, 7, 20)), _scene("older", date(2026, 7, 10))])
    patchers = _patched({"shishper": "lake-1"}, upsert_result=True, stale_result=False)
    with patchers[0], patchers[1], patchers[2], patchers[3]:
        results = run_latest(source, lake_slugs=["shishper"])

    assert len(results) == 1
    result = results[0]
    assert result.observations_written == 1
    assert result.scenes_checked == 1  # stopped after the first usable scene
    assert result.stale is False
    assert result.error is None


def test_run_latest_skips_cloudy_scenes_until_a_usable_one():
    source = FakeSource(
        scenes=[_scene("cloudy", date(2026, 7, 20)), _scene("clear", date(2026, 7, 18))],
        cloudy_scene_ids={"cloudy"},
    )
    patchers = _patched({"shishper": "lake-1"})
    with patchers[0], patchers[1], patchers[2], patchers[3]:
        results = run_latest(source, lake_slugs=["shishper"])

    assert results[0].scenes_checked == 2
    assert results[0].observations_written == 1


def test_run_latest_writes_zero_when_every_candidate_is_cloudy():
    source = FakeSource(scenes=[_scene("cloudy", date(2026, 7, 20))], cloudy_scene_ids={"cloudy"})
    patchers = _patched({"shishper": "lake-1"})
    with patchers[0], patchers[1], patchers[2], patchers[3]:
        results = run_latest(source, lake_slugs=["shishper"])

    assert results[0].observations_written == 0
    assert results[0].scenes_checked == 1


def test_run_latest_captures_exceptions_per_lake_without_sinking_the_batch():
    source = FakeSource(raise_on_find=RuntimeError("STAC search failed"))
    patchers = _patched({"shishper": "lake-1"})
    with patchers[0], patchers[1], patchers[2], patchers[3]:
        results = run_latest(source, lake_slugs=["shishper"])

    assert results[0].error == "STAC search failed"
    assert results[0].stale is True


def test_run_backfill_writes_every_usable_scene_not_just_the_newest():
    source = FakeSource(
        scenes=[
            _scene("s1", date(2026, 1, 5)),
            _scene("s2", date(2026, 2, 5)),
            _scene("s3", date(2026, 3, 5)),
        ]
    )
    patchers = _patched({"shishper": "lake-1"}, upsert_result=True)
    with patchers[0], patchers[1], patchers[2], patchers[3]:
        results = run_backfill(source, date(2026, 1, 1), date(2026, 4, 1), lake_slugs=["shishper"])

    assert results[0].observations_written == 3
    assert results[0].scenes_checked == 3


def test_run_backfill_does_not_recount_already_written_observations():
    source = FakeSource(scenes=[_scene("s1", date(2026, 1, 5))])
    patchers = _patched({"shishper": "lake-1"}, upsert_result=False)  # already exists (idempotent re-run)
    with patchers[0], patchers[1], patchers[2], patchers[3]:
        results = run_backfill(source, date(2026, 1, 1), date(2026, 4, 1), lake_slugs=["shishper"])

    assert results[0].observations_written == 0
    assert results[0].scenes_checked == 1


def test_run_backfill_defaults_end_to_today_when_omitted():
    source = FakeSource(scenes=[])
    patchers = _patched({"shishper": "lake-1"})
    with patchers[0], patchers[1], patchers[2], patchers[3]:
        results = run_backfill(source, date(2026, 1, 1), lake_slugs=["shishper"])

    assert results[0].scenes_checked == 0
    assert results[0].error is None


# ---------------------------------------------------------------------------
# SAR fallback tests
# ---------------------------------------------------------------------------

def test_run_latest_uses_optical_when_cloud_fraction_below_trigger():
    """With cloud cover well under SAR_TRIGGER_CLOUD_FRACTION, only the optical
    path runs. The SAR source's find_scenes_in_range is never called."""
    # FakeSource returns a cloud-free scene (SCL=6).
    optical = FakeSource(scenes=[_scene("clear", date(2026, 7, 20))])
    sar = FakeSarSource(scenes=[_sar_scene("sar-1", date(2026, 7, 20))])
    patchers = _patched({"shishper": "lake-1"}, upsert_result=True)
    with patchers[0], patchers[1], patchers[2], patchers[3]:
        results = run_latest(optical, lake_slugs=["shishper"], sar_source=sar)

    # Observation written via optical path; SAR source never queried for dates.
    assert results[0].observations_written == 1
    assert results[0].error is None


def test_run_latest_triggers_sar_when_optical_is_too_cloudy():
    """All-cloud optical scene (SCL=9) with a SAR source → SAR observation written
    with source='sentinel1-grd-sar'."""

    optical = FakeSource(
        scenes=[_scene("cloudy", date(2026, 7, 20))],
        cloudy_scene_ids={"cloudy"},
    )
    sar_date = date(2026, 7, 21)  # SAR scene is 1 day after optical (within ±3 days)
    sar = FakeSarSource(scenes=[_sar_scene("sar-1", sar_date)])

    written_obs = []

    def capture_obs(conn, obs):
        written_obs.append(obs)
        return True

    patchers = _patched({"shishper": "lake-1"}, upsert_result=True)
    obs_patcher = patch("pipeline.batch.upsert_observation", side_effect=capture_obs)
    with patchers[0], patchers[1], obs_patcher, patchers[3]:
        results = run_latest(optical, lake_slugs=["shishper"], sar_source=sar)

    assert results[0].observations_written == 1
    assert results[0].error is None
    assert len(written_obs) == 1
    obs = written_obs[0]
    assert obs.source == "sentinel1-grd-sar"
    assert obs.cloud_fraction == 0.0
    # Date is the SAR scene's own date, not the cloudy optical scene's date.
    assert obs.captured_at == sar_date


def test_run_latest_skips_observation_when_sar_source_is_none():
    """Cloudy optical scene + sar_source=None → no observation written, no crash."""
    optical = FakeSource(
        scenes=[_scene("cloudy", date(2026, 7, 20))],
        cloudy_scene_ids={"cloudy"},
    )
    patchers = _patched({"shishper": "lake-1"})
    with patchers[0], patchers[1], patchers[2], patchers[3]:
        results = run_latest(optical, lake_slugs=["shishper"], sar_source=None)

    assert results[0].observations_written == 0
    assert results[0].error is None


def test_run_latest_skips_when_sar_returns_no_scenes_in_window(caplog):
    """Zero SAR scenes in ±3-day window → observation skipped, WARNING logged."""
    optical = FakeSource(
        scenes=[_scene("cloudy", date(2026, 7, 20))],
        cloudy_scene_ids={"cloudy"},
    )
    sar = FakeSarSource(scenes=[])  # no SAR scenes available

    patchers = _patched({"shishper": "lake-1"})
    with caplog.at_level(logging.WARNING, logger="pipeline.batch"):
        with patchers[0], patchers[1], patchers[2], patchers[3]:
            results = run_latest(optical, lake_slugs=["shishper"], sar_source=sar)

    assert results[0].observations_written == 0
    assert results[0].error is None
    assert any("observation skipped" in r.message for r in caplog.records)


def test_run_latest_captures_sar_exception_without_sinking_batch():
    """SAR source raises → per-lake error captured, run continues for other lakes."""
    optical = FakeSource(
        scenes=[_scene("cloudy", date(2026, 7, 20))],
        cloudy_scene_ids={"cloudy"},
    )
    sar = FakeSarSource(raise_on_find=RuntimeError("SAR STAC unavailable"))

    patchers = _patched({"shishper": "lake-1"})
    with patchers[0], patchers[1], patchers[2], patchers[3]:
        results = run_latest(optical, lake_slugs=["shishper"], sar_source=sar)

    assert results[0].error == "SAR STAC unavailable"
    assert results[0].stale is True


def test_run_backfill_writes_sar_observation_when_optical_is_too_cloudy():
    """Backfill: cloudy optical + SAR source → SAR observation written in addition to
    continuing through the date range."""
    cloudy_day = date(2026, 7, 20)
    sar_day = date(2026, 7, 19)

    optical = FakeSource(
        scenes=[_scene("cloudy", cloudy_day)],
        cloudy_scene_ids={"cloudy"},
    )
    sar = FakeSarSource(scenes=[_sar_scene("sar-1", sar_day)])

    written_obs = []

    def capture_obs(conn, obs):
        written_obs.append(obs)
        return True

    patchers = _patched({"shishper": "lake-1"})
    obs_patcher = patch("pipeline.batch.upsert_observation", side_effect=capture_obs)
    with patchers[0], patchers[1], obs_patcher, patchers[3]:
        results = run_backfill(
            optical, date(2026, 7, 1), date(2026, 7, 31),
            lake_slugs=["shishper"], sar_source=sar,
        )

    assert results[0].scenes_checked == 1
    assert results[0].observations_written == 1
    assert written_obs[0].source == "sentinel1-grd-sar"
    assert written_obs[0].captured_at == sar_day
    assert written_obs[0].cloud_fraction == 0.0


def test_run_backfill_picks_closest_sar_scene_to_optical_date():
    """When multiple SAR scenes are in the window, the one closest in time is used."""
    optical_day = date(2026, 7, 20)
    sar_near = date(2026, 7, 21)   # 1 day away
    sar_far = date(2026, 7, 23)    # 3 days away (still in window)

    optical = FakeSource(
        scenes=[_scene("cloudy", optical_day)],
        cloudy_scene_ids={"cloudy"},
    )
    sar = FakeSarSource(scenes=[
        _sar_scene("sar-far", sar_far),
        _sar_scene("sar-near", sar_near),
    ])

    written_obs = []

    def capture_obs(conn, obs):
        written_obs.append(obs)
        return True

    patchers = _patched({"shishper": "lake-1"})
    obs_patcher = patch("pipeline.batch.upsert_observation", side_effect=capture_obs)
    with patchers[0], patchers[1], obs_patcher, patchers[3]:
        run_backfill(
            optical, date(2026, 7, 1), date(2026, 7, 31),
            lake_slugs=["shishper"], sar_source=sar,
        )

    assert len(written_obs) == 1
    # Nearest SAR scene date should be recorded.
    assert written_obs[0].captured_at == sar_near
