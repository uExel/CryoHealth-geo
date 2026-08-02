"""batch.py orchestrates db.py + stac_source.py + ndwi.py; all three are mocked/patched
here so these tests exercise only batch.py's own control flow (which scenes get
processed, when a lake is skipped, when staleness gets refreshed) — no real DB, no
real network, no real satellite imagery."""

from __future__ import annotations

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


def _scene(scene_id: str, day: date, cloud_pct: float = 5.0) -> SceneRef:
    return SceneRef(scene_id=scene_id, captured_at=day, cloud_cover_pct=cloud_pct, assets={})


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
