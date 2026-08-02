"""One batch pass, in two modes sharing the same per-scene logic:
- run_latest(): daily scheduled use — newest usable scene per lake, matches R2's
  "new cloud-free scene -> observation row, zero manual steps."
- run_backfill(): historical population over a date range — every usable scene per
  lake in the window, not just the newest.

Neither mode computes a hazard score or calls the alerts engine — that needs the real
hazard index (PRD §8), separate future work. This writes Observation rows and Lake.stale
flags only (R2's actual scope).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from uuid import uuid4

from pipeline.db import ObservationWrite, connect, lake_id_for_slug, refresh_staleness, upsert_observation
from pipeline.lakes import LAKES
from pipeline.ndwi import cloud_fraction, cloud_mask, compute_ndwi, water_area_km2
from pipeline.stac_source import SceneRef, SceneSource

logger = logging.getLogger(__name__)

USABLE_CLOUD_FRACTION_MAX = 0.5


@dataclass
class LakeResult:
    slug: str
    observations_written: int
    scenes_checked: int
    stale: bool
    error: str | None = None


def _process_scene(
    source: SceneSource, scene: SceneRef, bbox: tuple[float, float, float, float]
) -> tuple[float, float] | None:
    """Returns (area_km2, cloud_fraction) if the scene is usable at the AOI, else None."""
    bands = source.read_bands(scene, ["B03", "B08", "SCL"], bbox)
    usable = cloud_mask(bands["SCL"])
    clouds = cloud_fraction(usable)
    if clouds > USABLE_CLOUD_FRACTION_MAX:
        return None
    ndwi = compute_ndwi(bands["B03"], bands["B08"])
    return water_area_km2(ndwi, usable), clouds


def run_latest(source: SceneSource, lake_slugs: list[str] | None = None) -> list[LakeResult]:
    run_id = f"latest-{uuid4()}"
    results = []
    with connect() as conn:
        for slug in lake_slugs or list(LAKES):
            lake = LAKES[slug]
            lake_id = lake_id_for_slug(conn, slug)
            if lake_id is None:
                results.append(LakeResult(slug, 0, 0, stale=True, error="lake not seeded in DB"))
                continue

            written = 0
            checked = 0
            try:
                candidates = source.find_recent_scenes(lake.bbox(), limit=12)
                for scene in candidates:
                    checked += 1
                    outcome = _process_scene(source, scene, lake.bbox())
                    if outcome is not None:
                        area_km2, clouds = outcome
                        obs = ObservationWrite(
                            lake_id=lake_id,
                            captured_at=scene.captured_at,
                            area_km2=area_km2,
                            cloud_fraction=clouds,
                            scene_id=scene.scene_id,
                            run_id=run_id,
                        )
                        if upsert_observation(conn, obs):
                            written += 1
                        break  # newest usable is enough for the "latest" mode
            except Exception as exc:  # noqa: BLE001 — one lake's failure must not sink the batch
                logger.exception("run_latest failed for %s", slug)
                results.append(LakeResult(slug, written, checked, stale=True, error=str(exc)))
                continue

            stale = refresh_staleness(conn, lake_id)
            results.append(LakeResult(slug, written, checked, stale=stale))
    return results


def run_backfill(
    source: SceneSource, start: date, end: date | None = None, lake_slugs: list[str] | None = None
) -> list[LakeResult]:
    end = end or date.today()
    run_id = f"backfill-{start.isoformat()}-{uuid4()}"
    results = []
    with connect() as conn:
        for slug in lake_slugs or list(LAKES):
            lake = LAKES[slug]
            lake_id = lake_id_for_slug(conn, slug)
            if lake_id is None:
                results.append(LakeResult(slug, 0, 0, stale=True, error="lake not seeded in DB"))
                continue

            written = 0
            checked = 0
            try:
                candidates = source.find_scenes_in_range(lake.bbox(), start, end)
                for scene in candidates:
                    checked += 1
                    outcome = _process_scene(source, scene, lake.bbox())
                    if outcome is None:
                        continue
                    area_km2, clouds = outcome
                    obs = ObservationWrite(
                        lake_id=lake_id,
                        captured_at=scene.captured_at,
                        area_km2=area_km2,
                        cloud_fraction=clouds,
                        scene_id=scene.scene_id,
                        run_id=run_id,
                    )
                    if upsert_observation(conn, obs):
                        written += 1
            except Exception as exc:  # noqa: BLE001
                logger.exception("run_backfill failed for %s", slug)
                results.append(LakeResult(slug, written, checked, stale=True, error=str(exc)))
                continue

            stale = refresh_staleness(conn, lake_id)
            results.append(LakeResult(slug, written, checked, stale=stale))
    return results


__all__ = ["LakeResult", "run_backfill", "run_latest"]
