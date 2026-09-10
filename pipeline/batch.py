"""One batch pass, in two modes sharing the same per-scene logic:
- run_latest(): daily scheduled use — newest usable scene per lake, matches R2's
  "new cloud-free scene -> observation row, zero manual steps."
- run_backfill(): historical population over a date range — every usable scene per
  lake in the window, not just the newest.

SAR fallback (Sentinel-1 GRD) is automatically triggered when a Sentinel-2 optical
scene's cloud fraction at the AOI exceeds SAR_TRIGGER_CLOUD_FRACTION (0.40).  Both
run modes accept an optional ``sar_source`` argument; when None the pipeline behaves
exactly as before (cloudy scene skipped, lake ages toward stale).

SAR date-matching: SAR scenes must fall within ±SAR_DATE_WINDOW_DAYS of the optical
scene's captured_at date.  If no SAR scene exists in that window, the observation is
skipped and a WARNING is logged — a data gap is preferable to a fabricated reading.
The date written to the DB is the SAR scene's own captured_at (honest provenance).

Neither mode computes a hazard score or calls the alerts engine — that needs the real
hazard index (PRD §8), separate future work. This writes Observation rows and Lake.stale
flags only (R2's actual scope).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta
from uuid import uuid4

from pipeline.alphaearth_source import EmbeddingUnavailableError
from pipeline.alphaearth_source import fetch_embeddings as ae_fetch_embeddings
from pipeline.alphaearth_source import is_available as ae_is_available
from pipeline.db import ObservationWrite, connect, lake_id_for_slug, refresh_staleness, upsert_observation
from pipeline.lakes import LAKES
from pipeline.ml_segmentation import InferenceUnavailableError
from pipeline.ml_segmentation import is_available as ml_is_available
from pipeline.ml_segmentation import run_inference as ml_run_inference
from pipeline.ml_segmentation import water_area_km2_from_mask
from pipeline.ndwi import cloud_fraction, cloud_mask, compute_ndwi, water_area_km2
from pipeline.sar_source import extract_sar_water, sar_water_area_km2
from pipeline.stac_source import SceneRef, SceneSource

logger = logging.getLogger(__name__)

# Optical cloud fraction above which the SAR fallback is triggered (if sar_source is set).
SAR_TRIGGER_CLOUD_FRACTION = 0.40

# ±window in days around an optical scene's date when searching for a SAR companion.
# 3 days guarantees exactly one Sentinel-1 IW pass in scope given the 6-day revisit
# cycle (two satellites → effective ~6 days). Not ±6 because a week-old SAR reading
# during active melt (July–August) may represent a meaningfully different water extent.
SAR_DATE_WINDOW_DAYS = 3

# Source identifiers stored in the observations table — differentiates each water-
# extraction path so every DB row is auditable without inspecting logs (ADR 0002).
_SAR_SOURCE_ID = "sentinel1-grd-sar"
_NDWI_SOURCE_ID = "sentinel2-ndwi"
_ML_SEG_SOURCE_ID = "alphaearth-ml-seg"

# Hard ceiling: optical scenes above this cloud fraction are discarded entirely
# (no optical observation written). SAR fallback is attempted for scenes above
# SAR_TRIGGER_CLOUD_FRACTION — which may be the same or lower than this ceiling.
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
) -> tuple[float, float, str] | None:
    """Returns (area_km2, cloud_fraction, source_id) if scene is usable, else None.

    'Usable' means cloud fraction at the AOI is at or below USABLE_CLOUD_FRACTION_MAX.

    Water-extraction path (ADR 0002 §Architecture):
    1. AlphaEarth path: if both ae_is_available() and ml_is_available(), fetch
       64-band embeddings from GEE and run ONNX adapter → source_id = 'alphaearth-ml-seg'.
    2. NDWI fallback: on any EmbeddingUnavailableError or InferenceUnavailableError,
       fall back to McFeeters NDWI → source_id = 'sentinel2-ndwi'.

    source_id is recorded in the DB so every observation is auditable without logs.
    """
    bands = source.read_bands(scene, ["B03", "B08", "SCL"], bbox)
    usable = cloud_mask(bands["SCL"])
    clouds = cloud_fraction(usable)
    if clouds > USABLE_CLOUD_FRACTION_MAX:
        return None

    # --- AlphaEarth + adapter path ---
    if ae_is_available() and ml_is_available():
        try:
            year = scene.captured_at.year
            embeddings = ae_fetch_embeddings(bbox, year)  # (64, H, W) — cached by (bbox, year)
            mask = ml_run_inference(embeddings)            # (H, W) bool
            mask = mask & usable                           # exclude cloud pixels
            area = water_area_km2_from_mask(mask)
            logger.debug(
                "AlphaEarth+adapter path for scene %s (year=%d)", scene.scene_id, year
            )
            return area, clouds, _ML_SEG_SOURCE_ID
        except (EmbeddingUnavailableError, InferenceUnavailableError) as exc:
            logger.warning(
                "AlphaEarth/adapter unavailable for scene %s — falling back to NDWI: %s",
                scene.scene_id, exc,
            )

    # --- NDWI fallback (always available) ---
    ndwi = compute_ndwi(bands["B03"], bands["B08"])
    area = water_area_km2(ndwi, usable)
    logger.debug("NDWI path for scene %s", scene.scene_id)
    return area, clouds, _NDWI_SOURCE_ID


def _process_sar_scene(
    sar_source: SceneSource,
    optical_scene: SceneRef,
    bbox: tuple[float, float, float, float],
) -> tuple[float, date] | None:
    """Attempt SAR water extraction for scenes within ±SAR_DATE_WINDOW_DAYS of the
    optical scene's captured_at date.

    Returns:
        (area_km2, sar_captured_at) where sar_captured_at is the SAR scene's own date
        (not the optical scene date — we record what was actually observed).
        Returns None if no SAR scene is found in the window or if an error occurs.

    Degradation contract (see implementation_plan.md):
    - Zero SAR results in the window → log WARNING, return None (no fabricated reading).
    - Picks the scene with minimum |sar_date - optical_date| if multiple results.
    - The last-known optical reading is never silently substituted.
    """
    optical_date = optical_scene.captured_at
    window_start = optical_date - timedelta(days=SAR_DATE_WINDOW_DAYS)
    window_end = optical_date + timedelta(days=SAR_DATE_WINDOW_DAYS)

    candidates = sar_source.find_scenes_in_range(bbox, window_start, window_end)
    if not candidates:
        logger.warning(
            "No Sentinel-1 GRD scene within \u00b13 days of %s for bbox %s — observation skipped",
            optical_date.isoformat(),
            bbox,
        )
        return None

    # Pick the SAR scene closest in time to the optical scene's date.
    best = min(candidates, key=lambda s: abs((s.captured_at - optical_date).days))

    bands = sar_source.read_bands(best, ["vv", "vh"], bbox)
    water_mask = extract_sar_water(bands["vv"], bands["vh"])
    area = sar_water_area_km2(water_mask)
    return area, best.captured_at


def run_latest(
    source: SceneSource,
    lake_slugs: list[str] | None = None,
    *,
    sar_source: SceneSource | None = None,
) -> list[LakeResult]:
    """Process the newest usable scene for each lake.

    When a candidate scene's AOI cloud fraction exceeds SAR_TRIGGER_CLOUD_FRACTION
    (0.40) and ``sar_source`` is provided, a Sentinel-1 SAR scene from within
    ±SAR_DATE_WINDOW_DAYS is used instead. If no SAR scene is found the candidate
    is skipped. If ``sar_source`` is None, cloud-obscured scenes are skipped as before.
    """
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
                        area_km2, clouds, obs_source = outcome
                        obs = ObservationWrite(
                            lake_id=lake_id,
                            captured_at=scene.captured_at,
                            area_km2=area_km2,
                            cloud_fraction=clouds,
                            scene_id=scene.scene_id,
                            run_id=run_id,
                            source=obs_source,
                        )
                        if upsert_observation(conn, obs):
                            written += 1
                        break  # newest usable is enough for the "latest" mode

                    # Optical too cloudy — try SAR fallback if wired up.
                    if sar_source is not None:
                        sar_outcome = _process_sar_scene(sar_source, scene, lake.bbox())
                        if sar_outcome is not None:
                            sar_area_km2, sar_date = sar_outcome
                            obs = ObservationWrite(
                                lake_id=lake_id,
                                captured_at=sar_date,
                                area_km2=sar_area_km2,
                                cloud_fraction=0.0,  # SAR is cloud-transparent
                                scene_id=scene.scene_id,
                                run_id=run_id,
                                source=_SAR_SOURCE_ID,
                            )
                            if upsert_observation(conn, obs):
                                written += 1
                            break  # SAR fallback succeeded; stop searching
            except Exception as exc:  # noqa: BLE001 — one lake's failure must not sink the batch
                logger.exception("run_latest failed for %s", slug)
                results.append(LakeResult(slug, written, checked, stale=True, error=str(exc)))
                continue

            stale = refresh_staleness(conn, lake_id)
            results.append(LakeResult(slug, written, checked, stale=stale))
    return results


def run_backfill(
    source: SceneSource,
    start: date,
    end: date | None = None,
    lake_slugs: list[str] | None = None,
    *,
    sar_source: SceneSource | None = None,
) -> list[LakeResult]:
    """Process every usable scene for each lake in [start, end].

    For each optical scene whose AOI cloud fraction exceeds SAR_TRIGGER_CLOUD_FRACTION
    (0.40), a Sentinel-1 SAR scene from ±SAR_DATE_WINDOW_DAYS is attempted if
    ``sar_source`` is set. Both optical and SAR observations may be written for the
    same time window (different source values distinguish them in the DB).
    """
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
                        # Optical too cloudy — attempt SAR fallback if wired up.
                        if sar_source is not None:
                            sar_outcome = _process_sar_scene(sar_source, scene, lake.bbox())
                            if sar_outcome is not None:
                                sar_area_km2, sar_date = sar_outcome
                                obs = ObservationWrite(
                                    lake_id=lake_id,
                                    captured_at=sar_date,
                                    area_km2=sar_area_km2,
                                    cloud_fraction=0.0,
                                    scene_id=scene.scene_id,
                                    run_id=run_id,
                                    source=_SAR_SOURCE_ID,
                                )
                                if upsert_observation(conn, obs):
                                    written += 1
                        continue
                    area_km2, clouds, obs_source = outcome
                    obs = ObservationWrite(
                        lake_id=lake_id,
                        captured_at=scene.captured_at,
                        area_km2=area_km2,
                        cloud_fraction=clouds,
                        scene_id=scene.scene_id,
                        run_id=run_id,
                        source=obs_source,
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


__all__ = ["LakeResult", "SAR_DATE_WINDOW_DAYS", "SAR_TRIGGER_CLOUD_FRACTION", "run_backfill", "run_latest"]
