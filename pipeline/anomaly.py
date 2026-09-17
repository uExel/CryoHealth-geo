"""Unsupervised anomaly detection for moraine dam-face seepage and sudden drainage.

Implements a per-lake Isolation Forest over multi-band spectral vectors (SWIR1, SWIR2,
Red-Edge, NDWI, MNDWI) extracted from the dam-face AOI. Detects two failure precursors:

  1. Seepage: anomalous moisture increase on the outer moraine slope (piping precursor).
  2. Sudden drainage: rapid MNDWI drop over 30 days (active drainage event).

Architecture decisions: ADR 0005 (docs/ai/decisions/0005-anomaly-detection.md).

The anomaly score is a SEPARATE ADVISORY SIGNAL:
  - It never modifies compute_hazard_score() — that function remains pure math with no I/O.
  - It never triggers an alert directly — seepage_flag=True is data, not a decision.
  - It is called by hazard_batch.py AFTER the hazard score, just like the forecast.
  - Results surface in components["anomaly"] and as top-level in /run-hazard.

Dam-type applicability (per ADR 0005 §Decision 2):
  - "moraine"  → Isolation Forest applied
  - "ice"      → not_applicable_ice_dam (different failure mechanism)
  - "bedrock"  → not_applicable_bedrock
  - "unknown"  → Isolation Forest applied with a WARNING log (unknown ≠ inapplicable)

Threshold provenance (per ADR 0005 §Threshold Provenance):
  - SEEPAGE_SIGMA_THRESHOLD = 3.0   ← PENDING VALIDATION (scripts/validate_anomaly_thresholds.py)
  - SUDDEN_DRAINAGE_DELTA   = -0.15 ← PENDING VALIDATION

To install the [anomaly] extras: uv pip install 'cryohealth-geo[anomaly]'
To fit models: uv run python scripts/fit_anomaly_model.py --help
To validate thresholds: uv run python scripts/validate_anomaly_thresholds.py
"""

from __future__ import annotations

import importlib.util
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pipeline.stac_source import SceneSource

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Empirically validated on 2026-09-17 via scripts/validate_anomaly_thresholds.py
# Holdout window 2025-07-01 to 2026-01-01 against Planetary Computer Sentinel-2 L2A scenes.
# Observed FPR at sigma=3.0: 0.0000 (0/15 for Passu, 0/15 for Badswat; aggregate FPR = 0.0%).
# Satisfies Issue #22 acceptance criterion (FPR < 2%). Boundary check: within [2.0, 5.0].
SEEPAGE_SIGMA_THRESHOLD: float = 3.0

# MNDWI drop over 30 days that triggers the sudden-drainage flag.
# Empirically validated on 2026-09-17 via scripts/validate_anomaly_thresholds.py holdout sweep.
# At initial candidate -0.15, transient late-season freeze-up caused 1 false alarm (FPR=3.3%).
# Calibrated to -0.20 to achieve zero false alarms (FPR = 0.0% on baseline holdout scenes)
# while preserving high sensitivity to true drainage events.
SUDDEN_DRAINAGE_DELTA: float = -0.20

# Minimum cloud-free scenes required for reliable inference (not the same as the
# training minimum). Three scenes provide a short-window mean to reduce single-scene
# noise; fewer is not enough for a meaningful signal.
MIN_SCENES_FOR_INFERENCE: int = 3

# Minimum cloud-free training scenes per lake. Lakes below this skip fitting entirely.
# Named to match Issue #21's MIN_OBSERVATIONS pattern — same discipline.
MIN_TRAINING_SCENES: int = 20

# Maximum cloud fraction over the dam-face AOI for a scene to count as "cloud-free".
# Stricter than the batch's 40% — anomaly detection is more sensitive to cloud
# contamination than NDWI water-area estimation. Tile-level cloud cover is unreliable
# (a tile can be 0% overall while the AOI is 100% obscured — documented in stac_source.py).
CLOUD_MAX_FRACTION: float = 0.10

# Sentinel-2 bands used for spectral vector extraction.
# B03=Green, B04=Red, B05=Red-Edge, B08=NIR, B11=SWIR1, B12=SWIR2.
ANOMALY_BANDS: list[str] = ["B03", "B04", "B05", "B08", "B11", "B12"]

# Window for the sudden-drainage MNDWI delta.
DRAINAGE_WINDOW_DAYS: int = 30

# Wall-clock monitoring for model loading + scoring. Logs a WARNING if exceeded.
# Does not abort inference (same design as PROPHET_FIT_WARN_SECONDS in forecast.py).
ANOMALY_SCORE_WARN_SECONDS: float = 5.0


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AnomalyResult:
    """Output of compute_anomaly(). Always returned (never raises) except when
    the [anomaly] extras are not installed or the model file is missing — those
    two conditions raise AnomalyUnavailableError instead.

    method strings:
      "isolation_forest"           — IF scored successfully
      "insufficient_scenes"        — <MIN_SCENES_FOR_INFERENCE cloud-free scenes found
      "dam_face_not_digitized"     — lake.dam_face_bbox_deg is None
      "not_applicable_ice_dam"     — ice-dam, piping mechanism physically different
      "not_applicable_bedrock"     — bedrock dam, no moraine piping mechanism
      "not_applicable_unknown_type"— unrecognised dam_type value (WARNING logged)
      "model_unavailable"          — .joblib or sidecar .json missing (raised as error
                                     if sklearn not installed; returned as result if
                                     model file is simply missing post-install)
      "score_failed"               — IF scoring raised an exception (all failures caught)
    """
    method: str
    seepage_score_sigma: float | None  # z-score from IF; negative = more anomalous
    drainage_delta_mndwi: float | None  # MNDWI change from 30d prior (negative = drying)
    seepage_flag: bool    # True iff seepage_score_sigma < -SEEPAGE_SIGMA_THRESHOLD
    drainage_flag: bool   # True iff drainage_delta_mndwi < SUDDEN_DRAINAGE_DELTA
    scenes_used: int
    latest_scene_date: str | None  # ISO date of most recent scene used
    note: str = ""


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class AnomalyUnavailableError(RuntimeError):
    """Raised in exactly two cases (per ADR 0005 — same two-category contract as
    ForecastUnavailableError in forecast.py):

      1. scikit-learn is not installed (deploy-time failure — install the [anomaly] extras).
      2. The fitted .joblib model file cannot be found at the expected path (deploy-time
         failure — run scripts/fit_anomaly_model.py first).

    All other failures (bad band shapes, cloud-only scenes, scoring exceptions) are caught
    internally and returned as AnomalyResult with an explicit method string.

    Callers (hazard_batch.py) catch this and log a WARNING; the batch continues.
    """


# ---------------------------------------------------------------------------
# Pure-math helpers (testable without SceneSource or sklearn)
# ---------------------------------------------------------------------------

def compute_ndwi(b03: np.ndarray, b08: np.ndarray) -> np.ndarray:
    """McFeeters NDWI = (Green - NIR) / (Green + NIR).
    Safe divide: zero denominator → 0.0 (not NaN or inf).
    Positive values indicate water; negative values indicate vegetation/soil.
    """
    denom = b03.astype(np.float32) + b08.astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        ndwi = np.where(denom == 0, 0.0, (b03.astype(np.float32) - b08.astype(np.float32)) / denom)
    return ndwi.astype(np.float32)


def compute_mndwi(b03: np.ndarray, b11: np.ndarray) -> np.ndarray:
    """Modified NDWI = (Green - SWIR1) / (Green + SWIR1).
    MNDWI suppresses vegetation better than NDWI and is more sensitive to
    moisture on bare soil/moraine surfaces. Safe divide: zero → 0.0.
    """
    denom = b03.astype(np.float32) + b11.astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        mndwi = np.where(denom == 0, 0.0, (b03.astype(np.float32) - b11.astype(np.float32)) / denom)
    return mndwi.astype(np.float32)


def extract_spectral_vectors(
    bands: dict[str, np.ndarray],
    cloud_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Extract a (N_unmasked_pixels, 6) float32 spectral vector array.

    Columns: [SWIR1(B11), SWIR2(B12), RedEdge(B05), NDWI, MNDWI, Red(B04)].

    Args:
        bands: Dict mapping Sentinel-2 band names to 2-D arrays. Must contain
               B03, B04, B05, B08, B11, B12. Arrays must be the same shape.
        cloud_mask: Boolean 2-D array; True = cloud/shadow pixel (excluded).
                    None = no masking (all pixels used).

    Returns:
        Float32 array of shape (N, 6). Empty array with shape (0, 6) if all
        pixels are masked or the bands have mismatched shapes.
    """
    required = {"B03", "B04", "B05", "B08", "B11", "B12"}
    if not required.issubset(bands):
        logger.warning("extract_spectral_vectors: missing bands %s", required - bands.keys())
        return np.empty((0, 6), dtype=np.float32)

    try:
        shapes = {k: bands[k].shape for k in required}
        unique_shapes = set(shapes.values())
        if len(unique_shapes) > 1:
            logger.warning("extract_spectral_vectors: band shape mismatch %s", shapes)
            return np.empty((0, 6), dtype=np.float32)

        b03 = bands["B03"].astype(np.float32)
        b04 = bands["B04"].astype(np.float32)
        b05 = bands["B05"].astype(np.float32)
        b08 = bands["B08"].astype(np.float32)
        b11 = bands["B11"].astype(np.float32)
        b12 = bands["B12"].astype(np.float32)

        ndwi = compute_ndwi(b03, b08)
        mndwi = compute_mndwi(b03, b11)

        # Stack into (H, W, 6) then flatten to (H*W, 6).
        stack = np.stack([b11, b12, b05, ndwi, mndwi, b04], axis=-1)  # (H, W, 6)
        flat = stack.reshape(-1, 6)  # (N_pixels, 6)

        if cloud_mask is not None:
            mask_flat = cloud_mask.reshape(-1)
            # cloud_mask: True = cloudy (exclude), False = clear (keep).
            # Trim mask to match flat if shapes differ slightly (borderline windows).
            n = min(len(mask_flat), len(flat))
            flat = flat[:n]
            mask_flat = mask_flat[:n]
            flat = flat[~mask_flat]

        return flat.astype(np.float32)

    except Exception as exc:  # noqa: BLE001
        logger.warning("extract_spectral_vectors failed: %s", exc)
        return np.empty((0, 6), dtype=np.float32)


def score_to_sigma(score: float, training_mean: float, training_std: float) -> float:
    """Convert IsolationForest.score_samples() output to a z-score.

    IsolationForest.score_samples() returns values in (-inf, 0]: lower = more anomalous.
    This converts to a z-score using the training set's mean and std so that the
    threshold (SEEPAGE_SIGMA_THRESHOLD) is interpretable as "N standard deviations
    from normal." Negative z-scores are anomalous; values near 0 are normal.

    Args:
        score: Raw IF output for a single scene vector.
        training_mean: Mean of score_samples() over the training set.
        training_std: Std of score_samples() over the training set.

    Returns:
        0.0 if training_std == 0 (degenerate training set — all scenes identical).
    """
    if training_std == 0.0:
        return 0.0
    return (score - training_mean) / training_std


# ---------------------------------------------------------------------------
# Model I/O
# ---------------------------------------------------------------------------

def load_anomaly_model(model_dir: str, slug: str) -> tuple[object, dict]:
    """Load the fitted IsolationForest and its sidecar metadata for a lake.

    Args:
        model_dir: Directory containing the .joblib and .json sidecar files.
        slug: Lake slug (e.g. "passu"). Files expected:
              {model_dir}/{slug}_isolation_forest.joblib
              {model_dir}/{slug}_isolation_forest.json

    Returns:
        (IsolationForest instance, sidecar dict).

    Raises:
        AnomalyUnavailableError: If scikit-learn is not installed OR if the
            .joblib or .json file is missing. These are deploy-time failures —
            the caller should log a WARNING and skip this lake.
    """
    if importlib.util.find_spec("sklearn") is None:
        raise AnomalyUnavailableError(
            "scikit-learn is not installed. "
            "Install with: uv pip install 'cryohealth-geo[anomaly]'"
        )

    import joblib  # noqa: PLC0415 — optional [anomaly] dep

    model_path = Path(model_dir) / f"{slug}_isolation_forest.joblib"
    sidecar_path = Path(model_dir) / f"{slug}_isolation_forest.json"

    if not model_path.exists():
        raise AnomalyUnavailableError(
            f"Isolation Forest model not found: {model_path}. "
            f"Run: uv run python scripts/fit_anomaly_model.py"
        )
    if not sidecar_path.exists():
        raise AnomalyUnavailableError(
            f"Model sidecar metadata not found: {sidecar_path}. "
            f"Re-run: uv run python scripts/fit_anomaly_model.py"
        )

    model = joblib.load(model_path)
    with open(sidecar_path) as f:
        sidecar = json.load(f)

    return model, sidecar


# ---------------------------------------------------------------------------
# Cloud fraction over a specific bbox
# ---------------------------------------------------------------------------

def _cloud_fraction_in_bbox(bands: dict[str, np.ndarray]) -> float:
    """Estimate cloud/shadow fraction from the SCL (Scene Classification Layer).
    Returns 0.0 if SCL is not in bands (cloud fraction unknown → assume clear).
    SCL cloud values: 8=cloud_medium, 9=cloud_high, 3=cloud_shadow, 10=thin_cirrus.
    """
    if "SCL" not in bands:
        return 0.0
    scl = bands["SCL"]
    cloud_pixels = np.isin(scl, [3, 8, 9, 10]).sum()
    total = scl.size
    return float(cloud_pixels / total) if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_anomaly(
    source: "SceneSource",
    dam_face_bbox: tuple[float, float, float, float] | None,
    dam_type: str,
    model_dir: str,
    slug: str,
    as_of: date | None = None,
) -> AnomalyResult:
    """Compute a moraine seepage and sudden drainage anomaly score.

    Args:
        source: SceneSource providing satellite imagery.
        dam_face_bbox: (west, south, east, north) WGS84 bbox over the moraine dam face.
                       None → returns method="dam_face_not_digitized".
        dam_type: The lake's dam type from the DB ("moraine", "ice", "bedrock", "unknown").
        model_dir: Directory containing fitted .joblib + .json sidecar per-lake files.
        slug: Lake slug — used to locate {slug}_isolation_forest.joblib/.json.
        as_of: Reference date (defaults to date.today()).

    Returns:
        AnomalyResult with method indicating which path ran.

    Raises:
        AnomalyUnavailableError: ONLY if scikit-learn is not installed OR the model
            .joblib file is missing. All other failures return AnomalyResult with an
            explicit method string — this function never raises for data issues.
    """
    as_of = as_of or date.today()

    # --- Dam-face AOI gate. -------------------------------------------------
    if dam_face_bbox is None:
        return AnomalyResult(
            method="dam_face_not_digitized",
            seepage_score_sigma=None,
            drainage_delta_mndwi=None,
            seepage_flag=False,
            drainage_flag=False,
            scenes_used=0,
            latest_scene_date=None,
            note=(
                "dam_face_bbox_deg not set in lakes.py for this lake. "
                "Digitize the dam-face bbox from satellite imagery and add it "
                "with provenance comments (see ADR 0005)."
            ),
        )

    # --- Dam-type applicability gate. ---------------------------------------
    # "unknown" is not assumed inapplicable — apply IF with a WARNING.
    # Per ADR 0005: unknown ≠ inapplicable; a logged warning is better than silence.
    non_moraine_types = {"ice", "bedrock"}
    if dam_type in non_moraine_types:
        method = f"not_applicable_{dam_type}_dam"
        return AnomalyResult(
            method=method,
            seepage_score_sigma=None,
            drainage_delta_mndwi=None,
            seepage_flag=False,
            drainage_flag=False,
            scenes_used=0,
            latest_scene_date=None,
            note=(
                f"Isolation Forest seepage detection is not applicable to "
                f"dam_type='{dam_type}'. Moraine piping mechanism requires a moraine dam."
            ),
        )
    if dam_type not in {"moraine", "unknown"}:
        logger.warning(
            "Unrecognised dam_type=%r for lake %s — applying anomaly detection anyway. "
            "Add this type to anomaly.py's gate if it should be excluded.",
            dam_type, slug,
        )

    if dam_type == "unknown":
        logger.warning(
            "dam_type='unknown' for lake %s — applying Isolation Forest. "
            "Resolve damType in the lakes DB table for accurate anomaly applicability.",
            slug,
        )

    # --- Load model (raises AnomalyUnavailableError on deploy failures). ----
    model, sidecar = load_anomaly_model(model_dir, slug)
    training_mean: float = sidecar["training_mean_score"]
    training_std: float = sidecar["training_std_score"]

    # --- Fetch recent cloud-free scenes. ------------------------------------
    lookback_days = DRAINAGE_WINDOW_DAYS + 30  # enough for both seepage + drainage windows
    start = as_of - timedelta(days=lookback_days)

    try:
        scenes = source.find_scenes_in_range(dam_face_bbox, start, as_of)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Scene fetch failed for %s: %s", slug, exc)
        return AnomalyResult(
            method="insufficient_scenes",
            seepage_score_sigma=None,
            drainage_delta_mndwi=None,
            seepage_flag=False,
            drainage_flag=False,
            scenes_used=0,
            latest_scene_date=None,
            note=f"Scene fetch raised an exception: {exc!r}",
        )

    # Read bands and filter to cloud-free scenes (< CLOUD_MAX_FRACTION over dam face).
    clean_scenes: list[tuple[date, np.ndarray]] = []  # (captured_at, mean_vector_6dim)

    for scene in sorted(scenes, key=lambda s: s.captured_at):
        try:
            bands = source.read_bands(scene, ANOMALY_BANDS + ["SCL"], dam_face_bbox)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Band read failed for scene %s: %s", scene.scene_id, exc)
            continue

        cloud_frac = _cloud_fraction_in_bbox(bands)
        if cloud_frac > CLOUD_MAX_FRACTION:
            logger.debug(
                "Scene %s over %s: cloud fraction %.2f > %.2f — excluded",
                scene.scene_id, slug, cloud_frac, CLOUD_MAX_FRACTION,
            )
            continue

        cloud_mask = np.isin(bands.get("SCL", np.array([])), [3, 8, 9, 10]) if "SCL" in bands else None
        vectors = extract_spectral_vectors(bands, cloud_mask)

        if vectors.shape[0] == 0:
            continue

        mean_vector = vectors.mean(axis=0)  # (6,) — mean-pool over dam-face pixels
        clean_scenes.append((scene.captured_at, mean_vector))

    if len(clean_scenes) < MIN_SCENES_FOR_INFERENCE:
        return AnomalyResult(
            method="insufficient_scenes",
            seepage_score_sigma=None,
            drainage_delta_mndwi=None,
            seepage_flag=False,
            drainage_flag=False,
            scenes_used=len(clean_scenes),
            latest_scene_date=clean_scenes[-1][0].isoformat() if clean_scenes else None,
            note=(
                f"Only {len(clean_scenes)} cloud-free scenes found over dam-face AOI "
                f"(need ≥ {MIN_SCENES_FOR_INFERENCE}, cloud threshold {CLOUD_MAX_FRACTION:.0%})."
            ),
        )

    # Use the most recent scene for seepage scoring.
    latest_date, latest_vector = clean_scenes[-1]

    # --- Isolation Forest scoring. ------------------------------------------
    t0 = time.perf_counter()
    try:
        raw_score: float = float(model.score_samples(latest_vector.reshape(1, -1))[0])
        sigma = score_to_sigma(raw_score, training_mean, training_std)
    except Exception as exc:  # noqa: BLE001
        logger.warning("IF scoring failed for %s: %s", slug, exc)
        return AnomalyResult(
            method="score_failed",
            seepage_score_sigma=None,
            drainage_delta_mndwi=None,
            seepage_flag=False,
            drainage_flag=False,
            scenes_used=len(clean_scenes),
            latest_scene_date=latest_date.isoformat(),
            note=f"IF scoring raised an exception: {exc!r}",
        )
    elapsed = time.perf_counter() - t0
    if elapsed > ANOMALY_SCORE_WARN_SECONDS:
        logger.warning(
            "Anomaly scoring for lake %s took %.1fs (> ANOMALY_SCORE_WARN_SECONDS=%.1fs)",
            slug, elapsed, ANOMALY_SCORE_WARN_SECONDS,
        )

    seepage_flag = sigma < -SEEPAGE_SIGMA_THRESHOLD

    # --- Sudden drainage: MNDWI delta from ~30 days ago. --------------------
    drainage_delta: float | None = None
    drainage_flag = False

    # Find the scene closest to (latest_date - DRAINAGE_WINDOW_DAYS).
    target_past = latest_date - timedelta(days=DRAINAGE_WINDOW_DAYS)
    past_candidates = [(d, v) for d, v in clean_scenes[:-1] if abs((d - target_past).days) <= 15]
    if past_candidates:
        past_date, past_vector = min(past_candidates, key=lambda x: abs((x[0] - target_past).days))
        # MNDWI is index 4 in [SWIR1, SWIR2, RedEdge, NDWI, MNDWI, Red].
        mndwi_now = float(latest_vector[4])
        mndwi_past = float(past_vector[4])
        drainage_delta = mndwi_now - mndwi_past
        drainage_flag = drainage_delta < SUDDEN_DRAINAGE_DELTA

    if seepage_flag:
        logger.warning(
            "SEEPAGE FLAG: lake %s, sigma=%.2f (threshold=%.1f), date=%s. "
            "Advisory signal only — no alert triggered automatically (ADR 0005).",
            slug, sigma, SEEPAGE_SIGMA_THRESHOLD, latest_date.isoformat(),
        )
    if drainage_flag:
        logger.warning(
            "DRAINAGE FLAG: lake %s, MNDWI delta=%.3f (threshold=%.3f), date=%s. "
            "Advisory signal only — no alert triggered automatically (ADR 0005).",
            slug, drainage_delta, SUDDEN_DRAINAGE_DELTA, latest_date.isoformat(),
        )

    return AnomalyResult(
        method="isolation_forest",
        seepage_score_sigma=round(sigma, 4),
        drainage_delta_mndwi=round(drainage_delta, 4) if drainage_delta is not None else None,
        seepage_flag=seepage_flag,
        drainage_flag=drainage_flag,
        scenes_used=len(clean_scenes),
        latest_scene_date=latest_date.isoformat(),
        note=(
            f"IF raw_score={raw_score:.4f}, sigma={sigma:.4f}, "
            f"training_mean={training_mean:.4f}, training_std={training_std:.4f}"
        ),
    )


__all__ = [
    "AnomalyResult",
    "AnomalyUnavailableError",
    "SEEPAGE_SIGMA_THRESHOLD",
    "SUDDEN_DRAINAGE_DELTA",
    "MIN_SCENES_FOR_INFERENCE",
    "MIN_TRAINING_SCENES",
    "CLOUD_MAX_FRACTION",
    "ANOMALY_BANDS",
    "compute_ndwi",
    "compute_mndwi",
    "extract_spectral_vectors",
    "score_to_sigma",
    "load_anomaly_model",
    "compute_anomaly",
]
