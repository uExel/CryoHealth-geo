"""Segmentation adapter inference on AlphaEarth embeddings.

Runs a lightweight ONNX decoder head trained on HMA glacial lake labels on top
of 64-band AlphaEarth embeddings (fetched by pipeline/alphaearth_source.py),
producing a binary water mask at 10 m resolution.

Design contract (mirrors ndwi.py / alphaearth_source.py philosophy):
- Pure inference — no DB, no HTTP, no file I/O at call time.
- Raises InferenceUnavailableError for all "model not usable" conditions so the
  caller (batch.py) can fall back to NDWI cleanly without catching bare exceptions.
- The adapter takes AlphaEarth EMBEDDINGS as input (64 bands), not raw satellite
  bands — it is NOT a standalone segmentation model.

See docs/ai/decisions/0002-ml-segmentation.md (ADR 0002) for architecture rationale.
See docs/AI_MODEL_CARD.md for model specification and evaluation targets.

TODO(#19-training): this module is INFRASTRUCTURE — no trained adapter weights exist
yet. The is_available() probe returns False until ML_MODEL_PATH is set to a real
.onnx file. Adapter training and 92% IoU verification are tracked in Issue #19.
The 92% IoU acceptance criterion from Issue #18 is NOT met by this PR (see ADR 0002).

Environment variables (all optional):
    ML_MODEL_PATH             Local path to the ONNX adapter weights (.onnx file).
                              When unset, is_available() returns False and every
                              run_inference() call raises InferenceUnavailableError.
    ML_INFERENCE_TIMEOUT_S    Compute-bound timeout (default 2.5 s).
                              This is the original Issue #18 acceptance criterion,
                              now explicitly scoped to adapter inference only — see
                              ADR 0002 §Latency Criterion for the split rationale.
    ML_USE_CUDA               Set to '1' to prefer CUDA execution provider (default CPU).
"""

from __future__ import annotations

import logging
import os
import time

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Expected number of input channels: one per AlphaEarth embedding band.
_EXPECTED_INPUT_CHANNELS = 64

# Sigmoid threshold for converting logit output to binary water mask.
_SIGMOID_THRESHOLD = 0.5

# Pixel area for Sentinel-2 / AlphaEarth 10 m resolution.
_PIXEL_AREA_M2 = 10 * 10

# Default adapter inference timeout (seconds).
# This is the compute-bound target from Issue #18, explicitly scoped to inference
# only — GEE fetch latency is governed separately by EE_FETCH_TIMEOUT_S.
_DEFAULT_INFERENCE_TIMEOUT_S = 2.5

# ---------------------------------------------------------------------------
# Lazy ONNX session cache
# ---------------------------------------------------------------------------

_session: object | None = None  # onnxruntime.InferenceSession when loaded


def reset_session() -> None:
    """Force the next call to reload the ONNX session (useful in tests)."""
    global _session  # noqa: PLW0603
    _session = None


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class InferenceUnavailableError(RuntimeError):
    """Raised when the ONNX adapter cannot run.

    Callers should catch this and fall back to the NDWI path.
    Possible causes:
    - ML_MODEL_PATH is not set (no weights yet — pending Issue #19).
    - onnxruntime is not installed.
    - The weights file does not exist at the configured path.
    - Inference exceeded ML_INFERENCE_TIMEOUT_S.
    - ONNX runtime error during inference.
    """


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------


def _get_model_path() -> str:
    """Return the configured model path; raises InferenceUnavailableError if unset."""
    path = os.environ.get("ML_MODEL_PATH", "")
    if not path:
        raise InferenceUnavailableError(
            "ML_MODEL_PATH is not set — no adapter weights available. "
            "Adapter training is tracked in Issue #19. "
            "The NDWI fallback will be used instead."
        )
    return path


def _load_session() -> object:
    """Load (or return cached) ONNX InferenceSession.

    Raises InferenceUnavailableError if onnxruntime is not installed or
    the model path does not exist.
    """
    global _session  # noqa: PLW0603
    if _session is not None:
        return _session

    try:
        import onnxruntime as ort  # noqa: PLC0415
    except ImportError as exc:
        raise InferenceUnavailableError(
            "onnxruntime is not installed. Install it with: "
            "uv pip install 'cryohealth-geo[ml]'. "
            "The NDWI fallback will be used instead."
        ) from exc

    model_path = _get_model_path()

    if not os.path.isfile(model_path):
        raise InferenceUnavailableError(
            f"ONNX adapter weights not found at '{model_path}'. "
            "See Issue #19 for adapter training. NDWI fallback will be used."
        )

    use_cuda = os.environ.get("ML_USE_CUDA", "0") == "1"
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if use_cuda
        else ["CPUExecutionProvider"]
    )

    logger.info(
        "Loading ONNX segmentation adapter from %s (providers=%s)", model_path, providers
    )
    _session = ort.InferenceSession(model_path, providers=providers)
    logger.info(
        "ONNX adapter session loaded — inputs: %s",
        [i.name for i in _session.get_inputs()],
    )
    return _session


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def is_available() -> bool:
    """Return True if adapter inference can be attempted.

    Fast probe — checks:
    1. ML_MODEL_PATH env var is set.
    2. onnxruntime is importable.
    3. The model file exists at the configured path.

    Does NOT load the session or make any network call.
    """
    path = os.environ.get("ML_MODEL_PATH", "")
    if not path:
        return False
    if not os.path.isfile(path):
        return False
    try:
        import onnxruntime  # noqa: F401,PLC0415
    except ImportError:
        return False
    return True


def run_inference(
    embeddings: np.ndarray,
    *,
    timeout_s: float | None = None,
) -> np.ndarray:
    """Run the segmentation adapter on AlphaEarth embeddings.

    Args:
        embeddings: np.ndarray, shape (64, H, W) float32 — AlphaEarth embedding
                    patch as returned by alphaearth_source.fetch_embeddings().
        timeout_s: Compute-bound adapter inference timeout (seconds).
                   Defaults to ML_INFERENCE_TIMEOUT_S env var or 2.5 s.
                   This is the Issue #18 acceptance criterion, explicitly scoped
                   to compute only (see ADR 0002 §Latency Criterion).

    Returns:
        Binary water mask: np.ndarray bool, shape (H, W).

    Raises:
        InferenceUnavailableError: model unavailable, timeout exceeded, or ONNX error.
        ValueError: if embeddings has wrong shape.
    """
    if embeddings.ndim != 3 or embeddings.shape[0] != _EXPECTED_INPUT_CHANNELS:
        raise ValueError(
            f"embeddings must have shape (64, H, W), got {embeddings.shape}. "
            "Use alphaearth_source.fetch_embeddings() to produce the input."
        )

    h, w = embeddings.shape[1], embeddings.shape[2]

    # Build (1, 64, H, W) float32 tensor
    input_tensor = embeddings.astype(np.float32)[np.newaxis, ...]  # (1, 64, H, W)

    session = _load_session()

    if timeout_s is None:
        timeout_s = float(
            os.environ.get("ML_INFERENCE_TIMEOUT_S", _DEFAULT_INFERENCE_TIMEOUT_S)
        )

    input_name = session.get_inputs()[0].name
    t0 = time.monotonic()
    try:
        outputs = session.run(None, {input_name: input_tensor})
    except Exception as exc:
        raise InferenceUnavailableError(f"ONNX adapter inference failed: {exc}") from exc

    elapsed = time.monotonic() - t0
    if elapsed > timeout_s:
        raise InferenceUnavailableError(
            f"Adapter inference exceeded timeout ({elapsed:.2f} s > {timeout_s:.2f} s). "
            "Falling back to NDWI. Consider quantizing the adapter or increasing "
            "ML_INFERENCE_TIMEOUT_S."
        )

    logger.debug(
        "Adapter inference completed in %.3f s for patch (%d×%d)", elapsed, h, w
    )

    # Post-process: sigmoid → binary mask
    logits = outputs[0]  # (1, 1, H, W) or (1, H, W)
    if logits.ndim == 4:
        logits = logits[0, 0]  # → (H, W)
    elif logits.ndim == 3:
        logits = logits[0]  # → (H, W)

    probs = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
    return (probs >= _SIGMOID_THRESHOLD).astype(bool)


def water_area_km2_from_mask(mask: np.ndarray, pixel_area_m2: float = _PIXEL_AREA_M2) -> float:
    """Compute water area in km² from a binary water mask.

    Args:
        mask: Boolean (H × W) array. True = water pixel.
        pixel_area_m2: Pixel area in m². Default 100 m² (10 m × 10 m).

    Returns:
        Water area in km².
    """
    return float(mask.sum()) * pixel_area_m2 / 1_000_000


__all__ = [
    "InferenceUnavailableError",
    "is_available",
    "reset_session",
    "run_inference",
    "water_area_km2_from_mask",
]
