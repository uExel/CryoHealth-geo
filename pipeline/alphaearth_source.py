"""AlphaEarth embedding ingestion from Google Earth Engine.

Queries the public AlphaEarth Satellite Embedding dataset
(GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL) for a lake AOI bounding box and year,
returning a 64-band numpy array at 10 m resolution. No model weights are run
locally — Google DeepMind has already computed and published these embeddings
as an analysis-ready Earth Engine ImageCollection.

Design contract (mirrors ndwi.py / stac_source.py philosophy):
- All Earth Engine I/O is encapsulated here — callers receive a plain numpy array.
- Raises EmbeddingUnavailableError for all failure modes so callers can cleanly
  fall back to the NDWI path without catching bare exceptions.
- Embeddings are cached in-memory by (bbox, year) with a 24-hour TTL, matching
  meteo.py's _inputs_cache pattern. Annual data is immutable within a year, so
  re-fetching during backfill runs wastes GEE quota unnecessarily.

See docs/ai/decisions/0002-ml-segmentation.md (ADR 0002) for the full rationale.
See docs/AI_MODEL_CARD.md for dataset specification.

TODO(#19-training): once the segmentation adapter is trained and weights are
available, the embeddings fetched here feed directly into ml_segmentation.run_inference().

Environment variables (all optional — NDWI fallback used when absent):
    EE_SERVICE_ACCOUNT       Google service account email for EE auth.
    EE_PRIVATE_KEY_JSON      Path to service account JSON key file.
                             If neither is set, falls back to Application Default
                             Credentials (gcloud auth application-default login).
    EE_FETCH_TIMEOUT_S       Wall-clock timeout for GEE fetch (default 8 s).
                             Governs the network-bound fetch separately from the
                             2.5 s compute-bound adapter inference target — see ADR 0002.
"""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Verified GEE collection ID — confirmed against Earth Engine Data Catalog.
_GEE_COLLECTION = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"

# AlphaEarth produces 64-dimensional embeddings.
_N_EMBEDDING_BANDS = 64
_EMBEDDING_BAND_NAMES = [f"embedding_{i}" for i in range(_N_EMBEDDING_BANDS)]

# Default fetch timeout (seconds). Governs network-bound GEE call only.
# See ADR 0002 §Latency Criterion for the explicit split vs the 2.5 s adapter target.
_DEFAULT_FETCH_TIMEOUT_S = 8.0

# ---------------------------------------------------------------------------
# In-memory embedding cache (Issue 2 — matches meteo.py pattern)
# ---------------------------------------------------------------------------
# Annual embeddings are immutable within a year. Cache for 24 hours so that
# run_backfill() processing N scenes per lake per year makes only one GEE call.

_CACHE_TTL_SECONDS = 24 * 3600  # 24h — longer than meteo's 6h; data is truly static

# Key: (bbox_tuple, year: int)  Value: (cached_at: datetime, embeddings: np.ndarray)
_embedding_cache: dict[tuple, tuple[datetime, np.ndarray]] = {}


def clear_embedding_cache() -> None:
    """Clear the in-memory embedding cache. Intended for testing."""
    _embedding_cache.clear()


def _cache_get(key: tuple) -> np.ndarray | None:
    entry = _embedding_cache.get(key)
    if entry is None:
        return None
    cached_at, value = entry
    if (datetime.now(UTC) - cached_at).total_seconds() > _CACHE_TTL_SECONDS:
        del _embedding_cache[key]
        return None
    return value


def _cache_set(key: tuple, value: np.ndarray) -> None:
    _embedding_cache[key] = (datetime.now(UTC), value)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class EmbeddingUnavailableError(RuntimeError):
    """Raised when AlphaEarth embeddings cannot be fetched.

    Callers should catch this and fall back to the NDWI path.
    Possible causes:
    - earthengine-api not installed.
    - EE credentials not configured and no Application Default Credentials.
    - GEE fetch exceeded EE_FETCH_TIMEOUT_S.
    - GEE quota exceeded or API error.
    - No embedding data for the requested bbox / year.
    """


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------


def _authenticate_ee() -> None:
    """Initialise the Earth Engine API.

    Prefers service account credentials (EE_SERVICE_ACCOUNT + EE_PRIVATE_KEY_JSON).
    Falls back to Application Default Credentials if neither is set.

    Raises:
        EmbeddingUnavailableError: if authentication fails.
    """
    try:
        import ee  # noqa: PLC0415 — optional heavy dep, loaded lazily
    except ImportError as exc:
        raise EmbeddingUnavailableError(
            "earthengine-api is not installed. Install it with: "
            "uv pip install 'cryohealth-geo[ml]'. "
            "The NDWI fallback will be used instead."
        ) from exc

    if ee.data._credentials is not None:  # type: ignore[attr-defined]
        # Already initialised in this process — no-op.
        return

    service_account = os.environ.get("EE_SERVICE_ACCOUNT", "")
    key_path = os.environ.get("EE_PRIVATE_KEY_JSON", "")

    try:
        if service_account and key_path:
            credentials = ee.ServiceAccountCredentials(service_account, key_path)
            ee.Initialize(credentials)
            logger.debug("Earth Engine initialised via service account: %s", service_account)
        else:
            # Application Default Credentials (gcloud auth application-default login)
            ee.Initialize()
            logger.debug("Earth Engine initialised via Application Default Credentials")
    except Exception as exc:
        raise EmbeddingUnavailableError(
            f"Earth Engine authentication failed: {exc}. "
            "Set EE_SERVICE_ACCOUNT + EE_PRIVATE_KEY_JSON, or run "
            "'gcloud auth application-default login'."
        ) from exc


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def is_available() -> bool:
    """Return True if Earth Engine embeddings can be attempted.

    Checks:
    1. earthengine-api is importable.
    2. Either EE credentials env vars are set, or Application Default Credentials
       appear to be configured (existence of the ADC file is not checked — this is
       a fast probe, not a live connection test).

    Does NOT make a network call.
    """
    try:
        import ee  # noqa: F401,PLC0415
    except ImportError:
        return False

    # Consider available if service account creds are set OR if we might have ADC.
    service_account = os.environ.get("EE_SERVICE_ACCOUNT", "")
    key_path = os.environ.get("EE_PRIVATE_KEY_JSON", "")
    if service_account and key_path:
        return True

    # Check for ADC file presence (standard location).
    import pathlib  # noqa: PLC0415

    adc_path = pathlib.Path.home() / ".config" / "gcloud" / "application_default_credentials.json"
    return adc_path.exists()


def fetch_embeddings(
    bbox: tuple[float, float, float, float],
    year: int,
    *,
    timeout_s: float | None = None,
) -> np.ndarray:
    """Fetch AlphaEarth 64-band embeddings for a lake AOI from Google Earth Engine.

    Queries GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL, mosaics the annual composite
    to the requested bounding box, and returns a (64, H, W) float32 array aligned
    to the AOI at 10 m resolution.

    Results are cached in-memory by (bbox, year) for 24 hours — subsequent calls
    with the same arguments return the cached array without hitting GEE again.

    Args:
        bbox: (west, south, east, north) in WGS84 degrees.
        year: Calendar year to fetch embeddings for (2017–2024).
        timeout_s: GEE fetch wall-clock timeout. Defaults to EE_FETCH_TIMEOUT_S
                   env var or _DEFAULT_FETCH_TIMEOUT_S (8 s).
                   This governs the network-bound fetch only — the 2.5 s
                   compute-bound adapter inference target is separate (ADR 0002).

    Returns:
        np.ndarray, shape (64, H, W), dtype float32.
        H and W depend on AOI size at 10 m resolution (≈ 200×200 for a ~2 km lake).

    Raises:
        EmbeddingUnavailableError: if auth fails, GEE times out, quota exceeded,
            or no data exists for the requested bbox/year. Callers fall back to NDWI.
        ValueError: if year is outside the 2017–2024 available range.
    """
    if not (2017 <= year <= 2024):
        raise ValueError(
            f"AlphaEarth embeddings are available 2017–2024; requested year={year}."
        )

    cache_key = (bbox, year)
    cached = _cache_get(cache_key)
    if cached is not None:
        logger.debug(
            "Embedding cache hit for bbox=%s year=%d (shape %s)", bbox, year, cached.shape
        )
        return cached

    if timeout_s is None:
        timeout_s = float(os.environ.get("EE_FETCH_TIMEOUT_S", _DEFAULT_FETCH_TIMEOUT_S))

    _authenticate_ee()

    try:
        import ee  # noqa: PLC0415
        import concurrent.futures  # noqa: PLC0415

        west, south, east, north = bbox
        aoi = ee.Geometry.Rectangle([west, south, east, north])

        collection = ee.ImageCollection(_GEE_COLLECTION)
        image = (
            collection
            .filterDate(f"{year}-01-01", f"{year}-12-31")
            .filterBounds(aoi)
            .mosaic()
            .select(_EMBEDDING_BAND_NAMES)
        )

        def _fetch() -> np.ndarray:
            # sampleRectangle returns a dict of band_name → 2D list of pixel values.
            sample = image.sampleRectangle(region=aoi, defaultValue=0)
            arrays = []
            for band in _EMBEDDING_BAND_NAMES:
                band_data = sample.get(band).getInfo()  # list of lists
                arrays.append(np.array(band_data, dtype=np.float32))
            return np.stack(arrays, axis=0)  # (64, H, W)

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_fetch)
            try:
                result = future.result(timeout=timeout_s)
            except concurrent.futures.TimeoutError:
                raise EmbeddingUnavailableError(
                    f"GEE fetch timed out after {timeout_s:.1f} s for bbox={bbox} year={year}. "
                    f"Increase EE_FETCH_TIMEOUT_S or check GEE API status. "
                    f"Falling back to NDWI."
                )

    except EmbeddingUnavailableError:
        raise
    except Exception as exc:
        raise EmbeddingUnavailableError(
            f"GEE fetch failed for bbox={bbox} year={year}: {exc}"
        ) from exc

    if result.shape[0] != _N_EMBEDDING_BANDS:
        raise EmbeddingUnavailableError(
            f"Expected {_N_EMBEDDING_BANDS} embedding bands, got {result.shape[0]}. "
            "Check GEE collection schema."
        )

    logger.info(
        "AlphaEarth embeddings fetched: bbox=%s year=%d shape=%s",
        bbox, year, result.shape,
    )
    _cache_set(cache_key, result)
    return result


__all__ = [
    "EmbeddingUnavailableError",
    "clear_embedding_cache",
    "fetch_embeddings",
    "is_available",
]
