"""Sentinel-1 GRD (Ground Range Detected) C-band SAR scene source and water extraction.

Why this exists: monsoon cloud cover over the Karakoram / Hindukush routinely reaches
60–80% between June and September, precisely when melt-driven GLOF risk peaks and
optical Sentinel-2 NDWI observations drop out. SAR backscatter penetrates cloud and
precipitation, so a Sentinel-1 GRD IW-mode scene taken within ±3 days of the cloudy
optical window still gives a usable water-extent estimate.

Processing chain, applied in order inside extract_sar_water:
  1. Radiometric calibration: raw DN² → linear sigma-naught (σ₀), then → dB.
     Sentinel-1 GRD COGs on Planetary Computer are already calibrated to σ₀ in linear
     scale (float32); this module handles both the already-calibrated float32 case and
     the raw-uint16 amplitude case (amplitude → power → dB).
  2. Speckle filtering: Lee filter (default) or Gamma MAP — both vectorised NumPy,
     no SciPy dependency. Window size 5×5, ENL=4.4 (standard for IW GRD).
  3. Water thresholding: dual-polarisation fixed thresholds (VV < –14 dB AND VH < –21 dB)
     as the primary method; Otsu adaptive threshold available as a fallback for each band.

Threshold literature references (dual_pol_threshold docstring repeats these):
  - Pulvirenti et al. (2011) "An algorithm for operational flood mapping from
    Synthetic Aperture Radar (SAR) data using fuzzy logic", Natural Hazards and Earth
    System Sciences, 11(2), 529–540.  → –14 dB VV, –21 dB VH for open water.
  - ESA Climate Change Initiative Water Bodies product ATBD v2.1 (2020).

Pixel area: Sentinel-1 IW GRD is processed to 10×10 m ground range resolution
(same footprint as Sentinel-2 10 m bands) — pixel area constant matches ndwi.py.

# TODO(validation): compare SAR water extents against paired cloud-free S2 optical
# scenes from Shishper 2022 clear-sky window before this feeds a production hazard
# score. Synthetic tests in tests/test_sar_source.py confirm arithmetic correctness
# only. The ±10% acceptance criterion requires a real-world validation pass.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import planetary_computer
import rasterio
from pystac_client import Client
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds

from pipeline.stac_source import SceneRef

# ---------------------------------------------------------------------------
# STAC / source constants
# ---------------------------------------------------------------------------

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "sentinel-1-grd"
# Sentinel-1 IW GRD is posted at 10 m ground range resolution.
SAR_PIXEL_AREA_M2 = 10 * 10

# Speckle filter defaults — IW GRD nominal ENL.
_DEFAULT_WINDOW = 5
_DEFAULT_ENL = 4.4  # Equivalent Number of Looks for IW GRD (ESA documentation).

# Dual-polarisation fixed-threshold water detection defaults (Pulvirenti et al. 2011).
_VV_THRESH_DB = -14.0
_VH_THRESH_DB = -21.0


# ---------------------------------------------------------------------------
# Scene source
# ---------------------------------------------------------------------------


class Sentinel1GRDSource:
    """Planetary Computer–backed source for Sentinel-1 GRD IW dual-pol scenes.

    Returns SceneRef objects whose ``assets`` dict contains at least "vv" and "vh"
    keys (lowercase, matching the PC asset naming).  No authentication required.

    find_recent_scenes / find_scenes_in_range satisfy the SceneSource protocol so
    this can be passed as ``sar_source`` to pipeline/batch.py.
    """

    def __init__(self) -> None:
        self._client = Client.open(STAC_URL)

    def find_recent_scenes(
        self, bbox: tuple[float, float, float, float], limit: int = 12
    ) -> list[SceneRef]:
        return self._search(bbox, max_items=limit)

    def find_scenes_in_range(
        self, bbox: tuple[float, float, float, float], start: date, end: date
    ) -> list[SceneRef]:
        return self._search(
            bbox,
            max_items=None,
            datetime_range=f"{start.isoformat()}/{end.isoformat()}",
        )

    def _search(
        self,
        bbox: tuple[float, float, float, float],
        max_items: int | None,
        datetime_range: str | None = None,
    ) -> list[SceneRef]:
        # Filter to IW mode dual-pol (VV+VH) scenes only — these are the only
        # mode that provides both polarisations needed for dual-pol thresholding.
        search = self._client.search(
            collections=[COLLECTION],
            bbox=bbox,
            datetime=datetime_range,
            sortby=[{"field": "properties.datetime", "direction": "desc"}],
            max_items=max_items,
            query={
                "sar:instrument_mode": {"eq": "IW"},
                "sar:polarizations": {"contains": ["VV", "VH"]},
            },
        )
        refs = []
        for item in search.items():
            signed = planetary_computer.sign(item)
            assets = {k: a.href for k, a in signed.assets.items()}
            # Sentinel-1 GRD has no scene-level cloud cover — SAR is cloud-transparent.
            refs.append(
                SceneRef(
                    scene_id=item.id,
                    captured_at=item.datetime.date(),
                    cloud_cover_pct=0.0,
                    assets=assets,
                )
            )
        return refs

    def read_bands(
        self,
        scene: SceneRef,
        bands: list[str],
        bbox: tuple[float, float, float, float],
    ) -> dict[str, np.ndarray]:
        """Read SAR bands from Sentinel-1 GRD COGs.

        ``bands`` should be a subset of the asset keys available on the scene
        (typically ``["vv", "vh"]``).  The returned arrays are windowed to ``bbox``
        (WGS84) and pixel-aligned — same contract as all other SceneSource
        implementations so batch.py never needs to know which source it's talking to.

        Arrays are returned in linear power scale (float32 σ₀) ready for conversion
        to dB.  No speckle filtering or thresholding is applied here — those live in
        extract_sar_water so they're independently testable.
        """
        raw: dict[str, np.ndarray] = {}
        for band in bands:
            href = scene.assets[band]
            with rasterio.open(href) as src:
                utm_bbox = transform_bounds("EPSG:4326", src.crs, *bbox)
                window = from_bounds(*utm_bbox, transform=src.transform)
                data = src.read(1, window=window)
                raw[band] = data.astype(np.float32)

        # Clip all bands to the minimum shared shape in case pixel counts differ
        # by ±1 due to floating-point window rounding (same fix as stac_source.py).
        if raw:
            rows = min(a.shape[0] for a in raw.values())
            cols = min(a.shape[1] for a in raw.values())
            raw = {b: arr[:rows, :cols] for b, arr in raw.items()}

        return raw


# ---------------------------------------------------------------------------
# Radiometric conversion
# ---------------------------------------------------------------------------


def linear_to_db(linear: np.ndarray, *, is_amplitude: bool = False) -> np.ndarray:
    """Convert linear-scale SAR backscatter to decibels.

    Args:
        linear: Array of σ₀ values.  Planetary Computer Sentinel-1 GRD COGs are
                already in linear *power* scale (float32); pass ``is_amplitude=False``
                (the default).  If working with raw DN amplitude values (uint16 from
                unprocessed scenes) pass ``is_amplitude=True`` to apply the extra
                squaring step.
        is_amplitude: If True, input is amplitude (DN) and power = amplitude².

    Returns:
        float32 array in dB.  Zero-or-negative inputs (no-data / shadow areas) are
        mapped to –40 dB to avoid log(0); this is well below any real water signal.
    """
    arr = linear.astype(np.float32)
    if is_amplitude:
        # Amplitude-to-dB via the 20 log₁₀ formula (mathematically identical to
        # squaring first then 10 log₁₀, but avoids applying 20*log₁₀ to power).
        arr = np.where(arr > 0, arr, 1e-8)
        return (20.0 * np.log10(arr)).astype(np.float32)
    # Power-to-dB: 10 log₁₀(σ₀)
    arr = np.where(arr > 0, arr, 1e-8)
    return (10.0 * np.log10(arr)).astype(np.float32)



# ---------------------------------------------------------------------------
# Speckle filtering
# ---------------------------------------------------------------------------


def _box_mean_var(arr: np.ndarray, half: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute local mean and variance for each pixel using a (2*half+1)² window.

    Vectorised cumulative-sum approach — O(N) regardless of window size, avoids
    any SciPy / ndimage dependency.
    """
    cs = np.cumsum(np.cumsum(arr, axis=0), axis=1)
    cs2 = np.cumsum(np.cumsum(arr ** 2, axis=0), axis=1)
    h, w = arr.shape

    def _box_sum(c: np.ndarray) -> np.ndarray:
        # Pad so every pixel gets a full window (border pixels use clipped extent).
        r1 = np.clip(np.arange(h) - half - 1, -1, h - 1)
        r2 = np.clip(np.arange(h) + half, 0, h - 1)
        c1 = np.clip(np.arange(w) - half - 1, -1, w - 1)
        c2 = np.clip(np.arange(w) + half, 0, w - 1)

        def _get(r: np.ndarray, c_: np.ndarray) -> np.ndarray:
            ri = np.clip(r, 0, h - 1)
            ci = np.clip(c_, 0, w - 1)
            out = c[np.ix_(ri, ci)]
            # Zero out rows/cols that were originally -1 (before array start).
            out[r < 0, :] = 0
            out[:, c_ < 0] = 0
            return out

        return _get(r2, c2) - _get(r1, c2) - _get(r2, c1) + _get(r1, c1)

    # Window pixel counts (may be smaller at borders).
    r1 = np.clip(np.arange(h) - half - 1, -1, h - 1)
    r2 = np.clip(np.arange(h) + half, 0, h - 1)
    c1 = np.clip(np.arange(w) - half - 1, -1, w - 1)
    c2 = np.clip(np.arange(w) + half, 0, w - 1)
    n = np.outer((r2 - r1), (c2 - c1)).astype(np.float32)

    s = _box_sum(cs).astype(np.float32)
    s2 = _box_sum(cs2).astype(np.float32)
    mean = s / np.maximum(n, 1)
    var = np.maximum(s2 / np.maximum(n, 1) - mean ** 2, 0.0)
    return mean, var


def lee_filter(
    arr: np.ndarray,
    *,
    window_size: int = _DEFAULT_WINDOW,
    enl: float = _DEFAULT_ENL,
) -> np.ndarray:
    """Lee speckle filter for SAR imagery (linear power domain).

    Reduces multiplicative speckle while preserving edges. Operates on data in
    *linear power* scale — do NOT pass dB values.

    Lee (1980): "Digital image enhancement and noise filtering by use of local
    statistics", IEEE TPAMI, 2(2), 165–168.

    Args:
        arr: 2-D float array of linear σ₀ power values.
        window_size: Odd integer side length of the local statistics window (default 5).
        enl: Equivalent Number of Looks — controls how aggressively noise is smoothed.
             4.4 is the nominal ENL for Sentinel-1 IW GRD (ESA documentation).

    Returns:
        Speckle-filtered float32 array, same shape as ``arr``.
    """
    arr = arr.astype(np.float32)
    half = window_size // 2
    local_mean, local_var = _box_mean_var(arr, half)

    # Noise variance estimate from ENL: cu² = 1/ENL.
    cu_sq = 1.0 / enl
    # Weight: w = (σ²_local - cu²·μ²) / σ²_local
    # Clipped to [0, 1] so the filter is never more than the local mean.
    local_var_safe = np.maximum(local_var, 1e-10)
    weight = np.clip(
        (local_var - cu_sq * local_mean ** 2) / local_var_safe, 0.0, 1.0
    )
    return (local_mean + weight * (arr - local_mean)).astype(np.float32)


def gamma_map_filter(
    arr: np.ndarray,
    *,
    window_size: int = _DEFAULT_WINDOW,
    enl: float = _DEFAULT_ENL,
) -> np.ndarray:
    """Gamma MAP speckle filter for SAR imagery (linear power domain).

    Assumes a Gamma-distributed reflectivity model (appropriate for most natural
    land surfaces). Performs slightly better than Lee on homogeneous areas at the
    cost of minor additional computation.

    Lopes et al. (1990) "Maximum A Posteriori Speckle Filtering and First Order
    Texture Models in SAR Images", IGARSS, 2409–2412.

    Args:
        arr: 2-D float array of linear σ₀ power values.
        window_size: Odd integer side length (default 5).
        enl: Equivalent Number of Looks (default 4.4 for S1 IW GRD).

    Returns:
        Speckle-filtered float32 array, same shape as ``arr``.
    """
    arr = arr.astype(np.float32)
    half = window_size // 2
    local_mean, local_var = _box_mean_var(arr, half)

    # Gamma MAP weight:  b = (1 + 1/ENL) / (local_var/mean² + 1/ENL)
    cu_sq = 1.0 / enl
    local_mean_safe = np.maximum(local_mean, 1e-10)
    ci_sq = local_var / (local_mean_safe ** 2)
    b = (1.0 + cu_sq) / (ci_sq + cu_sq)
    b = np.clip(b, 0.0, 1.0)
    return (b * arr + (1.0 - b) * local_mean).astype(np.float32)


# ---------------------------------------------------------------------------
# Water thresholding
# ---------------------------------------------------------------------------


def otsu_threshold(
    band_db: np.ndarray,
    *,
    min_val: float = -30.0,
    max_val: float = 0.0,
    bins: int = 256,
) -> float:
    """Otsu (1979) adaptive bimodal threshold in dB space.

    Maximises inter-class variance between the two assumed Gaussian populations
    (open water at low backscatter vs land / vegetation at higher backscatter).
    Operates on valid dB values in [``min_val``, ``max_val``]; pixels outside that
    range are excluded from the histogram (they are likely no-data shadows or
    saturated returns).

    Returns:
        Threshold in dB. Pixels *below* this value are classified as water.
        Falls back to -14.0 dB (VV default) if the histogram is degenerate
        (all pixels in one bin).
    """
    flat = band_db.ravel().astype(np.float32)
    flat = flat[(flat >= min_val) & (flat <= max_val)]
    if flat.size == 0:
        return _VV_THRESH_DB  # safe fallback

    counts, bin_edges = np.histogram(flat, bins=bins, range=(min_val, max_val))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    total = counts.sum()
    if total == 0:
        return _VV_THRESH_DB

    # Efficient vectorised Otsu via cumulative sums.
    cumsum_n = np.cumsum(counts)
    cumsum_w = np.cumsum(counts * bin_centers)

    n1 = cumsum_n[:-1].astype(np.float64)
    n2 = total - n1
    valid = (n1 > 0) & (n2 > 0)
    if not valid.any():
        return _VV_THRESH_DB

    mu1 = cumsum_w[:-1] / np.maximum(n1, 1)
    mu2 = (cumsum_w[-1] - cumsum_w[:-1]) / np.maximum(n2, 1)

    with np.errstate(invalid="ignore"):
        sigma_b2 = np.where(valid, n1 * n2 * (mu1 - mu2) ** 2 / total ** 2, 0.0)

    best_idx = int(np.argmax(sigma_b2))
    return float(bin_centers[best_idx])


def dual_pol_threshold(
    vv_db: np.ndarray,
    vh_db: np.ndarray,
    *,
    vv_thresh: float = _VV_THRESH_DB,
    vh_thresh: float = _VH_THRESH_DB,
) -> np.ndarray:
    """Dual-polarisation fixed-threshold water mask.

    Returns a boolean array where ``True`` == open water pixel, defined as:
        VV < vv_thresh (default –14 dB)  AND  VH < vh_thresh (default –21 dB)

    The AND condition (requiring both bands to satisfy their threshold) reduces
    false positives from specular reflectors (corner reflectors, urban metallic
    surfaces) which can have very low VH but higher VV.

    Threshold references:
      - Pulvirenti, L. et al. (2011) "An algorithm for operational flood mapping
        from Synthetic Aperture Radar (SAR) data using fuzzy logic." Natural
        Hazards and Earth System Sciences, 11(2), 529–540.
        DOI: 10.5194/nhess-11-529-2011.
      - ESA Climate Change Initiative Water Bodies ATBD v2.1 (2020), Section 3.2.

    Args:
        vv_db: VV-polarisation backscatter in dB (post speckle-filter).
        vh_db: VH-polarisation backscatter in dB (post speckle-filter).
        vv_thresh: VV water threshold in dB (default –14.0).
        vh_thresh: VH water threshold in dB (default –21.0).

    Returns:
        Boolean ndarray, same shape as inputs. True = water.

    # TODO(validation): compare against paired S2/S1 scenes from Shishper 2022
    # clear-sky window before this feeds a production hazard score.
    # See issue acceptance criteria — synthetic tests satisfy code correctness
    # only; ±10% real-world accuracy requires a separate validation pass.
    """
    return (vv_db < vv_thresh) & (vh_db < vh_thresh)


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------


def extract_sar_water(
    vv_raw: np.ndarray,
    vh_raw: np.ndarray,
    *,
    method: str = "dual_pol",
    filter_type: str = "lee",
    window_size: int = _DEFAULT_WINDOW,
    enl: float = _DEFAULT_ENL,
    vv_thresh: float | None = None,
    vh_thresh: float | None = None,
    is_amplitude: bool = False,
) -> np.ndarray:
    """End-to-end SAR water extraction pipeline.

    Applies (in order): radiometric dB conversion → speckle filtering →
    water thresholding.

    Args:
        vv_raw: Raw VV backscatter from ``Sentinel1GRDSource.read_bands``
                (linear σ₀ float32, or amplitude uint16 if ``is_amplitude=True``).
        vh_raw: Raw VH backscatter, same scale as ``vv_raw``.
        method: ``"dual_pol"`` (default) — fixed VV+VH threshold.
                ``"otsu_vv"`` — Otsu threshold on VV only.
                ``"otsu_dual"`` — independent Otsu thresholds on both bands combined.
        filter_type: ``"lee"`` (default) or ``"gamma_map"``.
        window_size: Speckle filter kernel size (default 5).
        enl: Equivalent Number of Looks (default 4.4).
        vv_thresh: Override VV threshold for ``dual_pol`` method (dB).
        vh_thresh: Override VH threshold for ``dual_pol`` method (dB).
        is_amplitude: Set True if inputs are raw amplitude DN (uint16) rather than
                      linear power σ₀ (float32 from COGs).

    Returns:
        Boolean water mask array (True = water), same shape as inputs.

    Raises:
        ValueError: If an unrecognised ``method`` or ``filter_type`` is given.
    """
    if filter_type not in ("lee", "gamma_map"):
        raise ValueError(f"Unknown filter_type {filter_type!r}; choose 'lee' or 'gamma_map'")
    if method not in ("dual_pol", "otsu_vv", "otsu_dual"):
        raise ValueError(f"Unknown method {method!r}; choose 'dual_pol', 'otsu_vv', or 'otsu_dual'")

    # Step 1 — Convert to dB.
    vv_db = linear_to_db(vv_raw, is_amplitude=is_amplitude)
    vh_db = linear_to_db(vh_raw, is_amplitude=is_amplitude)

    # Step 2 — Speckle filter (applied in dB domain; equivalent to log-domain
    # filtering which is standard practice for multiplicative speckle reduction).
    _filter = lee_filter if filter_type == "lee" else gamma_map_filter

    # Filters operate on linear power — convert back, filter, convert again.
    # This keeps the filter's statistical assumptions correct.
    vv_lin = 10.0 ** (vv_db / 10.0)
    vh_lin = 10.0 ** (vh_db / 10.0)
    vv_filt = _filter(vv_lin, window_size=window_size, enl=enl)
    vh_filt = _filter(vh_lin, window_size=window_size, enl=enl)
    vv_db = linear_to_db(vv_filt)
    vh_db = linear_to_db(vh_filt)

    # Step 3 — Threshold.
    if method == "dual_pol":
        return dual_pol_threshold(
            vv_db,
            vh_db,
            vv_thresh=vv_thresh if vv_thresh is not None else _VV_THRESH_DB,
            vh_thresh=vh_thresh if vh_thresh is not None else _VH_THRESH_DB,
        )
    elif method == "otsu_vv":
        t = otsu_threshold(vv_db)
        return vv_db < t
    else:  # otsu_dual
        t_vv = otsu_threshold(vv_db)
        t_vh = otsu_threshold(vh_db)
        return (vv_db < t_vv) & (vh_db < t_vh)


def sar_water_area_km2(
    water_mask: np.ndarray, *, pixel_size_m: float = 10.0
) -> float:
    """Surface area of water pixels in km².

    Args:
        water_mask: Boolean 2-D array from ``extract_sar_water`` or
                    ``dual_pol_threshold``. True = water pixel.
        pixel_size_m: Ground sample distance in metres (default 10 m for IW GRD).

    Returns:
        Float area in km².
    """
    pixel_area_km2 = (pixel_size_m ** 2) / 1_000_000.0
    return float(water_mask.sum()) * pixel_area_km2


__all__ = [
    "Sentinel1GRDSource",
    "dual_pol_threshold",
    "extract_sar_water",
    "gamma_map_filter",
    "lee_filter",
    "linear_to_db",
    "otsu_threshold",
    "sar_water_area_km2",
]
