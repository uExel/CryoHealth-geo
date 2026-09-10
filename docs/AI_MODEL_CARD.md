# AI Model Card — AlphaEarth Segmentation Adapter v1

> Implements: [GitHub Issue #18](https://github.com/uExel/CryoHealth-geo/issues/18)
> ADR: [ADR 0002](ai/decisions/0002-ml-segmentation.md)
> Modules: `pipeline/alphaearth_source.py`, `pipeline/ml_segmentation.py`
> Last updated: 2026-09-10

> [!IMPORTANT]
> **The 92% IoU acceptance criterion from Issue #18 is NOT yet met.**
> This PR delivers the full inference infrastructure and a fallback-safe stub.
> Adapter training and IoU verification are tracked in Issue #19.

---

## System Overview

```
GEE: GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL
  → 64-band AlphaEarth embeddings (10 m, precomputed globally)
  → ONNX segmentation adapter (trained on HMA lake labels — Issue #19)
  → Binary water mask → area_km2
  → observations.source = "alphaearth-ml-seg"

Fallback (always available):
  → McFeeters NDWI (pipeline/ndwi.py, pure math)
  → observations.source = "sentinel2-ndwi"
```

---

## AlphaEarth Embedding Dataset

| Property | Value |
|---|---|
| **Earth Engine collection** | `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL` |
| **Dimensions** | 64 bands (`embedding_0` … `embedding_63`) |
| **Resolution** | 10 m |
| **Temporal coverage** | Annual, 2017–2024 (updated annually) |
| **Coverage** | Global |
| **Access** | Public — standard `earthengine-api`, no approval required |
| **Source model** | Google DeepMind AlphaEarth Foundations (precomputed by Google; not run locally) |

---

## Segmentation Adapter (Stub — pending Issue #19)

| Property | Value |
|---|---|
| **Input** | `(1, 64, H, W)` float32 — AlphaEarth embeddings |
| **Output** | `(H, W)` float32 logit → sigmoid → binary mask (threshold 0.5) |
| **Architecture** | Lightweight decoder head (2–4 conv layers, ~1–2 MB) |
| **Format** | ONNX Runtime ≥ 1.17, opset 17 |
| **Weights** | Not yet trained — pending Issue #19 |
| **IoU target** | ≥ 0.92 on shadow-affected HMA test set — **not yet verified** |

---

## Latency Targets (Split per ADR 0002 §Latency Criterion)

| Component | Target | Nature | Env override |
|---|---|---|---|
| GEE embedding fetch | ≤ 8 s (configurable) | Network-bound | `EE_FETCH_TIMEOUT_S` |
| ONNX adapter inference | **< 2.5 s** on CPU | Compute-bound | `ML_INFERENCE_TIMEOUT_S` |

The original Issue #18 criterion ("< 2.5 s per lake AOI") is now explicitly scoped
to adapter inference only. See ADR 0002 §Latency Criterion for the rationale.

---

## Caching

Embedding fetches are cached in-memory by `(bbox, year)` with a 24-hour TTL
(data is immutable within a year). This prevents redundant GEE calls during
`run_backfill()` which processes multiple scenes per lake per year.

---

## Fallback Behavior

```
ae_is_available() AND ml_is_available()
    → fetch embeddings → run adapter → source = "alphaearth-ml-seg"
    → on EmbeddingUnavailableError or InferenceUnavailableError:
        → NDWI fallback → source = "sentinel2-ndwi"
Otherwise (no credentials / no weights):
    → NDWI fallback directly → source = "sentinel2-ndwi"
```

Every observation written to the DB records which path produced it. No silent ML.

---

## Known Failure Modes

| Failure Mode | Impact | Mitigation |
|---|---|---|
| GEE API timeout or quota | Embedding fetch fails | `EmbeddingUnavailableError` → NDWI fallback |
| No adapter weights (current state) | Inference path unavailable | `is_available()` returns False → NDWI fallback |
| Year outside 2017–2024 | No GEE data | `ValueError` raised at fetch; caller logs and continues |
| EE credentials missing | Auth fails | `EmbeddingUnavailableError` → NDWI fallback |

---

## Infrastructure Requirements

| Component | Requirement |
|---|---|
| Python | ≥ 3.12 |
| earthengine-api | ≥ 0.1.390 |
| onnxruntime | ≥ 1.17 (CPU) or onnxruntime-gpu |
| scipy | ≥ 1.14 (for boundary smoothing) |
| GEE credentials | Service account JSON or `gcloud auth application-default login` |
| Adapter weights | Not yet available — pending Issue #19 |

Install ML extras: `uv pip install 'cryohealth-geo[ml]'`

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `EE_SERVICE_ACCOUNT` | _(unset)_ | GEE service account email |
| `EE_PRIVATE_KEY_JSON` | _(unset)_ | Path to service account key JSON |
| `EE_FETCH_TIMEOUT_S` | `8` | GEE fetch timeout (seconds) |
| `ML_MODEL_PATH` | _(unset — NDWI fallback)_ | Path to ONNX adapter weights |
| `ML_INFERENCE_TIMEOUT_S` | `2.5` | Adapter inference timeout (seconds) |
| `ML_USE_CUDA` | `0` | Set to `1` for CUDA execution provider |

---

## Pending (Issue #19)

- [ ] Source / produce labeled HMA glacial lake training data
- [ ] Train lightweight ONNX decoder on AlphaEarth 64-band embeddings
- [ ] Verify IoU ≥ 0.92 on shadow-affected test set (n ≥ 20 lakes, 2023–2025)
- [ ] Verify adapter inference < 2.5 s on CPU
- [ ] Upload weights to S3, set `ML_MODEL_PATH` in production environment

---

## Contact

Module author: laiba-107
Repository: [uExel/CryoHealth-geo](https://github.com/uExel/CryoHealth-geo)
Issue #18: https://github.com/uExel/CryoHealth-geo/issues/18
Issue #19: (to be opened)
