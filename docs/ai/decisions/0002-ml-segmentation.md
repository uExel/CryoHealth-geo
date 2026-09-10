# ADR 0002 — ML Segmentation via AlphaEarth Embeddings for Glacial Water Extraction

## Status

Accepted — 2026-09-10
Author: laiba-107 (Collaborator)
Implements: [GitHub Issue #18](https://github.com/uExel/CryoHealth-geo/issues/18)
Follow-up: Issue #19 — HMA adapter training and 92% IoU verification

---

## Context

`pipeline/ndwi.py` uses McFeeters (1996) NDWI with a zero threshold to detect open
water in Sentinel-2 L2A scenes. This approach has been explicitly designed as "pure
math, no I/O" (see `CLAUDE.md`) and works well for clear, unambiguous glacial lakes.

Three documented failure modes degrade accuracy in High Mountain Asia (HMA) terrain:

1. **Deep mountain shadows** — near-zero reflectance in all bands satisfies
   `(Green − NIR) / (Green + NIR) > 0`, producing false-positive water pixels.
2. **Turbid glacial rock-flour silt** — fine sediment suppresses reflectance at the
   lake margin; the NDWI water signal fades before the true shoreline, causing
   systematic underestimation of lake extent.
3. **Mixed slush/ice-lake boundaries** — partially melted snow at lake margins
   straddles the NDWI zero boundary non-deterministically between scenes, inflating
   apparent seasonal variance in growth-rate calculations.

These errors propagate into `pipeline/hazard.py`'s growth-rate score, potentially
producing spurious high-tier GLOF alerts. An ML segmentation approach using
pre-computed geospatial embeddings has been proposed to resolve these failure modes.

---

## Decision

**Augment the NDWI water-extraction path with a two-stage ML pipeline:**

1. **AlphaEarth embedding fetch** (`pipeline/alphaearth_source.py`): query the
   Google Earth Engine dataset `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL` for each lake
   AOI. This dataset provides precomputed 64-dimensional embeddings at 10 m resolution,
   globally available 2017–2024, derived from Google DeepMind's AlphaEarth Foundations
   model. No model weights are run locally — Google has already computed and published
   these embeddings as an analysis-ready Earth Engine ImageCollection.

2. **Segmentation adapter inference** (`pipeline/ml_segmentation.py`): run a
   lightweight ONNX decoder head trained on HMA glacial lake labels on top of the
   AlphaEarth embeddings to produce a binary water mask.

Subject to the following constraints:

### Constraints

| Constraint | Rationale |
|---|---|
| NDWI path remains the **default fallback** | If either the GEE fetch or adapter inference is unavailable, the pipeline degrades to pure-math NDWI rather than erroring or producing no output |
| `pipeline/ndwi.py` remains **pure numpy math** | Boundary smoothing and vectorization go in new `pipeline/geom.py` — ndwi.py's zero-dependency contract is intact |
| ML path is **audited per observation** | `observations.source` is `"alphaearth-ml-seg"` vs `"sentinel2-ndwi"` — no silent ML |
| Model output feeds only **water area (km²)** | Same scalar contract as NDWI today — no new DB fields, no new schema |
| **Embeddings are cached** in-memory by `(bbox, year)` | Annual data is static; re-fetching for each scene in a backfill run would waste GEE quota unnecessarily |

### Why this satisfies the "never silent ML" alerting policy

1. The ML model outputs a **binary water mask** — same type as NDWI. No tier decision.
2. Tier decisions remain entirely in `pipeline/hazard.py` (pure deterministic math).
3. The `source` column in the DB records provenance for every written observation.
4. If the ML path is unavailable, NDWI produces the observation — nothing is silently suppressed.

---

## Latency Criterion (Explicit Redefinition — Issue 1)

The original issue #18 acceptance criterion states *"inference runtime < 2.5 seconds
per lake AOI on CPU."* This criterion is ambiguous about whether it includes network
fetch time. This ADR formally splits it into two independently governed targets:

| Component | Target | Nature | On Breach |
|---|---|---|---|
| GEE embedding fetch | ≤ `EE_FETCH_TIMEOUT_S` (default **8 s**) | Network-bound — not controllable | `EmbeddingUnavailableError` → NDWI fallback |
| ONNX adapter inference | **< 2.5 s** on CPU | Compute-bound — controllable | `InferenceUnavailableError` → NDWI fallback |

The original "< 2.5 s" criterion is retained as-is for the compute portion (adapter
inference), which is the portion this service can actually control. The network fetch
is governed by a configurable timeout rather than a hard acceptance criterion, since
GEE API latency is outside this service's control.

**This redefinition is recorded here explicitly** — it is not a silent code choice.

---

## Caching Strategy (Issue 2)

AlphaEarth annual embeddings are immutable within a year: the same `(bbox, year)` pair
always returns the same 64-band array. `run_backfill()` processes N optical scenes per
lake per year, which without a cache would trigger N identical GEE API calls.

`pipeline/alphaearth_source.py` uses an in-memory cache keyed by `(bbox, year)` with
a 24-hour TTL, modelled on the `meteo.py` `_inputs_cache` pattern:

```python
_embedding_cache: dict[tuple, tuple[datetime, np.ndarray]] = {}
_CACHE_TTL_SECONDS = 24 * 3600  # annual data; longer TTL than meteo's 6h
```

A `clear_embedding_cache()` function is exported for tests, matching `clear_meteo_cache()`.

---

## Two-PR Split (Issue 3)

The 92% IoU acceptance criterion **cannot be verified** without a trained adapter.
Rather than defer this silently, this plan explicitly splits work into two tracked items:

**This PR (Issue #18):**
- Full infrastructure: fetch → embed → adapt → mask → fallback
- Adapter path guarded by `ml_segmentation.is_available()` — falls back to NDWI at
  merge time (no weights yet)
- `# TODO(#19-training): replace stub with real adapter weights` visible in module docstring
- This ADR and `docs/AI_MODEL_CARD.md` explicitly state the 92% IoU criterion is **not
  yet met** — it is verified in the follow-up

**Follow-up Issue #19:**
- Source / produce labeled HMA training data
- Train and export ONNX adapter decoder on AlphaEarth 64-band embeddings
- Verify IoU ≥ 0.92 on shadow-affected test set
- Upload weights to S3, set `ML_MODEL_PATH` in production environment
- Close the acceptance criterion left open by this PR

---

## AlphaEarth Dataset Specification

| Property | Value |
|---|---|
| Earth Engine collection | `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL` |
| Embedding dimensions | 64 bands (named `embedding_0` … `embedding_63`) |
| Spatial resolution | 10 m |
| Temporal coverage | Annual, 2017–2024 (updated annually) |
| Coverage | Global |
| Access | Public — standard `earthengine-api`, no special approval required |
| Source model | Google DeepMind AlphaEarth Foundations (precomputed, not run locally) |

---

## Segmentation Adapter Specification (Stub — pending Issue #19)

| Property | Value |
|---|---|
| Input | `(1, 64, H, W)` float32 — AlphaEarth embeddings |
| Output | `(H, W)` float32 logit → sigmoid → binary mask |
| Architecture | Lightweight decoder head (2–4 conv layers, ~1–2 MB) |
| Format | ONNX Runtime ≥ 1.17, opset 17 |
| Weights | Not yet trained — see Issue #19 |
| IoU target | ≥ 0.92 on shadow-affected HMA test set — **not yet verified** |

---

## Consequences

- **New optional dependency group `[ml]`**: `earthengine-api>=0.1.390`,
  `onnxruntime>=1.17`, `scipy>=1.14`. Base install unchanged; NDWI fallback always available.
- **New env vars**: `EE_SERVICE_ACCOUNT`, `EE_PRIVATE_KEY_JSON`, `EE_FETCH_TIMEOUT_S`,
  `ML_MODEL_PATH`, `ML_INFERENCE_TIMEOUT_S`. Documented in `.env.example`.
- **No DB schema changes** — `observations.source` already accepts free-form strings.
- **No CryoHealth-api changes** — scalar `area_km2` contract unchanged.

---

## References

- `CLAUDE.md` — documents `ndwi.py` as "pure math"
- `docs/HAZARD_METHODOLOGY.md` — hazard scoring consumes `area_km2`
- `pipeline/meteo.py` — cache pattern (`_inputs_cache`, TTL, `clear_meteo_cache`)
- Issue #18: https://github.com/uExel/CryoHealth-geo/issues/18
- Issue #19: (to be opened — HMA adapter training + IoU verification)
- GEE Dataset: https://developers.google.com/earth-engine/datasets
