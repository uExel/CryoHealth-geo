# AI Model Card — Moraine Dam-Face Anomaly Detection v1

> Implements: [GitHub Issue #22](https://github.com/uExel/CryoHealth-geo/issues/22)
> ADR: [ADR 0005](ai/decisions/0005-anomaly-detection.md)
> Modules: `pipeline/anomaly.py`, `scripts/fit_anomaly_model.py`, `scripts/validate_anomaly_thresholds.py`
> Last updated: 2026-09-15

> [!IMPORTANT]
> **Advisory Signal Only — Never Auto-Triggers Alerts.**
> Per ADR 0005 and CryoHealth safety policy (*"Tier/alert policy is code + humans, never silent ML"*),
> anomaly scores and flags (`seepage_flag=True`, `drainage_flag=True`) surface as advisory metadata
> only. They never directly trigger emergency inspection alerts or modify the deterministic
> `compute_hazard_score()`. Escalation requires an audited human reason.

---

## System Overview

```
Sentinel-2 L2A (dam_face_bbox_deg)
  → Bands: B03 (Green), B04 (Red), B05 (RedEdge), B08 (NIR), B11 (SWIR1), B12 (SWIR2)
  → Cloud Mask: SCL cloud/shadow filtering (max 10% cloud over dam-face AOI)
  → Indices: NDWI = (B03 - B08)/(B03 + B08), MNDWI = (B03 - B11)/(B03 + B11)
  → Patch vector extraction: (N_unmasked_pixels, 6) → Mean-pooled scene vector (1, 6)
  → Per-Lake Isolation Forest (.joblib + sidecar .json)
  → score_samples() → Normalized to z-score (sigma) against training distribution
  → Seepage signal: sigma < -SEEPAGE_SIGMA_THRESHOLD (3.0σ)
  → Sudden drainage signal: MNDWI_today - MNDWI_30d_prior < SUDDEN_DRAINAGE_DELTA (-0.15)
  → Result: AnomalyResult attached to HazardRunResult and components["anomaly"]
```

---

## Model Architecture & Training

| Property | Value |
|---|---|
| **Model Type** | Unsupervised Isolation Forest (`sklearn.ensemble.IsolationForest`) |
| **Strategy** | **Per-lake** models (each lake has a distinct moraine lithology, aspect, and baseline) |
| **Features (6-D)** | `[SWIR1(B11), SWIR2(B12), RedEdge(B05), NDWI, MNDWI, Red(B04)]` |
| **Hyperparameters** | `n_estimators=100`, `contamination=0.05`, `random_state=42` |
| **Training Input** | Historical cloud-free scenes (2023–2026 backfill, `cloud_fraction < 0.10`) |
| **Training Gate** | `MIN_TRAINING_SCENES = 20` (skips lake with warning if insufficient) |
| **Inference Gate** | `MIN_SCENES_FOR_INFERENCE = 3` (reduces single-scene transient noise) |
| **Storage** | `.cache/anomaly/{slug}_isolation_forest.joblib` + `{slug}_isolation_forest.json` sidecar |

---

## Model Sidecar & Provenance

To guarantee auditability in disaster-risk governance, every fitted model generates a `.json` sidecar detailing the exact training baseline:

```json
{
  "slug": "passu",
  "fitted_at": "2026-09-15T12:00:00Z",
  "n_training_scenes": 28,
  "n_estimators": 100,
  "contamination": 0.05,
  "training_mean_score": -0.4852,
  "training_std_score": 0.0410,
  "band_means": [412.3, 498.1, 310.4, 0.12, 0.18, 260.5],
  "band_stds": [22.1, 18.4, 15.2, 0.03, 0.03, 14.1],
  "training_scenes": [
    {"scene_id": "S2A_MSIL2A_20230601...", "captured_at": "2023-06-01", "cloud_fraction": 0.02}
  ]
}
```

This provenance allows investigators to audit exactly which scenes established "normal" if an anomalous flag is challenged.

---

## Dam-Type Applicability (ADR 0005 §Decision 2)

Moraine piping and seepage detection is physically valid **only** on unconsolidated moraine dam slopes. Applying spectral soil-moisture anomaly detection to ice-dammed or supra-glacial lakes produces uninterpretable noise.

Runtime behavior is gated on the `damType` attribute from the database (`LakeStaticInputs.dam_type`):

| `dam_type` | Action | `AnomalyResult.method` |
|---|---|---|
| `"moraine"` | Isolation Forest inference | `"isolation_forest"` |
| `"ice"` | Bypassed (failure mode is ice fracture/buoyancy) | `"not_applicable_ice_dam"` |
| `"bedrock"` | Bypassed (no piping failure mode) | `"not_applicable_bedrock"` |
| `"unknown"` | Evaluated with a log warning | `"isolation_forest"` |
| Lake has no `dam_face_bbox_deg` | Skipped | `"dam_face_not_digitized"` |

---

## Precursor Signals

### 1. Seepage / Piping Precursor
- **Mechanism:** Water infiltrating internal moraine drainage channels saturates the outer toe, increasing surface moisture and shifting SWIR/MNDWI signatures.
- **Metric:** `seepage_score_sigma = (score - training_mean) / training_std`.
- **Flag Condition:** `seepage_score_sigma < -SEEPAGE_SIGMA_THRESHOLD` (default: `3.0σ`).
- **Target FPR:** `< 2.0%` false positive rate on baseline historical non-event scenes.

### 2. Sudden Drainage Precursor
- **Mechanism:** Rapid tunnel formation or moraine breach rapidly drains water, sharply dropping MNDWI across the lake/dam interface.
- **Metric:** `drainage_delta_mndwi = MNDWI_today - MNDWI_30d_prior`.
- **Flag Condition:** `drainage_delta_mndwi < SUDDEN_DRAINAGE_DELTA` (default: `-0.15`).

---

## Retraining & Operational Cadence

- **Offline Training:** Run annually via `scripts/fit_anomaly_model.py`.
- **Threshold Calibration:** Verified via walk-forward hold-out validation with `scripts/validate_anomaly_thresholds.py`.
- **Staleness Warnings:**
  - If existing model is `< 30 days` old: fitting warns against premature refit unless `--force` is used.
  - If existing model is `> 365 days` old: fitting and batch execution warn that an annual refit is due.

---

## Error Handling & Fallbacks

Identical two-category contract as `pipeline/forecast.py`:

| Condition | Behavior | Caller Impact |
|---|---|---|
| `scikit-learn` or `joblib` missing | Raises `AnomalyUnavailableError` | `hazard_batch.py` logs WARNING; batch continues without anomaly |
| Model `.joblib` or `.json` missing | Raises `AnomalyUnavailableError` | `hazard_batch.py` logs WARNING; batch continues without anomaly |
| Dam face not digitized (`None`) | Returns `AnomalyResult(method="dam_face_not_digitized")` | Clean advisory result |
| Ice / Bedrock dam type | Returns `AnomalyResult(method="not_applicable_...")` | Clean advisory result |
| Too cloudy / < 3 inference scenes | Returns `AnomalyResult(method="insufficient_scenes")` | Clean advisory result |
| STAC fetch / scoring exception | Returns `AnomalyResult(method="score_failed")` | Clean advisory result |

`pipeline/hazard.py` and `compute_hazard_score()` remain 100% pure math with zero I/O and zero dependency on `pipeline/anomaly.py`.
