# ADR 0005 — Anomaly Detection: Advisory Signal, Dam-Face AOI, and Scope

> Status: **Accepted**
> Implements: [GitHub Issue #22](https://github.com/uExel/CryoHealth-geo/issues/22)
> Module: `pipeline/anomaly.py`
> Date: 2026-09-15

---

## Context

Issue #22 proposes unsupervised anomaly detection for early moraine seepage and
sudden drainage, using spectral vectors (SWIR, Red-Edge, NDWI, MNDWI) over the dam
face. The original issue states "trigger critical inspection alerts if seepage anomaly
threshold exceeds 3σ."

This was flagged `agent:needs-human` (shaan360, 2026-09-01) because directly routing
an ML model's output to alert creation conflicts with CryoHealth-api's explicit
requirement that creating or overriding an alert requires a mandatory audited human
reason. This ADR records the decisions that resolve that flag.

---

## Decision 1 — Advisory Signal Only; No Auto-Alert

**The anomaly score surfaces as an advisory field and never directly triggers alerts.**

When `seepage_score_sigma < -SEEPAGE_SIGMA_THRESHOLD` (i.e., `seepage_flag=True`):
- `AnomalyResult.seepage_flag = True` is stored
- Result is merged into `components["anomaly"]` in the CryoHealth-api POST body
  and exposed as a top-level `anomaly` field in the `/run-hazard` response
- **No call to any alert-creation endpoint is made by `pipeline/anomaly.py` or
  `pipeline/hazard_batch.py`**

Rationale (same as ADR 0004 §"never silent ML"):

An Isolation Forest trained on historical spectral vectors has no ground-truth label
for "seepage is occurring." A low score means "this scene's spectral signature is
unusual relative to the training baseline" — which also includes cloud shadow, seasonal
ice-melt exposure, sensor calibration drift, and transient snow on the moraine face.
The model cannot distinguish between these causes. Routing such a signal directly to
an alert would violate CryoHealth-api's CLAUDE.md requirement:
*"Tier/alert policy is code + humans, never silent ML."*

The score is still operationally valuable: a responder seeing `seepage_score_sigma =
-4.2` alongside a "high" hazard tier has more information than without it. The
decision to escalate that information into a formal inspection alert — with an audited
human reason — remains with the responder.

If CryoHealth-api introduces a "flag for human review" mechanism distinct from a full
alert in the future, `seepage_flag=True` is the natural input to that mechanism. Adding
such a pathway requires a CryoHealth-api change and is out of scope for Issue #22.

---

## Decision 2 — Dam-Face AOI: Per-Lake Explicit Bbox; Moraine-Applicable Lakes Only

**Anomaly detection is physically meaningful only for moraine-dammed lakes.**

The moraine piping/seepage mechanism — sub-surface water migrating through a moraine
dam and emerging as moisture anomalies on the outer slope — requires a moraine dam
structure. Ice-dammed lakes (e.g., Shishper) and supra-glacial/englacial lakes
(e.g., Ghulkin) have different failure mechanisms. Applying the same spectral anomaly
detector to a supra-glacial lake's "dam face" (which is glacier ice, not moraine
material) would produce meaningless signals.

Dam-type applicability is determined at runtime from the `damType` field in the
`lakes` DB table — the same field already used by `hazard_batch.py` for the hazard
score's dam-type component. `anomaly.py` receives `dam_type: str` as a parameter
and returns an explicit `AnomalyResult` method string for non-applicable cases:

| `dam_type` value | `AnomalyResult.method` |
|---|---|
| `"moraine"` | `"isolation_forest"` (or a data-gate fallback) |
| `"ice"` | `"not_applicable_ice_dam"` |
| `"unknown"` | `"isolation_forest"` (applied with a WARNING log — unknown dam type is not assumed inapplicable) |
| `"bedrock"` | `"not_applicable_bedrock"` |
| Any other value | `"not_applicable_unknown_type"` with WARNING |

**Dam-face AOI**: each applicable lake requires an explicit `dam_face_bbox_deg`
in `pipeline/lakes.py` — a small rectangular bbox over the moraine outer slope,
downslope of the outlet point. This is distinct from the lake centroid AOI (which
covers the lake surface) and the outlet point (the D8 seed). The dam-face bbox must
be manually digitized per lake from satellite imagery with provenance comments
(source imagery date, confidence level), using the same discipline established for
`outlet_lon/outlet_lat` in ADR 0003.

Lakes with `dam_face_bbox_deg = None` return `method="dam_face_not_digitized"` —
the anomaly run skips that lake gracefully, and the batch continues.

**Rejected alternative (Option B — NDWI mask)**: using the existing centroid bbox
and masking out water pixels (NDWI > 0.1) to isolate surrounding terrain. Rejected
because the residual pixels include glacier ice, lateral moraines, and bedrock terrain
that is not the dam face — adding noise irrelevant to piping detection. The 400-metre
outlet offset established in ADR 0003 showed that precision matters for point-sourced
geospatial algorithms; the same reasoning applies here.

---

## Decision 3 — Per-Lake Isolation Forest; Annual Refit

**One IsolationForest model per lake** (not a single pooled model).

Each moraine's spectral baseline differs by rock type, debris cover, and snow
fraction. A pooled model would flag Badswat's dark basalt moraine as anomalous simply
because its reflectance differs from Ghulkin's lighter granite — a false positive with
no physical meaning. Per-lake models isolate each lake's spectral "normal."

**Training data gate**: minimum `MIN_TRAINING_SCENES = 20` cloud-free scenes over the
dam-face AOI (cloud fraction < 10% over the dam-face bbox, stricter than the batch's
40% threshold because anomaly detection is more sensitive to cloud contamination than
NDWI water-area estimation). Lakes below this minimum skip fitting and return
`method="insufficient_training_scenes"` at inference time.

**Cloud-free threshold**: 10% cloud fraction over the dam-face AOI specifically.
Using the tile-level cloud cover (which can be 0% even when the AOI is 100% obscured
— documented in `stac_source.py`) would contaminate the training set.

**Training window**: 2023-01-01 to 2026-01-01 (same as Issue #21's calibration
dataset, for consistency across ML components).

**Refit cadence**: annual. The moraine's spectral baseline changes slowly (geological
timescale); monthly retraining would add noise from seasonal snow and vegetation
variation without capturing real baseline drift. The sidecar JSON records
`training_date` and the fit script warns if the model is > 365 days old.

**Sidecar JSON scene provenance**: the sidecar must record
`training_scenes: [{scene_id, captured_at, cloud_fraction}]` — not just the
training_date — so any future anomaly flag can be traced to exactly which scenes
defined "normal." This is required for the same reason `HazardResult.components`
stores raw inputs: a stored score must be auditable from what's in the record alone.

---

## Decision 4 — Isolation Forest Only; Autoencoder Deferred

**Isolation Forest is the required implementation for Issue #22. Autoencoder is a
noted possible future enhancement, explicitly out of scope for this issue's acceptance
criteria.**

Rationale: an Autoencoder (neural network reconstruction model) requires training
infrastructure, significantly more data per lake to learn a reliable reconstruction,
and adds PyTorch/ONNX as a new dependency class. Isolation Forest is interpretable,
fast to fit (scikit-learn, ~50 MB), and directly satisfies the acceptance criteria.
The detector interface is designed as a swappable component so an Autoencoder can be
added later without touching `hazard_batch.py`. Same pattern as Issue #21 (Prophet
required, TFT deferred).

---

## Threshold Provenance

Both thresholds are initial candidates, explicitly marked `PENDING VALIDATION`:

- `SEEPAGE_SIGMA_THRESHOLD = 3.0` — the 3σ stated in the issue is a prior, not an
  empirical finding. The z-score is derived from the IsolationForest's
  `score_samples()` output normalized by the training set's mean and std. The
  `scripts/validate_anomaly_thresholds.py` script computes historical FPR at 2σ–5σ;
  the acceptance criterion is FPR < 2% on baseline scenes. The validated value is
  committed to `anomaly.py` with a provenance comment (script name, run date,
  FPR at chosen threshold) before merge.

- `SUDDEN_DRAINAGE_DELTA = -0.15` — MNDWI drop of 0.15 in 30 days as a proxy for
  sudden drainage. Also pending validation against the historical dataset.

Note: IsolationForest's raw `score_samples()` output is in `(-inf, 0]` (lower =
more anomalous). Converting to a z-score using training-set mean and std produces
a value comparable to "number of standard deviations from normal." The threshold
is applied to this z-score, not to the raw IF score. This conversion and its
assumptions are documented in `pipeline/anomaly.py`.

---

## Consequences

- `pipeline/hazard.py` and `compute_hazard_score()` are **not modified**
- No new alert pathways are created
- The `[anomaly]` optional extras group (scikit-learn, joblib) follows the same
  lightweight optional pattern as `[forecast]` and `[ml]`
- Six `dam_face_bbox_deg` values must be digitized for moraine-applicable lakes
  before `fit_anomaly_model.py` can run — this is a pre-merge human action
- The sidecar JSON's `training_scenes` list creates an auditable record of what
  "normal" was defined against for each lake
