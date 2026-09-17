# GitHub Milestones & Issues: AI/ML Architecture & Pipeline Improvement Roadmap

This document breaks down the approved AI/ML and engineering advancements for **CryoHealth-geo** into copy-pasteable GitHub Milestones and actionable Issues, aligned with the approved 5-feature implementation plan.

---

## 🎯 Milestone Overview

| Milestone | Title | Target Timeline | Objective |
| :--- | :--- | :--- | :--- |
| **M1** | `Phase 1: Multi-Sensor SAR & Meteorological Forcing` | Sprint 1 (Weeks 1–4) | All-weather Sentinel-1 SAR water detection and Open-Meteo thermal forcing. |
| **M2** | `Phase 2: AlphaEarth Segmentation & Hydro Flow-Paths` | Sprint 2 (Weeks 5–8) | AlphaEarth-powered glacial lake segmentation and DEM D8 flow-path flood exposure corridor modeling. |
| **M3** | `Phase 3: Predictive AI Forecasting` | Sprint 3 (Weeks 9–10) | Empirically calibrated 14-day predictive lake expansion forecasting. |

---

# 📦 Milestone 1: Multi-Sensor SAR & Meteorological Forcing

**Milestone Name:** `Phase 1: Multi-Sensor SAR & Meteorological Forcing`  
**Description:** Overcome Karakoram summer monsoon cloud cover using Sentinel-1 C-band SAR radar imagery and integrate real-time thermal melting data into hazard scoring.

---

### 📝 Issue 1.1: Sentinel-1 SAR Radar Ingestion Pipeline (Cloud-Penetrating Water Detection)
- **Title:** `[Feature] Implement Sentinel-1 SAR (GRD) ingestion and backscatter thresholding in pipeline/sar_source.py`
- **Milestone:** `Phase 1: Multi-Sensor SAR & Meteorological Forcing`
- **Labels:** `feature`, `earth-observation`, `sar`, `radar`

#### Problem Statement
Monsoon cloud cover (often reaching 60–80% cloud cover between June and September) obscures optical Sentinel-2 imagery during peak melting periods when GLOF risks are highest.

#### Proposed Changes
- Create `pipeline/sar_source.py` implementing `SceneSource` for Sentinel-1 Ground Range Detected (GRD) C-band data (VV and VH polarizations).
- Implement radiometric calibration, speckle filtering (Lee filter / Gamma MAP), and dual-polarization water thresholding ($VV < -14\text{ dB}$ and $VH < -21\text{ dB}$).
- Merge SAR-derived water extents with optical NDWI extents in `pipeline/batch.py` with cloud masking metadata.
- No database schema changes — writes into the existing `observations` table with `source="sentinel1-sar"`.

#### Implementation Tasks
- [ ] Create `pipeline/sar_source.py` with STAC discovery for `sentinel-1-grd`.
- [ ] Implement speckle filtering and Otsu / adaptive bimodal thresholding for SAR water extraction.
- [ ] Wire `sar_source` into `pipeline/batch.py` to automatically trigger when Sentinel-2 scene cloud cover exceeds 40%.
- [ ] Add unit tests with synthetic radar backscatter rasters in `tests/test_sar_source.py`.

#### Acceptance Criteria
- [ ] Pipeline extracts water surface area from Sentinel-1 scenes even during 100% cloud cover.
- [ ] SAR water extent error within ±10% of cloud-free Sentinel-2 optical ground truth.
- [ ] Zero changes required to `CryoHealth-api` contract or database schema.

---

### 📝 Issue 1.2: Meteorological & Thermal Melt Forcing Integration
- **Title:** `[Feature] Integrate Open-Meteo temperature/precipitation forcing into hazard scoring`
- **Milestone:** `Phase 1: Multi-Sensor SAR & Meteorological Forcing`
- **Labels:** `feature`, `meteorology`, `hazard-model`

#### Problem Statement
Current hazard scores in `pipeline/hazard.py` only use historical area growth without considering active heatwaves or freezing-level shifts that rapidly trigger glacier melt and dam destabilization.

#### Proposed Changes
- Create `pipeline/weather.py` fetching 7-day temperature history/forecast, freezing level altitude (0°C isotherm), and daily maximum temperature anomalies via the Open-Meteo Free API.
- Implement an in-memory 6-hour TTL cache to avoid redundant API queries during batch passes and testing.
- Update `pipeline/hazard.py` to accept an optional `thermal_forcing` parameter and apply a thermal risk multiplier (e.g., 1.2x if freezing level exceeds 5,000m or daily max is >+2.5°C above the 30-day baseline), keeping the total score bounded in [0, 1].
- Document the formula in `docs/HAZARD_METHODOLOGY.md`.

#### Implementation Tasks
- [ ] Implement `pipeline/weather.py` with caching and request rate limiting.
- [ ] Update `pipeline/hazard.py` to incorporate the meteorological sub-index into the composite hazard score.
- [ ] Update `pipeline/hazard_batch.py` to query lake-coordinate weather metrics.
- [ ] Add unit tests in `tests/test_weather.py`, including edge cases for missing weather data.

#### Acceptance Criteria
- [ ] Temperature anomalies >+4°C above seasonal mean dynamically raise the hazard score.
- [ ] Graceful degradation: if the weather API is unavailable, hazard calculations fall back to optical-only scoring without failing the batch.
- [ ] Zero changes required to `CryoHealth-api` contract or database schema.

---

# 📦 Milestone 2: AlphaEarth Segmentation & Hydro Flow-Paths

**Milestone Name:** `Phase 2: AlphaEarth Segmentation & Flow-Path Modeling`  
**Description:** Replace empirical band-ratio thresholds with AlphaEarth-powered water boundary segmentation, and replace circular exposure buffers with true hydrological valley flow accumulation corridors.

---

### 📝 Issue 2.1: Glacial Water Boundary Segmentation via AlphaEarth Embeddings
- **Title:** `[AI/ML] Integrate AlphaEarth foundation model embeddings & fine-tuned segmentation adapter for high-turbidity glacial water extraction`
- **Milestone:** `Phase 2: AlphaEarth Segmentation & Flow-Path Modeling`
- **Labels:** `ai-ml`, `alphaearth`, `foundation-models`, `segmentation`, `training-required`

#### Problem Statement
Fixed NDWI zero-threshold boundaries fail in complex high-altitude terrain due to deep mountain shadows (misclassified as water), turbid glacial rock-flour silt (low reflectance), and mixed slush/ice lake boundaries.

**Note:** unlike other features in this roadmap, this requires a trained model, labeled data, and inference infrastructure — a deliberate scope increase from the original zero-training implementation plan, approved separately.

#### Proposed Changes
- Create `pipeline/alphaearth_source.py` and `pipeline/ml_segmentation.py` with ONNX / TensorRT runtime support.
- Ingest pre-trained AlphaEarth geospatial foundation model embeddings (multi-spectral + SAR + DEM fusion) to extract semantic representations of high-altitude glacial landscapes.
- Attach a lightweight segmentation decoder/adapter on top of AlphaEarth embeddings, fine-tuned on High Mountain Asia (HMA) glacial lake datasets.
- Fall back to standard NDWI spectral thresholding if model inference is disabled or GPU/CPU resources exceed defined thresholds.
- Document training data provenance and inference infrastructure requirements (GPU/CPU, expected cost) in `docs/AI_MODEL_CARD.md`.

#### Implementation Tasks
- [ ] Source or produce labeled HMA glacial lake training data; document provenance.
- [ ] Package AlphaEarth embedding extraction pipeline and fine-tuned segmentation adapter weights into repository asset storage / S3 bucket.
- [ ] Implement multi-spectral patch inference (B2, B3, B4, B8, B11, SAR VV/VH, DEM) through the AlphaEarth feature extractor in `pipeline/ml_segmentation.py`.
- [ ] Add boundary polygon smoothing and vectorization to GeoJSON in `pipeline/ndwi.py`.
- [ ] Add test suite with benchmark accuracy metrics in `tests/test_ml_segmentation.py`.
- [ ] Document inference infrastructure requirements and cost implications in `docs/AI_MODEL_CARD.md`.

#### Acceptance Criteria
- [ ] Segmentation precision and recall on shadow-affected test lakes exceeds 95% IoU with AlphaEarth embeddings.
- [ ] Inference runtime < 2.5 seconds per lake AOI.
- [ ] NDWI fallback verified to trigger correctly when inference is unavailable.
- [ ] Zero changes required to `CryoHealth-api` contract or database schema.

---

### 📝 Issue 2.2: Topographic Hydrological Valley Flow-Path Modeling (D8 Inundation Corridor)
- **Title:** `[Geospatial] Replace radial buffer exposure calculation with DEM D8 flow accumulation modeling`
- **Milestone:** `Phase 2: AlphaEarth Segmentation & Flow-Path Modeling`
- **Labels:** `geospatial`, `hydrology`, `dem`, `exposure`

#### Problem Statement
`pipeline/exposure.py` currently estimates downstream vulnerable population using a radial Euclidean buffer around the lake. In steep alpine valleys, floodwaves follow canyon channels — radial buffers overestimate non-valley populations while underestimating downstream valley settlements 15–30km away.

#### Proposed Changes
- Refactor `pipeline/exposure.py` using the Copernicus GLO-30 DEM.
- Implement D8 flow direction and downslope path accumulation from the lake outlet, 20–50 km downstream.
- Intersect the resulting hydrological inundation corridor (e.g. 500m valley width along the flow path) with the local WorldPop population raster.
- Preserve the existing `population_within_buffer()` interface for backward compatibility.

#### Implementation Tasks
- [ ] Implement DEM flow direction, pit filling, and flow accumulation routines in `pipeline/dem.py`.
- [ ] Compute downstream flowline geometry up to 50 km from the lake outlet in `pipeline/exposure.py`.
- [ ] Calculate true downstream exposed population and critical infrastructure counts.
- [ ] Update unit tests in `tests/test_exposure.py` and benchmark against historical GLOF flood paths.

#### Acceptance Criteria
- [ ] Population exposure is calculated strictly along the downstream river corridor.
- [ ] Verified against historical GLOF inundation footprint for Shishper and Passu lakes.
- [ ] Zero changes required to `CryoHealth-api` contract or database schema.

---

# 📦 Milestone 3: Predictive AI Forecasting

**Milestone Name:** `Phase 3: Predictive AI Forecasting`  
**Description:** Enable proactive early warnings with an empirically calibrated 14-day predictive lake expansion model.

---

### 📝 Issue 3.1: Predictive 14-Day Growth Forecasting & Empirical Calibration
- **Title:** `[AI/ML] Build 14-day predictive lake surface expansion model with Prophet and empirical threshold calibration`
- **Milestone:** `Phase 3: Predictive AI Forecasting`
- **Labels:** `ai-ml`, `forecasting`, `time-series`

#### Problem Statement
Current hazard scoring is reactive, evaluating only historical expansion. Early warning requires a 14-day forecast so responders have lead time before a lake reaches catastrophic overtopping volume.

**Note:** Prophet is the required primary implementation for this issue. A Temporal Fusion Transformer and AlphaEarth temporal embeddings are noted as a possible future enhancement but are explicitly out of scope for this issue's acceptance criteria.

#### Proposed Changes
- Create `pipeline/forecast.py` implementing a Prophet time-series fit on historical lake area observations.
- Compute the posterior probability of expansion P(Area at T+14 > current Area); trigger risk elevation only if this probability is ≥85%.
- Gate predictions: fall back to a deterministic 30-day linear moving average if observation count < 15 or history < 180 days; discard the forecast if uncertainty bounds exceed the calibrated threshold.
- Create `scripts/calibrate_forecast_thresholds.py` to run walk-forward backtesting across the 2023–2026 backfilled observation dataset, sweeping uncertainty cutoffs (15%–60%) to empirically determine the confidence threshold and validate the 85% posterior probability cutoff.

#### Implementation Tasks
- [ ] Implement time-series pre-processing and feature engineering in `pipeline/forecast.py`.
- [ ] Implement the minimum-data gate (15 observations / 180 days) and linear moving-average fallback.
- [ ] Implement the posterior probability gate and low-confidence discard logic.
- [ ] Build `scripts/calibrate_forecast_thresholds.py` for walk-forward backtesting against real historical data.
- [ ] Integrate forecast confidence bounds into `pipeline/hazard.py`.
- [ ] Provide forecasted area endpoints in `pipeline/service.py`.
- [ ] Create unit and backtesting tests in `tests/test_forecast.py`.

#### Acceptance Criteria
- [ ] Model outputs 14-day forecasted lake area with confidence bounds (p10/p50/p90).
- [ ] Calibration script reports empirical coverage, false alarm rate, and rejection rate across the swept threshold range, with the final 85% posterior probability cutoff justified by this data (not assumed).
- [ ] Sparse-data lakes (<15 observations or <180 days) correctly fall back to the linear moving average without failing the batch.
- [ ] Zero changes required to `CryoHealth-api` contract or database schema.

---

# 📦 Future Extensions (Not Yet Approved)

**Description:** Additional ideas surfaced during roadmap planning. None are part of current-sprint scope — each requires a feasibility check, a calibration/validation plan, and separate supervisor approval before being scheduled into a sprint.

---

### 📝 Issue F.1: Self-Healing Multi-Provider STAC Failover Engine
- **Title:** `[Resilience] Build multi-provider satellite ingestion failover (CDSE <-> Planetary Computer <-> USGS)`
- **Milestone:** `Future Extensions (Not Yet Approved)`
- **Labels:** `infrastructure`, `resilience`, `stac`, `deferred`
- **Status:** Deferred — precautionary. No CDSE downtime or rate-limiting has been observed in production. Adds infrastructure complexity (circuit breaker, multi-provider band/CRS standardization) without a documented incident justifying it. Revisit if a real outage occurs.

#### Problem Statement
European Space Agency Copernicus Data Space Ecosystem (CDSE) or Planetary Computer STAC APIs periodically experience rate limits, authentication timeouts, or indexing lags, stalling daily pipeline jobs.

#### Proposed Changes
- Create a composite `MultiProviderSceneSource` in `pipeline/source_manager.py`.
- Implement automated retry with exponential backoff and circuit breaker failover: `CDSE` → `Planetary Computer` → `Earth Search (Element84 AWS)`.

#### Implementation Tasks
- [ ] Implement provider circuit breaker and fallback router in `pipeline/source_manager.py`.
- [ ] Standardize band-name mapping and coordinate system reprojections across providers.
- [ ] Add integration tests in `tests/test_source_manager.py` verifying fallback upon simulated 503/429 HTTP status codes.

#### Acceptance Criteria
- [ ] Ingestion automatically switches to secondary STAC provider without throwing unhandled exceptions.
- [ ] Metrics log provider health and failover events.

---

### 📝 Issue F.2: InSAR Moraine Deformation & Glacier Velocity Tracking
- **Title:** `[Feature] Integrate InSAR surface velocity and moraine displacement tracking into dynamic hazard scoring`
- **Milestone:** `Future Extensions (Not Yet Approved)`
- **Labels:** `feature`, `insar`, `glaciology`, `hazard-model`, `deferred`
- **Status:** Deferred — technically demanding (phase unwrapping, atmospheric correction, coherence loss over snow/ice), unvalidated data coverage/revisit frequency over target lakes, and no calibration plan for the 0.5 m/day surge threshold (same trap the forecast threshold avoided via backtesting). Needs a feasibility check and a calibration plan before scoping into a sprint.

#### Problem Statement
Ice-dammed lakes burst because ice dams surge and fracture. Dam stability is currently static metadata. Dynamic dam displacement velocities are needed to anticipate breach events weeks before failure.

#### Proposed Changes
- Ingest ITS_LIVE / Copernicus Sentinel-1 InSAR surface velocity grids for glaciers and moraine dams.
- Extract displacement velocity vectors ($v > 0.5\text{ m/day}$ surge indicator) in `pipeline/hazard.py`.
- Incorporate dynamic dam stability factor into hazard computation.

#### Implementation Tasks
- [ ] Verify ITS_LIVE / Sentinel-1 InSAR data coverage and revisit frequency for target lakes (feasibility check).
- [ ] Create `pipeline/insar.py` to query velocity grids around lake dam coordinates.
- [ ] Add displacement threshold metrics to `pipeline/hazard.py`.
- [ ] Design a walk-forward-style calibration plan for the surge threshold, similar to Issue 3.1's forecast calibration.
- [ ] Update `docs/HAZARD_METHODOLOGY.md` with deformation formulas and weight allocations.
- [ ] Write unit tests with mocked velocity fields in `tests/test_insar.py`.

#### Acceptance Criteria
- [ ] Glacial dam surge acceleration triggers an elevated dam stability risk score.
- [ ] Surge threshold is empirically justified via calibration, not assumed.

---

### 📝 Issue F.3: AlphaEarth Latent Embedding Drift for Moraine Dam Seepage & Failure Anomaly Detection
- **Title:** `[AI/ML] Implement unsupervised anomaly detection for moraine seepage using AlphaEarth latent embedding drift`
- **Milestone:** `Future Extensions (Not Yet Approved)`
- **Labels:** `ai-ml`, `alphaearth`, `anomaly-detection`, `spectral-analysis`, `deferred`
- **Status:** Deferred — not in the approved plan; a third distinct use of AlphaEarth (alongside segmentation and the forecasting stretch goal) that needs deliberate sequencing rather than parallel adoption. Scarce labeled failure examples make validation difficult.

#### Problem Statement
Sub-surface moraine piping or sudden internal drainage causes rapid spectral and moisture shifts on the dam face before full catastrophic dam collapse occurs. Labeled failure examples are extremely scarce.

#### Proposed Changes
- Create `pipeline/anomaly.py` utilizing AlphaEarth zero-shot / few-shot latent representation embeddings.
- Compute cosine distance and Mahalanobis distance between current dam-face embeddings and historical baseline distributions.
- Flag sudden anomalous moisture increases or structural shifts on the outer moraine slope ($>3\sigma$ drift) as potential piping/seepage.

#### Implementation Tasks
- [ ] Extract multi-band dam face patch embeddings using AlphaEarth in `pipeline/anomaly.py`.
- [ ] Compute latent drift metrics across historical cloud-free scene baselines.
- [ ] Output anomaly score and trigger critical inspection alerts if seepage anomaly threshold exceeds $3\sigma$.
- [ ] Add unit tests in `tests/test_anomaly.py`.

#### Acceptance Criteria
- [ ] Successfully detects subtle moisture leakage and pre-breach dam face deformation without requiring large supervised training sets.
- [ ] Low false positive rate (<1.5% on baseline historical scenes).

---

### 📝 Issue F.4: Event-Driven Satellite Acquisition (STAC Notifications & Webhooks)
- **Title:** `[Automation] Implement event-driven STAC Webhook triggers for immediate near-real-time processing`
- **Milestone:** `Future Extensions (Not Yet Approved)`
- **Labels:** `automation`, `fastapi`, `webhooks`, `deferred`
- **Status:** Deferred — not in the approved plan. Reasonable automation improvement, revisit once core features are stable.

#### Problem Statement
Relying solely on once-daily cron polling delays alert generation by up to 24 hours after a satellite has completed an orbit pass and published data.

#### Proposed Changes
- Expose an authenticated webhook endpoint `POST /webhooks/stac-event` in `pipeline/service.py`.
- Listen for Copernicus Data Space / AWS SNS notifications for new Sentinel-2 and Sentinel-1 acquisitions intersecting Gilgit-Baltistan bounding boxes.
- Enqueue immediate lake observation and hazard scoring upon receipt of webhook.

#### Implementation Tasks
- [ ] Add webhook payload parser and HMAC-SHA256 signature verification in `pipeline/service.py`.
- [ ] Trigger background asynchronous lake worker without blocking the webhook response.
- [ ] Add tests for webhook payload ingestion in `tests/test_service.py`.

#### Acceptance Criteria
- [ ] Valid STAC webhook triggers an end-to-end lake observation within 3 minutes of scene availability.
- [ ] Unauthenticated or malformed webhook requests are rejected with 401/400 status.

---

### 📝 Issue F.5: Automated LLM Situation Reports & Multi-Channel Alert Dispatcher
- **Title:** `[Feature] Generate automated LLM executive situation briefings (PDF/JSON) and multi-channel dispatching`
- **Milestone:** `Future Extensions (Not Yet Approved)`
- **Labels:** `feature`, `genai`, `reporting`, `alerts`, `deferred`
- **Status:** Deferred — not in the approved plan. Introduces a new external LLM API dependency (cost/reliability) not addressed anywhere in current scope.

#### Problem Statement
Emergency response agencies (NDMA/PDMA/District Administration) require concise, natural-language situation reports detailing why a lake changed tier, projected flood arrival times, and action checklists.

#### Proposed Changes
- Create `pipeline/report_generator.py` utilizing structured LLM output (Gemini / Claude / Llama 3) to generate natural language executive briefings from hazard telemetry.
- Generate high-resolution PDF briefings containing true-color satellite imagery, NDWI water boundary contours, 90-day time-series growth charts, and downstream flowline maps.
- Provide webhook integration for instant alert dispatch to WhatsApp Business API, Telegram, and SMS gateways.

#### Implementation Tasks
- [ ] Implement prompt templates and structured JSON schema parsing in `pipeline/report_generator.py`.
- [ ] Implement automated report rendering with Matplotlib plots and ReportLab/WeasyPrint PDF generator.
- [ ] Wire multi-channel alert webhook notifications in `pipeline/hazard_client.py`.
- [ ] Add test cases verifying report generation in `tests/test_report_generator.py`.

#### Acceptance Criteria
- [ ] Generates complete PDF situation brief within 5 seconds of a tier change to "High" or "Critical".
- [ ] Natural language summary correctly explains the driving factors (e.g. "Area grew by 24% over 30 days during active +3.5°C heatwave").