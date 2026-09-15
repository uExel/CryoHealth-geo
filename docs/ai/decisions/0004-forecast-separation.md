# ADR 0004 — Prophet Forecast as a Separate Advisory Signal

## Status

Accepted — 2026-09-15
Author: laiba-107 (Collaborator)
Implements: [GitHub Issue #21](https://github.com/uExel/CryoHealth-geo/issues/21)
Resolves: `agent:needs-human` flag on the architecture question in that issue

---

## Context

The hazard scoring pipeline (`pipeline/hazard.py`, `compute_hazard_score()`) is a
pure-math composite of seven risk signals evaluated at the current observation date.
Two of those signals — `area_growth` (30d + 90d growth rate) and `seasonal_anomaly`
(current month vs. prior-year same-month mean) — directly measure the historical
lake area expansion series from the Observations table.

Issue #21 introduces a **14-day Prophet time-series forecast** of that same area-growth
series. The architecture question flagged in the issue was: should this forecast be
blended into `compute_hazard_score()`, or kept as a separate output?

---

## Decision

The forecast is a **separate advisory signal**, not an input to `compute_hazard_score()`.

`pipeline/forecast.py` and `compute_hazard_score()` are **completely independent**:
- `forecast.py` is not imported by `hazard.py`
- `compute_hazard_score()` signature and behavior are unchanged
- The forecast result is attached to `HazardRunResult` in `hazard_batch.py` *after*
  `compute_hazard_score()` has already returned — never as an input to it

---

## Rationale

### 1. Blending would double-count the same underlying signal

Prophet's projection extrapolates the same `area_growth` and `seasonal_anomaly`
series that already appear in the composite score. Adding a 14-day projection of
that same signal back into the score creates a loop: the score would reflect both
the historical growth *and* a model's prediction of *more* growth based on the same
history. The composite would amplify the area signal beyond its stated weight (0.27
for `area_growth` + 0.135 for `seasonal_anomaly`) without any corresponding weight
rebalancing or methodology documentation — a silent scope drift of the same type
that the additive/multiplier and AlphaEarth corrections identified and blocked.

### 2. "What is now" and "what is predicted" must stay auditable separately

`HazardScore.components` is designed so that any stored score can be reproduced
from the stored component values alone (PRD §9 R3, HAZARD_METHODOLOGY.md §Reproducibility).
A score that blends present-state signals with a 14-day probabilistic projection
collapses two different epistemic claims — what the data shows today vs. what a model
predicts for two weeks from now — into a single number that can no longer be audited
by either claim alone. Responders interpreting the tier need to know whether "critical"
reflects present measurements or a forecast.

### 3. "Never silent ML" — the ML output must be clearly labeled

The AlphaEarth ADR (0002) established: ML outputs must be separately labeled, not
dissolved into composite figures. A forecast probability is exactly the kind of
advisory signal that benefits from a named, separately reported field rather than
contributing an undeclared fraction to a score tier.

### 4. `compute_hazard_score()` is defined as pure math in the methodology contract

HAZARD_METHODOLOGY.md and CLAUDE.md establish that `compute_hazard_score()` takes
already-fetched data, contains no I/O, and produces a deterministic result from
its inputs. A time-series model fit is not deterministic in the same sense (Stan
sampling introduces variance), and injecting it into `compute_hazard_score()` would
break that contract — and the tests that depend on it.

---

## Implementation Contract

### `pipeline/forecast.py`

- `compute_forecast(observations, as_of)` — entry point
- Raises `ForecastUnavailableError` **only if `prophet` is not installed** (optional
  `[forecast]` extras group, consistent with `[ml]` from ADR 0002)
- All other errors (bad data, Prophet fit failure) are caught internally and returned
  as `ForecastResult(method="insufficient_confidence")` — never re-raised
- `PROPHET_FIT_WARN_SECONDS = 10.0` — wall-clock monitoring via `time.perf_counter()`;
  does **not** abort the fit; a slow fit logs a WARNING but its result is still used.
  Named "warn" not "timeout" because it does not enforce a timeout — naming matches
  actual behavior, consistent with the `ForecastUnavailableError` naming fix above

### `pipeline/hazard_batch.py`

- `compute_hazard_score()` called first; `compute_forecast()` called afterward
- Forecast merged into `components_with_forecast` dict *at the point of* calling
  `report_hazard_score()` — `HazardResult` object is never mutated
- `HazardRunResult.forecast: dict | None` carries the result to the `/run-hazard`
  response — a separate field, not folded into `score` or `tier`

### Two response shapes — intentional, not inconsistent

| Context | Forecast location | Why |
|---|---|---|
| CryoHealth-api `POST /alerts/hazard-scores` | `components["forecast"]` (nested) | `components` is the existing opaque JSONB blob; top-level DTO keys are fixed |
| `/run-hazard` response to callers | `forecast` at top level alongside `score`/`tier` | CryoHealth-geo's own response format, not constrained by the DTO |

This is a deliberate two-surface design, documented here to prevent a future reader
from "fixing" the shape difference.

---

## Threshold Provenance

Both `EXPANSION_PROB_THRESHOLD = 0.85` and `UNCERTAINTY_CUTOFF = 0.35` are **initial
candidates, not empirically justified values**. The calibration script
(`scripts/calibrate_forecast_thresholds.py`) sweeps:

- `prob_threshold`: 0.50–0.99 (11 points)
- `uncertainty_cutoff`: 0.15–0.60 (10 points)

Both with a boundary-detection guard: if the recommended value lands at the min or max
of the swept range, the script prints a warning that the range must be widened before
trusting the result.

Final chosen values must be committed to `forecast.py` with the calibration run date
and the output CSV filename in a comment — same provenance discipline as the SAR
`-14 dB` / `-21 dB` thresholds and the AlphaEarth training data provenance in the
model card. The acceptance criterion in Issue #21 is explicit on this: the 85% cutoff
must be *justified by data*, not assumed.

---

## Consequences

- `compute_hazard_score()` has zero changes — pure-math contract maintained
- `METHODOLOGY_VERSION` is unchanged — no tier formula change
- `HazardResult` (frozen dataclass) is not mutated
- Prophet is an optional `[forecast]` extra — base install and all prior functionality
  unaffected when not installed
- Runtime: Prophet fit is ~3–8 s per lake (benchmark); 6-lake daily batch adds ~50 s
  total, well within the 24-hour window. `PROPHET_FIT_WARN_SECONDS = 10.0` monitors
  for outliers
- Calibration is a prerequisite for the acceptance criterion — thresholds in the
  initial PR are explicitly marked "pending calibration" in code comments

---

## References

- Issue #21: https://github.com/uExel/CryoHealth-geo/issues/21
- ADR 0002: `docs/ai/decisions/0002-ml-segmentation.md` (optional extras pattern,
  "never silent ML" precedent)
- `pipeline/forecast.py`, `scripts/calibrate_forecast_thresholds.py`
- `docs/HAZARD_METHODOLOGY.md` (pure-math contract, reproducibility requirement)
