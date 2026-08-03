# Hazard index methodology

Version `1.0` (`pipeline/hazard.py`'s `METHODOLOGY_VERSION` — bump this whenever weights
or thresholds change, and update this doc in the same commit; the two must never drift).

This is a **prototype methodology** per PRD §8: a transparent, documented composite
score, not a calibrated or peer-reviewed hazard model. None of the six monitored lakes
has had a GLOF event in the observed period, so there is no labeled outcome to calibrate
weights or thresholds against yet — they're a defensible starting point, not a final
answer. Revisit if/when there's a real event to learn from.

## Reproducibility

Every `HazardScore` row's `components` field stores everything needed to recompute the
score by hand: the raw inputs, the normalized per-component risk values, the weights and
thresholds used *at the time*, and this doc's version tag. A score is reproducible from
what's in the database alone — you should never need `pipeline/hazard.py`'s source to
verify one, only this document and the stored `components`.

## Composite score

Six components, each normalized to a risk value in `[0, 1]`, combined as a weighted sum:

```
score = Σ (weight_i × risk_i)
```

| Component | Weight | Why this weight |
|---|---|---|
| Area growth rate (30d/90d) | 0.30 | The most direct precursor signal — a lake that's actively expanding is the closest thing to a real-time warning this composite has. |
| Historical GLOF record | 0.15 | A proven precedent at this specific lake is strong evidence, but it's a static fact, not a live signal. |
| Dam type | 0.20 | Moraine dams fail far more often than bedrock in the GLOF literature — a strong static predictor of *whether* an outburst can happen at all. |
| Glacier contact | 0.10 | Real but weaker on its own — matters more for whether a lake *can* keep growing than for predicting an imminent event. |
| Slope | 0.10 | Matters more for downstream energy/exposure once a breach happens than for triggering probability — weighted accordingly. |
| Seasonal anomaly | 0.15 | Real signal, but noisier and partially redundant with area growth — demoted to avoid double-counting the same underlying trend. |

Weights sum to 1.0 (enforced by an assertion in `pipeline/hazard.py`).

### Area growth rate (30d / 90d)

Fractional change in `areaKm2` from the closest observation to (today − 30 days) [or
90 days] to the latest observation, using the closest available observation within a
15-day tolerance of the target date. `None` (no risk contribution) if no observation
falls within tolerance — a missing signal is scored as zero risk, not fabricated.

Only **growth** counts as risk; a shrinking lake is not scored as more hazardous. This is
a deliberate, documented simplification: rapid shrink can mean an active drainage event
already underway, which is a different, more urgent signal than a precursor index is
designed to catch, and isn't handled by this composite (**known limitation**, not solved
here — a real-time drainage detector would be a different mechanism).

Growth is clamped to `[0, 50%]` and linearly mapped to `[0, 1]` risk (50%+ growth over
the window = maximum risk from that window). The 30d and 90d windows are averaged with
equal weight before applying the component's 0.30 weight.

### Seasonal anomaly

Fractional deviation of the latest observed area from the mean area observed in the same
calendar month in *other* years. `None` if there's no other year's data for that month
(a lake with under a year of history has no seasonal baseline yet). Only positive anomaly
(larger than the seasonal norm) counts as risk, clamped to `[0, 30%]` → `[0, 1]`.

### Dam type

| Dam type | Risk |
|---|---|
| moraine | 1.0 |
| ice | 0.8 |
| unknown | 0.5 |
| bedrock | 0.2 |

`unknown` is scored as moderate risk, not zero — an unclassified dam is not assumed safe.

### Glacier contact

`1.0` if the lake is still in contact with its parent glacier (can keep receiving
ice/water, calving-triggered surges remain possible), else `0.3` (detached lakes carry
real but lower residual risk, not none).

### Slope

Mean terrain slope (degrees) within the lake's AOI, from the Copernicus GLO-30 DEM
(`pipeline/dem.py`). Clamped to `[0°, 40°]` → `[0, 1]` risk.

### Historical GLOF record

`1.0` if `Lake.historicalGlof` is true, else `0.0`.

## Tier bands

Checked highest-first; the first threshold the score meets or exceeds wins:

| Score | Tier |
|---|---|
| ≥ 0.65 | critical |
| ≥ 0.45 | high |
| ≥ 0.25 | watch |
| < 0.25 | normal |

## Exposure (prioritization only — does not affect the tier)

Per PRD §8, downstream population exposure informs *prioritization*, not the hazard
tier itself, and is computed separately from the weighted score above.

**Scope note**: PRD §8 describes "facilities within modeled flow path buffer" — true
hydrological flow-path modeling (flow accumulation/routing from the DEM) is a
significant scope item on its own and isn't required for the tier. This methodology
computes exposure as real WorldPop population (`pipeline/exposure.py`, Pakistan, 2025,
100m, constrained product) within a **straight-line 5km square buffer** around the
lake's point coordinate — an honest, documented simplification of the eventual
flow-routed version, using real population data and a real computed number, not a
fabricated one. Stored in `components.exposure`; real flow-path modeling remains
future work.

## Known limitations (honest, not hidden)

- Thresholds are a documented starting point, not calibrated against a real GLOF event.
- Growth-only risk scoring misses rapid shrink as a possible active-drainage signal.
- Exposure uses a straight-line buffer, not real flow-path routing.
- Facilities/population *within* the buffer aren't broken out by type (school, clinic,
  settlement) — only a total population count.
- Small, sub-pixel-scale lakes (e.g. Khurdopin, ~0.01–0.17 km²) have genuinely noisy
  day-to-day area readings even at zero cloud fraction — real NDWI detection instability
  at that size, not a data pipeline bug. When the observation closest to a growth
  window's target date is one of these noisy readings, the growth percentage can be
  large and spurious; the 50% saturation clamp bounds any single such outlier's
  contribution to the same maximum as a real 50%+ growth, so it can't distort the
  composite beyond that ceiling, but it doesn't eliminate the underlying noise. (A real,
  reproducible instance of this was found and fixed during this methodology's initial
  live verification: `_observations`'s SQL query lacked `ORDER BY "capturedAt"`, so when
  two observations tied for closest-to-target-date, which one won was non-deterministic
  — and it initially picked a noisy near-zero khurdopin reading over its more
  representative same-distance neighbor. Fixed by ordering the query; ties now
  deterministically resolve to the earlier date.)
