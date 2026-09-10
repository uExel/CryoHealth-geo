# ADR 0003 — D8 Flow Accumulation for Downstream Population Exposure

## Status

Accepted — 2026-09-10
Author: laiba-107 (Collaborator)
Implements: [GitHub Issue #19](https://github.com/uExel/CryoHealth-geo/issues/19)
Validation sub-issue: [Issue #26](https://github.com/uExel/CryoHealth-geo/issues/26)

---

## Context

`pipeline/exposure.py` used a straight-line 5 km square buffer around each lake centroid
to estimate downstream exposed population. This approach was explicitly documented as a
simplification in the module docstring, `HAZARD_METHODOLOGY.md`, and each returned dict's
`note` field. It was always intended as a PoC placeholder pending proper flow routing.

In steep alpine valleys (all six monitored Karakoram/Gilgit-Baltistan lakes), the error
profile of the radial buffer is systematic and directional:

- **Overestimates**: includes hillside and ridge populations that a channelled GLOF
  floodwave cannot reach
- **Underestimates**: misses valley-floor settlements 10–50 km downstream that are
  within the drainage corridor but outside the 5 km radius

This directly affects the prioritization component of `HazardScore.components.exposure`
(used for triage, not the tier — PRD §8 is explicit on this).

---

## Decision

Replace the radial buffer with a **D8 flow-accumulation corridor** computed from the
Copernicus GLO-30 DEM (already used in `pipeline/dem.py` for slope). The pipeline:

1. Fetch a DEM bounding box covering the lake outlet + 50 km downstream
2. Fill depressions (Priority-Flood algorithm)
3. Compute D8 steepest-descent flow direction
4. Trace the downstream flowline from the lake outlet up to 50 km
5. Buffer the flowline by 500 m half-width → corridor mask
6. Intersect corridor mask with WorldPop 100 m raster → exposed population

---

## Algorithm Choices

### D8, not D-Infinity

D-Infinity (Tarboton 1997) distributes flow between two neighbors using an angular
interpolation. This is superior for continuous flow accumulation maps but produces a
**fractional flow path** — not a single discrete corridor. D8 assigns all flow to one
downslope neighbor, producing a single connected flowline. For the purpose of tracing
one inundation corridor (not a continuous flow distribution), D8 is the right choice.

### Pure NumPy, No External Flow-Routing Library

Available alternatives: pysheds, whitebox, TauDEM, richdem. All require either native
compiled extensions or external binary dependencies that are difficult to pin and audit.

The three D8 operations needed (pit fill, flow direction, flowline trace) are each
straightforward NumPy operations:
- Pit filling: Priority-Flood using `heapq` (stdlib) — O(n log n)
- D8 direction: vectorized 8-neighbor slope comparison with `np.pad` — O(n)
- Flowline trace: iterative cell walk following direction codes — O(path length)

No new mandatory compiled dependency is introduced. This keeps the base install clean.
`scipy.ndimage.distance_transform_edt` is used for the corridor mask in `exposure.py`
and is already an optional `[ml]` extra; a numpy fallback is provided.

### Outlet Coordinates, Not Lake Centroid

D8 flowline tracing is sensitive to the seed cell. Starting from the lake centroid
produces a meaningless path for lakes where the centroid is in open water — the lowest
point on the dam (where water exits) is the physically correct seed. For Shishper, the
ice dam toe is ~3.5 km from the centroid, which corresponds to a ~350-pixel offset in
GLO-30 — enough to route the flowline into a different tributary valley entirely.

`LakeAoi` now carries `outlet_lon` / `outlet_lat` with per-lake provenance comments.
All six outlet coordinates are manual estimates from Google Maps satellite imagery
(2026-09-10). **Visual verification required before production use** — see comment per
lake in `pipeline/lakes.py`. Shishper is flagged as dynamic (ice dam changes yearly).

### Corridor Half-Width: 500 m (Configurable)

Karakoram valley floors vary: Hassanabad Nallah is 150–400 m wide; the lower Hunza
gorges widen to 600–1200 m. A 500 m half-width (1 km total corridor) is a conservative
starting estimate that captures the valley floor without extending onto slopes.
Configurable via `EXPOSURE_CORRIDOR_HALF_WIDTH_M` env var.

### Backward-Compatible Wrapper

`population_within_buffer(lon, lat)` remains the public API called by `hazard_batch.py`.
It now internally:
1. Looks up the lake's outlet coordinates by centroid proximity (exact for 6 lakes)
2. Calls `population_along_flowline(outlet_lon, outlet_lat)`
3. Falls back to the original radial buffer on `FlowRoutingError` with a WARNING log

No changes to `hazard_batch.py` or `hazard.py`. The `population_within_buffer` key is
always present in the returned dict. A new `method` field records which path ran
(`"d8_flowline_corridor"` or `"radial_buffer_fallback"`).

---

## Validation Status (Acceptance Criterion)

The issue acceptance criterion requires validation against historical Shishper and Passu
GLOF inundation footprints. This data is not publicly available as GeoJSON/Shapefile
(confirmed: Dartmouth Flood Observatory, Copernicus EMS, HDX searched 2026-09-10).

**This PR**: delivers the D8 infrastructure. The `@pytest.mark.slow` validation test
has `TODO(#26-validation)` — the footprint sourcing is tracked in Issue #26 (sub-issue
of #19). `HAZARD_METHODOLOGY.md` explicitly states this criterion is pending #26.

---

## Consequences

- **No new mandatory dependencies** — base install unchanged; scipy optional for
  corridor mask (degrades gracefully to numpy fallback)
- **DEM fetch is larger** — full 50 km corridor bbox vs. small AOI for slope.
  Cached by `(outlet_lon, outlet_lat, max_km)` via `@functools.cache`
- **GEE/network call** — the DEM fetch hits Planetary Computer STAC (same as slope).
  For hazard batch runs that already call `mean_slope_degrees()`, the DEM tiles may
  already be cached by the HTTP client
- **HAZARD_METHODOLOGY.md version unchanged** — exposure is a prioritization component
  not in the scored formula; no METHODOLOGY_VERSION bump needed

---

## References

- `pipeline/dem.py` — existing DEM fetch + slope calculation
- `pipeline/exposure.py` — existing WorldPop fetch
- `docs/HAZARD_METHODOLOGY.md` §Exposure — updated in this PR
- Issue #19: https://github.com/uExel/CryoHealth-geo/issues/19
- Issue #26 (validation): https://github.com/uExel/CryoHealth-geo/issues/26
- Wang & Liu (2006) — Priority-Flood pit filling algorithm
- Tarboton (1997) — D-Infinity (referenced for contrast; D8 chosen instead)
