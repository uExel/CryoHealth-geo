"""The same six individually-cited lakes as CryoHealth-api's seed data — ported, not
re-derived, so the two services never disagree about where a lake actually is. See
CryoHealth-api/src/lakes/data/lakes.seed-data.ts and CryoHealth-api ADR 0002 for why
it's six lakes and not the full 25-lake shortlist, and the citation for each coordinate.

outlet_lon / outlet_lat (added Issue #19) — the dam-toe / outlet point used as the D8
flow-routing seed. Distinct from the lake centroid: a GLOF propagates from the lowest
point of the dam structure, not the lake's geometric centre. A centroid-to-outlet offset
of hundreds of metres translates directly to a different drainage channel in D8 routing
(see ADR 0003). All outlet coordinates are manual estimates from Google Maps satellite
imagery (2026-09-10) — individual provenance notes are per-lake below.
REVIEW REQUIRED before production: verify all six visually against current satellite
imagery, especially Shishper (ice dam — outlet position changes with surge cycle).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LakeAoi:
    slug: str
    name: str
    lon: float   # Lake centroid longitude, WGS84
    lat: float   # Lake centroid latitude, WGS84
    # Dam outlet / toe coordinates — D8 flow-routing seed (Issue #19, ADR 0003).
    # Required: no default. Must be explicitly set per lake with provenance comment.
    outlet_lon: float
    outlet_lat: float
    # Half-width of the centroid AOI bounding box in degrees. ~0.01deg is roughly 1.1km
    # at this latitude — enough margin around a point coordinate for a small alpine lake
    # without pulling in unrelated terrain. Real production AOIs should come from the
    # lake's actual boundary polygon (CryoHealth-api's Lake.boundary, currently unused)
    # once one is digitized — this is a deliberate PoC simplification.
    half_width_deg: float = 0.01

    def bbox(self) -> tuple[float, float, float, float]:
        return (
            self.lon - self.half_width_deg,
            self.lat - self.half_width_deg,
            self.lon + self.half_width_deg,
            self.lat + self.half_width_deg,
        )


LAKES: dict[str, LakeAoi] = {
    aoi.slug: aoi
    for aoi in [
        LakeAoi(
            "shishper", "Shishper (Hassanabad) glacial lake",
            lon=74.61, lat=36.40,
            # Outlet: toe of the Shishper Glacier surge lobe where it dams the
            # Hassanabad Nallah (ice-dammed lake — NOT a fixed moraine dam).
            # The ice-dam position is dynamic: it advances/retreats with the
            # surge cycle, so this coordinate represents the approximate 2022–2025
            # dam position near the glacier-nallah confluence.
            # Source: manual digitizing from Google Maps satellite imagery, 2026-09-10.
            # Cross-reference: study area in literature at ~36.42°N, 74.50°E–74.61°E
            # (Muhammad et al. 2023, Mondal et al. 2026).
            # Confidence: LOW — ice dam is dynamic; re-verify annually.
            # TODO: verify against latest Planet/Sentinel-2 imagery before production.
            outlet_lon=74.552, outlet_lat=36.421,
        ),
        LakeAoi(
            "khurdopin", "Khurdopin glacial lake",
            lon=75.5083, lat=36.3383,
            # Outlet: approximate moraine/ice dam toe at the lower end of the
            # Khurdopin surge lake where it drains into the Shimshal River.
            # Source: manual digitizing from Google Maps satellite imagery, 2026-09-10.
            # Confidence: MEDIUM — moraine dam, relatively stable position.
            # TODO: verify visually before production use.
            outlet_lon=75.498, outlet_lat=36.330,
        ),
        LakeAoi(
            "badswat", "Badswat glacial lake",
            lon=74.0775, lat=36.4855,
            # Outlet: approximate moraine dam toe; lake drains toward the
            # Bualtar/Hispar drainage.
            # Source: manual digitizing from Google Maps satellite imagery, 2026-09-10.
            # Confidence: MEDIUM.
            # TODO: verify visually before production use.
            outlet_lon=74.070, outlet_lat=36.482,
        ),
        LakeAoi(
            "passu", "Passu glacial pond",
            lon=74.77, lat=36.47,
            # Outlet: small cirque pond; outlet close to centroid given lake size
            # (~0.05–0.08 km²). Dam toe drains toward the Hunza River gorge.
            # Source: manual digitizing from Google Maps satellite imagery, 2026-09-10.
            # Confidence: MEDIUM (small offset from centroid expected).
            # TODO: verify visually before production use.
            outlet_lon=74.773, outlet_lat=36.466,
        ),
        LakeAoi(
            "ghulkin", "Ghulkin glacial pond",
            lon=74.856, lat=36.4653,
            # Outlet: small pond near Ghulkin Glacier snout; outlet on the
            # downvalley moraine edge draining toward the Hunza River.
            # Source: manual digitizing from Google Maps satellite imagery, 2026-09-10.
            # Confidence: MEDIUM.
            # TODO: verify visually before production use.
            outlet_lon=74.852, outlet_lat=36.462,
        ),
        LakeAoi(
            "batura", "Batura glacier snout ponds",
            lon=74.65, lat=36.53,
            # Outlet: snout pond at Batura Glacier terminus; drains into the
            # Hunza River via the Batura Nallah.
            # Source: manual digitizing from Google Maps satellite imagery, 2026-09-10.
            # Confidence: MEDIUM.
            # TODO: verify visually before production use.
            outlet_lon=74.647, outlet_lat=36.526,
        ),
    ]
}
