"""The same six individually-cited lakes as CryoHealth-api's seed data — ported, not
re-derived, so the two services never disagree about where a lake actually is. See
CryoHealth-api/src/lakes/data/lakes.seed-data.ts and CryoHealth-api ADR 0002 for why
it's six lakes and not the full 25-lake shortlist, and the citation for each coordinate.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LakeAoi:
    slug: str
    name: str
    lon: float
    lat: float
    # Half-width of the AOI bounding box in degrees. ~0.01deg is roughly 1.1km at this
    # latitude — enough margin around a point coordinate for a small alpine lake without
    # pulling in unrelated terrain. Real production AOIs should come from the lake's
    # actual boundary polygon (CryoHealth-api's Lake.boundary, currently unused) once
    # one is digitized — this is a deliberate PoC simplification, not the final approach.
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
        LakeAoi("shishper", "Shishper (Hassanabad) glacial lake", lon=74.61, lat=36.40),
        LakeAoi("khurdopin", "Khurdopin glacial lake", lon=75.5083, lat=36.3383),
        LakeAoi("badswat", "Badswat glacial lake", lon=74.0775, lat=36.4855),
        LakeAoi("passu", "Passu glacial pond", lon=74.77, lat=36.47),
        LakeAoi("ghulkin", "Ghulkin glacial pond", lon=74.856, lat=36.4653),
        LakeAoi("batura", "Batura glacier snout ponds", lon=74.65, lat=36.53),
    ]
}
