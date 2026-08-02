"""python -m pipeline.poc --lake <slug>

Fetches the latest usable Sentinel-2 L2A scene for a lake's AOI from Planetary
Computer's anonymous STAC API, computes NDWI, masks cloud/shadow/cirrus via the scene's
SCL band, and prints the water area estimate. See ADR 0001 for why Planetary Computer
(not GEE or CDSE, neither reachable without credentials this session doesn't have).
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from pipeline.lakes import LAKES
from pipeline.ndwi import cloud_fraction, cloud_mask, compute_ndwi, water_area_km2
from pipeline.stac_source import PlanetaryComputerSource


# SCL ships at 20m/pixel vs B03/B08's 10m — nearest-neighbor upsample (exact 2x) to
# align them, since SCL is a categorical classification (interpolating it would invent
# classes that don't exist).
def _upsample_scl_to_10m(scl: np.ndarray) -> np.ndarray:
    return np.repeat(np.repeat(scl, 2, axis=0), 2, axis=1)


def _crop_to_common_shape(*arrays: np.ndarray) -> list[np.ndarray]:
    """Each band is windowed independently against a reprojected bbox, and
    from_bounds() rounds fractional pixel edges per-band — confirmed live against real
    Planetary Computer imagery, where B03 came back one row taller than 2x SCL despite
    both nominally covering the same AOI. Cropping to the smallest common shape is the
    PoC-appropriate fix; production should resample onto one common grid (e.g. a
    rasterio WarpedVRT) instead of relying on independent windows lining up."""
    rows = min(a.shape[0] for a in arrays)
    cols = min(a.shape[1] for a in arrays)
    return [a[:rows, :cols] for a in arrays]


# Scene-level cloud_cover% is a whole-tile average and can look fine while the AOI
# itself is fully obscured (confirmed live) — this is the AOI-level bar a scene must
# clear to count as "usable" instead of being skipped for the next-most-recent one.
USABLE_CLOUD_FRACTION_MAX = 0.5


def run(lake_slug: str) -> float:
    lake = LAKES.get(lake_slug)
    if lake is None:
        print(f"Unknown lake '{lake_slug}'. Known: {', '.join(sorted(LAKES))}", file=sys.stderr)
        raise SystemExit(2)

    source = PlanetaryComputerSource()
    bbox = lake.bbox()

    candidates = source.find_recent_scenes(bbox)
    if not candidates:
        print(f"No Sentinel-2 scenes found at all for {lake.name} (bbox {bbox}).", file=sys.stderr)
        raise SystemExit(1)

    for scene in candidates:
        bands = source.read_bands(scene, ["B03", "B08", "SCL"], bbox)
        green, nir, scl = bands["B03"], bands["B08"], bands["SCL"]
        scl_upsampled = _upsample_scl_to_10m(scl)
        green, nir, scl_aligned = _crop_to_common_shape(green, nir, scl_upsampled)

        usable = cloud_mask(scl_aligned)
        clouds = cloud_fraction(usable)
        print(
            f"  checked {scene.scene_id} ({scene.captured_at}): AOI cloud_fraction={clouds:.3f}",
            file=sys.stderr,
        )
        if clouds <= USABLE_CLOUD_FRACTION_MAX:
            ndwi = compute_ndwi(green, nir)
            area_km2 = water_area_km2(ndwi, usable)
            print(f"lake:          {lake.name} ({lake.slug})")
            print(f"scene:         {scene.scene_id}")
            print(f"captured:      {scene.captured_at}")
            print(f"cloud_cover:   {scene.cloud_cover_pct:.1f}% (scene-level, from STAC metadata)")
            print(f"cloud_fraction:{clouds:.3f} (AOI-level, from SCL mask)")
            print(f"water_area_km2:{area_km2:.4f}")
            return area_km2

    # Matches R2's staleness rule: no usable scene in the search window -> say so
    # explicitly, never fall back to a silently stale or bogus 0 value.
    print(
        f"STALE: no scene with AOI cloud_fraction <= {USABLE_CLOUD_FRACTION_MAX} "
        f"found for {lake.name} in the last {len(candidates)} candidates.",
        file=sys.stderr,
    )
    raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lake", required=True, choices=sorted(LAKES), help="lake slug")
    args = parser.parse_args()
    run(args.lake)


if __name__ == "__main__":
    main()
