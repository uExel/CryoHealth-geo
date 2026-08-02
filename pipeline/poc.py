"""python -m pipeline.poc --lake <slug> [--source planetary-computer|cdse]

Fetches the latest usable Sentinel-2 L2A scene for a lake's AOI, computes NDWI, masks
cloud/shadow/cirrus via the scene's SCL band, and prints the water area estimate.

--source planetary-computer (default): Microsoft Planetary Computer's anonymous STAC
  API — no credentials needed, real imagery, not the production platform (ADR 0001).
--source cdse: Copernicus Data Space Ecosystem, the production platform ADR 0001
  chose. Needs CDSE_CLIENT_ID / CDSE_CLIENT_SECRET env vars (a free account + OAuth2
  client at dataspace.copernicus.eu).

Every SceneSource returns bands already pixel-aligned — this file doesn't need to know
which source-specific quirks (UTM reprojection, resolution alignment, ...) applied.
"""

from __future__ import annotations

import argparse
import sys

from pipeline.lakes import LAKES
from pipeline.ndwi import cloud_fraction, cloud_mask, compute_ndwi, water_area_km2
from pipeline.stac_source import PlanetaryComputerSource, SceneSource

# Scene-level cloud_cover% is a whole-tile average and can look fine while the AOI
# itself is fully obscured (confirmed live) — this is the AOI-level bar a scene must
# clear to count as "usable" instead of being skipped for the next-most-recent one.
USABLE_CLOUD_FRACTION_MAX = 0.5


def _make_source(name: str) -> SceneSource:
    if name == "cdse":
        from pipeline.cdse_source import CdseSource

        return CdseSource()
    return PlanetaryComputerSource()


def run(lake_slug: str, source_name: str = "planetary-computer") -> float:
    lake = LAKES.get(lake_slug)
    if lake is None:
        print(f"Unknown lake '{lake_slug}'. Known: {', '.join(sorted(LAKES))}", file=sys.stderr)
        raise SystemExit(2)

    source = _make_source(source_name)
    bbox = lake.bbox()

    candidates = source.find_recent_scenes(bbox, 12)
    if not candidates:
        print(f"No Sentinel-2 scenes found at all for {lake.name} (bbox {bbox}).", file=sys.stderr)
        raise SystemExit(1)

    for scene in candidates:
        bands = source.read_bands(scene, ["B03", "B08", "SCL"], bbox)
        green, nir, scl = bands["B03"], bands["B08"], bands["SCL"]

        usable = cloud_mask(scl)
        clouds = cloud_fraction(usable)
        print(
            f"  checked {scene.scene_id} ({scene.captured_at}): AOI cloud_fraction={clouds:.3f}",
            file=sys.stderr,
        )
        if clouds <= USABLE_CLOUD_FRACTION_MAX:
            ndwi = compute_ndwi(green, nir)
            area_km2 = water_area_km2(ndwi, usable)
            print(f"source:        {source_name}")
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
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--lake", required=True, choices=sorted(LAKES), help="lake slug")
    parser.add_argument(
        "--source", choices=["planetary-computer", "cdse"], default="planetary-computer", help="scene source"
    )
    args = parser.parse_args()
    run(args.lake, args.source)


if __name__ == "__main__":
    main()
