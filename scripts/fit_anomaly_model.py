"""Offline Isolation Forest fitting for anomaly detection (Issue #22).

Reads historical cloud-free scenes over each lake's dam-face AOI, extracts spectral
vectors, fits one IsolationForest per lake, and serializes to .joblib + .json sidecar.

Usage:
    uv run python scripts/fit_anomaly_model.py \\
        --start-date 2023-01-01 \\
        --end-date 2026-01-01 \\
        --output-dir .cache/anomaly \\
        --n-estimators 100 \\
        --contamination 0.05 \\
        [--force]

Requires:
    - Real DB + SceneSource access (DB_HOST/DB_PORT/DB_USER/DB_PASSWORD/DB_NAME set)
    - [anomaly] extras: uv pip install 'cryohealth-geo[anomaly]'

Per ADR 0005:
    - Per-lake models (not pooled) — each lake's moraine has a different spectral baseline
    - MIN_TRAINING_SCENES = 20 cloud-free scenes required; lakes below this are skipped
    - Cloud-free threshold = 10% over dam-face AOI (stricter than batch's 40%)
    - Training window: 2023–2026 (same as Issue #21 calibration dataset)
    - Annual refit: sidecar JSON records training_date AND training_scenes list
    - Sidecar training_scenes list is REQUIRED for future anomaly-flag auditing

After running:
    1. Review the per-lake scene counts in the output.
    2. Lakes with n_scenes < MIN_TRAINING_SCENES are skipped — run longer backfill
       or check cloud-cover for those lakes.
    3. Run validate_anomaly_thresholds.py to confirm FPR < 2%.
    4. Update SEEPAGE_SIGMA_THRESHOLD in pipeline/anomaly.py with the validated value
       and a provenance comment (script, run date, FPR at chosen threshold).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from datetime import date, timedelta
from pathlib import Path

# Ensure repo root is on sys.path when script is invoked directly
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


def _check_deps() -> None:
    import importlib.util
    missing = [name for name in ["sklearn", "joblib"] if importlib.util.find_spec(name) is None]
    if missing:
        print(f"ERROR: missing [anomaly] extras: {missing}", file=sys.stderr)
        print("Install with: uv pip install 'cryohealth-geo[anomaly]'", file=sys.stderr)
        sys.exit(1)


def _fit_lake(
    slug: str,
    dam_face_bbox: tuple[float, float, float, float],
    start: date,
    end: date,
    n_estimators: int,
    contamination: float,
    output_dir: Path,
    force: bool,
    max_scenes: int = 35,
) -> dict:
    """Fit and save an IsolationForest model for one lake. Returns a status dict."""
    import joblib  # noqa: PLC0415
    from sklearn.ensemble import IsolationForest  # noqa: PLC0415
    from pipeline.anomaly import (  # noqa: PLC0415
        ANOMALY_BANDS,
        CLOUD_MAX_FRACTION,
        MIN_TRAINING_SCENES,
        extract_spectral_vectors,
    )
    from pipeline.stac_source import PlanetaryComputerSource  # noqa: PLC0415

    model_path = output_dir / f"{slug}_isolation_forest.joblib"
    sidecar_path = output_dir / f"{slug}_isolation_forest.json"

    # Staleness check: warn if existing model is fresh (< 30d) or very old (> 365d).
    if model_path.exists() and not force:
        existing = json.loads(sidecar_path.read_text()) if sidecar_path.exists() else {}
        if "training_date" in existing:
            from datetime import datetime  # noqa: PLC0415
            fit_date = datetime.fromisoformat(existing["training_date"]).date()
            age_days = (date.today() - fit_date).days
            if age_days < 30:
                print(
                    f"  SKIPPED {slug}: existing model is {age_days} days old (< 30 days). "
                    f"Use --force to overwrite."
                )
                return {"slug": slug, "status": "skipped_fresh", "age_days": age_days}
            if age_days > 365:
                print(f"  WARNING {slug}: existing model is {age_days} days old (> 365 days). "
                      "Annual refit recommended — this run will overwrite it.")
        else:
            print(f"  WARNING {slug}: existing model has no training_date in sidecar. "
                  "Use --force to overwrite or manually delete and re-run.")
            return {"slug": slug, "status": "skipped_no_date"}

    print(f"\n  [{slug}] Fetching scenes {start} -> {end} ...")
    source = PlanetaryComputerSource()

    try:
        scenes = source.find_scenes_in_range(dam_face_bbox, start, end)
    except Exception as exc:  # noqa: BLE001
        print(f"  ERROR {slug}: scene fetch failed: {exc}", file=sys.stderr)
        return {"slug": slug, "status": "scene_fetch_error", "error": str(exc)}

    print(f"  [{slug}] {len(scenes)} candidate scenes. Filtering for cloud fraction < {CLOUD_MAX_FRACTION:.0%} ...", flush=True)

    scene_vectors: list[tuple[str, date, float, list[float]]] = []  # (scene_id, date, cloud_frac, vector)

    for scene in sorted(scenes, key=lambda s: s.captured_at):
        # Step 1: Read SCL band only to filter cloudy scenes quickly
        try:
            scl_dict = source.read_bands(scene, ["SCL"], dam_face_bbox)
        except Exception:
            continue

        if "SCL" in scl_dict:
            import numpy as np  # noqa: PLC0415
            scl = scl_dict["SCL"]
            cloud_frac = float(np.isin(scl, [3, 8, 9, 10]).sum() / scl.size) if scl.size > 0 else 0.0
            if cloud_frac > CLOUD_MAX_FRACTION:
                continue
            cloud_mask = np.isin(scl, [3, 8, 9, 10])
        else:
            cloud_frac = 0.0
            cloud_mask = None

        # Step 2: Scene is clear over dam-face AOI — read the 6 spectral bands
        try:
            bands = source.read_bands(scene, ANOMALY_BANDS, dam_face_bbox)
        except Exception:
            continue

        vectors = extract_spectral_vectors(bands, cloud_mask)
        if vectors.shape[0] == 0:
            continue

        mean_vec = vectors.mean(axis=0).tolist()  # (6,)
        scene_vectors.append((scene.scene_id, scene.captured_at, cloud_frac, mean_vec))
        print(f"    [{slug}] Clean scene #{len(scene_vectors)}: {scene.captured_at} (cloud {cloud_frac:.1%})", flush=True)

        if max_scenes and len(scene_vectors) >= max_scenes:
            print(f"    [{slug}] Reached target clean scenes ({max_scenes}). Proceeding to fit.", flush=True)
            break

    n_scenes = len(scene_vectors)
    print(f"  [{slug}] {n_scenes} cloud-free scenes with usable vectors.", flush=True)

    if n_scenes < MIN_TRAINING_SCENES:
        print(
            f"  SKIPPED {slug}: {n_scenes} scenes < MIN_TRAINING_SCENES={MIN_TRAINING_SCENES}. "
            f"Extend the training window or check cloud-cover for this lake.",
            flush=True,
        )
        return {"slug": slug, "status": "insufficient_scenes", "n_scenes": n_scenes}

    import numpy as np  # noqa: PLC0415
    X = np.array([sv[3] for sv in scene_vectors], dtype=np.float32)  # (N, 6)

    # Compute per-band statistics for the sidecar (used for z-score conversion at inference).
    band_means = X.mean(axis=0).tolist()
    band_stds = X.std(axis=0).tolist()

    print(f"  [{slug}] Fitting IsolationForest (n_estimators={n_estimators}, contamination={contamination}) ...")
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=42,  # reproducibility
        )
        model.fit(X)
    elapsed = time.perf_counter() - t0

    # Training-set score statistics (used by score_to_sigma() at inference time).
    train_scores = model.score_samples(X)
    training_mean_score = float(train_scores.mean())
    training_std_score = float(train_scores.std())

    print(f"  [{slug}] Fit complete in {elapsed:.1f}s. "
          f"training_mean={training_mean_score:.4f}, training_std={training_std_score:.4f}")

    # Serialize model.
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    # Sidecar JSON — includes training_scenes list for future anomaly-flag auditing.
    sidecar = {
        "slug": slug,
        "n_scenes": n_scenes,
        "training_date": date.today().isoformat(),
        "date_range": {"start": start.isoformat(), "end": end.isoformat()},
        "n_estimators": n_estimators,
        "contamination": contamination,
        "band_order": ["SWIR1(B11)", "SWIR2(B12)", "RedEdge(B05)", "NDWI", "MNDWI", "Red(B04)"],
        "band_means": band_means,
        "band_stds": band_stds,
        "training_mean_score": training_mean_score,
        "training_std_score": training_std_score,
        # REQUIRED for future anomaly-flag auditing (ADR 0005 §Decision 3).
        # If a future seepage flag looks suspicious, this list identifies exactly
        # which scenes defined "normal" for this lake's model.
        "training_scenes": [
            {
                "scene_id": sv[0],
                "captured_at": sv[1].isoformat(),
                "cloud_fraction": round(sv[2], 4),
            }
            for sv in scene_vectors
        ],
    }
    sidecar_path.write_text(json.dumps(sidecar, indent=2))

    print(f"  [{slug}] Saved: {model_path}")
    print(f"  [{slug}] Saved: {sidecar_path}")
    return {"slug": slug, "status": "ok", "n_scenes": n_scenes, "elapsed_s": round(elapsed, 1)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit per-lake Isolation Forest models for anomaly detection (Issue #22)."
    )
    parser.add_argument("--start-date", required=True, help="ISO date, e.g. 2023-01-01")
    parser.add_argument("--end-date", required=True, help="ISO date, e.g. 2026-01-01")
    parser.add_argument("--output-dir", default=".cache/anomaly")
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument(
        "--contamination",
        type=float,
        default=0.05,
        help="Expected fraction of anomalous scenes in training set. "
             "5%% is a documented starting point — adjust if validation FPR is too high.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing models even if < 30 days old.",
    )
    parser.add_argument(
        "--slugs",
        default=None,
        help="Comma-separated lake slugs. Defaults to all lakes with dam_face_bbox set.",
    )
    parser.add_argument(
        "--max-scenes",
        type=int,
        default=35,
        help="Target number of clean scenes to acquire (minimum 20). Defaults to 35.",
    )
    args = parser.parse_args()

    _check_deps()

    from pipeline.lakes import LAKES  # noqa: PLC0415

    start = date.fromisoformat(args.start_date)
    end = date.fromisoformat(args.end_date)
    output_dir = Path(args.output_dir)

    if args.slugs:
        slugs = [s.strip() for s in args.slugs.split(",")]
    else:
        slugs = [slug for slug, lake in LAKES.items() if lake.dam_face_bbox_deg is not None]

    if not slugs:
        print(
            "No lakes with dam_face_bbox_deg set. "
            "Digitize dam-face bboxes in pipeline/lakes.py first (see ADR 0005)."
        )
        sys.exit(0)

    print(f"Fitting Isolation Forest models for {len(slugs)} lake(s): {slugs}", flush=True)
    print(f"Training window: {start} -> {end}", flush=True)
    print(f"Parameters: n_estimators={args.n_estimators}, contamination={args.contamination}, max_scenes={args.max_scenes}", flush=True)
    print(f"Output: {output_dir.resolve()}\n", flush=True)

    results = []
    for slug in slugs:
        lake = LAKES.get(slug)
        if lake is None:
            print(f"  WARNING: slug {slug!r} not in LAKES — skipping")
            continue
        bbox = lake.dam_face_bbox_deg
        if bbox is None:
            print(f"  SKIPPED {slug}: dam_face_bbox_deg is None (not digitized or not applicable)")
            results.append({"slug": slug, "status": "no_bbox"})
            continue

        result = _fit_lake(
            slug, bbox, start, end, args.n_estimators, args.contamination, output_dir, args.force, args.max_scenes
        )
        results.append(result)

    print("\n" + "=" * 60)
    print("SUMMARY:")
    for r in results:
        status_str = r.get("status", "?")
        extra = ""
        if "n_scenes" in r:
            extra = f", n_scenes={r['n_scenes']}"
        if "elapsed_s" in r:
            extra += f", {r['elapsed_s']}s"
        print(f"  {r['slug']:15s}  {status_str}{extra}")

    ok_count = sum(1 for r in results if r.get("status") == "ok")
    print(f"\n{ok_count}/{len(results)} models fitted successfully.")
    if ok_count > 0:
        print(
            "\nNext step: run scripts/validate_anomaly_thresholds.py to confirm FPR < 2%.\n"
            "Then update SEEPAGE_SIGMA_THRESHOLD in pipeline/anomaly.py with the\n"
            "validated value and a provenance comment."
        )


if __name__ == "__main__":
    main()
