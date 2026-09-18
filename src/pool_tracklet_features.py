"""
pool_tracklet_features.py

Gathers frame-level Person Re-Identification (ReID) visual embeddings belonging
to each tracking trajectory (tracklet), computes their mean embedding, and
re-normalizes the result to unit L2 norm (||f||_2 = 1.0).

Outputs:
1. data_manifests/tracklet_features.csv
   Columns: view, track_id, start_frame, end_frame, start_time, end_time, num_detections
2. data_storage/data/tracklet_embeddings.npz
   Compressed archive keyed by f"{view}_{track_id}" with shape (512,) float32 arrays.

Usage:
    python -m src.pool_tracklet_features
    python -m src.pool_tracklet_features --force
"""

# Group 1 — standard library
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

# Group 2 — third-party
import numpy as np

# Group 3 — local / project
from config import DATA_DIR
from src.extract_frames import VIEWS as VIEW_OFFSETS

# --- Constants & Paths ------------------------------------------------------
VIEWS = list(VIEW_OFFSETS.keys())
DEFAULT_FPS = 25.0
LOW_CONFIDENCE_THRESHOLD = 3  # Tracklets with fewer detections are flagged

MANIFEST_PATH = (
    Path(__file__).resolve().parent.parent / "data_manifests" / "tracklet_features.csv"
)
OUTPUT_NPZ_PATH = DATA_DIR / "tracklet_embeddings.npz"


def compute_iou(box1: list[float] | np.ndarray, box2: list[float] | np.ndarray) -> float:
    """Compute spatial Intersection over Union (IoU) between two bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    a2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = a1 + a2 - inter

    return float(inter / union) if union > 0.0 else 0.0


def normalize_l2(vector: np.ndarray) -> np.ndarray:
    """Normalize a 1D vector to unit L2 norm."""
    norm = np.linalg.norm(vector)
    if norm > 1e-12:
        return (vector / norm).astype(np.float32)
    return vector.astype(np.float32)


def load_view_embeddings(view_name: str) -> dict:
    """Load frame-level ReID embeddings and metadata archive for a given view."""
    emb_path = DATA_DIR / "embeddings" / f"{view_name}_embeddings.npz"
    if not emb_path.exists():
        raise FileNotFoundError(f"ReID embeddings archive not found: {emb_path}")

    with np.load(emb_path) as data:
        return {
            "embeddings": data["embeddings"],
            "frame_indices": data["frame_indices"],
            "boxes": data["boxes"],
            "confidences": data["confidences"],
        }


def load_view_detections(view_name: str) -> list[dict]:
    """Load tracked detections CSV for a given view."""
    csv_path = DATA_DIR / "detections" / f"{view_name}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Detections CSV not found: {csv_path}")

    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def match_and_pool_view(
    view_name: str,
    fps: float = DEFAULT_FPS,
) -> tuple[list[dict], dict[str, np.ndarray], list[int]]:
    """
    Match frame-level detections to ReID embeddings for a view, group by track_id,
    and pool into average unit L2-normalized embeddings.

    Returns:
        tracklet_rows: List of metadata dicts for CSV manifest.
        tracklet_embeddings: Dict mapping f"{view}_{track_id}" to (512,) vector.
        low_conf_ids: List of track IDs with fewer than LOW_CONFIDENCE_THRESHOLD detections.
    """
    emb_data = load_view_embeddings(view_name)
    csv_rows = load_view_detections(view_name)

    embs = emb_data["embeddings"]
    emb_frames = emb_data["frame_indices"]
    emb_boxes = emb_data["boxes"]
    emb_confs = emb_data["confidences"]

    # Index embeddings by frame index for fast candidate lookup
    emb_by_frame = defaultdict(list)
    for i, (f, b, c) in enumerate(zip(emb_frames, emb_boxes, emb_confs)):
        emb_by_frame[int(f)].append({
            "idx": i,
            "box": b,
            "conf": float(c),
            "used": False,
        })

    # Group matched embeddings and frame indices by track_id
    track_embs = defaultdict(list)
    track_frames = defaultdict(list)

    matched_count = 0
    for r in csv_rows:
        frame_idx = int(r["frame_idx"])
        track_id = int(r["track_id"])
        conf = float(r["confidence"])
        box = [float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"])]

        candidates = emb_by_frame.get(frame_idx, [])
        available = [c for c in candidates if not c["used"]]

        if not available:
            continue

        # Match using combined spatial IoU and confidence closeness
        best_score = -999.0
        best_cand = None
        for cand in available:
            score = compute_iou(box, cand["box"]) - 10.0 * abs(conf - cand["conf"])
            if score > best_score:
                best_score = score
                best_cand = cand

        if best_cand is not None:
            best_cand["used"] = True
            emb_idx = best_cand["idx"]
            track_embs[track_id].append(embs[emb_idx])
            track_frames[track_id].append(frame_idx)
            matched_count += 1

    print(
        f"[INFO] {view_name}: matched {matched_count}/{len(csv_rows)} "
        f"detections to ReID embeddings across {len(track_embs)} tracklets"
    )

    tracklet_rows = []
    tracklet_embeddings = {}
    low_conf_ids = []

    # Sort tracklets by track_id
    for track_id in sorted(track_embs.keys()):
        raw_embs = np.array(track_embs[track_id], dtype=np.float32)
        frames = sorted(track_frames[track_id])
        num_dets = len(frames)

        start_frame = frames[0]
        end_frame = frames[-1]
        start_time = round(start_frame / fps, 2)
        end_time = round(end_frame / fps, 2)

        # Average all frame embeddings belonging to this tracklet
        mean_emb = np.mean(raw_embs, axis=0)
        pooled_emb = normalize_l2(mean_emb)

        key = f"{view_name}_{track_id}"
        tracklet_embeddings[key] = pooled_emb

        tracklet_rows.append({
            "view": view_name,
            "track_id": track_id,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "start_time": f"{start_time:.2f}",
            "end_time": f"{end_time:.2f}",
            "num_detections": num_dets,
        })

        if num_dets < LOW_CONFIDENCE_THRESHOLD:
            low_conf_ids.append(track_id)

    return tracklet_rows, tracklet_embeddings, low_conf_ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run pooling even if output files already exist",
    )
    args = parser.parse_args()

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_NPZ_PATH.parent.mkdir(parents=True, exist_ok=True)

    if MANIFEST_PATH.exists() and OUTPUT_NPZ_PATH.exists() and not args.force:
        print(f"[SKIP] Outputs already present at {MANIFEST_PATH} and {OUTPUT_NPZ_PATH} (use --force to re-run)")
        return

    all_manifest_rows = []
    all_tracklet_embeddings = {}
    view_stats = {}

    for view_name in VIEWS:
        rows, embs, low_conf = match_and_pool_view(view_name, fps=DEFAULT_FPS)
        all_manifest_rows.extend(rows)
        all_tracklet_embeddings.update(embs)

        avg_dets = (
            sum(r["num_detections"] for r in rows) / len(rows) if rows else 0.0
        )
        view_stats[view_name] = {
            "num_tracklets": len(rows),
            "avg_detections": avg_dets,
            "low_conf": low_conf,
        }

    # Write data_manifests/tracklet_features.csv
    fieldnames = [
        "view",
        "track_id",
        "start_frame",
        "end_frame",
        "start_time",
        "end_time",
        "num_detections",
    ]
    with open(MANIFEST_PATH, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_manifest_rows)

    print(f"[DONE] Manifest written to {MANIFEST_PATH} ({len(all_manifest_rows)} rows)")

    # Save data_storage/data/tracklet_embeddings.npz
    np.savez_compressed(OUTPUT_NPZ_PATH, **all_tracklet_embeddings)
    print(
        f"[DONE] Pooled tracklet embeddings written to {OUTPUT_NPZ_PATH} "
        f"({len(all_tracklet_embeddings)} arrays, dim=512)"
    )

    # Print summary report
    print("\n" + "=" * 70)
    print("TRACKLET POOLING SUMMARY REPORT")
    print("=" * 70)
    total_tracklets = sum(s["num_tracklets"] for s in view_stats.values())
    total_low_conf = sum(len(s["low_conf"]) for s in view_stats.values())

    for view_name, stats in view_stats.items():
        print(f"View: {view_name}")
        print(f"  - Total tracklets: {stats['num_tracklets']}")
        print(f"  - Average detections/tracklet: {stats['avg_detections']:.1f}")
        if stats["low_conf"]:
            print(
                f"  - [LOW-CONFIDENCE] {len(stats['low_conf'])} tracklet(s) with < {LOW_CONFIDENCE_THRESHOLD} detections: "
                f"{stats['low_conf']}"
            )
        else:
            print("  - Low-confidence tracklets: None")
        print()

    print(f"Overall Total Tracklets: {total_tracklets}")
    print(f"Total Low-Confidence Tracklets (< {LOW_CONFIDENCE_THRESHOLD} dets): {total_low_conf}")
    print("=" * 70)


if __name__ == "__main__":
    main()
