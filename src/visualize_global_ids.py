"""
visualize_global_ids.py

Multi-Camera Global Identity Sanity-Check Visualizer.

Samples multi-view global identities resolved across camera views during Stage 2,
extracts representative bounding box person crops from each view, and arranges
them into a synchronized grid (rows = global_id, columns = view1, view2, view3).

Outputs:
    data_storage/outputs/global_id_sanity_check.png

Usage:
    python -m src.visualize_global_ids
    python -m src.visualize_global_ids --num-samples 10
"""

# Group 1 — standard library
import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

# Group 2 — third-party
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Group 3 — local / project
from config import DATA_DIR, OUTPUTS_DIR

# --- Module-Level Constants & Paths -----------------------------------------
DEFAULT_GLOBAL_ID_PATH = (
    Path(__file__).resolve().parent.parent / "data_manifests" / "global_identities.csv"
)
DEFAULT_OUTPUT_PATH = OUTPUTS_DIR / "global_id_sanity_check.png"
VIEWS = ["view1", "view2", "view3"]
DEFAULT_FPS = 25.0


def load_global_identities(identities_path: Path) -> dict[int, list[dict]]:
    """
    Load global identities manifest and group tracklets by global_id.

    Returns:
        Dict mapping global_id -> list of {"view": str, "track_id": int}
    """
    if not identities_path.exists():
        raise FileNotFoundError(f"Global identities file not found: {identities_path}")

    clusters = defaultdict(list)
    with open(identities_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            clusters[int(r["global_id"])].append({
                "view": r["view"],
                "track_id": int(r["track_id"]),
            })
    return clusters


def load_view_detections(view_name: str) -> dict[int, list[dict]]:
    """
    Load all detections for a view, grouped by track_id.
    """
    csv_path = DATA_DIR / "detections" / f"{view_name}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Detections CSV not found: {csv_path}")

    dets_by_track = defaultdict(list)
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            dets_by_track[int(r["track_id"])].append({
                "frame_idx": int(r["frame_idx"]),
                "box": [float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"])],
                "conf": float(r.get("confidence", 0.0)),
            })
    return dets_by_track


def extract_crop(
    view_name: str,
    frame_idx: int,
    box: list[float],
) -> np.ndarray:
    """
    Read frame from disk and extract bounding box crop.
    Returns RGB image resized to standard aspect ratio (256x128).
    """
    frame_path = DATA_DIR / view_name / f"frame_{frame_idx:04d}.jpg"
    frame = cv2.imread(str(frame_path))
    if frame is None:
        blank = np.full((256, 128, 3), 230, dtype=np.uint8)
        return blank

    h, w = frame.shape[:2]
    x1 = max(0, min(int(round(box[0])), w - 1))
    y1 = max(0, min(int(round(box[1])), h - 1))
    x2 = max(x1 + 1, min(int(round(box[2])), w))
    y2 = max(y1 + 1, min(int(round(box[3])), h))

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return np.full((256, 128, 3), 230, dtype=np.uint8)

    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    crop_resized = cv2.resize(crop_rgb, (128, 256), interpolation=cv2.INTER_LINEAR)
    return crop_resized


def get_representative_crop(
    view_name: str,
    track_id: int,
    detections: list[dict],
) -> tuple[np.ndarray, int, float, float]:
    """
    Select the best representative detection frame and crop for a tracklet.
    Selects from the top 20% highest-confidence detections closest to tracklet temporal midpoint.

    Returns:
        crop: RGB image (256, 128, 3).
        frame_idx: Selected frame number.
        timestamp: Physical time in seconds.
        conf: Detection confidence.
    """
    if not detections:
        blank = np.full((256, 128, 3), 235, dtype=np.uint8)
        return blank, 0, 0.0, 0.0

    # Sort by frame_idx to find temporal median
    detections_sorted = sorted(detections, key=lambda d: d["frame_idx"])
    mid_idx = len(detections_sorted) // 2
    mid_frame = detections_sorted[mid_idx]["frame_idx"]

    # Filter detections with highest confidence, prefer ones near temporal midpoint
    top_confs = sorted(detections, key=lambda d: d["conf"], reverse=True)
    high_conf_cutoff = top_confs[min(len(top_confs) - 1, max(1, len(top_confs) // 5))]["conf"]
    candidate_pool = [d for d in detections if d["conf"] >= high_conf_cutoff]

    best_det = min(candidate_pool, key=lambda d: abs(d["frame_idx"] - mid_frame))

    crop = extract_crop(view_name, best_det["frame_idx"], best_det["box"])
    timestamp = best_det["frame_idx"] / DEFAULT_FPS
    return crop, best_det["frame_idx"], timestamp, best_det["conf"]


def create_blank_tile(label: str = "Not Observed") -> np.ndarray:
    """Generate a clean placeholder tile for camera views not in the cluster."""
    tile = np.full((256, 128, 3), 245, dtype=np.uint8)
    cv2.putText(
        tile,
        "N/A",
        (35, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (160, 160, 160),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        tile,
        label,
        (15, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (140, 140, 140),
        1,
        cv2.LINE_AA,
    )
    return tile


def select_sample_clusters(
    clusters: dict[int, list[dict]],
    num_samples: int = 10,
    seed: int = 42,
) -> list[int]:
    """
    Select diverse sample of multi-view clusters, prioritizing 3-view clusters.
    """
    three_view = [gid for gid, members in clusters.items() if len(members) == 3]
    two_view = [gid for gid, members in clusters.items() if len(members) == 2]

    random.seed(seed)
    # Take mostly 3-view clusters, supplemented by 2-view clusters
    num_3 = min(len(three_view), int(round(num_samples * 0.7)))
    num_2 = min(len(two_view), num_samples - num_3)

    sampled_3 = random.sample(three_view, num_3) if len(three_view) >= num_3 else three_view
    sampled_2 = random.sample(two_view, num_2) if len(two_view) >= num_2 else two_view

    selected = sorted(sampled_3 + sampled_2)
    return selected[:num_samples]


def plot_sanity_check_grid(
    selected_gids: list[int],
    clusters: dict[int, list[dict]],
    view_detections: dict[str, dict[int, list[dict]]],
    out_path: Path,
) -> None:
    """
    Render comparative multi-view grid figure and save to out_path.
    """
    num_rows = len(selected_gids)
    num_cols = len(VIEWS)

    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(num_cols * 3.2, num_rows * 3.6),
        squeeze=False,
    )

    # Title header
    fig.suptitle(
        "Stage 2: Multi-Camera Multi-Target (MCMT) Association Sanity Check\n"
        "Synchronized Person ReID Clusters Across CCTV Camera Views",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )

    for row_idx, gid in enumerate(selected_gids):
        cluster_members = {m["view"]: m["track_id"] for m in clusters[gid]}
        view_count = len(cluster_members)

        for col_idx, view_name in enumerate(VIEWS):
            ax = axes[row_idx, col_idx]

            if view_name in cluster_members:
                track_id = cluster_members[view_name]
                dets = view_detections[view_name].get(track_id, [])
                crop, frame_idx, timestamp, conf = get_representative_crop(
                    view_name, track_id, dets
                )
                ax.imshow(crop)

                # Column headers on top row
                if row_idx == 0:
                    title_top = f"Camera View: {view_name.upper()}\n"
                else:
                    title_top = ""

                ax.set_title(
                    f"{title_top}Track #{track_id} | F:{frame_idx:04d}\nt={timestamp:.2f}s | conf={conf:.2f}",
                    fontsize=9,
                    fontweight="medium",
                )
                # Green border for valid detection
                for spine in ax.spines.values():
                    spine.set_edgecolor("#2ecc71")
                    spine.set_linewidth(2.0)
            else:
                blank = create_blank_tile()
                ax.imshow(blank)
                if row_idx == 0:
                    title_top = f"Camera View: {view_name.upper()}\n"
                else:
                    title_top = ""
                ax.set_title(f"{title_top}Not Observed", fontsize=9, color="#7f8c8d")
                for spine in ax.spines.values():
                    spine.set_edgecolor("#bdc3c7")
                    spine.set_linewidth(1.0)

            ax.set_xticks([])
            ax.set_yticks([])

            # Label global ID on leftmost axis
            if col_idx == 0:
                ax.set_ylabel(
                    f"Global ID #{gid}\n({view_count} views)",
                    fontsize=10,
                    fontweight="bold",
                    color="#2c3e50",
                    rotation=0,
                    labelpad=45,
                    va="center",
                )

    plt.tight_layout(rect=[0.05, 0.01, 0.98, 0.98])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"[DONE] Multi-view sanity-check grid saved to: {out_path.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize multi-camera global identity associations."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_GLOBAL_ID_PATH,
        help="Path to global_identities.csv",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output image path",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of global identities to sample (default: 10)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sample selection",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Multi-Camera Global Identity Visualizer")
    print("=" * 70)
    print(f"Manifest   : {args.manifest}")
    print(f"Output     : {args.out}")
    print(f"Num Samples: {args.num_samples}")
    print("-" * 70)

    # 1. Load global identities
    clusters = load_global_identities(args.manifest)
    multi_view_count = sum(1 for members in clusters.values() if len(members) >= 2)
    print(f"[INFO] Loaded {len(clusters)} global identities ({multi_view_count} multi-view).")

    # 2. Select samples
    selected_gids = select_sample_clusters(
        clusters, num_samples=args.num_samples, seed=args.seed
    )
    print(f"[INFO] Selected global IDs for inspection: {selected_gids}")

    # 3. Load view detections
    print("[INFO] Loading view detections...")
    view_detections = {v: load_view_detections(v) for v in VIEWS}

    # 4. Render sanity check grid
    print("[INFO] Rendering multi-view grid...")
    plot_sanity_check_grid(selected_gids, clusters, view_detections, args.out)
    print("=" * 70)


if __name__ == "__main__":
    main()
