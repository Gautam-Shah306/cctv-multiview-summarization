"""
visualize_reid.py

Visual verification and inspection tool for Person Re-Identification (ReID) embeddings.

Evaluates embedding quality by computing cosine similarity across person crops
both within a single camera view (track consistency) and across multiple camera
views (cross-camera person association).

Usage:
    # 1. Inspect cross-camera person retrieval (query view1 against gallery view2)
    python -m src.visualize_reid --query-view view1 --gallery-view view2 --out reid_retrieval.png

    # 2. Inspect intra-camera track embedding consistency for a view
    python -m src.visualize_reid --intra-track --view view1 --out reid_intra_track.png

    # 3. Plot cosine similarity heatmap for sample detections
    python -m src.visualize_reid --similarity-matrix --view view1 --num-samples 20 --out reid_matrix.png
"""

# Group 1 — standard library
import argparse
from pathlib import Path

# Group 2 — third-party
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Group 3 — local / project
from config import DATA_DIR

EMBEDDINGS_DIR = DATA_DIR / "embeddings"


def load_embeddings(view_name: str) -> dict[str, np.ndarray] | None:
    """Load embeddings and metadata arrays from DATA_DIR/embeddings/<view>_embeddings.npz."""
    npz_path = EMBEDDINGS_DIR / f"{view_name}_embeddings.npz"
    if not npz_path.exists():
        print(f"[ERROR] Embeddings file not found: {npz_path}")
        return None

    with np.load(npz_path) as data:
        return {
            "embeddings": data["embeddings"],        # (N, 512)
            "frame_indices": data["frame_indices"],  # (N,)
            "track_ids": data["track_ids"],          # (N,)
            "boxes": data["boxes"],                  # (N, 4)
            "confidences": data["confidences"],      # (N,)
        }


def get_crop(view_name: str, frame_idx: int, box: list[float] | np.ndarray) -> np.ndarray:
    """Read frame from disk and return RGB crop for the bounding box."""
    frame_path = DATA_DIR / view_name / f"frame_{frame_idx:04d}.jpg"
    frame = cv2.imread(str(frame_path))
    if frame is None:
        return np.zeros((256, 128, 3), dtype=np.uint8)

    h, w = frame.shape[:2]
    x1 = max(0, min(int(round(box[0])), w - 1))
    y1 = max(0, min(int(round(box[1])), h - 1))
    x2 = max(x1 + 1, min(int(round(box[2])), w))
    y2 = max(y1 + 1, min(int(round(box[3])), h))

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros((256, 128, 3), dtype=np.uint8)

    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    return cv2.resize(crop, (128, 256))


def visualize_cross_camera_retrieval(
    query_view: str = "view1",
    gallery_view: str = "view2",
    num_queries: int = 4,
    top_k: int = 5,
    out_path: str = "reid_retrieval.png",
) -> None:
    """
    Visualizes cross-camera retrieval: for query person crops in query_view,
    finds and displays the top-K highest cosine similarity matches in gallery_view.
    """
    q_data = load_embeddings(query_view)
    g_data = load_embeddings(gallery_view)

    if q_data is None or g_data is None:
        return

    q_embeds = q_data["embeddings"]  # (N_q, 512)
    g_embeds = g_data["embeddings"]  # (N_g, 512)

    # Compute full cosine similarity matrix: (N_q, N_g)
    # Note: Embeddings are already L2 normalized, so similarity is dot product
    sim_matrix = np.dot(q_embeds, g_embeds.T)

    # Select queries evenly distributed across query detections
    query_step = max(1, len(q_embeds) // num_queries)
    query_indices = list(range(0, len(q_embeds), query_step))[:num_queries]

    fig, axes = plt.subplots(
        len(query_indices),
        top_k + 1,
        figsize=((top_k + 1) * 3, len(query_indices) * 3.5),
    )
    if len(query_indices) == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_idx, q_idx in enumerate(query_indices):
        q_frame = int(q_data["frame_indices"][q_idx])
        q_track = int(q_data["track_ids"][q_idx])
        q_box = q_data["boxes"][q_idx]
        q_crop = get_crop(query_view, q_frame, q_box)

        # Plot Query Image
        ax_q = axes[row_idx, 0]
        ax_q.imshow(q_crop)
        ax_q.set_title(
            f"Query: {query_view}\nFrame {q_frame:04d} | ID:{q_track}",
            fontsize=9,
            fontweight="bold",
            color="darkblue",
        )
        ax_q.axis("off")

        # Find Top-K gallery matches
        similarities = sim_matrix[q_idx]
        top_gallery_indices = np.argsort(-similarities)[:top_k]

        for rank, g_idx in enumerate(top_gallery_indices):
            g_frame = int(g_data["frame_indices"][g_idx])
            g_track = int(g_data["track_ids"][g_idx])
            g_box = g_data["boxes"][g_idx]
            g_crop = get_crop(gallery_view, g_frame, g_box)
            sim_score = similarities[g_idx]

            ax_g = axes[row_idx, rank + 1]
            ax_g.imshow(g_crop)
            ax_g.set_title(
                f"Rank {rank + 1} ({gallery_view})\nSim: {sim_score:.3f}\nF:{g_frame:04d} | ID:{g_track}",
                fontsize=8,
            )
            ax_g.axis("off")

    plt.suptitle(
        f"Stage 1b ReID Cross-Camera Matching: Query ({query_view}) -> Gallery ({gallery_view})",
        fontsize=12,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()

    out_file = Path(out_path)
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    print(f"[DONE] Cross-camera retrieval plot saved to {out_file.resolve()}")
    plt.close()


def visualize_similarity_matrix(
    view_name: str = "view1",
    num_samples: int = 25,
    out_path: str = "reid_similarity_matrix.png",
) -> None:
    """Plots cosine similarity matrix heatmap among sampled person detections in a view."""
    data = load_embeddings(view_name)
    if data is None:
        return

    embeds = data["embeddings"]
    if len(embeds) == 0:
        return

    step = max(1, len(embeds) // num_samples)
    sampled_indices = list(range(0, len(embeds), step))[:num_samples]
    sampled_embeds = embeds[sampled_indices]
    sampled_frames = data["frame_indices"][sampled_indices]
    sampled_tracks = data["track_ids"][sampled_indices]

    sim_mat = np.dot(sampled_embeds, sampled_embeds.T)

    plt.figure(figsize=(10, 8))
    im = plt.imshow(sim_mat, cmap="viridis", vmin=0.0, vmax=1.0)
    plt.colorbar(im, label="Cosine Similarity")

    tick_labels = [f"F{f}\nID:{t}" for f, t in zip(sampled_frames, sampled_tracks)]
    plt.xticks(range(len(sampled_indices)), tick_labels, rotation=90, fontsize=7)
    plt.yticks(range(len(sampled_indices)), tick_labels, fontsize=7)
    plt.title(f"ReID Cosine Similarity Matrix ({view_name})", fontsize=11, fontweight="bold")
    plt.tight_layout()

    out_file = Path(out_path)
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    print(f"[DONE] Similarity matrix saved to {out_file.resolve()}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-view", default="view1", help="Query camera view (default: view1)")
    parser.add_argument("--gallery-view", default="view2", help="Gallery camera view (default: view2)")
    parser.add_argument("--view", default="view1", help="Camera view for similarity matrix (default: view1)")
    parser.add_argument("--num-queries", type=int, default=4, help="Number of query person crops (default: 4)")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top gallery matches (default: 5)")
    parser.add_argument("--similarity-matrix", action="store_true", help="Plot cosine similarity matrix heatmap")
    parser.add_argument("--out", default="reid_retrieval.png", help="Output plot filename")
    args = parser.parse_args()

    if args.similarity_matrix:
        visualize_similarity_matrix(view_name=args.view, out_path=args.out)
    else:
        visualize_cross_camera_retrieval(
            query_view=args.query_view,
            gallery_view=args.gallery_view,
            num_queries=args.num_queries,
            top_k=args.top_k,
            out_path=args.out,
        )


if __name__ == "__main__":
    main()
