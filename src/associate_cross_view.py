"""
associate_cross_view.py

Cross-View Association & Multi-Camera Multi-Target (MCMT) Clustering.

Constructs an affinity graph across overlapping CCTV camera views by computing
pairwise cosine similarity between 512-d ReID tracklet embeddings, enforces
spatio-temporal feasibility constraints (temporal overlap / proximity), and
resolves consistent global identities using hierarchical agglomerative clustering.

Key Outputs:
1. data_manifests/cross_view_graph.csv:
   All candidate cross-view edges with cosine similarity and physical time gap.
2. data_manifests/global_identities.csv:
   Mapping of (view, track_id) to global_id, including low-confidence singletons.

Usage:
    python -m src.associate_cross_view
    python -m src.associate_cross_view --max-time-gap 2.0 --similarity-threshold 0.5
"""

# Group 1 — standard library
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

# Group 2 — third-party
import numpy as np
from sklearn.cluster import AgglomerativeClustering

# Group 3 — local / project
from config import (
    CROSS_VIEW_MAX_TIME_GAP,
    CROSS_VIEW_MIN_DETECTIONS,
    CROSS_VIEW_MIN_SIMILARITY,
    DATA_DIR,
)

# --- Module-Level Constants & Paths -----------------------------------------
DEFAULT_MANIFEST_PATH = (
    Path(__file__).resolve().parent.parent / "data_manifests" / "tracklet_features.csv"
)
DEFAULT_NPZ_PATH = DATA_DIR / "tracklet_embeddings.npz"
DEFAULT_GRAPH_PATH = (
    Path(__file__).resolve().parent.parent / "data_manifests" / "cross_view_graph.csv"
)
DEFAULT_GLOBAL_ID_PATH = (
    Path(__file__).resolve().parent.parent / "data_manifests" / "global_identities.csv"
)


def load_tracklet_manifest(
    manifest_path: Path,
    min_detections: int = CROSS_VIEW_MIN_DETECTIONS,
) -> tuple[list[dict], list[dict]]:
    """
    Load tracklet manifest and separate candidate tracklets from low-confidence singletons.

    Args:
        manifest_path: Path to tracklet_features.csv.
        min_detections: Minimum detection count for candidate matching.

    Returns:
        candidates: Tracklets meeting min_detections threshold.
        low_conf: Tracklets with fewer detections to be preserved as singletons.
    """
    if not manifest_path.exists():
        raise FileNotFoundError(f"Tracklet manifest not found: {manifest_path}")

    candidates = []
    low_conf = []

    with open(manifest_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            item = {
                "view": r["view"],
                "track_id": int(r["track_id"]),
                "start_frame": int(r["start_frame"]),
                "end_frame": int(r["end_frame"]),
                "start_time": float(r["start_time"]),
                "end_time": float(r["end_time"]),
                "num_detections": int(r["num_detections"]),
            }
            if item["num_detections"] >= min_detections:
                candidates.append(item)
            else:
                low_conf.append(item)

    return candidates, low_conf


def load_tracklet_embeddings(npz_path: Path) -> dict[str, np.ndarray]:
    """
    Load pooled ReID embeddings from compressed npz archive.

    Args:
        npz_path: Path to tracklet_embeddings.npz.

    Returns:
        Dict mapping f"{view}_{track_id}" to unit-L2 512-d array.
    """
    if not npz_path.exists():
        raise FileNotFoundError(f"Tracklet embeddings archive not found: {npz_path}")

    with np.load(npz_path) as data:
        return {k: data[k] for k in data.files}


def compute_time_gap(t_a: dict, t_b: dict) -> float:
    """
    Compute physical time gap between two tracklets in seconds.
    Returns 0.0 if time intervals overlap.
    """
    s_a, e_a = t_a["start_time"], t_a["end_time"]
    s_b, e_b = t_b["start_time"], t_b["end_time"]

    if s_a <= e_b and s_b <= e_a:
        return 0.0
    if s_b > e_a:
        return s_b - e_a
    return s_a - e_b


def build_affinity_graph(
    candidates: list[dict],
    embeddings: dict[str, np.ndarray],
    max_time_gap: float = CROSS_VIEW_MAX_TIME_GAP,
) -> tuple[list[dict], list[dict], np.ndarray]:
    """
    Compute pairwise cosine similarities across views and construct affinity distance matrix.

    Returns:
        all_candidate_edges: List of all cross-view pairs with similarity and time gap.
        surviving_edges: List of edges that satisfy the spatio-temporal filter.
        dist_matrix: (N, N) precomputed distance matrix (1.0 - similarity) for clustering.
    """
    num_candidates = len(candidates)
    for idx, c in enumerate(candidates):
        c["idx"] = idx
        key = f"{c['view']}_{c['track_id']}"
        if key not in embeddings:
            raise KeyError(f"Missing embedding for candidate: {key}")
        c["embedding"] = embeddings[key]

    # Initialize distance matrix with 2.0 (maximum distance for unit-norm cosine metric)
    dist_matrix = np.full((num_candidates, num_candidates), 2.0, dtype=np.float32)
    np.fill_diagonal(dist_matrix, 0.0)

    all_candidate_edges = []
    surviving_edges = []

    for i in range(num_candidates):
        t_a = candidates[i]
        for j in range(i + 1, num_candidates):
            t_b = candidates[j]

            # Skip same-view comparisons (same camera cannot observe the same person as 2 tracks)
            if t_a["view"] == t_b["view"]:
                continue

            # Embeddings are unit L2 normalized; dot product equals cosine similarity
            sim = float(np.dot(t_a["embedding"], t_b["embedding"]))
            sim = max(-1.0, min(1.0, sim))
            time_gap = compute_time_gap(t_a, t_b)

            edge = {
                "view_a": t_a["view"],
                "track_id_a": t_a["track_id"],
                "view_b": t_b["view"],
                "track_id_b": t_b["track_id"],
                "cosine_similarity": round(sim, 4),
                "time_gap": round(time_gap, 2),
            }
            all_candidate_edges.append(edge)

            # Feasibility filter: physical temporal proximity
            if time_gap <= max_time_gap:
                surviving_edges.append(edge)
                distance = max(0.0, 1.0 - sim)
                dist_matrix[i, j] = distance
                dist_matrix[j, i] = distance

    return all_candidate_edges, surviving_edges, dist_matrix


def cluster_identities(
    candidates: list[dict],
    dist_matrix: np.ndarray,
    similarity_threshold: float = CROSS_VIEW_MIN_SIMILARITY,
    linkage: str = "complete",
) -> tuple[list[list[dict]], int]:
    """
    Perform agglomerative clustering on affinity distance matrix and enforce
    the cluster-validity constraint (at most one tracklet per camera view per cluster).

    Args:
        candidates: List of candidate tracklet dicts.
        dist_matrix: (N, N) pairwise distance matrix.
        similarity_threshold: Minimum cosine similarity required to merge.
        linkage: Linkage criterion ('complete' or 'average').

    Returns:
        resolved_clusters: List of clusters, where each cluster is a list of tracklet dicts.
        conflicts_resolved: Total count of same-view conflicts resolved by splitting.
    """
    distance_threshold = 1.0 - similarity_threshold

    clustering = AgglomerativeClustering(
        metric="precomputed",
        linkage=linkage,
        distance_threshold=distance_threshold,
        n_clusters=None,
    )
    labels = clustering.fit_predict(dist_matrix)

    initial_clusters = defaultdict(list)
    for idx, label in enumerate(labels):
        initial_clusters[label].append(candidates[idx])

    resolved_clusters = []
    conflicts_resolved = 0

    for label, members in initial_clusters.items():
        view_map = defaultdict(list)
        for m in members:
            view_map[m["view"]].append(m)

        has_conflict = any(len(v_list) > 1 for v_list in view_map.values())
        if not has_conflict:
            resolved_clusters.append(members)
            continue

        # --- Resolve same-view conflicts by retaining highest-similarity tracklet
        valid_cluster = []
        evicted = []

        # First collect non-conflicted views as anchor members
        anchors = [v_list[0] for v, v_list in view_map.items() if len(v_list) == 1]

        for view_name, v_members in view_map.items():
            if len(v_members) == 1:
                valid_cluster.append(v_members[0])
            else:
                conflicts_resolved += len(v_members) - 1

                # If anchors exist, score by average similarity to anchors
                if anchors:
                    best_m = None
                    best_score = -float("inf")
                    for m in v_members:
                        score = float(np.mean([
                            float(np.dot(m["embedding"], a["embedding"]))
                            for a in anchors
                        ]))
                        if score > best_score:
                            best_score = score
                            best_m = m
                    valid_cluster.append(best_m)
                    evicted.extend([m for m in v_members if m is not best_m])
                else:
                    # If no anchors from other views, keep tracklet with most detections
                    v_members_sorted = sorted(
                        v_members, key=lambda x: x["num_detections"], reverse=True
                    )
                    valid_cluster.append(v_members_sorted[0])
                    evicted.extend(v_members_sorted[1:])

        resolved_clusters.append(valid_cluster)
        # Each evicted tracklet forms its own singleton cluster
        for e in evicted:
            resolved_clusters.append([e])

    return resolved_clusters, conflicts_resolved


def write_candidate_graph(candidate_edges: list[dict], out_path: Path) -> None:
    """Save all cross-view candidate edges to CSV."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "view_a",
        "track_id_a",
        "view_b",
        "track_id_b",
        "cosine_similarity",
        "time_gap",
    ]
    with open(out_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(candidate_edges)


def write_global_identities(
    clusters: list[list[dict]],
    low_conf_tracklets: list[dict],
    out_path: Path,
) -> list[dict]:
    """
    Assign consistent, sequential global_id values and write manifest.

    Returns:
        All assigned row dicts with fields (view, track_id, global_id).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    current_global_id = 1

    # Sort clusters: multi-view clusters first (by descending view count, then start time)
    def cluster_sort_key(c: list[dict]) -> tuple[int, float]:
        min_start = min(m["start_time"] for m in c)
        return (-len(c), min_start)

    sorted_clusters = sorted(clusters, key=cluster_sort_key)

    for cluster in sorted_clusters:
        # Sort cluster members canonically by view
        for m in sorted(cluster, key=lambda x: x["view"]):
            rows.append({
                "view": m["view"],
                "track_id": m["track_id"],
                "global_id": current_global_id,
            })
        current_global_id += 1

    # Preserved low-confidence tracklets as singletons
    sorted_low_conf = sorted(
        low_conf_tracklets, key=lambda x: (x["view"], x["track_id"])
    )
    for lc in sorted_low_conf:
        rows.append({
            "view": lc["view"],
            "track_id": lc["track_id"],
            "global_id": current_global_id,
        })
        current_global_id += 1

    fieldnames = ["view", "track_id", "global_id"]
    with open(out_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-View Association & MCMT Clustering for Multi-View CCTV."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help="Path to tracklet_features.csv manifest",
    )
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=DEFAULT_NPZ_PATH,
        help="Path to tracklet_embeddings.npz archive",
    )
    parser.add_argument(
        "--max-time-gap",
        type=float,
        default=CROSS_VIEW_MAX_TIME_GAP,
        help="Max allowed physical time gap (seconds) between views (default: 2.0)",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=CROSS_VIEW_MIN_SIMILARITY,
        help="Minimum cosine similarity for cross-view clustering (default: 0.5)",
    )
    parser.add_argument(
        "--min-detections",
        type=int,
        default=CROSS_VIEW_MIN_DETECTIONS,
        help="Minimum detections for tracklet to enter candidate matching (default: 3)",
    )
    parser.add_argument(
        "--linkage",
        type=str,
        default="complete",
        choices=["complete", "average"],
        help="Agglomerative clustering linkage criterion (default: complete)",
    )
    parser.add_argument(
        "--out-graph",
        type=Path,
        default=DEFAULT_GRAPH_PATH,
        help="Output path for candidate edges CSV",
    )
    parser.add_argument(
        "--out-manifest",
        type=Path,
        default=DEFAULT_GLOBAL_ID_PATH,
        help="Output path for global identities CSV",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Stage 2: Cross-View Association & MCMT Clustering")
    print("=" * 70)
    print(f"Tracklet Manifest : {args.manifest}")
    print(f"Embeddings Archive: {args.embeddings}")
    print(f"Max Time Gap      : {args.max_time_gap} s")
    print(f"Similarity Thresh : {args.similarity_threshold} (dist <= {1.0 - args.similarity_threshold:.2f})")
    print(f"Min Detections    : {args.min_detections}")
    print(f"Linkage Criterion : {args.linkage}")
    print("-" * 70)

    # 1. Load data
    candidates, low_conf = load_tracklet_manifest(args.manifest, args.min_detections)
    embeddings = load_tracklet_embeddings(args.embeddings)

    total_tracklets = len(candidates) + len(low_conf)
    print(f"[INFO] Loaded {total_tracklets} total tracklets:")
    print(f"       - Candidates for cross-view matching (>= {args.min_detections} dets): {len(candidates)}")
    print(f"       - Excluded low-confidence singletons  (< {args.min_detections} dets) : {len(low_conf)}")

    # 2. Build affinity graph
    all_edges, surviving_edges, dist_matrix = build_affinity_graph(
        candidates, embeddings, max_time_gap=args.max_time_gap
    )
    write_candidate_graph(all_edges, args.out_graph)

    print(f"[INFO] Total cross-view candidate edges computed: {len(all_edges)}")
    print(f"[INFO] Candidate edges saved to: {args.out_graph}")
    print(
        f"[INFO] Edges surviving time-gap filter (<= {args.max_time_gap:.1f}s): {len(surviving_edges)} "
        f"({(len(surviving_edges) / len(all_edges) * 100):.1f}% of total)"
    )

    # 3. Perform agglomerative clustering & conflict resolution
    clusters, conflicts_resolved = cluster_identities(
        candidates,
        dist_matrix,
        similarity_threshold=args.similarity_threshold,
        linkage=args.linkage,
    )

    # 4. Write final global identities
    assigned_rows = write_global_identities(clusters, low_conf, args.out_manifest)

    # 5. Compute and report cluster statistics
    total_clusters = len(set(r["global_id"] for r in assigned_rows))
    view_span_counts = defaultdict(int)

    cluster_to_views = defaultdict(set)
    for r in assigned_rows:
        cluster_to_views[r["global_id"]].add(r["view"])

    for gid, views in cluster_to_views.items():
        view_span_counts[len(views)] += 1

    print("-" * 70)
    print("MCMT Clustering Results & Statistics:")
    print(f"  • Total Global Identity Clusters : {total_clusters}")
    print(f"  • Same-View Conflicts Resolved   : {conflicts_resolved}")
    print("  • Cluster Breakdown by Camera View Span:")
    print(f"      - Spanning all 3 views       : {view_span_counts[3]} clusters")
    print(f"      - Spanning 2 views           : {view_span_counts[2]} clusters")
    print(f"      - Single-view tracklets      : {view_span_counts[1]} clusters (including {len(low_conf)} low-conf singletons)")
    print(f"[INFO] Global identities saved to: {args.out_manifest}")
    print("=" * 70)


if __name__ == "__main__":
    main()
