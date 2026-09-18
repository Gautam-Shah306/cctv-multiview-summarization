"""
keyframe_selection.py

Novelty #1: Adaptive Keyframe Selection via Fused DINO ViT & OSNet ReID Embeddings.

Replaces heuristic pixel-level difference thresholding with a learned, content-aware
redundancy metric combining:
1. Global full-frame Vision Transformer scene embeddings (DINO)
2. Fine-grained pedestrian identity embeddings (OSNet ReID)

Keyframe Decision Rule:
    - Frames are grouped into temporal shots (default: 10s = 250 frames at 25 fps).
    - Seed frame of each shot is unconditionally accepted.
    - Candidate frame t computes fused redundancy score against all accepted keyframes j in shot:
          s_redund(t, j) = alpha * s_D(d_t, d_j) + (1 - alpha) * s_R(r_t, r_j)
      (Falls back to s_D alone if either frame lacks detected persons).
    - Frame t is accepted as a new keyframe iff:
          max_j(s_redund(t, j)) < epsilon_shot
      where epsilon_shot = mu_shot - k * sigma_shot computed adaptively from observed
      within-shot redundancy scores.

Outputs:
    data_manifests/keyframe_decisions_{view}.csv
    Columns: view, frame_idx, shot_id, accepted, max_redund_score, epsilon_used

Usage:
    python -m src.keyframe_selection --view view1
    python -m src.keyframe_selection --view all --shot-window-sec 10.0 --alpha 0.5 --k 1.0
"""

# Group 1 — standard library
import argparse
import csv
from collections import defaultdict
from pathlib import Path

# Group 2 — third-party
import numpy as np

# Group 3 — local / project
from config import (
    DATA_DIR,
    DINOV3_EMBEDDINGS_DIR,
    KEYFRAME_ALPHA,
    KEYFRAME_DEFAULT_EPSILON,
    KEYFRAME_K,
    KEYFRAME_SHOT_WINDOW_SEC,
)
from src.extract_frames import VIEWS as VIEW_CONFIGS

# --- Module Constants & Paths -----------------------------------------------
SUPPORTED_VIEWS = list(VIEW_CONFIGS.keys())
DEFAULT_FPS = 25.0
MANIFESTS_DIR = Path(__file__).resolve().parent.parent / "data_manifests"


def load_dino_embeddings(view_name: str) -> dict[int, np.ndarray]:
    """
    Load DINO embeddings archive and map frame_idx -> (384,) unit-norm embedding.
    """
    npz_path = DINOV3_EMBEDDINGS_DIR / f"{view_name}_dinov3.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"DINO embeddings not found: {npz_path}. "
            f"Please run 'python -m src.extract_dinov3_features --view {view_name}' first."
        )

    with np.load(npz_path) as data:
        embeddings = data["embeddings"]
        frame_indices = data["frame_indices"]

    mapping = {}
    for idx, f_idx in enumerate(frame_indices):
        mapping[int(f_idx)] = embeddings[idx]
    return mapping


def load_dominant_reid_embeddings(view_name: str) -> dict[int, np.ndarray]:
    """
    Load ReID embeddings and select the dominant (highest confidence) person
    embedding for each frame.

    Returns:
        Mapping of frame_idx -> (512,) unit-norm dominant person embedding.
    """
    reid_path = DATA_DIR / "embeddings" / f"{view_name}_embeddings.npz"
    if not reid_path.exists():
        print(f"[WARNING] ReID embeddings not found at {reid_path}. Falling back to DINO alone.")
        return {}

    with np.load(reid_path) as data:
        embeddings = data["embeddings"]
        frame_indices = data["frame_indices"]
        confidences = data["confidences"]

    # Group candidate detections by frame index
    frame_dets = defaultdict(list)
    for idx, (f_idx, conf) in enumerate(zip(frame_indices, confidences)):
        frame_dets[int(f_idx)].append((float(conf), embeddings[idx]))

    dominant_reid = {}
    for f_idx, candidates in frame_dets.items():
        # Pick detection with highest confidence
        best_emb = max(candidates, key=lambda x: x[0])[1]
        dominant_reid[f_idx] = best_emb

    return dominant_reid


def load_all_reid_embeddings(view_name: str) -> dict[int, np.ndarray]:
    """
    Load ReID embeddings and group all per-detection embeddings by frame.

    Returns:
        Mapping of frame_idx -> (N, 512) array of unit-norm person embeddings,
        where N is the number of detected persons in that frame.
    """
    reid_path = DATA_DIR / "embeddings" / f"{view_name}_embeddings.npz"
    if not reid_path.exists():
        print(f"[WARNING] ReID embeddings not found at {reid_path}. Falling back to DINO alone.")
        return {}

    with np.load(reid_path) as data:
        embeddings = data["embeddings"]
        frame_indices = data["frame_indices"]

    frame_dets = defaultdict(list)
    for idx, f_idx in enumerate(frame_indices):
        frame_dets[int(f_idx)].append(embeddings[idx])

    all_reid = {}
    for f_idx, candidates in frame_dets.items():
        all_reid[f_idx] = np.array(candidates, dtype=np.float32)

    return all_reid


def compute_greedy_set_reid_similarity(
    r_t: np.ndarray,
    r_j: np.ndarray,
    agg_method: str = "mean",
) -> float:
    """
    Compute set-based ReID similarity s_R(t, j) using greedy max-similarity matching.

    1. Compute pairwise cosine similarity matrix between all persons in frame t and j.
    2. Greedily pick the highest-similarity unmatched pair until one side is exhausted.
    3. Aggregate matched-pair similarities via mean or max.
    """
    # r_t: (N_t, 512), r_j: (N_j, 512)
    sim_matrix = np.dot(r_t, r_j.T)
    sim_matrix = np.clip(sim_matrix, -1.0, 1.0)

    n_t, n_j = sim_matrix.shape
    num_matches = min(n_t, n_j)
    if num_matches == 0:
        return 0.0

    working_matrix = sim_matrix.copy()
    matched_sims = []
    for _ in range(num_matches):
        idx = np.argmax(working_matrix)
        row, col = np.unravel_index(idx, working_matrix.shape)
        matched_sims.append(float(working_matrix[row, col]))
        working_matrix[row, :] = -np.inf
        working_matrix[:, col] = -np.inf

    if agg_method == "max":
        return float(np.max(matched_sims))
    elif agg_method == "mean":
        return float(np.mean(matched_sims))
    else:
        raise ValueError(f"Unknown aggregation method: {agg_method}. Choose 'mean' or 'max'.")


def compute_pairwise_redundancy(
    d_t: np.ndarray,
    d_j: np.ndarray,
    r_t: np.ndarray | None,
    r_j: np.ndarray | None,
    alpha: float = KEYFRAME_ALPHA,
    reid_matching: str = "greedy",
    agg_method: str = "mean",
    debug: bool = False,
    frame_pair: tuple[int, int] | None = None,
) -> tuple[float, bool]:
    """
    Compute fused multimodal redundancy score between candidate frame t and accepted frame j.

    s_redund(t, j) = alpha * s_D + (1 - alpha) * s_R
    Falls back to s_D alone if either frame lacks detected person ReID embeddings.

    Returns:
        (score, fallback_triggered)
    """
    # DINO cosine similarity (unit-L2 normalized dot product)
    s_d = float(np.dot(d_t, d_j))
    s_d = max(-1.0, min(1.0, s_d))

    fallback = (r_t is None or r_j is None or len(r_t) == 0 or len(r_j) == 0)

    if not fallback:
        if reid_matching == "greedy":
            assert r_t.ndim == 2 and r_j.ndim == 2, (
                f"Greedy matching requires 2D arrays (N, 512), got {r_t.shape} and {r_j.shape}"
            )
            s_r = compute_greedy_set_reid_similarity(r_t, r_j, agg_method=agg_method)
            if debug:
                pair_str = (
                    f" frame t={frame_pair[0]} (N_t={len(r_t)}) vs keyframe j={frame_pair[1]} (N_j={len(r_j)})"
                    if frame_pair else ""
                )
                print(f"[DEBUG:GREEDY] compute_greedy_set_reid_similarity executed for{pair_str} -> s_R={s_r:.4f}")
        elif reid_matching == "dominant":
            v_t = r_t[0] if r_t.ndim == 2 else r_t
            v_j = r_j[0] if r_j.ndim == 2 else r_j
            assert v_t.ndim == 1 and v_j.ndim == 1, (
                f"Dominant matching requires 1D vectors (512,), got {v_t.shape} and {v_j.shape}"
            )
            s_r = float(np.dot(v_t, v_j))
            s_r = max(-1.0, min(1.0, s_r))
            if debug:
                pair_str = (
                    f" frame t={frame_pair[0]} vs keyframe j={frame_pair[1]}"
                    if frame_pair else ""
                )
                print(f"[DEBUG:DOMINANT] 1-to-1 dot product executed for{pair_str} -> s_R={s_r:.4f}")
        else:
            raise ValueError(f"Unknown reid_matching mode: {reid_matching}")
        score = alpha * s_d + (1.0 - alpha) * s_r
    else:
        # Fallback to scene visual similarity alone
        score = s_d

    return float(score), fallback


def select_keyframes_for_view(
    view_name: str,
    shot_window_sec: float = KEYFRAME_SHOT_WINDOW_SEC,
    alpha: float = KEYFRAME_ALPHA,
    k: float = KEYFRAME_K,
    default_epsilon: float = KEYFRAME_DEFAULT_EPSILON,
    fps: float = DEFAULT_FPS,
    reid_matching: str = "greedy",
    agg_method: str = "mean",
    debug: bool = False,
    max_debug_prints: int = 5,
) -> tuple[list[dict], dict]:
    """
    Execute adaptive within-shot keyframe selection for a given view.

    Returns:
        decision_rows: List of decision dicts for manifest logging.
        metrics: Summary metrics dict.
    """
    dino_map = load_dino_embeddings(view_name)
    if reid_matching == "dominant":
        reid_map = load_dominant_reid_embeddings(view_name)
    else:
        reid_map = load_all_reid_embeddings(view_name)

    sorted_frames = sorted(dino_map.keys())
    total_frames = len(sorted_frames)

    frames_per_shot = max(1, int(round(shot_window_sec * fps)))

    # Segment frames into shots
    shots = defaultdict(list)
    for f_idx in sorted_frames:
        shot_id = f_idx // frames_per_shot
        shots[shot_id].append(f_idx)

    decision_rows = []
    accepted_keyframe_indices = []
    debug_prints_count = 0

    for shot_id in sorted(shots.keys()):
        shot_frames = shots[shot_id]
        shot_keyframes = []
        shot_observed_scores = []

        for frame_t in shot_frames:
            d_t = dino_map[frame_t]
            r_t = reid_map.get(frame_t, None)
            num_persons_t = len(r_t) if r_t is not None else 0

            # Rule: Always accept the first frame of each shot (seed keyframe)
            if len(shot_keyframes) == 0:
                accepted = True
                max_redund = 0.0
                eps_used = default_epsilon
                shot_keyframes.append(frame_t)
                accepted_keyframe_indices.append(frame_t)
                best_num_persons_j = 0
                best_fallback = 0
            else:
                # Compare against all previously accepted keyframes in this shot
                pairwise_scores = []
                fallback_flags = []
                num_persons_j_list = []

                for frame_j in shot_keyframes:
                    d_j = dino_map[frame_j]
                    r_j = reid_map.get(frame_j, None)

                    should_debug = False
                    if debug and debug_prints_count < max_debug_prints:
                        # Print debug for frames with 2+ persons on both sides (or >=1 for dominant)
                        n_t_check = len(r_t) if (r_t is not None and r_t.ndim == 2) else (1 if r_t is not None else 0)
                        n_j_check = len(r_j) if (r_j is not None and r_j.ndim == 2) else (1 if r_j is not None else 0)
                        if (reid_matching == "greedy" and n_t_check >= 2 and n_j_check >= 2) or (reid_matching == "dominant" and r_t is not None and r_j is not None):
                            should_debug = True
                            debug_prints_count += 1

                    score, fallback = compute_pairwise_redundancy(
                        d_t,
                        d_j,
                        r_t,
                        r_j,
                        alpha=alpha,
                        reid_matching=reid_matching,
                        agg_method=agg_method,
                        debug=should_debug,
                        frame_pair=(frame_t, frame_j),
                    )
                    pairwise_scores.append(score)
                    fallback_flags.append(fallback)
                    num_persons_j_list.append(len(r_j) if r_j is not None else 0)
                    shot_observed_scores.append(score)

                best_match_idx = int(np.argmax(pairwise_scores))
                max_redund = pairwise_scores[best_match_idx]
                best_fallback = int(fallback_flags[best_match_idx])
                best_num_persons_j = num_persons_j_list[best_match_idx]

                # Compute adaptive threshold epsilon_shot = mu - k * sigma
                if len(shot_observed_scores) >= 2:
                    mu = float(np.mean(shot_observed_scores))
                    sigma = float(np.std(shot_observed_scores))
                    eps_used = mu - k * sigma
                else:
                    eps_used = default_epsilon

                # Keyframe acceptance condition
                accepted = max_redund < eps_used
                if accepted:
                    shot_keyframes.append(frame_t)
                    accepted_keyframe_indices.append(frame_t)

            decision_rows.append({
                "view": view_name,
                "frame_idx": frame_t,
                "shot_id": shot_id,
                "accepted": int(accepted),
                "max_redund_score": round(max_redund, 4),
                "epsilon_used": round(eps_used, 4),
                "num_persons_t": num_persons_t,
                "num_persons_j": best_num_persons_j,
                "fallback_triggered": best_fallback,
            })

    # Compute compression metrics
    num_accepted = len(accepted_keyframe_indices)
    compression_ratio = total_frames / num_accepted if num_accepted > 0 else 0.0
    reduction_pct = (1.0 - (num_accepted / total_frames)) * 100.0 if total_frames > 0 else 0.0
    avg_shot_size = total_frames / len(shots) if shots else 0.0
    avg_keyframes_per_shot = num_accepted / len(shots) if shots else 0.0

    metrics = {
        "view": view_name,
        "total_frames": total_frames,
        "accepted_keyframes": num_accepted,
        "num_shots": len(shots),
        "avg_shot_size": avg_shot_size,
        "avg_keyframes_per_shot": avg_keyframes_per_shot,
        "compression_ratio": compression_ratio,
        "reduction_pct": reduction_pct,
    }

    return decision_rows, metrics


def save_keyframe_decisions(
    view_name: str,
    decision_rows: list[dict],
    suffix: str | None = None,
) -> Path:
    """Save keyframe decisions manifest to data_manifests/keyframe_decisions_{view}[_{suffix}].csv."""
    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    if suffix:
        clean_suffix = f"_{suffix.lstrip('_')}"
        out_path = MANIFESTS_DIR / f"keyframe_decisions_{view_name}{clean_suffix}.csv"
    else:
        out_path = MANIFESTS_DIR / f"keyframe_decisions_{view_name}.csv"

    fieldnames = [
        "view",
        "frame_idx",
        "shot_id",
        "accepted",
        "max_redund_score",
        "epsilon_used",
        "num_persons_t",
        "num_persons_j",
        "fallback_triggered",
    ]
    with open(out_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(decision_rows)

    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Adaptive Keyframe Selection using DINO ViT and OSNet ReID Embeddings."
    )
    parser.add_argument(
        "--view",
        type=str,
        default="all",
        choices=["all"] + SUPPORTED_VIEWS,
        help="View to process (default: all)",
    )
    parser.add_argument(
        "--shot-window-sec",
        type=float,
        default=KEYFRAME_SHOT_WINDOW_SEC,
        help=f"Fixed shot duration in seconds (default: {KEYFRAME_SHOT_WINDOW_SEC})",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=KEYFRAME_ALPHA,
        help=f"DINO weight vs ReID weight alpha in [0, 1] (default: {KEYFRAME_ALPHA})",
    )
    parser.add_argument(
        "--k",
        type=float,
        default=KEYFRAME_K,
        help=f"Standard deviation multiplier k for epsilon = mu - k * sigma (default: {KEYFRAME_K})",
    )
    parser.add_argument(
        "--default-epsilon",
        type=float,
        default=KEYFRAME_DEFAULT_EPSILON,
        help=f"Baseline epsilon before sufficient shot statistics (default: {KEYFRAME_DEFAULT_EPSILON})",
    )
    parser.add_argument(
        "--reid-matching",
        type=str,
        default="greedy",
        choices=["greedy", "dominant"],
        help="ReID matching strategy: set-based 'greedy' or 'dominant' single person (default: greedy)",
    )
    parser.add_argument(
        "--reid-aggregation",
        "--reid-agg",
        dest="reid_aggregation",
        type=str,
        default="mean",
        choices=["mean", "max"],
        help="Aggregation for matched pair similarities: 'mean' or 'max' (default: mean)",
    )
    parser.add_argument(
        "--output-suffix",
        type=str,
        default=None,
        help="Suffix for output CSV filename. If omitted, defaults to '<matching>_k<k>' "
             "(e.g. setbased_k1.0) to preserve baseline without overwriting. "
             "Pass empty string '' to write unsuffixed.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print debug trace of ReID matching calls for sample multi-person frames.",
    )

    args = parser.parse_args()

    # Suffix handling
    if args.output_suffix is not None:
        suffix = args.output_suffix if args.output_suffix != "" else None
    else:
        suffix = f"setbased_k{args.k}" if args.reid_matching == "greedy" else f"dominant_k{args.k}"

    print("=" * 75)
    print("Novelty #1: Adaptive Keyframe Selection (DINO ViT + OSNet ReID)")
    print("=" * 75)
    print(f"Target View      : {args.view}")
    print(f"Shot Window      : {args.shot_window_sec:.1f} s (~{int(args.shot_window_sec * DEFAULT_FPS)} frames)")
    print(f"Fusion Alpha     : {args.alpha:.2f} (DINO: {args.alpha:.2f}, ReID: {1.0 - args.alpha:.2f})")
    print(f"Adaptive Factor k: {args.k:.2f} (epsilon = mu - {args.k:.2f} * sigma)")
    print(f"Baseline Epsilon : {args.default_epsilon:.2f}")
    print(f"ReID Matching    : {args.reid_matching} (set-based)" if args.reid_matching == "greedy" else f"ReID Matching    : {args.reid_matching}")
    print(f"ReID Aggregation : {args.reid_aggregation}")
    print(f"Output Suffix    : {suffix if suffix else '[None - writing baseline keyframe_decisions_{view}.csv]'}")
    print("-" * 75)

    views_to_process = SUPPORTED_VIEWS if args.view == "all" else [args.view]
    all_metrics = []

    for v in views_to_process:
        print(f"[INFO] Processing keyframe selection for {v}...")
        decision_rows, metrics = select_keyframes_for_view(
            view_name=v,
            shot_window_sec=args.shot_window_sec,
            alpha=args.alpha,
            k=args.k,
            default_epsilon=args.default_epsilon,
            fps=DEFAULT_FPS,
            reid_matching=args.reid_matching,
            agg_method=args.reid_aggregation,
            debug=args.debug,
        )
        out_csv = save_keyframe_decisions(v, decision_rows, suffix=suffix)
        all_metrics.append(metrics)
        print(f"       -> Decisions logged to: {out_csv}")

    print("=" * 75)
    print("KEYFRAME SELECTION SUMMARY REPORT & COMPRESSION METRICS")
    print("=" * 75)
    print(
        f"{'View':<8} | {'Total Frames':<12} | {'Keyframes':<10} | "
        f"{'Shots':<6} | {'Avg Shot':<10} | {'Compression':<12} | {'Reduction %':<10}"
    )
    print("-" * 75)

    total_all_frames = sum(m["total_frames"] for m in all_metrics)
    total_all_keyframes = sum(m["accepted_keyframes"] for m in all_metrics)

    for m in all_metrics:
        print(
            f"{m['view']:<8} | {m['total_frames']:<12} | {m['accepted_keyframes']:<10} | "
            f"{m['num_shots']:<6} | {m['avg_shot_size']:<10.1f} | "
            f"{m['compression_ratio']:<11.2f}x | {m['reduction_pct']:<9.1f}%"
        )

    if len(all_metrics) > 1:
        overall_comp = total_all_frames / total_all_keyframes if total_all_keyframes > 0 else 0.0
        overall_reduct = (1.0 - (total_all_keyframes / total_all_frames)) * 100.0 if total_all_frames > 0 else 0.0
        print("-" * 75)
        print(
            f"{'OVERALL':<8} | {total_all_frames:<12} | {total_all_keyframes:<10} | "
            f"{sum(m['num_shots'] for m in all_metrics):<6} | {'-':<10} | "
            f"{overall_comp:<11.2f}x | {overall_reduct:<9.1f}%"
        )
    print("=" * 75)


if __name__ == "__main__":
    main()
