"""
build_summary.py

Produces a first-pass multi-view summary by reconciling per-view keyframe shots,
cross-view global identities, and object detection data.

This script operates purely on structured CSV manifests and does not require GPU
or model inference. By default, it now uses the stitched v2 tracklet and global 
identity files for cleaner cross-view clustering.

Usage:
    python -m src.build_summary
"""

# Group 1 — standard library
import csv
import sys
from collections import defaultdict
from pathlib import Path

# Group 2 — third-party
import numpy as np
import pandas as pd

# Group 3 — local / project
from config import DATA_DIR

# --- Module-Level Constants ---
VIEWS = ["view1", "view2", "view3"]
MANIFESTS_DIR = Path(__file__).resolve().parent.parent / "data_manifests"
CLUSTER_TOLERANCE_FRAMES = 5
FRAME_WIDTH = 360
FRAME_HEIGHT = 288
TOTAL_AREA = FRAME_WIDTH * FRAME_HEIGHT
MIN_OVERLAP_FRAC = 0.25
BROAD_ID_THRESHOLD = 0.15


def reconstruct_shot_windows(suffix: str = "") -> list[dict]:
    """
    For each view, load keyframe_decisions_viewX.csv, filter to accepted == 1,
    group by shot_id, and compute window bounds and best keyframe.
    """
    all_shots = []
    for view in VIEWS:
        filename = f"keyframe_decisions_{view}_{suffix}.csv" if suffix else f"keyframe_decisions_{view}.csv"
        path = MANIFESTS_DIR / filename
        if not path.exists():
            print(f"[ERROR] Required input file missing: {path}", file=sys.stderr)
            sys.exit(1)
        
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"[ERROR] Failed to read {path}: {e}", file=sys.stderr)
            sys.exit(1)
            
        if df.empty:
            print(f"[INFO] {view}: No keyframe decisions found (empty CSV).")
            continue
            
        accepted = df[df["accepted"] == 1]
        if accepted.empty:
            print(f"[INFO] {view}: No accepted keyframes found.")
            continue
            
        grouped = df.groupby("shot_id")
        
        for shot_id, group in grouped:
            acc_group = group[group["accepted"] == 1]
            if acc_group.empty:
                continue
                
            window_start = int(group["frame_idx"].min())
            window_end = int(group["frame_idx"].max())
            
            # Keyframe is the one with the lowest max_redund_score in the accepted group
            best_idx = acc_group["max_redund_score"].idxmin()
            keyframe_idx = int(acc_group.loc[best_idx, "frame_idx"])
            
            all_shots.append({
                "view": view,
                "shot_id": int(shot_id),
                "window_start_frame": window_start,
                "window_end_frame": window_end,
                "keyframe_frame_idx": keyframe_idx
            })
            
    return all_shots



def link_shots_to_global_identities(shots: list[dict], tf_path: Path, gi_path: Path) -> list[dict]:
    """
    Link each shot to global identities by finding overlapping tracklets.
    """
    if not tf_path.exists():
        print(f"[ERROR] Required input file missing: {tf_path}", file=sys.stderr)
        sys.exit(1)
    if not gi_path.exists():
        print(f"[ERROR] Required input file missing: {gi_path}", file=sys.stderr)
        sys.exit(1)
        
    tf_df = pd.read_csv(tf_path)
    gi_df = pd.read_csv(gi_path)
    
    if tf_df.empty or gi_df.empty:
        print("[INFO] Tracklet or global identities CSV is empty. No identities will be linked.")
        gi_map = {}
    else:
        # Create mapping from (view, track_id) to global_id
        gi_map = gi_df.set_index(["view", "track_id"])["global_id"].to_dict()
    
    shot_to_overlaps = defaultdict(list)
    global_id_shot_counts = defaultdict(int)
    
    for shot in shots:
        view = shot["view"]
        s_start = shot["window_start_frame"]
        s_end = shot["window_end_frame"]
        s_frames = s_end - s_start + 1
        
        if not tf_df.empty:
            view_tf = tf_df[tf_df["view"] == view]
            # Overlap condition: not (t_end < s_start or t_start > s_end)
            overlaps = view_tf[~((view_tf["end_frame"] < s_start) | (view_tf["start_frame"] > s_end))]
            
            for _, row in overlaps.iterrows():
                t_start = int(row["start_frame"])
                t_end = int(row["end_frame"])
                o_start = max(s_start, t_start)
                o_end = min(s_end, t_end)
                overlap_frames = max(0, o_end - o_start + 1)
                
                if overlap_frames / s_frames >= MIN_OVERLAP_FRAC:
                    track_id = int(row["stitched_track_id"]) if "stitched_track_id" in row else int(row["track_id"])
                    key = (view, track_id)
                    if key in gi_map:
                        shot_to_overlaps[id(shot)].append((gi_map[key], overlap_frames))
                        
    # Count distinct shots per global_id for filtering
    for shot in shots:
        unique_gids = set(gid for gid, _ in shot_to_overlaps[id(shot)])
        for gid in unique_gids:
            global_id_shot_counts[gid] += 1
            
    # Identify broad IDs
    total_shots = len(shots)
    broad_ids = set()
    for gid, count in global_id_shot_counts.items():
        if count / total_shots > BROAD_ID_THRESHOLD:
            broad_ids.add(gid)
            print(f"[INFO] Excluded suspiciously broad Global ID {gid} (linked to {count}/{total_shots} shots, >{BROAD_ID_THRESHOLD*100:.0f}%)")
            
    zero_count = 0
    one_count = 0
    multi_count = 0
    
    for shot in shots:
        overlaps = shot_to_overlaps[id(shot)]
        linked_gids = set(gid for gid, _ in overlaps)
        shot["linked_global_ids"] = list(linked_gids)
        shot["filtered_linked_ids"] = list(linked_gids - broad_ids)
        
        # Dominant ID: max overlap_frames, excluding broad_ids
        dominant_id = None
        max_overlap = -1
        
        gid_overlaps = defaultdict(int)
        for gid, frames in overlaps:
            gid_overlaps[gid] += frames
            
        for gid, frames in gid_overlaps.items():
            if gid not in broad_ids and frames > max_overlap:
                max_overlap = frames
                dominant_id = gid
                
        shot["dominant_global_id"] = dominant_id
        
        if len(linked_gids) == 0:
            zero_count += 1
        elif len(linked_gids) == 1:
            one_count += 1
        else:
            multi_count += 1
            
    print(f"[INFO] Identity linking: {zero_count} shots with 0 IDs, {one_count} with 1 ID, {multi_count} with >1 IDs.")
    return shots


def compute_vq_score(shots: list[dict]) -> list[dict]:
    """
    Compute the View Quality (VQ) score for each shot based on the real VQ score
    computed via camera calibration (loaded from data_manifests/vq_scores_viewX.csv).
    """
    shots_by_view = defaultdict(list)
    for i, shot in enumerate(shots):
        shots_by_view[shot["view"]].append((i, shot))
        
    for view, view_shots in shots_by_view.items():
        vq_path = MANIFESTS_DIR / f"vq_scores_{view}.csv"
        if not vq_path.exists():
            print(f"[ERROR] Required input file missing: {vq_path}", file=sys.stderr)
            sys.exit(1)
            
        vq_df = pd.read_csv(vq_path)
        
        for idx, shot in view_shots:
            if vq_df.empty:
                shots[idx]["vq_score"] = 0.0
                continue
                
            # Filter detections to this shot's window
            mask = (vq_df["frame_idx"] >= shot["window_start_frame"]) & (vq_df["frame_idx"] <= shot["window_end_frame"])
            shot_dets = vq_df[mask]
            
            if shot_dets.empty:
                shots[idx]["vq_score"] = 0.0
                continue
                
            # Aggregate across the shot: we use the max score to represent the best view of any person in that shot
            shots[idx]["vq_score"] = float(shot_dets["vq_score"].max())
            
    return shots


def build_event_clusters(shots: list[dict], max_slot_span_frames: int = 750) -> tuple[list[dict], list[list[dict]]]:
    """
    TWO-PHASE CLUSTERING:
    Phase 1: Time-slot bucketing (no identity involved) to prevent identity sharing
             from ever chaining across unrelated moments in the video.
    Phase 2: Identity-based merging restricted to within each time slot.
    """
    # Phase 1: Time-slot bucketing
    sorted_shots = sorted(shots, key=lambda s: s["window_start_frame"])
    time_slots = []
    
    for shot in sorted_shots:
        if not time_slots:
            time_slots.append([shot])
            continue
            
        last_slot = time_slots[-1]
        
        # Calculate span of last slot
        slot_min_start = min(s["window_start_frame"] for s in last_slot)
        slot_max_end = max(s["window_end_frame"] for s in last_slot)
        
        # Check time overlap with tolerance
        overlap = not (
            shot["window_end_frame"] < slot_min_start - CLUSTER_TOLERANCE_FRAMES or
            shot["window_start_frame"] > slot_max_end + CLUSTER_TOLERANCE_FRAMES
        )
        
        # Check if merging would exceed max_slot_span_frames
        new_min_start = min(slot_min_start, shot["window_start_frame"])
        new_max_end = max(slot_max_end, shot["window_end_frame"])
        new_span = new_max_end - new_min_start
        
        if overlap and new_span <= max_slot_span_frames:
            last_slot.append(shot)
        else:
            time_slots.append([shot])
            
    print(f"[INFO] Phase 1: Grouped {len(shots)} shots into {len(time_slots)} time slots.")
    slot_sizes = defaultdict(int)
    for slot in time_slots:
        slot_sizes[len(slot)] += 1
    for size in sorted(slot_sizes.keys()):
        print(f"  Slot size {size}: {slot_sizes[size]} slot(s)")

    # Phase 2: Identity-based merging within each slot only
    unclustered = []
    clusters = []
    
    for slot in time_slots:
        slot_unclustered = []
        slot_to_cluster = []
        
        for shot in slot:
            if shot.get("dominant_global_id") is None:
                slot_unclustered.append(shot)
            else:
                slot_to_cluster.append(shot)
                
        slot_clusters = []
        for shot in slot_to_cluster:
            matching_cluster_indices = []
            for i, cluster in enumerate(slot_clusters):
                for c_shot in cluster:
                    # Time overlap within the slot
                    overlap_time = not (
                        shot["window_end_frame"] < c_shot["window_start_frame"] - CLUSTER_TOLERANCE_FRAMES or
                        shot["window_start_frame"] > c_shot["window_end_frame"] + CLUSTER_TOLERANCE_FRAMES
                    )
                    
                    if not overlap_time:
                        continue
                        
                    if shot["view"] == c_shot["view"]:
                        # SAME VIEW: strict dominant ID match
                        if shot["dominant_global_id"] == c_shot["dominant_global_id"]:
                            matching_cluster_indices.append(i)
                            break
                    else:
                        # DIFFERENT VIEWS: looser shared ID match
                        shared_ids = set(shot["filtered_linked_ids"]).intersection(set(c_shot["filtered_linked_ids"]))
                        if shared_ids:
                            matching_cluster_indices.append(i)
                            break 
                            
            if not matching_cluster_indices:
                slot_clusters.append([shot])
            elif len(matching_cluster_indices) == 1:
                slot_clusters[matching_cluster_indices[0]].append(shot)
            else:
                # Merge multiple clusters
                merged_cluster = [shot]
                for idx in sorted(matching_cluster_indices, reverse=True):
                    merged_cluster.extend(slot_clusters.pop(idx))
                slot_clusters.append(merged_cluster)
                
        unclustered.extend(slot_unclustered)
        clusters.extend(slot_clusters)
        
    print(f"[INFO] Phase 2: Built {len(clusters)} cross-view event clusters.")
    return unclustered, clusters


def select_representatives_and_assemble(unclustered: list[dict], clusters: list[list[dict]]) -> list[dict]:
    """
    Pick the shot with the highest vq_score for each cluster, logging drops.
    Assemble final chronological summary.
    """
    final_summary = []
    
    for i, cluster in enumerate(clusters):
        cluster_id = i + 1
        if len(cluster) == 1:
            rep = cluster[0]
            rep["cluster_id"] = cluster_id
            final_summary.append(rep)
            continue
            
        # Pick highest vq_score
        rep = max(cluster, key=lambda s: s["vq_score"])
        rep["cluster_id"] = cluster_id
        final_summary.append(rep)
        
        # Log drops
        rep_desc = f"{rep['view']}/shot_{rep['shot_id']}"
        for shot in cluster:
            if shot is not rep:
                dropped_desc = f"{shot['view']}/shot_{shot['shot_id']}"
                print(f"[INFO] Cluster {cluster_id}: dropped {dropped_desc} in favor of {rep_desc}")
                
    for shot in unclustered:
        shot["cluster_id"] = None
        final_summary.append(shot)
        
    # Sort chronologically by window_start_frame
    final_summary.sort(key=lambda s: s["window_start_frame"])
    
    # Assign sequence_order
    for i, shot in enumerate(final_summary):
        shot["sequence_order"] = i + 1
        
    return final_summary


def write_output(final_summary: list[dict], out_path: Path) -> None:
    """
    Write data_manifests/final_summary.csv.
    """
    fieldnames = [
        "sequence_order", "view", "shot_id", "window_start_frame", "window_end_frame",
        "keyframe_frame_idx", "vq_score", "cluster_id", "linked_global_ids"
    ]
    
    # Format linked_global_ids
    for shot in final_summary:
        shot["linked_global_ids"] = ";".join(map(str, sorted(shot.get("linked_global_ids", []))))
    
    with open(out_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(final_summary)
        
    print(f"[INFO] Wrote {len(final_summary)} shots to {out_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tracklet-manifest",
        type=Path,
        default=MANIFESTS_DIR / "tracklet_features_stitched.csv",
        help="Path to tracklet manifest",
    )
    parser.add_argument(
        "--global-identities",
        type=Path,
        default=MANIFESTS_DIR / "global_identities_v2.csv",
        help="Path to global identities manifest",
    )
    parser.add_argument(
        "--out-summary",
        type=Path,
        default=MANIFESTS_DIR / "final_summary_v2.csv",
        help="Output path for final summary",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="",
        help="Suffix for keyframe decisions files (e.g. 'w150')",
    )
    parser.add_argument(
        "--max-slot-span-frames",
        type=int,
        default=750,
        help="Maximum frame span for a single time slot cluster (default: 750)",
    )
    args = parser.parse_args()
    
    print("[INFO] Starting Multi-View Summary Build...")
    
    print("[INFO] Reconstructing shot windows...")
    shots = reconstruct_shot_windows(suffix=args.suffix)
    print(f"[INFO] Loaded {len(shots)} total shots.")
    
    print("[INFO] Linking shots to global identities...")
    shots = link_shots_to_global_identities(shots, args.tracklet_manifest, args.global_identities)
    
    print("[INFO] Computing View Quality scores...")
    shots = compute_vq_score(shots)
    
    print("[INFO] Building cross-view event clusters...")
    unclustered, clusters = build_event_clusters(shots, max_slot_span_frames=args.max_slot_span_frames)
    
    print("[INFO] Selecting cluster representatives...")
    final_summary = select_representatives_and_assemble(unclustered, clusters)
    
    # TASK 3 SAFETY NET: Drop any shot with ZERO linked global identities
    valid_summary = []
    dropped_count = 0
    for shot in final_summary:
        if len(shot.get("linked_global_ids", [])) == 0:
            print(f"[INFO] Rule-Based Path: Dropping shot {shot['view']}/shot_{shot.get('shot_id', 'unknown')} (frames {shot['window_start_frame']}-{shot['window_end_frame']}) - ZERO linked global identities.")
            dropped_count += 1
        else:
            valid_summary.append(shot)
    
    print(f"[INFO] Rule-Based Identity Filter: dropped {dropped_count} shots.")
    final_summary = valid_summary
    
    print("[INFO] Writing output manifest...")
    write_output(final_summary, args.out_summary)
    
    # Print [DONE] summary
    view_counts = defaultdict(int)
    for s in final_summary:
        view_counts[s["view"]] += 1
        
    print("\n[DONE] Summary Assembly Complete")
    print("-" * 40)
    print(f"Total Shots In:          {len(shots)}")
    print(f"Total Clusters Found:    {len(clusters)}")
    
    # Cluster size distribution
    cluster_sizes = defaultdict(int)
    for c in clusters:
        cluster_sizes[len(c)] += 1
    print("Cluster Size Distribution:")
    for size in sorted(cluster_sizes.keys()):
        print(f"  Size {size}: {cluster_sizes[size]} cluster(s)")
        
    print(f"\nTotal Final Shots Out:   {len(final_summary)}")
    print("Breakdown by View (Out):")
    for v in VIEWS:
        print(f"  {v}: {view_counts[v]}")
    print("-" * 40)


if __name__ == "__main__":
    main()
