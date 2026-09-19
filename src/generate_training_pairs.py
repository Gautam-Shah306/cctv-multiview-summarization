import sys
import csv
import random
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

# Group 3 - local / project
from config import DATA_DIR
from src.build_summary import (
    reconstruct_shot_windows,
    link_shots_to_global_identities,
    compute_vq_score,
    MANIFESTS_DIR,
    CLUSTER_TOLERANCE_FRAMES
)

def generate_pairs():
    # 1. Reconstruct and link shots
    print("[INFO] Loading and linking shots...")
    shots = reconstruct_shot_windows(suffix="")
    
    tf_path = MANIFESTS_DIR / "tracklet_features_stitched.csv"
    gi_path = MANIFESTS_DIR / "global_identities_v2.csv"
    
    shots = link_shots_to_global_identities(shots, tf_path, gi_path)
    shots = compute_vq_score(shots)
    
    # Sort shots primarily by start frame to make time comparison easier
    shots.sort(key=lambda x: x["window_start_frame"])
    
    print(f"[INFO] Processed {len(shots)} shots.")
    
    positives = []
    hard_negatives = []
    easy_negatives = []
    
    MAX_SLOT_SPAN = 750
    
    # 2. Generate Pairs
    print("[INFO] Generating positive and negative pairs...")
    
    for i in range(len(shots)):
        for j in range(i + 1, len(shots)):
            s_a = shots[i]
            s_b = shots[j]
            
            # Extract linked IDs (filtered of broad IDs as per build_summary)
            # Actually, let's use all linked_global_ids or filtered?
            # We'll use filtered_linked_ids because broad IDs are noise.
            ids_a = set(s_a.get("filtered_linked_ids", []))
            ids_b = set(s_b.get("filtered_linked_ids", []))
            
            shared_ids = ids_a.intersection(ids_b)
            
            # Temporal gap
            # Gap > 0 means they don't overlap. Gap <= 0 means overlap.
            gap = max(0, max(s_a["window_start_frame"], s_b["window_start_frame"]) - 
                         min(s_a["window_end_frame"], s_b["window_end_frame"]))
            
            # Classify
            if shared_ids:
                if s_a["view"] != s_b["view"]:
                    # Cross-view Positive (assuming any time overlap/closeness, but we'll take all with shared ID)
                    if gap <= CLUSTER_TOLERANCE_FRAMES:
                        positives.append((s_a, s_b, "positive_cross_view"))
                else:
                    # Intra-view Positive
                    if gap <= MAX_SLOT_SPAN:
                        positives.append((s_a, s_b, "positive_intra_view"))
            else:
                if s_a["view"] != s_b["view"]:
                    # TIGHTENED: Hard negatives must be <= 500 frames apart (2x shot window)
                    if gap <= 500:
                        hard_negatives.append((s_a, s_b, "hard_negative"))
                    else:
                        easy_negatives.append((s_a, s_b, "easy_negative"))
                else:
                    # Same view without shared IDs. If they are close in time, it might just be different people.
                    # We'll consider them easy negatives to prioritize cross-view hard negatives.
                    easy_negatives.append((s_a, s_b, "easy_negative"))
                    
    print(f"[INFO] Found {len(positives)} total positive pairs.")
    print(f"[INFO] Found {len(hard_negatives)} total hard negative pairs.")
    print(f"[INFO] Found {len(easy_negatives)} total easy negative pairs.")
    
    # 3. Balancing
    # We want 1:1 positive:negative.
    # Negatives should be 50% hard, 50% easy (or whatever ratio maxes out hard negatives)
    target_neg_count = len(positives)
    
    target_hard = min(len(hard_negatives), target_neg_count // 2)
    target_easy = target_neg_count - target_hard
    
    # If not enough easy negatives (unlikely), take more hard ones
    if target_easy > len(easy_negatives):
        target_easy = len(easy_negatives)
        target_hard = min(len(hard_negatives), target_neg_count - target_easy)
        
    sampled_hard = random.sample(hard_negatives, target_hard) if target_hard > 0 else []
    sampled_easy = random.sample(easy_negatives, target_easy) if target_easy > 0 else []
    
    final_pairs = []
    # Add positives
    for s_a, s_b, ptype in positives:
        final_pairs.append({
            "shot_A_id": s_a["shot_id"],
            "shot_A_view": s_a["view"],
            "shot_B_id": s_b["shot_id"],
            "shot_B_view": s_b["view"],
            "label": 1,
            "pair_type": ptype,
            "shot_A_vq": s_a.get("vq_score", 0.0),
            "shot_B_vq": s_b.get("vq_score", 0.0),
            "shot_A_start": s_a["window_start_frame"],
            "shot_A_end": s_a["window_end_frame"],
            "shot_B_start": s_b["window_start_frame"],
            "shot_B_end": s_b["window_end_frame"],
            "shared_ids": ";".join(map(str, sorted(set(s_a.get("filtered_linked_ids", [])).intersection(set(s_b.get("filtered_linked_ids", []))))))
        })
        
    # Add negatives
    for s_a, s_b, ptype in (sampled_hard + sampled_easy):
        final_pairs.append({
            "shot_A_id": s_a["shot_id"],
            "shot_A_view": s_a["view"],
            "shot_B_id": s_b["shot_id"],
            "shot_B_view": s_b["view"],
            "label": 0,
            "pair_type": ptype,
            "shot_A_vq": s_a.get("vq_score", 0.0),
            "shot_B_vq": s_b.get("vq_score", 0.0),
            "shot_A_start": s_a["window_start_frame"],
            "shot_A_end": s_a["window_end_frame"],
            "shot_B_start": s_b["window_start_frame"],
            "shot_B_end": s_b["window_end_frame"],
            "shared_ids": ""
        })
        
    print(f"[INFO] Final dataset: {len(positives)} pos, {len(sampled_hard)} hard neg, {len(sampled_easy)} easy neg.")
    
    # Print sanity check
    print("\n--- SANITY CHECK: 3 POSITIVE PAIRS ---")
    for row in random.sample(final_pairs[:len(positives)], min(3, len(positives))):
        print(f"Pos ({row['pair_type']}): {row['shot_A_view']}/shot_{row['shot_A_id']} ({row['shot_A_start']}-{row['shot_A_end']}) "
              f"& {row['shot_B_view']}/shot_{row['shot_B_id']} ({row['shot_B_start']}-{row['shot_B_end']}) "
              f"| Shared IDs: {row['shared_ids']}")
              
    print("\n--- SANITY CHECK: 3 HARD NEGATIVE PAIRS ---")
    hards = [r for r in final_pairs if r["pair_type"] == "hard_negative"]
    for row in random.sample(hards, min(3, len(hards))):
        print(f"Hard Neg: {row['shot_A_view']}/shot_{row['shot_A_id']} ({row['shot_A_start']}-{row['shot_A_end']}) "
              f"& {row['shot_B_view']}/shot_{row['shot_B_id']} ({row['shot_B_start']}-{row['shot_B_end']}) "
              f"| Shared IDs: {row['shared_ids']}")
              
    print("-" * 40)
    
    # 4. Extract unique shots and their features
    unique_shots = {}
    for pair in final_pairs:
        k_a = f"{pair['shot_A_view']}_shot_{pair['shot_A_id']}"
        if k_a not in unique_shots:
            # Find the shot object
            s_a = next(s for s in shots if s["view"] == pair['shot_A_view'] and s["shot_id"] == pair['shot_A_id'])
            unique_shots[k_a] = s_a
            
        k_b = f"{pair['shot_B_view']}_shot_{pair['shot_B_id']}"
        if k_b not in unique_shots:
            s_b = next(s for s in shots if s["view"] == pair['shot_B_view'] and s["shot_id"] == pair['shot_B_id'])
            unique_shots[k_b] = s_b
            
    print(f"[INFO] Extracting features for {len(unique_shots)} unique shots...")
    
    dino_features = {}
    reid_features = {}
    
    # Load all DINO npz files
    dino_data = {}
    for view in ["view1", "view2", "view3"]:
        dino_path = DATA_DIR / "dinov3_embeddings" / f"{view}_dinov3.npz"
        if dino_path.exists():
            d = np.load(dino_path)
            # Map frame_idx -> embedding
            frames = d["frame_indices"]
            embeds = d["embeddings"]
            dino_data[view] = {frames[i]: embeds[i] for i in range(len(frames))}
        else:
            dino_data[view] = {}
            
    # Load ReID tracklet features
    tf_df = pd.read_csv(tf_path)
    tracklet_npz = np.load(DATA_DIR / "tracklet_embeddings_stitched.npz")
    
    for shot_key, shot in unique_shots.items():
        view = shot["view"]
        
        # DINOv3
        kf = int(shot["keyframe_frame_idx"])
        if kf in dino_data[view]:
            dino_features[shot_key] = dino_data[view][kf]
        else:
            print(f"[WARNING] Missing DINOv3 embedding for {shot_key} at frame {kf}. Using zeros.")
            dino_features[shot_key] = np.zeros(384, dtype=np.float32)
            
        # ReID
        # Find all tracklets in this view that overlap the shot window
        s_start = shot["window_start_frame"]
        s_end = shot["window_end_frame"]
        
        view_tf = tf_df[tf_df["view"] == view]
        overlaps = view_tf[~((view_tf["end_frame"] < s_start) | (view_tf["start_frame"] > s_end))]
        
        shot_reids = []
        for _, row in overlaps.iterrows():
            t_id = int(row["stitched_track_id"] if "stitched_track_id" in row else row["track_id"])
            t_key = f"{view}_{t_id}"
            if t_key in tracklet_npz:
                shot_reids.append(tracklet_npz[t_key])
                
        if shot_reids:
            reid_features[shot_key] = np.vstack(shot_reids)
        else:
            # Empty shot fallback
            print(f"[WARNING] No ReID embeddings found for {shot_key}. Using zeros.")
            reid_features[shot_key] = np.zeros((1, 512), dtype=np.float32)
            
    # 5. Write outputs
    df_pairs = pd.DataFrame(final_pairs)
    # Add an explicit pair_id
    df_pairs.insert(0, "pair_id", df_pairs.index)
    
    out_csv = MANIFESTS_DIR / "training_pairs.csv"
    df_pairs.to_csv(out_csv, index=False)
    
    out_dino = MANIFESTS_DIR / "training_features_dino.npz"
    np.savez(out_dino, **dino_features)
    
    out_reid = MANIFESTS_DIR / "training_features_reid.npz"
    np.savez(out_reid, **reid_features)
    
    print(f"\n[DONE] Wrote {len(df_pairs)} pairs to {out_csv}")
    print(f"[DONE] Wrote DINOv3 embeddings to {out_dino}")
    print(f"[DONE] Wrote ReID embeddings (un-pooled, exact set of tracklets) to {out_reid}")

if __name__ == "__main__":
    generate_pairs()
