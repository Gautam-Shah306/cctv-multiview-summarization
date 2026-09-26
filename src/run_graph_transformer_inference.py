"""
run_graph_transformer_inference.py

Retrains the sparse Graph Transformer on all available labeled pairs and uses it to predict
cross-view relationships and construct event clusters, exporting a final summary video.

Usage:
    python -m src.run_graph_transformer_inference
"""

import argparse
import itertools
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data

from src.train_graph_transformer_w150_sparse import GraphTransformerW150
from src.build_graph_w150_sparse import build_graph_w150_sparse

# --- Module-Level Constants ---
MANIFESTS_DIR = Path(__file__).resolve().parent.parent / "data_manifests"
EDGE_PREDICTION_THRESHOLD = 0.5
TRAINING_EPOCHS = 150
TRAINING_LR = 1e-3
TRAINING_WEIGHT_DECAY = 1e-4
INFERENCE_BATCH_SIZE = 1000
MIN_OVERLAP_FRAC = 0.25

def get_vq_score(shot_key: str, nodes_info: dict) -> float:
    return float(nodes_info[shot_key]["vq"])

def retrain_model(data: Data, device: torch.device) -> GraphTransformerW150:
    """Retrain on all available labeled pairs to maximize training data for inference."""
    print(f"[INFO] Retraining model on ALL labeled training pairs (612 edges) for {TRAINING_EPOCHS} epochs...")
    model = GraphTransformerW150().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=TRAINING_LR, weight_decay=TRAINING_WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss()
    
    # Use all edges in the data for training
    train_mask = torch.ones(data.edge_index.shape[1], dtype=torch.bool, device=device)
    
    for epoch in range(1, TRAINING_EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        train_edge_index = data.edge_index[:, train_mask]
        z = model(data.x, train_edge_index) 
        out = model.predict_edges(z, train_edge_index)
        loss = criterion(out, data.y[train_mask])
        loss.backward()
        optimizer.step()
        
    return model

def build_connected_components(edge_list: list[tuple[int, int]], num_nodes: int) -> list[list[int]]:
    """Build connected components from the predicted edge list."""
    adj = {i: set() for i in range(num_nodes)}
    for u, v in edge_list:
        adj[u].add(v)
        adj[v].add(u)
        
    visited = set()
    clusters = []
    
    for i in range(num_nodes):
        if i not in visited:
            cluster = set()
            queue = [i]
            visited.add(i)
            while queue:
                curr = queue.pop(0)
                cluster.add(curr)
                for neighbor in adj[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            clusters.append(list(cluster))
    return clusters

def export_summary_csv(clusters: list[list[int]], node_keys: list[str], nodes_info: dict, out_name: str) -> pd.DataFrame:
    """Format final shots, apply identity safety check, and export to CSV."""
    final_shots = []
    for c_idx, cluster in enumerate(clusters):
        rep_idx = max(cluster, key=lambda idx: get_vq_score(node_keys[idx], nodes_info))
        rep_key = node_keys[rep_idx]
        info = nodes_info[rep_key]
        
        final_shots.append({
            "sequence_order": "", 
            "view": info["view"],
            "window_start_frame": info["start"],
            "window_end_frame": info["end"],
            "cluster_id": c_idx,
            "vq_score": info["vq"],
            "cluster_size": len(cluster)
        })
    
    final_shots.sort(key=lambda x: x["window_start_frame"])
    
    # TASK 3 SAFETY NET: Drop shots with zero linked global identities
    tf_df = pd.read_csv(MANIFESTS_DIR / "tracklet_features_stitched.csv")
    gi_df = pd.read_csv(MANIFESTS_DIR / "global_identities_v2.csv")
    
    if tf_df.empty or gi_df.empty:
        gi_map = {}
    else:
        gi_map = gi_df.set_index(["view", "track_id"])["global_id"].to_dict()
        
    valid_shots = []
    dropped_count = 0
    for shot in final_shots:
        view = shot["view"]
        s_start = shot["window_start_frame"]
        s_end = shot["window_end_frame"]
        s_frames = s_end - s_start + 1
        
        linked_gids = set()
        if not tf_df.empty:
            view_tf = tf_df[tf_df["view"] == view]
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
                        linked_gids.add(gi_map[key])
                        
        if len(linked_gids) == 0:
            print(f"[INFO] GT Path: Dropping shot {view} (frames {s_start}-{s_end}) - ZERO linked global identities.")
            dropped_count += 1
        else:
            valid_shots.append(shot)
            
    print(f"[INFO] GT Path Identity Filter: dropped {dropped_count} shots.")
    final_shots = valid_shots
    
    out_path = MANIFESTS_DIR / out_name
    df = pd.DataFrame(final_shots)
    if not df.empty:
        df["sequence_order"] = range(1, len(df) + 1)
        df.to_csv(out_path, index=False)
        print(f"[INFO] Wrote {out_name} with {len(df)} sequences.")
    else:
        print(f"[WARNING] No valid sequences left to write for {out_name}.")
        
    return df

def generate_video(input_csv: str, output_mp4: str) -> None:
    """Calls assemble_summary_video script."""
    cmd = ["python", "-m", "src.assemble_summary_video", "--input", str(MANIFESTS_DIR / input_csv), "--output", str(MANIFESTS_DIR / output_mp4)]
    subprocess.run(cmd, check=True)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Running on {device}...")
    
    # 1. Load Data
    data = build_graph_w150_sparse()
    data = data.to(device)
    
    pairs_df = pd.read_csv(MANIFESTS_DIR / "training_pairs_w150.csv")
    nodes_info = {}
    for _, row in pairs_df.iterrows():
        k_a = f"{row['shot_A_view']}_shot_{row['shot_A_id']}"
        nodes_info[k_a] = {"view": row['shot_A_view'], "vq": row['shot_A_vq'], "start": row['shot_A_start'], "end": row['shot_A_end']}
        
        k_b = f"{row['shot_B_view']}_shot_{row['shot_B_id']}"
        nodes_info[k_b] = {"view": row['shot_B_view'], "vq": row['shot_B_vq'], "start": row['shot_B_start'], "end": row['shot_B_end']}
        
    node_keys = sorted(list(nodes_info.keys()))
    num_nodes = len(node_keys)
    
    # 2. Train Model
    model = retrain_model(data, device)
    model.eval()
    
    # 3. Predict all possible edges
    print(f"[INFO] Predicting all possible pairwise edges for {num_nodes} nodes...")
    all_pairs = list(itertools.combinations(range(num_nodes), 2))
    u_idx = [p[0] for p in all_pairs]
    v_idx = [p[1] for p in all_pairs]
    
    with torch.no_grad():
        z = model(data.x, data.edge_index)
        probs = []
        for i in range(0, len(u_idx), INFERENCE_BATCH_SIZE):
            u_b = torch.tensor(u_idx[i:i+INFERENCE_BATCH_SIZE], device=device)
            v_b = torch.tensor(v_idx[i:i+INFERENCE_BATCH_SIZE], device=device)
            edge_idx_b = torch.stack([u_b, v_b], dim=0)
            out = model.predict_edges(z, edge_idx_b)
            p = torch.sigmoid(out).cpu().numpy()
            probs.extend(p)
            
    # 4. Build clusters
    pred_edges = []
    for (u, v), p in zip(all_pairs, probs):
        if p > EDGE_PREDICTION_THRESHOLD:
            pred_edges.append((u, v))
            
    clusters = build_connected_components(pred_edges, num_nodes)
    
    cross_view_count = 0
    for cluster in clusters:
        views = set([nodes_info[node_keys[idx]]["view"] for idx in cluster])
        if len(views) > 1:
            cross_view_count += 1
            
    print(f"\n--- GRAPH TRANSFORMER (150-frame) CLUSTERS ---")
    print(f"[INFO] Total nodes: {num_nodes}")
    print(f"[INFO] Total clusters: {len(clusters)}")
    print(f"[INFO] Cross-view clusters: {cross_view_count}")
    
    # 5. Export final summary
    gt_df = export_summary_csv(clusters, node_keys, nodes_info, "final_summary_graph_transformer.csv")
    
    # 6. Per-View Summaries
    print("\n[INFO] Generating Per-View Summaries...")
    from src.build_summary import reconstruct_shot_windows
    
    rb_all_shots = reconstruct_shot_windows("")
    gt_all_shots = reconstruct_shot_windows("w150")
    
    for view in ["view1", "view2", "view3"]:
        rb_shots = [s for s in rb_all_shots if s["view"] == view]
        for i, s in enumerate(rb_shots):
            s["sequence_order"] = i + 1
        rb_out = f"per_view_summary_{view}_ruleBased.csv"
        pd.DataFrame(rb_shots).to_csv(MANIFESTS_DIR / rb_out, index=False)
        
        gt_shots = [s for s in gt_all_shots if s["view"] == view]
        for i, s in enumerate(gt_shots):
            s["sequence_order"] = i + 1
        gt_out = f"per_view_summary_{view}_graphTransformer.csv"
        pd.DataFrame(gt_shots).to_csv(MANIFESTS_DIR / gt_out, index=False)
        
        print(f"[INFO] Generating video for {rb_out}...")
        generate_video(rb_out, f"per_view_summary_{view}_ruleBased_video.mp4")
        print(f"[INFO] Generating video for {gt_out}...")
        generate_video(gt_out, f"per_view_summary_{view}_graphTransformer_video.mp4")

    # 7. Generate Final Video
    print("\n[INFO] Generating final summary video...")
    generate_video("final_summary_graph_transformer.csv", "final_summary_graph_transformer_video.mp4")
    
    # 8. Comparison Report
    print("\n===========================================================")
    print("DIRECT COMPARISON REPORT")
    print("===========================================================")
    rb_final = pd.read_csv(MANIFESTS_DIR / "final_summary.csv")
    
    print(f"RULE-BASED (250-frame canonical pipeline):")
    print(f"  Final Events: {len(rb_final)}")
    
    print(f"\nGRAPH-TRANSFORMER (150-frame sparse model):")
    print(f"  Final Events: {len(gt_df) if gt_df is not None and not gt_df.empty else 0}")
    print(f"  Total Clusters: {len(clusters)} (from 81 total shots)")
    print(f"  Cross-view clusters: {cross_view_count}")
    
    print(f"\n[OVERLAP ANALYSIS]")
    overlap_count = 0
    agreement_count = 0
    if gt_df is not None and not gt_df.empty:
        for _, rb_row in rb_final.iterrows():
            rb_st = rb_row["window_start_frame"]
            rb_en = rb_row["window_end_frame"]
            overlaps = gt_df[
                (gt_df["view"] == rb_row["view"]) & 
                (gt_df["window_start_frame"] <= rb_en) & 
                (gt_df["window_end_frame"] >= rb_st)
            ]
            if not overlaps.empty:
                overlap_count += 1
                rb_center = (rb_st + rb_en) / 2
                gt_centers = (overlaps["window_start_frame"] + overlaps["window_end_frame"]) / 2
                
                for gt_center in gt_centers:
                    if abs(rb_center - gt_center) < 75:
                        agreement_count += 1
                        break
                        
        print(f"  Overlapping Events: {overlap_count}")
        print(f"  Representative Agreement: {agreement_count}/{overlap_count} ({agreement_count/max(1, overlap_count)*100:.1f}%)")
    print("===========================================================")

if __name__ == "__main__":
    main()
