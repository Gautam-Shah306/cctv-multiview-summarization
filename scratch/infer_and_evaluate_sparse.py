import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import numpy as np
import pandas as pd
import subprocess
import itertools

from src.train_graph_transformer_w150_sparse import GraphTransformerW150
from src.build_graph_w150_sparse import build_graph_w150_sparse

MANIFESTS_DIR = Path(__file__).resolve().parent.parent / "data_manifests"

def get_vq_score(shot_key, nodes_info):
    return float(nodes_info[shot_key]["vq"])

def retrain_model(data, device):
    """Retrain on all available labeled pairs to maximize training data for inference."""
    print("[INFO] Retraining model on ALL labeled training pairs (612 edges) for final inference...")
    model = GraphTransformerW150().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    
    # Use all edges in the data for training
    train_mask = torch.ones(data.edge_index.shape[1], dtype=torch.bool, device=device)
    
    for epoch in range(1, 150): # 150 epochs is enough for convergence on this tiny graph
        model.train()
        optimizer.zero_grad()
        train_edge_index = data.edge_index[:, train_mask]
        z = model(data.x, train_edge_index) 
        out = model.predict_edges(z, train_edge_index)
        loss = criterion(out, data.y[train_mask])
        loss.backward()
        optimizer.step()
        
    return model

def build_connected_components(edge_list, num_nodes):
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

def export_summary_csv(clusters, node_keys, nodes_info, out_name):
    final_shots = []
    for c_idx, cluster in enumerate(clusters):
        # Pick representative based on VQ score
        rep_idx = max(cluster, key=lambda idx: get_vq_score(node_keys[idx], nodes_info))
        rep_key = node_keys[rep_idx]
        info = nodes_info[rep_key]
        
        final_shots.append({
            "sequence_order": "", # Fill later
            "view": info["view"],
            "window_start_frame": info["start"],
            "window_end_frame": info["end"],
            "cluster_id": c_idx,
            "vq_score": info["vq"],
            "cluster_size": len(cluster)
        })
    
    # Sort chronologically by start frame
    final_shots.sort(key=lambda x: x["window_start_frame"])
    
    out_path = MANIFESTS_DIR / out_name
    df = pd.DataFrame(final_shots)
    df["sequence_order"] = range(1, len(df) + 1)
    df.to_csv(out_path, index=False)
    print(f"[DONE] Wrote {out_name} with {len(df)} sequences.")
    return df

def generate_video(input_csv, output_mp4):
    cmd = ["python", "-m", "src.assemble_summary_video", "--input", str(MANIFESTS_DIR / input_csv), "--output", str(MANIFESTS_DIR / output_mp4)]
    subprocess.run(cmd, check=True)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    
    # We predict using the learned graph representations. But what edge_index do we pass to the GAT?
    # The GAT is transductive and passes messages across existing known edges.
    # We will pass the ALL POSSIBLE edges as the structural edge_index? No, GAT uses the graph structure to form node embeddings.
    # The standard way is to use the known sparse graph to generate embeddings `z`, then predict links between ANY pairs in `z`.
    # But wait, what is the graph structure at inference time? We shouldn't use the labels to build the graph.
    # The original model used the 612 labeled pairs as the structural graph.
    # Let's use the known 612 edges to generate node embeddings `z` (as the model was trained).
    # Then we predict all pairwise connections using `z`.
    with torch.no_grad():
        z = model(data.x, data.edge_index)
        
        # Predict in batches to avoid OOM
        batch_size = 1000
        probs = []
        for i in range(0, len(u_idx), batch_size):
            u_b = torch.tensor(u_idx[i:i+batch_size], device=device)
            v_b = torch.tensor(v_idx[i:i+batch_size], device=device)
            edge_idx_b = torch.stack([u_b, v_b], dim=0)
            out = model.predict_edges(z, edge_idx_b)
            p = torch.sigmoid(out).cpu().numpy()
            probs.extend(p)
            
    # 4. Build clusters (p > 0.5)
    pred_edges = []
    for (u, v), p in zip(all_pairs, probs):
        if p > 0.5:
            pred_edges.append((u, v))
            
    clusters = build_connected_components(pred_edges, num_nodes)
    
    cross_view_count = 0
    for cluster in clusters:
        views = set([nodes_info[node_keys[idx]]["view"] for idx in cluster])
        if len(views) > 1:
            cross_view_count += 1
            
    print(f"\\n--- GRAPH TRANSFORMER (150-frame) CLUSTERS ---")
    print(f"Total nodes: {num_nodes}")
    print(f"Total clusters: {len(clusters)}")
    print(f"Cross-view clusters: {cross_view_count}")
    
    # 5. Export final summary
    gt_df = export_summary_csv(clusters, node_keys, nodes_info, "final_summary_graph_transformer.csv")
    
    # 6. Per-View Summaries
    print("\\n[INFO] Generating Per-View Summaries...")
    # Graph Transformer per view: The clusters before cross-view dedup.
    # Actually, a per-view summary is just the set of ALL representative shots for that view.
    # No, wait. Per-view summary is what the pipeline produces BEFORE cross-view fusion.
    # The Graph Transformer operates directly on the 81 shots (which were already filtered per-view by VQ in the 150-frame sweep!).
    # So the per-view summary for the Graph Transformer IS just the 81 shots from `keyframe_decisions_viewX_w150.csv`.
    # And for ruleBased, it's the 48 shots from `keyframe_decisions_viewX.csv`.
    
    from src.build_summary import reconstruct_shot_windows
    
    rb_all_shots = reconstruct_shot_windows("")
    gt_all_shots = reconstruct_shot_windows("w150")
    
    for view in ["view1", "view2", "view3"]:
        # Rule-based (250-frame)
        rb_shots = [s for s in rb_all_shots if s["view"] == view]
        for i, s in enumerate(rb_shots):
            s["sequence_order"] = i + 1
        rb_out = f"per_view_summary_{view}_ruleBased.csv"
        pd.DataFrame(rb_shots).to_csv(MANIFESTS_DIR / rb_out, index=False)
        
        # Graph Transformer (150-frame)
        gt_shots = [s for s in gt_all_shots if s["view"] == view]
        for i, s in enumerate(gt_shots):
            s["sequence_order"] = i + 1
        gt_out = f"per_view_summary_{view}_graphTransformer.csv"
        pd.DataFrame(gt_shots).to_csv(MANIFESTS_DIR / gt_out, index=False)
        
        # Generate Videos
        print(f"Generating video for {rb_out}...")
        generate_video(rb_out, f"per_view_summary_{view}_ruleBased_video.mp4")
        print(f"Generating video for {gt_out}...")
        generate_video(gt_out, f"per_view_summary_{view}_graphTransformer_video.mp4")

    # 7. Generate Final Video
    print("\\n[INFO] Generating final summary video...")
    generate_video("final_summary_graph_transformer.csv", "final_summary_graph_transformer_video.mp4")
    
    # 8. Comparison Report
    print("\\n===========================================================")
    print("DIRECT COMPARISON REPORT")
    print("===========================================================")
    rb_final = pd.read_csv(MANIFESTS_DIR / "final_summary.csv")
    
    print(f"RULE-BASED (250-frame canonical pipeline):")
    print(f"  Final Events: {len(rb_final)}")
    # We can't easily count cross-view clusters from final_summary.csv because it doesn't store cluster info, 
    # but we know from earlier it was 8 events from 48 shots.
    
    print(f"\\nGRAPH-TRANSFORMER (150-frame sparse model):")
    print(f"  Final Events: {len(gt_df)}")
    print(f"  Total Clusters: {len(clusters)} (from 81 total shots)")
    print(f"  Cross-view clusters: {cross_view_count}")
    
    print(f"\\n[OVERLAP ANALYSIS]")
    # For shots in BOTH summaries (overlapping time windows), how often do they agree?
    overlap_count = 0
    agreement_count = 0
    for _, rb_row in rb_final.iterrows():
        rb_st = rb_row["window_start_frame"]
        rb_en = rb_row["window_end_frame"]
        # Find overlapping GT events
        # Check if ANY gt_df row overlaps with this time window and same view
        overlaps = gt_df[
            (gt_df["view"] == rb_row["view"]) & 
            (gt_df["window_start_frame"] <= rb_en) & 
            (gt_df["window_end_frame"] >= rb_st)
        ]
        if not overlaps.empty:
            overlap_count += 1
            # Did they pick the same representative time window?
            # They have different window sizes (250 vs 150), so exact match is impossible.
            # But we can check if the centers of the windows are close (e.g., within 50 frames).
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
