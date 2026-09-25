"""
build_graph.py

Constructs the cross-view graph structure for the Graph Transformer model
from the dense (non-sparse) feature tracklets and matching CSV pairs.

Usage:
    python -m src.build_graph
"""

import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Check for PyG
try:
    import torch
    from torch_geometric.data import Data
except ImportError:
    print("[ERROR] PyTorch Geometric (torch_geometric) or PyTorch is not installed/available.")
    sys.exit(1)

MANIFESTS_DIR = Path(__file__).resolve().parent.parent / "data_manifests"

def build_graph() -> Data:
    # Load dataset
    pairs_df = pd.read_csv(MANIFESTS_DIR / "training_pairs.csv")
    
    # Load features
    dino_features = np.load(MANIFESTS_DIR / "training_features_dino.npz")
    reid_features = np.load(MANIFESTS_DIR / "training_features_reid.npz")
    
    # 1. Identify all unique nodes
    nodes = {}
    for _, row in pairs_df.iterrows():
        # Shot A
        k_a = f"{row['shot_A_view']}_shot_{row['shot_A_id']}"
        if k_a not in nodes:
            nodes[k_a] = {
                "view": row['shot_A_view'],
                "vq": row['shot_A_vq'],
                "start": row['shot_A_start']
            }
        # Shot B
        k_b = f"{row['shot_B_view']}_shot_{row['shot_B_id']}"
        if k_b not in nodes:
            nodes[k_b] = {
                "view": row['shot_B_view'],
                "vq": row['shot_B_vq'],
                "start": row['shot_B_start']
            }
            
    node_keys = sorted(list(nodes.keys()))
    node_to_idx = {k: i for i, k in enumerate(node_keys)}
    num_nodes = len(node_keys)
    
    print(f"[INFO] Found {num_nodes} unique nodes (shots).")
    
    # Determine max frame for timestamp normalization
    max_frame = max(n["start"] for n in nodes.values())
    if max_frame == 0: max_frame = 1 # avoid div by zero
    
    # 2. Build Node Features
    x_list = []
    
    for k in node_keys:
        info = nodes[k]
        
        # DINOv3 (384)
        dino = dino_features[k]
        
        # VQ score (1)
        vq = np.array([info["vq"]], dtype=np.float32)
        
        # Normalized timestamp (1)
        # Using start frame divided by max frame
        ts = np.array([info["start"] / max_frame], dtype=np.float32)
        
        # View ID one-hot (3)
        view_onehot = np.zeros(3, dtype=np.float32)
        if info["view"] == "view1": view_onehot[0] = 1.0
        elif info["view"] == "view2": view_onehot[1] = 1.0
        elif info["view"] == "view3": view_onehot[2] = 1.0
        
        # ReID (512)
        # We explicitly mean-pool the (N, 512) set-based embeddings here to create a 
        # FIXED SIZE graph node representation. 
        # NOTE: This node-level collapse decision is specific to creating fixed-size
        # node embeddings for the Graph Transformer, and does NOT contradict the earlier 
        # decision to maintain set-based unpooled embeddings for pair matching/retrieval.
        reid_set = reid_features[k]
        if len(reid_set.shape) > 1 and reid_set.shape[0] > 0:
            reid_mean = np.mean(reid_set, axis=0)
        else:
            reid_mean = np.zeros(512, dtype=np.float32)
            
        # Concatenate
        node_feat = np.concatenate([dino, vq, ts, view_onehot, reid_mean])
        x_list.append(node_feat)
        
    x = torch.tensor(np.vstack(x_list), dtype=torch.float32)
    print(f"[INFO] Node feature matrix shape: {x.shape} (Dims: {x.shape[1]})")
    
    # Validate no NaN/Inf
    if torch.isnan(x).any() or torch.isinf(x).any():
        print("[ERROR] Node feature matrix contains NaN or Inf values!")
        sys.exit(1)
    print("[INFO] Validated: No NaN or Inf values in node features.")
    
    # 3. Build Edges
    edge_index_list = []
    edge_attr_list = []
    
    for _, row in pairs_df.iterrows():
        k_a = f"{row['shot_A_view']}_shot_{row['shot_A_id']}"
        k_b = f"{row['shot_B_view']}_shot_{row['shot_B_id']}"
        
        idx_a = node_to_idx[k_a]
        idx_b = node_to_idx[k_b]
        
        # Undirected edges (add both directions)
        edge_index_list.append([idx_a, idx_b])
        edge_index_list.append([idx_b, idx_a])
        
        # Target label
        lbl = float(row["label"])
        edge_attr_list.append(lbl)
        edge_attr_list.append(lbl)
        
    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr_list, dtype=torch.float32)
    
    print(f"[INFO] Graph built with {edge_index.shape[1] // 2} unique undirected edges ({edge_index.shape[1]} directional edges).")
    
    # 4. Create PyG Data object
    data = Data(x=x, edge_index=edge_index, y=edge_attr)
    
    print("\n--- GRAPH SUMMARY ---")
    print(data)
    
    print("\n--- SANITY CHECK: 2 SAMPLE NODE VECTORS ---")
    sample_indices = random.sample(range(num_nodes), 2)
    for idx in sample_indices:
        feat = x[idx]
        key = node_keys[idx]
        print(f"\nNode: {key} (Index: {idx})")
        print(f"  DINOv3 [0:3]     : {feat[0:3].tolist()} ...")
        print(f"  VQ Score         : {feat[384].item():.4f}")
        print(f"  Timestamp (norm) : {feat[385].item():.4f}")
        print(f"  View One-Hot     : {feat[386:389].tolist()}")
        print(f"  ReID Mean [0:3]  : {feat[389:392].tolist()} ...")
        
    print("-" * 40)
    
    return data
    
if __name__ == "__main__":
    build_graph()
