import sys
import pandas as pd
import numpy as np
import random
from pathlib import Path
from sklearn.decomposition import PCA
from config import DRIVE_ROOT

try:
    import torch
    from torch_geometric.data import Data
except ImportError:
    print("[ERROR] PyTorch Geometric (torch_geometric) or PyTorch is not installed/available.")
    sys.exit(1)

MANIFESTS_DIR = Path(__file__).resolve().parent.parent / "data_manifests"
DRIVE_MANIFESTS_DIR = DRIVE_ROOT / "data_manifests"

def build_graph_w150_sparse():
    # Load dataset
    pairs_df = pd.read_csv(MANIFESTS_DIR / "training_pairs_w150.csv")
    
    # Load SPARCELY SAMPLED features from Google Drive where they were saved by Colab
    dino_features = np.load(DRIVE_MANIFESTS_DIR / "training_features_dino_w150_sparse.npz")
    reid_features = np.load(DRIVE_MANIFESTS_DIR / "training_features_reid_w150_sparse.npz")
    
    # 1. Identify all unique nodes and filter pairs
    nodes = {}
    valid_pairs = []
    skipped_pairs_count = 0
    
    for _, row in pairs_df.iterrows():
        k_a = f"{row['shot_A_view']}_shot_{row['shot_A_id']}"
        k_b = f"{row['shot_B_view']}_shot_{row['shot_B_id']}"
        
        missing = None
        if k_a not in dino_features: missing = (k_a, "DINO")
        elif k_a not in reid_features: missing = (k_a, "ReID")
        elif k_b not in dino_features: missing = (k_b, "DINO")
        elif k_b not in reid_features: missing = (k_b, "ReID")
        
        if missing:
            print(f"[INFO] Skipped pair: shot_key {missing[0]} missing from {missing[1]} features.")
            skipped_pairs_count += 1
            continue
            
        valid_pairs.append(row)
        
        if k_a not in nodes:
            nodes[k_a] = {"view": row['shot_A_view'], "vq": row['shot_A_vq'], "start": row['shot_A_start']}
        if k_b not in nodes:
            nodes[k_b] = {"view": row['shot_B_view'], "vq": row['shot_B_vq'], "start": row['shot_B_start']}
            
    pairs_df = pd.DataFrame(valid_pairs)
    print(f"[INFO] Total skipped pairs due to missing sparse features: {skipped_pairs_count}")

            
    node_keys = sorted(list(nodes.keys()))
    node_to_idx = {k: i for i, k in enumerate(node_keys)}
    num_nodes = len(node_keys)
    
    print(f"[INFO] Found {num_nodes} unique nodes (shots).")
    
    # 2. Extract raw features for PCA
    raw_dino_list = []
    raw_reid_list = []
    
    for k in node_keys:
        raw_dino_list.append(dino_features[k])
        reid_set = reid_features[k]
        if len(reid_set.shape) > 1 and reid_set.shape[0] > 0:
            reid_mean = np.mean(reid_set, axis=0) # pool multiple tracks per shot for final representation
        else:
            reid_mean = np.zeros(512, dtype=np.float32)
        raw_reid_list.append(reid_mean)
        
    raw_dino_arr = np.vstack(raw_dino_list)
    raw_reid_arr = np.vstack(raw_reid_list)
    
    # PCA Reduction
    dino_pca = PCA(n_components=64)
    reid_pca = PCA(n_components=64)
    
    dino_reduced = dino_pca.fit_transform(raw_dino_arr)
    reid_reduced = reid_pca.fit_transform(raw_reid_arr)
    
    print(f"[INFO] DINOv3 PCA (384 -> 64) Explained Variance: {np.sum(dino_pca.explained_variance_ratio_):.4f}")
    print(f"[INFO] ReID PCA (512 -> 64) Explained Variance: {np.sum(reid_pca.explained_variance_ratio_):.4f}")
    
    max_frame = max(n["start"] for n in nodes.values())
    if max_frame == 0: max_frame = 1 
    
    # 3. Build Node Features
    x_list = []
    
    for i, k in enumerate(node_keys):
        info = nodes[k]
        
        dino = dino_reduced[i]
        vq = np.array([info["vq"]], dtype=np.float32)
        ts = np.array([info["start"] / max_frame], dtype=np.float32)
        
        view_onehot = np.zeros(3, dtype=np.float32)
        if info["view"] == "view1": view_onehot[0] = 1.0
        elif info["view"] == "view2": view_onehot[1] = 1.0
        elif info["view"] == "view3": view_onehot[2] = 1.0
        
        reid = reid_reduced[i]
        
        node_feat = np.concatenate([dino, vq, ts, view_onehot, reid])
        x_list.append(node_feat)
        
    x = torch.tensor(np.vstack(x_list), dtype=torch.float32)
    print(f"[INFO] New node feature matrix shape: {x.shape} (Dims: {x.shape[1]})")
    
    if torch.isnan(x).any() or torch.isinf(x).any():
        print("[ERROR] Node feature matrix contains NaN or Inf values!")
        sys.exit(1)
        
    # 4. Build Edges
    edge_index_list = []
    edge_attr_list = []
    
    for _, row in pairs_df.iterrows():
        k_a = f"{row['shot_A_view']}_shot_{row['shot_A_id']}"
        k_b = f"{row['shot_B_view']}_shot_{row['shot_B_id']}"
        
        idx_a = node_to_idx[k_a]
        idx_b = node_to_idx[k_b]
        
        edge_index_list.append([idx_a, idx_b])
        edge_index_list.append([idx_b, idx_a])
        
        lbl = float(row["label"])
        edge_attr_list.append(lbl)
        edge_attr_list.append(lbl)
        
    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr_list, dtype=torch.float32)
    
    print(f"[INFO] Graph built with {edge_index.shape[1] // 2} unique undirected edges.")
    
    data = Data(x=x, edge_index=edge_index, y=edge_attr)
    return data

if __name__ == "__main__":
    build_graph_w150_sparse()
