"""
train_graph_transformer.py

Trains a Graph Transformer (GAT) to predict cross-view re-identification 
links based on dense tracklet node features.

Usage:
    python -m src.train_graph_transformer
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch_geometric.nn import GATConv

from src.build_graph import build_graph

class GraphTransformer(torch.nn.Module):
    def __init__(self, in_channels=901, hidden_channels=64, out_channels=64, dropout=0.4):
        super().__init__()
        # Project raw features to smaller dim to mitigate overfitting on this very small dataset
        self.proj = nn.Linear(in_channels, hidden_channels)
        
        # 2 layers of GATConv. Kept deliberately small (64 dims) due to 48-node overfitting risk.
        self.conv1 = GATConv(hidden_channels, hidden_channels, heads=2, concat=False)
        self.conv2 = GATConv(hidden_channels, out_channels, heads=2, concat=False)
        
        self.dropout = dropout
        
        # Edge classifier: takes concatenated [u, v] node features
        self.edge_classifier = nn.Sequential(
            nn.Linear(out_channels * 2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x, edge_index):
        # 1. Project 901 -> 64
        x = self.proj(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # 2. Graph Attention Message Passing
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        
        return x

    def predict_edges(self, z, edge_index):
        # z: node embeddings [N, out_channels]
        u = z[edge_index[0]]
        v = z[edge_index[1]]
        edge_feat = torch.cat([u, v], dim=-1)
        return self.edge_classifier(edge_feat).squeeze(-1)

def train() -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Device: {device}. (Note: User requested Colab GPU. Running in available environment.)")
    
    # Disable build_graph prints to keep console clean
    import os
    import sys as _sys
    class HiddenPrints:
        def __enter__(self):
            self._original_stdout = _sys.stdout
            _sys.stdout = open(os.devnull, 'w')
        def __exit__(self, exc_type, exc_val, exc_tb):
            _sys.stdout.close()
            _sys.stdout = self._original_stdout

    print("[INFO] Building PyTorch Geometric graph data object...")
    with HiddenPrints():
        data = build_graph()
        
    data = data.to(device)
    
    # 80/20 Stratified Edge Split
    # Since edges are added bidirectionally in build_graph, we group them into unique pairs (i * 2, i * 2 + 1)
    num_pairs = data.edge_index.shape[1] // 2
    pair_labels = data.y[::2].cpu().numpy()
    
    train_idx, val_idx = train_test_split(
        np.arange(num_pairs),
        test_size=0.2,
        stratify=pair_labels,
        random_state=42
    )
    
    train_edges = np.concatenate([train_idx * 2, train_idx * 2 + 1])
    val_edges = np.concatenate([val_idx * 2, val_idx * 2 + 1])
    
    train_mask = torch.zeros(data.edge_index.shape[1], dtype=torch.bool, device=device)
    val_mask = torch.zeros(data.edge_index.shape[1], dtype=torch.bool, device=device)
    train_mask[train_edges] = True
    val_mask[val_edges] = True
    
    print(f"[INFO] Train Edges (directional): {train_mask.sum().item()} | Val Edges (directional): {val_mask.sum().item()}")
    
    model = GraphTransformer().to(device)
    
    # Conservative learning rate, reduced weight decay so it actually trains
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    # Binary classification for redundant/same-event edges
    criterion = nn.BCEWithLogitsLoss()
    
    best_val_loss = float('inf')
    best_metrics = {}
    patience = 30
    patience_counter = 0
    
    MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = MODELS_DIR / "graph_transformer.pt"
    
    print("\n[INFO] Starting training...")
    
    for epoch in range(1, 301):
        model.train()
        optimizer.zero_grad()
        
        # Message passing uses ONLY training edges to prevent label leakage
        train_edge_index = data.edge_index[:, train_mask]
        
        z = model(data.x, train_edge_index) 
        out = model.predict_edges(z, train_edge_index)
        
        loss = criterion(out, data.y[train_mask])
        loss.backward()
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            # Keep message passing restricted to train edges, but predict on val edges
            z_val = model(data.x, train_edge_index)
            val_out = model.predict_edges(z_val, data.edge_index[:, val_mask])
            val_loss = criterion(val_out, data.y[val_mask]).item()
            
            # Since edges are duplicated (bidirectional), we can evaluate on all directional val edges
            preds = (torch.sigmoid(val_out) > 0.5).float().cpu().numpy()
            labels = data.y[val_mask].cpu().numpy()
            
            acc = accuracy_score(labels, preds)
            prec = precision_score(labels, preds, zero_division=0)
            rec = recall_score(labels, preds, zero_division=0)
            f1 = f1_score(labels, preds, zero_division=0)
            
            # Confusion Matrix
            from sklearn.metrics import confusion_matrix
            tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
            
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_metrics = {
                'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1, 'epoch': epoch,
                'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp
            }
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f} | Val F1: {f1:.4f} | Val Acc: {acc:.4f}")
            
        if patience_counter >= patience:
            print(f"\n[INFO] Early stopping triggered at epoch {epoch}")
            break
            
    print("\n--- FINAL VALIDATION METRICS ---")
    print(f"Best Epoch : {best_metrics['epoch']}")
    print(f"Accuracy   : {best_metrics['acc']:.4f}")
    print(f"Precision  : {best_metrics['prec']:.4f}")
    print(f"Recall     : {best_metrics['rec']:.4f}")
    print(f"F1 Score   : {best_metrics['f1']:.4f}")
    print("\n--- CONFUSION MATRIX ---")
    print(f"True Positives (TP) : {best_metrics['tp']}")
    print(f"True Negatives (TN) : {best_metrics['tn']}")
    print(f"False Positives (FP): {best_metrics['fp']}")
    print(f"False Negatives (FN): {best_metrics['fn']}")
    
    if best_metrics['acc'] > 0.98:
        print("\n[WARNING] Validation performance looks suspiciously perfect (near 100%).")
        print("Given the small dataset (48 nodes, 326 pairs), this is likely OVERFITTING/MEMORIZATION")
        print("rather than true generalization. This is a known limitation of using deep learning on tiny datasets.")
        
    size_mb = save_path.stat().st_size / (1024 * 1024)
    print(f"\n[DONE] Saved best model to {save_path.resolve()} ({size_mb:.2f} MB)")

if __name__ == "__main__":
    train()
