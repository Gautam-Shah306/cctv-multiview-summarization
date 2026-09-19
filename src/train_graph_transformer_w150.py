import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from src.build_graph_w150 import build_graph_w150

class GraphTransformerW150(torch.nn.Module):
    def __init__(self, in_channels=133, hidden_channels=64, out_channels=64, dropout=0.4):
        super().__init__()
        self.proj = nn.Linear(in_channels, hidden_channels)
        self.conv1 = GATConv(hidden_channels, hidden_channels, heads=2, concat=False)
        self.conv2 = GATConv(hidden_channels, out_channels, heads=2, concat=False)
        self.dropout = dropout
        
        self.edge_classifier = nn.Sequential(
            nn.Linear(out_channels * 2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x, edge_index):
        x = self.proj(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        
        return x

    def predict_edges(self, z, edge_index):
        u = z[edge_index[0]]
        v = z[edge_index[1]]
        edge_feat = torch.cat([u, v], dim=-1)
        return self.edge_classifier(edge_feat).squeeze(-1)

def train_kfold():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Device: {device}.")
    
    import os
    import sys as _sys
    class HiddenPrints:
        def __enter__(self):
            self._original_stdout = _sys.stdout
            _sys.stdout = open(os.devnull, 'w')
        def __exit__(self, exc_type, exc_val, exc_tb):
            _sys.stdout.close()
            _sys.stdout = self._original_stdout

    print("[INFO] Building PyTorch Geometric graph data object with PCA...")
    with HiddenPrints():
        data = build_graph_w150()
        
    data = data.to(device)
    
    num_pairs = data.edge_index.shape[1] // 2
    pair_labels = data.y[::2].cpu().numpy()
    
    print(f"[INFO] Dataset has {num_pairs} pairs ({data.x.shape[0]} nodes, {data.x.shape[1]} dims). Starting 5-Fold CV...")
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    fold_metrics = []
    
    MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(num_pairs), pair_labels)):
        print(f"\n--- FOLD {fold+1}/5 ---")
        
        train_edges = np.concatenate([train_idx * 2, train_idx * 2 + 1])
        val_edges = np.concatenate([val_idx * 2, val_idx * 2 + 1])
        
        train_mask = torch.zeros(data.edge_index.shape[1], dtype=torch.bool, device=device)
        val_mask = torch.zeros(data.edge_index.shape[1], dtype=torch.bool, device=device)
        train_mask[train_edges] = True
        val_mask[val_edges] = True
        
        model = GraphTransformerW150().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss()
        
        best_val_loss = float('inf')
        best_fold_metrics = {}
        patience = 30
        patience_counter = 0
        
        save_path = MODELS_DIR / f"graph_transformer_w150_fold{fold+1}.pt"
        
        for epoch in range(1, 301):
            model.train()
            optimizer.zero_grad()
            
            train_edge_index = data.edge_index[:, train_mask]
            
            z = model(data.x, train_edge_index) 
            out = model.predict_edges(z, train_edge_index)
            
            loss = criterion(out, data.y[train_mask])
            loss.backward()
            optimizer.step()
            
            model.eval()
            with torch.no_grad():
                z_val = model(data.x, train_edge_index)
                val_out = model.predict_edges(z_val, data.edge_index[:, val_mask])
                val_loss = criterion(val_out, data.y[val_mask]).item()
                
                preds = (torch.sigmoid(val_out) > 0.5).float().cpu().numpy()
                labels = data.y[val_mask].cpu().numpy()
                
                acc = accuracy_score(labels, preds)
                prec = precision_score(labels, preds, zero_division=0)
                rec = recall_score(labels, preds, zero_division=0)
                f1 = f1_score(labels, preds, zero_division=0)
                tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
                
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_fold_metrics = {
                    'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1, 'epoch': epoch,
                    'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp
                }
                torch.save(model.state_dict(), save_path)
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                break
                
        fold_metrics.append(best_fold_metrics)
        print(f"Fold {fold+1} Best Epoch: {best_fold_metrics['epoch']} | Val Loss: {best_val_loss:.4f} | F1: {best_fold_metrics['f1']:.4f} | Acc: {best_fold_metrics['acc']:.4f}")
        print(f"Fold {fold+1} CM: TP={best_fold_metrics['tp']}, TN={best_fold_metrics['tn']}, FP={best_fold_metrics['fp']}, FN={best_fold_metrics['fn']}")

    print("\n=======================================================")
    print("5-FOLD CROSS-VALIDATION RESULTS")
    print("=======================================================")
    
    accs = [m['acc'] for m in fold_metrics]
    precs = [m['prec'] for m in fold_metrics]
    recs = [m['rec'] for m in fold_metrics]
    f1s = [m['f1'] for m in fold_metrics]
    
    print(f"Accuracy  : {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"Precision : {np.mean(precs):.4f} ± {np.std(precs):.4f}")
    print(f"Recall    : {np.mean(recs):.4f} ± {np.std(recs):.4f}")
    print(f"F1 Score  : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    
    print("\n--- COMPARISON TO BASELINE ---")
    baseline_f1 = 0.6573
    diff = np.mean(f1s) - baseline_f1
    if diff > 0:
        print(f"The mean CV F1 ({np.mean(f1s):.4f}) EXCEEDS the original baseline (0.6573) by +{diff:.4f}!")
    else:
        print(f"The mean CV F1 ({np.mean(f1s):.4f}) is WORSE than the original baseline (0.6573) by {diff:.4f}.")

if __name__ == "__main__":
    train_kfold()
