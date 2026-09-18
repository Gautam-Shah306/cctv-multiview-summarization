import csv
from collections import defaultdict
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path.cwd()))

from config import CHECKPOINTS, DATA_DIR, REID_IMAGE_SIZE
from src.extract_reid_features import preprocess_crop
from src.models.reid_model import build_reid_model

def load_first_valid_crop(view, tid):
    if tid == -1: return None
    detections_csv = DATA_DIR / "detections" / f"{view}.csv"
    with open(detections_csv) as f:
        rows = list(csv.DictReader(f))
    for det in rows:
        if int(det["track_id"]) == tid:
            frame_idx = int(det["frame_idx"])
            frame_path = DATA_DIR / view / f"frame_{frame_idx:04d}.jpg"
            img = cv2.imread(str(frame_path))
            if img is None:
                continue
            img_h, img_w = img.shape[:2]
            x1 = max(0, min(int(round(float(det["x1"]))), img_w - 1))
            y1 = max(0, min(int(round(float(det["y1"]))), img_h - 1))
            x2 = max(x1 + 1, min(int(round(float(det["x2"]))), img_w))
            y2 = max(y1 + 1, min(int(round(float(det["y2"]))), img_h))
            crop = img[y1:y2, x1:x2]
            return preprocess_crop(crop, REID_IMAGE_SIZE)
    return None

manifest = Path("data_manifests/manual_crossview_eval.csv")
with open(manifest) as f:
    eval_pairs = list(csv.DictReader(f))

crops_data_a = []
crops_data_b = []
labels = []

for r in eval_pairs:
    v_a = r['view_a']
    tid_a = int(r['track_id_a'])
    v_b = r['view_b']
    tid_b = int(r['track_id_b'])
    same = int(r['same_person'])
    
    crop1 = load_first_valid_crop(v_a, tid_a)
    crop2 = load_first_valid_crop(v_b, tid_b)
    
    if crop1 is not None and crop2 is not None:
        crops_data_a.append(crop1)
        crops_data_b.append(crop2)
        labels.append(same)

print(f"Loaded {len(labels)} manual cross-view pairs.")
tensor_a = torch.stack(crops_data_a)
tensor_b = torch.stack(crops_data_b)

model_post = build_reid_model(pretrained=True, device="cpu")

with torch.no_grad():
    embs_a = model_post(tensor_a, normalize=True).numpy()
    embs_b = model_post(tensor_b, normalize=True).numpy()

same, diff = [], []
for i in range(len(labels)):
    sim = float(np.dot(embs_a[i], embs_b[i]))
    if labels[i] == 1:
        same.append(sim)
    else:
        diff.append(sim)

print(f"Num same-person pairs: {len(same)}")
print(f"Num diff-person pairs: {len(diff)}")

if len(same) > 0 and len(diff) > 0:
    print(f"Same-person similarity: min={min(same):.4f}, mean={np.mean(same):.4f}, max={max(same):.4f}, std={np.std(same):.4f}")
    print(f"Diff-person similarity: min={min(diff):.4f}, mean={np.mean(diff):.4f}, max={max(diff):.4f}, std={np.std(diff):.4f}")

    def evaluate_threshold(t):
        tp = sum(1 for s in same if s >= t)
        fn = len(same) - tp
        fp = sum(1 for d in diff if d >= t)
        tn = len(diff) - fp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        return precision, recall, f1

    p_075, r_075, f1_075 = evaluate_threshold(0.75)
    print(f"Metrics at threshold 0.75: Precision={p_075:.4f}, Recall={r_075:.4f}, F1={f1_075:.4f}")

    print("\nFine Sweep (0.95 to 0.995):")
    best_thresh = 0
    best_f1 = 0
    for t in np.arange(0.95, 0.996, 0.005):
        p, r, f1 = evaluate_threshold(t)
        print(f"  Thresh={t:.3f} | P={p:.4f}, R={r:.4f}, F1={f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    print(f"\nBest empirical threshold (F1): {best_thresh:.3f} (F1: {best_f1:.4f})")
