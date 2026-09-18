import csv
from pathlib import Path
from collections import defaultdict
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path.cwd()))

from config import CHECKPOINTS, DATA_DIR, REID_IMAGE_SIZE
from src.extract_reid_features import preprocess_crop
from src.models.reid_model import OSNet

class OSNetFixed(OSNet):
    def __init__(self, blocks=[2, 2, 2], channels=[64, 256, 384, 512], feature_dim=512):
        super().__init__(blocks=blocks, channels=channels, feature_dim=feature_dim)
        self.fc = nn.Sequential(
            nn.Linear(channels[3], feature_dim, bias=True),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
        )

def build_postfix_model():
    model = OSNetFixed()
    weights_path = CHECKPOINTS / "osnet_x1_0_market1501.pth"
    state_dict = torch.load(str(weights_path), map_location="cpu")
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    
    model_dict = model.state_dict()
    remap_dict = {}
    for k, v in state_dict.items():
        key = k.replace("module.", "")
        if key.startswith("conv2.2.0."):
            key = key.replace("conv2.2.0.", "transition1.0.")
        elif key.startswith("conv3.2.0."):
            key = key.replace("conv3.2.0.", "transition2.0.")
            
        if key in model_dict and model_dict[key].shape == v.shape:
            remap_dict[key] = v
            
    model.load_state_dict(remap_dict, strict=False)
    model.eval()
    return model

detections_csv = DATA_DIR / "detections" / "view1.csv"
with open(detections_csv) as f:
    rows = list(csv.DictReader(f))

tracks = defaultdict(list)
for r in rows:
    tid = int(r["track_id"])
    if tid != -1:
        tracks[tid].append(r)

# Pick 20 valid tracklets
valid_tids = [tid for tid, l in tracks.items() if len(l) > 10]
selected_tids = valid_tids[:20]

crops_data = []
crop_labels = []

for tid in selected_tids:
    det_list = tracks[tid]
    idx_a = 0
    idx_b = min(len(det_list) - 1, max(1, len(det_list) // 2))
    for sub_idx, det in enumerate([det_list[idx_a], det_list[idx_b]]):
        frame_idx = int(det["frame_idx"])
        frame_path = DATA_DIR / "view1" / f"frame_{frame_idx:04d}.jpg"
        img = cv2.imread(str(frame_path))
        if img is None:
            continue
        
        img_h, img_w = img.shape[:2]
        x1 = max(0, min(int(round(float(det["x1"]))), img_w - 1))
        y1 = max(0, min(int(round(float(det["y1"]))), img_h - 1))
        x2 = max(x1 + 1, min(int(round(float(det["x2"]))), img_w))
        y2 = max(y1 + 1, min(int(round(float(det["y2"]))), img_h))
        
        crop = img[y1:y2, x1:x2]
        tensor = preprocess_crop(crop, REID_IMAGE_SIZE)
        crops_data.append(tensor)
        label = f"P{tid}_F{frame_idx}"
        crop_labels.append(label)

batch_tensor = torch.stack(crops_data)
model_post = build_postfix_model()

with torch.no_grad():
    x = model_post.conv1(batch_tensor)
    x = model_post.maxpool(x)
    x = model_post.conv2(x)
    x = model_post.transition1(x)
    x = model_post.conv3(x)
    x = model_post.transition2(x)
    x = model_post.conv4(x)
    x = model_post.conv5(x)
    pooled = model_post.global_avgpool(x).view(x.size(0), -1)
    embs_post_pooled = F.normalize(pooled, p=2, dim=1).numpy()

sim_mat = np.dot(embs_post_pooled, embs_post_pooled.T)

same, diff = [], []
n_crops = len(crop_labels)
for i in range(n_crops):
    for j in range(i + 1, n_crops):
        if i // 2 == j // 2:
            same.append(sim_mat[i, j])
        else:
            diff.append(sim_mat[i, j])

print(f"Num same-person pairs: {len(same)}")
print(f"Num diff-person pairs: {len(diff)}")
print(f"Same-person similarity: min={min(same):.4f}, mean={np.mean(same):.4f}, max={max(same):.4f}")
print(f"Diff-person similarity: min={min(diff):.4f}, mean={np.mean(diff):.4f}, max={max(diff):.4f}")

best_thresh = 0
best_f1 = 0
for t in np.linspace(0.0, 1.0, 101):
    tp = sum(1 for s in same if s >= t)
    fn = len(same) - tp
    fp = sum(1 for d in diff if d >= t)
    tn = len(diff) - fp
    
    if tp + fp == 0:
        precision = 0
    else:
        precision = tp / (tp + fp)
        
    if tp + fn == 0:
        recall = 0
    else:
        recall = tp / (tp + fn)
        
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
        
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = t

print(f"Best empirical threshold (F1): {best_thresh:.3f} (F1: {best_f1:.4f})")
