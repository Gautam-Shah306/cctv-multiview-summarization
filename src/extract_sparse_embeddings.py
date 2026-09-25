import argparse
import sys
import os
import cv2
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from torchvision import transforms
from transformers import AutoModel

from config import DATA_DIR, DINOV3_MODEL, REID_MODEL, REID_WEIGHTS, REID_IMAGE_SIZE
from src.models.reid_model import build_reid_model
from src.extract_dinov3_features import get_image_transform

MANIFESTS_DIR = Path(__file__).resolve().parent.parent / "data_manifests"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

def preprocess_reid_crop(crop_bgr, target_size=(256, 128)):
    h, w = target_size
    resized = cv2.resize(crop_bgr, (w, h), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    normalized = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(normalized.transpose(2, 0, 1))
    return tensor

def load_all_detections():
    print("[INFO] Loading all detections from CSVs...")
    all_dets = {}
    for view in ["view1", "view2", "view3"]:
        csv_path = DATA_DIR / "detections" / f"{view}.csv"
        view_dets = defaultdict(list)
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                f_idx = int(row["frame_idx"])
                view_dets[f_idx].append(row.to_dict())
        all_dets[view] = view_dets
    return all_dets

def get_nearest_frame_detections(view_dets, target_frame, max_dist=30):
    if target_frame in view_dets and len(view_dets[target_frame]) > 0:
        return target_frame, view_dets[target_frame]
    
    # Fallback search
    available_frames = list(view_dets.keys())
    if not available_frames:
        return None, []
        
    nearest = min(available_frames, key=lambda f: abs(f - target_frame))
    dist = abs(nearest - target_frame)
    if dist <= max_dist:
        return nearest, view_dets[nearest]
    return None, []

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {device}")
    
    # 1. Identify Shots
    pairs_path = MANIFESTS_DIR / "training_pairs_w150.csv"
    if not pairs_path.exists():
        print(f"[ERROR] {pairs_path} not found.")
        sys.exit(1)
        
    pairs_df = pd.read_csv(pairs_path)
    shots = {}
    for _, row in pairs_df.iterrows():
        for prefix in ["shot_A", "shot_B"]:
            k = f"{row[prefix + '_view']}_shot_{row[prefix + '_id']}"
            if k not in shots:
                shots[k] = {
                    "view": row[prefix + '_view'],
                    "start": row[prefix + '_start'],
                    "end": row[prefix + '_end']
                }
                
    print(f"[INFO] Identified {len(shots)} unique shots to process.")
    
    # 2. Determine sample frames per shot
    frames_to_process = defaultdict(set) # view -> set of frames
    shot_samples = {}
    for shot_key, info in shots.items():
        st = info["start"]
        en = info["end"]
        span = en - st
        f1 = st
        f2 = int(st + 1/3 * span)
        f3 = int(st + 2/3 * span)
        f4 = en
        samples = [f1, f2, f3, f4]
        shot_samples[shot_key] = samples
        for f in samples:
            frames_to_process[info["view"]].add(f)
            
    # 3. Load Models
    print(f"[INFO] Loading DINO model ({DINOV3_MODEL})...")
    dino_model = AutoModel.from_pretrained(DINOV3_MODEL).to(device)
    dino_model.eval()
    for p in dino_model.parameters(): p.requires_grad = False
    
    dino_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    dino_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    
    print(f"[INFO] Loading OSNet ReID model ({REID_MODEL})...")
    reid_model = build_reid_model(model_name=REID_MODEL, weights_name=REID_WEIGHTS, pretrained=True, device=device)
    reid_model.eval()
    
    all_dets = load_all_detections()
    
    # To map tracklet IDs consistently across stitched tracks
    stitched_path = MANIFESTS_DIR / "tracklet_features_stitched.csv"
    stitch_map = {}
    if stitched_path.exists():
        import ast
        stitch_df = pd.read_csv(stitched_path)
        for _, row in stitch_df.iterrows():
            if "original_track_ids" in row:
                try:
                    orig_ids = ast.literal_eval(row["original_track_ids"])
                    for orig_id in orig_ids:
                        stitch_map[(row["view"], int(orig_id))] = int(row["stitched_track_id"])
                except:
                    pass
            
    dino_features = {}
    reid_features = {}
    
    # Process view by view to minimize directory hopping
    for view in ["view1", "view2", "view3"]:
        view_frames = sorted(list(frames_to_process[view]))
        if not view_frames:
            continue
            
        print(f"[INFO] Processing {len(view_frames)} target frames for {view}...")
        
        # Precompute per-frame DINO embeddings for required frames
        frame_dino = {}
        # Precompute per-frame ReID embeddings for required bounding boxes
        # Map: frame -> track_id -> tensor
        frame_reid = defaultdict(dict)
        
        view_dir = DATA_DIR / view
        view_det = all_dets[view]
        
        for f_idx in view_frames:
            img_path = view_dir / f"frame_{f_idx:04d}.jpg"
            if not img_path.exists():
                print(f"[WARNING] Missing image: {img_path}")
                continue
                
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue
                
            # --- DINOv3 ---
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_resized = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
            d_tensor = (torch.from_numpy(rgb_resized).permute(2, 0, 1).unsqueeze(0).to(device, dtype=torch.float32) / 255.0)
            d_pixel_vals = (d_tensor - dino_mean) / dino_std
            
            with torch.no_grad():
                out = dino_model(pixel_values=d_pixel_vals)
                cls = out.last_hidden_state[:, 0, :]
                cls_norm = torch.nn.functional.normalize(cls, p=2, dim=-1)
            frame_dino[f_idx] = cls_norm.cpu().numpy()[0]
            
            # --- ReID ---
            nearest_idx, dets = get_nearest_frame_detections(view_det, f_idx)
            if nearest_idx is not None:
                if nearest_idx != f_idx:
                    dist = abs(nearest_idx - f_idx)
                    print(f"[LOG] ReID Fallback Triggered: target frame {f_idx} in {view} used detections from nearest frame {nearest_idx} (distance: {dist})")
                    
                img_h, img_w = frame.shape[:2]
                for det in dets:
                    raw_tid = int(det.get("track_id", -1))
                    tid = stitch_map.get((view, raw_tid), raw_tid)
                    
                    x1 = max(0, min(int(round(det["x1"])), img_w - 1))
                    y1 = max(0, min(int(round(det["y1"])), img_h - 1))
                    x2 = max(x1 + 1, min(int(round(det["x2"])), img_w))
                    y2 = max(y1 + 1, min(int(round(det["y2"])), img_h))
                    
                    crop = frame[y1:y2, x1:x2]
                    if crop.size > 0:
                        crop_t = preprocess_reid_crop(crop, REID_IMAGE_SIZE).unsqueeze(0).to(device)
                        with torch.no_grad():
                            feat = reid_model(crop_t, normalize=True)
                        frame_reid[f_idx][tid] = feat.cpu().numpy()[0]
                        
        # Compile per-shot features for this view
        for shot_key, samples in shot_samples.items():
            if shots[shot_key]["view"] != view:
                continue
                
            # DINO pooling (mean across available frames)
            d_feats = []
            for f in samples:
                if f in frame_dino:
                    d_feats.append(frame_dino[f])
            if d_feats:
                dino_features[shot_key] = np.mean(d_feats, axis=0)
            else:
                print(f"[INFO] Excluded {shot_key} from sparse output (missing DINO features).")
                continue
                
            # ReID pooling (group by track_id across frames, mean pool per track)
            track_feats = defaultdict(list)
            for f in samples:
                for tid, feat in frame_reid[f].items():
                    track_feats[tid].append(feat)
                    
            r_feats = []
            for tid, feats in track_feats.items():
                r_feats.append(np.mean(feats, axis=0))
                
            if len(r_feats) > 1:
                print(f"[INFO] Multiple Tracks Found: Shot {shot_key} contains {len(r_feats)} distinct identities across its 4 frames. Maintaining as separate (N, 512) vectors.")
                
            if r_feats:
                reid_features[shot_key] = np.vstack(r_feats)
            else:
                print(f"[INFO] Excluded {shot_key} from sparse output (missing ReID features).")
                if shot_key in dino_features:
                    del dino_features[shot_key]
                continue
                
    # Save outputs
    out_dino = MANIFESTS_DIR / "training_features_dino_w150_sparse.npz"
    np.savez(out_dino, **dino_features)
    
    out_reid = MANIFESTS_DIR / "training_features_reid_w150_sparse.npz"
    np.savez(out_reid, **reid_features)
    
    print(f"\\n[DONE] Wrote sparse DINOv3 embeddings to {out_dino.name}")
    print(f"[DONE] Wrote sparse ReID embeddings to {out_reid.name}")

if __name__ == "__main__":
    main()
