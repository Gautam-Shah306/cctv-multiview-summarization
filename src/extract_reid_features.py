"""
extract_reid_features.py

Extracts deep Person Re-Identification (ReID) visual feature embeddings
for all person bounding boxes detected in Stage 1a.

Stage 1b of the proposed deep-learning pipeline:
1. Loads per-view person detections from DATA_DIR/detections/<view>.csv.
2. Crops person bounding boxes from synchronized video frames.
3. Preprocesses crops (256x128, ImageNet mean/std normalization).
4. Runs batched inference via pretrained OSNet (Market-1501) with L2 normalization.
5. Saves bulky feature embeddings to DATA_DIR/embeddings/<view>_embeddings.npz (Drive-side, gitignored).
6. Writes a lightweight run summary to data_manifests/reid_summary.csv (git-tracked).

Usage:
    python -m src.extract_reid_features
    python -m src.extract_reid_features --view view1
    python -m src.extract_reid_features --force
    python -m src.extract_reid_features --batch-size 64
"""

# Group 1 — standard library
import argparse
import csv
import shutil
import sys
import tempfile
from pathlib import Path

# Group 2 — third-party
import cv2
import numpy as np
import torch
import torch.nn as nn

# Group 3 — local / project
from config import (
    DATA_DIR,
    REID_BATCH_SIZE,
    REID_EMBEDDING_DIM,
    REID_HALF,
    REID_IMAGE_SIZE,
    REID_MODEL,
    REID_WEIGHTS,
)
from src.extract_frames import VIEWS as VIEW_OFFSETS
from src.models.reid_model import build_reid_model

# --- Config: ReID paths and constants ---------------------------------------
VIEWS = list(VIEW_OFFSETS.keys())
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
MANIFEST_PATH = (
    Path(__file__).resolve().parent.parent / "data_manifests" / "reid_summary.csv"
)

# ImageNet normalization statistics
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)


def load_detections(view_name: str) -> list[dict]:
    """Load detection rows for a view from DATA_DIR/detections/<view>.csv."""
    csv_path = DATA_DIR / "detections" / f"{view_name}.csv"
    if not csv_path.exists():
        print(f"[ERROR] Detections CSV not found: {csv_path}", file=sys.stderr)
        return []

    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "frame_idx": int(row["frame_idx"]),
                "view": row["view"],
                "track_id": int(row.get("track_id", -1)),
                "x1": float(row["x1"]),
                "y1": float(row["y1"]),
                "x2": float(row["x2"]),
                "y2": float(row["y2"]),
                "confidence": float(row.get("confidence", 0.0)),
                "class_id": int(row.get("class_id", 0)),
            })
    return rows


def preprocess_crop(crop_bgr: np.ndarray, target_size: tuple[int, int]) -> torch.Tensor:
    """
    Preprocess a BGR image crop into an ImageNet-standardized PyTorch tensor.
    Args:
        crop_bgr: BGR image patch as uint8 numpy array.
        target_size: Tuple (height, width), e.g. (256, 128).
    Returns:
        Tensor of shape (3, H, W) normalized with ImageNet mean and std.
    """
    h, w = target_size
    resized = cv2.resize(crop_bgr, (w, h), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    normalized = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(normalized.transpose(2, 0, 1))
    return tensor


def extract_view_features(
    view_name: str,
    model: nn.Module,
    force: bool = False,
    batch_size: int = REID_BATCH_SIZE,
    device: str = "cuda",
) -> int:
    """
    Extract ReID feature embeddings for all detected person bounding boxes in a view.
    Saves compressed .npz archive to DATA_DIR/embeddings/<view>_embeddings.npz.
    """
    out_path = EMBEDDINGS_DIR / f"{view_name}_embeddings.npz"
    if out_path.exists() and not force:
        print(f"[SKIP] {view_name}: ReID embeddings already present at {out_path} (use --force to re-run)")
        with np.load(out_path) as data:
            return len(data["embeddings"])

    detections = load_detections(view_name)
    if not detections:
        print(f"[WARNING] No detections found for {view_name}")
        return 0

    frames_dir = DATA_DIR / view_name
    local_cache_dir = Path(tempfile.gettempdir()) / "frame_cache" / view_name
    local_cache_dir.mkdir(parents=True, exist_ok=True)

    # Cache frames locally for high-speed I/O
    frame_indices_needed = sorted(set(d["frame_idx"] for d in detections))
    print(f"[INFO] Caching {len(frame_indices_needed)} active frames for {view_name} to local disk")
    for idx in frame_indices_needed:
        src_frame = frames_dir / f"frame_{idx:04d}.jpg"
        dst_frame = local_cache_dir / f"frame_{idx:04d}.jpg"
        if not dst_frame.exists() and src_frame.exists():
            shutil.copy(src_frame, dst_frame)

    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

    # Group detections by frame index for sequential reading
    dets_by_frame: dict[int, list[dict]] = {}
    for d in detections:
        dets_by_frame.setdefault(d["frame_idx"], []).append(d)

    crops_tensor_list = []
    metadata_list = []
    all_embeddings = []

    use_half = REID_HALF and device.startswith("cuda")
    print(f"[INFO] Extracting ReID features on {view_name} ({len(detections)} detections, batch_size={batch_size}, half_precision={use_half})")

    for idx in sorted(dets_by_frame.keys()):
        frame_file = local_cache_dir / f"frame_{idx:04d}.jpg"
        if not frame_file.exists():
            frame_file = frames_dir / f"frame_{idx:04d}.jpg"

        frame = cv2.imread(str(frame_file))
        if frame is None:
            continue

        img_h, img_w = frame.shape[:2]

        for det in dets_by_frame[idx]:
            x1 = max(0, min(int(round(det["x1"])), img_w - 1))
            y1 = max(0, min(int(round(det["y1"])), img_h - 1))
            x2 = max(x1 + 1, min(int(round(det["x2"])), img_w))
            y2 = max(y1 + 1, min(int(round(det["y2"])), img_h))

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            tensor = preprocess_crop(crop, REID_IMAGE_SIZE)
            crops_tensor_list.append(tensor)
            metadata_list.append({
                "frame_idx": det["frame_idx"],
                "track_id": det["track_id"],
                "box": [det["x1"], det["y1"], det["x2"], det["y2"]],
                "confidence": det["confidence"],
            })

            # Process in mini-batches
            if len(crops_tensor_list) >= batch_size:
                batch_tensor = torch.stack(crops_tensor_list).to(device)
                if use_half:
                    batch_tensor = batch_tensor.half()

                with torch.no_grad():
                    feats = model(batch_tensor, normalize=True)
                    all_embeddings.append(feats.float().cpu().numpy())

                crops_tensor_list.clear()

    # Process remaining residual batch
    if crops_tensor_list:
        batch_tensor = torch.stack(crops_tensor_list).to(device)
        if use_half:
            batch_tensor = batch_tensor.half()

        with torch.no_grad():
            feats = model(batch_tensor, normalize=True)
            all_embeddings.append(feats.float().cpu().numpy())

        crops_tensor_list.clear()

    if not all_embeddings:
        print(f"[ERROR] No valid crops processed for {view_name}", file=sys.stderr)
        return 0

    embeddings_array = np.concatenate(all_embeddings, axis=0)  # Shape (N, 512)
    frame_indices = np.array([m["frame_idx"] for m in metadata_list], dtype=np.int32)
    track_ids = np.array([m["track_id"] for m in metadata_list], dtype=np.int32)
    boxes = np.array([m["box"] for m in metadata_list], dtype=np.float32)
    confidences = np.array([m["confidence"] for m in metadata_list], dtype=np.float32)

    np.savez_compressed(
        out_path,
        embeddings=embeddings_array,
        frame_indices=frame_indices,
        track_ids=track_ids,
        boxes=boxes,
        confidences=confidences,
    )

    print(f"[DONE] {view_name}: {len(embeddings_array)} embeddings (dim={embeddings_array.shape[1]}) written to {out_path}")
    return len(embeddings_array)


def write_manifest(summary_rows: list[dict]) -> Path:
    """Write a lightweight per-view run summary to data_manifests/reid_summary.csv."""
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    existing_rows = {}
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH, newline="") as f:
            for row in csv.DictReader(f):
                existing_rows[row["view"]] = row

    for row in summary_rows:
        existing_rows[row["view"]] = row

    fieldnames = [
        "view",
        "detections_processed",
        "embedding_dim",
        "model",
        "weights",
        "batch_size",
    ]
    with open(MANIFEST_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(existing_rows.values())

    print(f"[DONE] ReID manifest written to {MANIFEST_PATH}")
    return MANIFEST_PATH


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true",
                        help="Re-extract embeddings even if output already exists")
    parser.add_argument("--view", choices=VIEWS, default=None,
                        help="Run ReID extraction on a single view only (default: all views)")
    parser.add_argument("--batch-size", type=int, default=REID_BATCH_SIZE,
                        help=f"Inference batch size (default: {REID_BATCH_SIZE})")
    parser.add_argument("--weights", type=str, default=REID_WEIGHTS,
                        help=f"Pretrained weights filename (default: {REID_WEIGHTS})")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Initializing ReID model ({REID_MODEL}) on {device}")
    model = build_reid_model(
        model_name=REID_MODEL,
        weights_name=args.weights,
        pretrained=True,
        device=device,
    )
    if REID_HALF and device.startswith("cuda"):
        model.half()

    views_to_run = [args.view] if args.view else VIEWS

    summary_rows = []
    for view_name in views_to_run:
        num_extracted = extract_view_features(
            view_name=view_name,
            model=model,
            force=args.force,
            batch_size=args.batch_size,
            device=device,
        )
        summary_rows.append({
            "view": view_name,
            "detections_processed": num_extracted,
            "embedding_dim": REID_EMBEDDING_DIM,
            "model": REID_MODEL,
            "weights": args.weights,
            "batch_size": args.batch_size,
        })

    write_manifest(summary_rows)


if __name__ == "__main__":
    main()
