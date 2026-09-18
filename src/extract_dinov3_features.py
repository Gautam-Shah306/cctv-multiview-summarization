"""
extract_dinov3_features.py

Novelty #1: Full-Frame Vision Transformer Feature Extraction.

Extracts frozen, self-supervised Vision Transformer (DINO) embeddings for every
frame across CCTV camera views. DINO features capture global scene layout, background
dynamics, and visual context to complement pedestrian-centric ReID embeddings.

Model Note:
    Defaults to `facebook/dinov2-small` (ViT-Small/14, 22M parameters, 384-d output),
    which is ungated, lightweight, and pre-cached locally. Official Meta DINOv3 weights
    (`facebook/dinov3-*`) are supported via `--model-name` for environments with gated
    Hugging Face access tokens configured.

Outputs:
    data_storage/data/dinov3_embeddings/{view}_dinov3.npz
    Arrays:
        - embeddings: (N, 384) float32, unit-L2 normalized (||d||_2 = 1.0)
        - frame_indices: (N,) int32 sorted frame numbers

Usage:
    python -m src.extract_dinov3_features --view view1
    python -m src.extract_dinov3_features --view all --batch-size 32
"""

# Group 1 — standard library
import argparse
import sys
import time
from pathlib import Path

# Group 2 — third-party
import cv2
import numpy as np
import torch
from torchvision import transforms
from transformers import AutoModel

# Group 3 — local / project
from config import DATA_DIR, DINOV3_EMBEDDINGS_DIR, DINOV3_MODEL
from src.extract_frames import VIEWS as VIEW_CONFIGS

# --- Module Constants & Defaults --------------------------------------------
SUPPORTED_VIEWS = list(VIEW_CONFIGS.keys())
DEFAULT_BATCH_SIZE = 32
IMAGE_SIZE = (224, 224)


def get_device() -> torch.device:
    """Return CUDA device if available, otherwise CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_dino_model(model_name: str, device: torch.device) -> AutoModel:
    """
    Load and freeze pretrained DINO Vision Transformer model.

    Args:
        model_name: HuggingFace model identifier.
        device: Target execution device.

    Returns:
        Frozen AutoModel in evaluation mode.
    """
    print(f"[INFO] Loading DINO model: {model_name} onto {device}...")
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def get_image_transform() -> transforms.Compose:
    """Return optimized Torchvision preprocessing pipeline for DINO ViT."""
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def get_view_frame_paths(view_name: str) -> list[tuple[int, Path]]:
    """
    Gather and sort all pre-extracted frames for a view.

    Returns:
        List of (frame_idx, Path) tuples sorted ascending by frame index.
    """
    view_dir = DATA_DIR / view_name
    if not view_dir.exists():
        raise FileNotFoundError(f"Frame directory not found: {view_dir}")

    frame_files = list(view_dir.glob("frame_*.jpg"))
    if not frame_files:
        raise FileNotFoundError(f"No frames found in {view_dir}")

    indexed_paths = []
    for p in frame_files:
        try:
            f_idx = int(p.stem.split("_")[1])
            indexed_paths.append((f_idx, p))
        except (IndexError, ValueError):
            continue

    indexed_paths.sort(key=lambda x: x[0])
    return indexed_paths


def extract_view_dinov3(
    view_name: str,
    model: AutoModel,
    device: torch.device,
    transform: transforms.Compose,
    batch_size: int = DEFAULT_BATCH_SIZE,
    force: bool = False,
) -> Path:
    """
    Extract DINO embeddings for all frames of a given view.

    Args:
        view_name: Camera view identifier (e.g. 'view1').
        model: Frozen DINO model.
        device: PyTorch device.
        transform: Image transform pipeline.
        batch_size: Inference batch size.
        force: Overwrite existing embeddings if True.

    Returns:
        Path to the saved npz archive.
    """
    DINOV3_EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DINOV3_EMBEDDINGS_DIR / f"{view_name}_dinov3.npz"

    if out_path.exists() and not force:
        print(f"[INFO] {view_name} DINO embeddings already exist at {out_path}. Skipping.")
        return out_path

    indexed_frames = get_view_frame_paths(view_name)
    total_frames = len(indexed_frames)
    print(f"[INFO] Extracting DINO embeddings for {view_name}: {total_frames} frames (batch_size={batch_size})...")

    all_embeddings = []
    all_frame_indices = []

    t_start = time.time()
    num_batches = (total_frames + batch_size - 1) // batch_size

    # Vectorized ImageNet mean and std tensors
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    for b_idx in range(num_batches):
        batch_items = indexed_frames[b_idx * batch_size : (b_idx + 1) * batch_size]
        batch_indices = [item[0] for item in batch_items]
        batch_imgs = []

        for _, img_path in batch_items:
            bgr = cv2.imread(str(img_path))
            if bgr is None:
                rgb = np.zeros((224, 224, 3), dtype=np.uint8)
            else:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                rgb = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
            batch_imgs.append(rgb)

        np_batch = np.stack(batch_imgs)
        # Convert to tensor: (B, C, H, W) normalized to [0, 1]
        tensor_batch = (
            torch.from_numpy(np_batch).permute(0, 3, 1, 2).to(device=device, dtype=torch.float32)
            / 255.0
        )
        # ImageNet normalization
        pixel_values = (tensor_batch - mean) / std

        with torch.no_grad():
            outputs = model(pixel_values=pixel_values)
            # CLS token embedding from the last hidden state
            cls_embeddings = outputs.last_hidden_state[:, 0, :]
            # Unit-L2 normalization
            norm_embeddings = torch.nn.functional.normalize(cls_embeddings, p=2, dim=-1)

        all_embeddings.append(norm_embeddings.cpu().numpy().astype(np.float32))
        all_frame_indices.extend(batch_indices)

        processed = min(total_frames, (b_idx + 1) * batch_size)
        if (b_idx + 1) % 25 == 0 or (b_idx + 1) == num_batches:
            elapsed = time.time() - t_start
            rate = processed / elapsed if elapsed > 0 else 0
            eta = (total_frames - processed) / rate if rate > 0 else 0
            print(
                f"       [{view_name}] {processed}/{total_frames} frames "
                f"({rate:.1f} fps, ETA: {eta:.1f}s)"
            )

    embeddings_matrix = np.vstack(all_embeddings)
    frame_indices_arr = np.array(all_frame_indices, dtype=np.int32)

    total_time = time.time() - t_start
    print(
        f"[INFO] Completed {view_name}: {embeddings_matrix.shape[0]} embeddings "
        f"extracted in {total_time:.1f}s ({total_time / total_frames * 1000:.1f} ms/frame)."
    )

    np.savez_compressed(
        out_path,
        embeddings=embeddings_matrix,
        frame_indices=frame_indices_arr,
    )
    print(f"[DONE] Saved {view_name} DINO embeddings to: {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract full-frame DINO ViT embeddings for multi-view CCTV summarization."
    )
    parser.add_argument(
        "--view",
        type=str,
        default="all",
        choices=["all"] + SUPPORTED_VIEWS,
        help="View to process (default: all)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DINOV3_MODEL,
        help=f"Pretrained ViT model identifier (default: {DINOV3_MODEL})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Inference batch size (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-extraction even if embeddings file already exists",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Novelty #1: Full-Frame DINO Feature Extraction")
    print("=" * 70)
    print(f"Model ID  : {args.model_name}")
    print(f"Target    : {args.view}")
    print(f"Batch Size: {args.batch_size}")
    print("-" * 70)

    device = get_device()
    model = load_dino_model(args.model_name, device)
    transform = get_image_transform()

    views_to_process = SUPPORTED_VIEWS if args.view == "all" else [args.view]

    for v in views_to_process:
        extract_view_dinov3(
            view_name=v,
            model=model,
            device=device,
            transform=transform,
            batch_size=args.batch_size,
            force=args.force,
        )

    print("=" * 70)
    print("[SUCCESS] All targeted DINO feature extractions complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
