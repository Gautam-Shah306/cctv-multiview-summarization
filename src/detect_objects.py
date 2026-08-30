"""
detect_objects.py

Runs per-view person detection over the extracted, synced frame sequences
using a pretrained YOLOv8 model (Ultralytics).

Stage 1a of the proposed deep-learning pipeline (see
Proposed_Methodology_Changes.md, Section 3.1): detector upgrade only. The
ReID embedding head (Stage 1b) is deferred to a follow-up script until this
stage is verified end-to-end.

Only the COCO "person" class (class_id = 0) is kept, matching the base
paper's person-presence use case and keeping output volume manageable.

Detections are written per-view as CSV files under DATA_DIR/detections/
(bulky, Drive-side, gitignored). A lightweight run summary is written to
data_manifests/detections_summary.csv (git-tracked), consistent with the
project's existing bulky-vs-lightweight data split.

Usage:
    python -m src.detect_objects
    python -m src.detect_objects --view view1
    python -m src.detect_objects --force
"""

# Group 1 — standard library
import csv
import shutil
import sys
import tempfile
from pathlib import Path

# Group 2 — third-party
import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Group 3 — local / project
from config import (
    CHECKPOINTS,
    DATA_DIR,
    DETECTION_BATCH_SIZE,
    DETECTION_CONFIDENCE,
    DETECTION_HALF,
    DETECTION_IMG_SIZE,
    DETECTION_MODEL,
    DETECTION_TRACKER,
    MOTION_GATING_THRESHOLD,
)
from src.extract_frames import VIEWS as VIEW_OFFSETS  # reuse canonical view names

# --- Config: detection-specific constants -----------------------------------
PERSON_CLASS_ID = 0  # COCO class id for "person"
VIEWS = list(VIEW_OFFSETS.keys())

DETECTIONS_DIR = DATA_DIR / "detections"
MANIFEST_PATH = (
    Path(__file__).resolve().parent.parent / "data_manifests" / "detections_summary.csv"
)


def load_model(model_name: str) -> YOLO:
    """Load YOLO weights, caching the download under CHECKPOINTS."""
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    weights_path = CHECKPOINTS / model_name

    model = YOLO(str(weights_path) if weights_path.exists() else model_name)

    if not weights_path.exists():
        downloaded = Path(model_name)
        if downloaded.exists():
            shutil.move(str(downloaded), str(weights_path))
        print(f"[INFO] Downloaded {model_name} -> {weights_path}")
    else:
        print(f"[INFO] Loaded cached weights from {weights_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"[INFO] Using device: {device} (half_precision={DETECTION_HALF and device == 'cuda'})")

    return model


def check_motion(frame_path: Path, prev_gray: np.ndarray | None, threshold: int) -> tuple[bool, np.ndarray | None]:
    """Check if current frame contains motion relative to the previous frame."""
    curr = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
    if curr is None:
        return True, prev_gray
    if prev_gray is None:
        return True, curr
    diff = cv2.absdiff(curr, prev_gray)
    active = int(np.count_nonzero(diff > 25)) > threshold
    return active, curr


def detect_view(
    view_name: str,
    model: YOLO,
    force: bool = False,
    batch_size: int = DETECTION_BATCH_SIZE,
    track: bool = False,
    motion_gating: bool = False,
) -> int:
    """
    Run detection/tracking over frames in a view using batched inference.
    Writes CSV rows to DATA_DIR/detections/<view>.csv.
    """
    frames_dir = DATA_DIR / view_name
    out_path = DETECTIONS_DIR / f"{view_name}.csv"

    if out_path.exists() and not force:
        frame_count = len(list(frames_dir.glob("frame_*.jpg")))
        print(f"[SKIP] {view_name}: detections already present at {out_path} (use --force to re-run)")
        return frame_count

    frame_paths = sorted(frames_dir.glob("frame_*.jpg"))

    local_cache_dir = Path(tempfile.gettempdir()) / "frame_cache" / view_name
    local_cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Caching {view_name} frames to local disk ({local_cache_dir})")
    local_frame_paths = []
    for p in frame_paths:
        local_path = local_cache_dir / p.name
        if not local_path.exists():
            shutil.copy(p, local_path)
        local_frame_paths.append(local_path)
    print(f"[INFO] {view_name}: {len(local_frame_paths)} frames cached locally")

    if not local_frame_paths:
        print(f"[ERROR] No frames found in {frames_dir}", file=sys.stderr)
        sys.exit(1)

    DETECTIONS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    use_half = DETECTION_HALF and torch.cuda.is_available()
    # Only pass half= when True; passing half=False triggers a deprecation
    # warning on every call in recent Ultralytics versions.
    half_kwargs = {"half": True} if use_half else {}

    print(f"[INFO] Running detection on {view_name} ({len(local_frame_paths)} frames, batch_size={batch_size}, track={track}, motion_gating={motion_gating})")

    prev_gray = None

    if track:
        # --- Tracking mode: feed one frame at a time to preserve ByteTrack's
        # Kalman-filter state across frames. Batching breaks temporal continuity
        # and causes all track_ids to fall back to -1.
        for frame_num, frame_p in enumerate(local_frame_paths):
            if motion_gating:
                active, prev_gray = check_motion(frame_p, prev_gray, MOTION_GATING_THRESHOLD)
                if not active:
                    continue

            result = model.track(
                source=str(frame_p),
                conf=DETECTION_CONFIDENCE,
                imgsz=DETECTION_IMG_SIZE,
                classes=[PERSON_CLASS_ID],
                tracker=DETECTION_TRACKER,
                persist=True,
                verbose=False,
                **half_kwargs,
            )[0]

            frame_idx = int(frame_p.stem.split("_")[1])
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = [round(v, 2) for v in box.xyxy[0].tolist()]
                    track_id = int(box.id[0]) if box.id is not None else -1
                    rows.append({
                        "frame_idx": frame_idx,
                        "view": view_name,
                        "track_id": track_id,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "confidence": round(float(box.conf[0]), 4),
                        "class_id": PERSON_CLASS_ID,
                    })

            if (frame_num + 1) % 100 == 0 or (frame_num + 1) == len(local_frame_paths):
                print(f"[INFO] {view_name}: {frame_num + 1}/{len(local_frame_paths)} frames tracked")

    else:
        # --- Detection-only mode: high-throughput batched GPU/CPU inference
        for i in range(0, len(local_frame_paths), batch_size):
            batch_paths = local_frame_paths[i:i + batch_size]
            active_paths = []

            if motion_gating:
                for p in batch_paths:
                    active, prev_gray = check_motion(p, prev_gray, MOTION_GATING_THRESHOLD)
                    if active:
                        active_paths.append(p)
            else:
                active_paths = batch_paths

            if not active_paths:
                continue

            results = model.predict(
                source=[str(p) for p in active_paths],
                batch=len(active_paths),
                conf=DETECTION_CONFIDENCE,
                imgsz=DETECTION_IMG_SIZE,
                classes=[PERSON_CLASS_ID],
                verbose=False,
                **half_kwargs,
            )

            for frame_p, result in zip(active_paths, results):
                frame_idx = int(frame_p.stem.split("_")[1])
                boxes = result.boxes
                if boxes is None:
                    continue

                for box in boxes:
                    x1, y1, x2, y2 = [round(v, 2) for v in box.xyxy[0].tolist()]
                    rows.append({
                        "frame_idx": frame_idx,
                        "view": view_name,
                        "track_id": -1,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "confidence": round(float(box.conf[0]), 4),
                        "class_id": PERSON_CLASS_ID,
                    })

            processed = min(i + batch_size, len(local_frame_paths))
            if processed % (batch_size * 5) == 0 or processed == len(local_frame_paths):
                print(f"[INFO] {view_name}: {processed}/{len(local_frame_paths)} frames processed")

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["frame_idx", "view", "track_id", "x1", "y1", "x2", "y2", "confidence", "class_id"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"[INFO] {view_name}: {len(rows)} detections written to {out_path}")
    return len(local_frame_paths)


def write_manifest(summary_rows: list[dict]) -> Path:
    """Write a lightweight per-view run summary to data_manifests/detections_summary.csv."""
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    existing_rows = {}
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH) as f:
            for row in csv.DictReader(f):
                existing_rows[row["view"]] = row

    for row in summary_rows:
        existing_rows[row["view"]] = row

    fieldnames = ["view", "frames_processed", "model", "confidence_threshold", "batch_size", "tracked"]
    with open(MANIFEST_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(existing_rows.values())

    print(f"[DONE] Manifest written to {MANIFEST_PATH}")
    return MANIFEST_PATH


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true",
                         help="Re-run detection even if output already exists")
    parser.add_argument("--view", choices=VIEWS, default=None,
                         help="Run detection on a single view only (default: all views)")
    parser.add_argument("--batch-size", type=int, default=DETECTION_BATCH_SIZE,
                         help=f"Inference batch size (default: {DETECTION_BATCH_SIZE})")
    parser.add_argument("--track", action="store_true",
                         help="Enable ByteTrack multi-object tracking across frames")
    parser.add_argument("--motion-gating", action="store_true",
                         help="Skip static frames with no significant motion")
    args = parser.parse_args()

    model = load_model(DETECTION_MODEL)
    views_to_run = [args.view] if args.view else VIEWS

    summary_rows = []
    for view_name in views_to_run:
        frames_processed = detect_view(
            view_name,
            model,
            force=args.force,
            batch_size=args.batch_size,
            track=args.track,
            motion_gating=args.motion_gating,
        )
        summary_rows.append({
            "view": view_name,
            "frames_processed": frames_processed,
            "model": DETECTION_MODEL,
            "confidence_threshold": DETECTION_CONFIDENCE,
            "batch_size": args.batch_size,
            "tracked": args.track,
        })

    write_manifest(summary_rows)


if __name__ == "__main__":
    main()