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
import sys
from pathlib import Path

# add to Group 1 (standard library) imports
import shutil

# Group 2 — third-party
from ultralytics import YOLO
import torch

# Group 3 — local / project
from config import DATA_DIR, CHECKPOINTS, DETECTION_MODEL, DETECTION_CONFIDENCE, DETECTION_IMG_SIZE
from src.extract_frames import VIEWS as VIEW_OFFSETS  # reuse canonical view names

# --- Config: detection-specific constants -----------------------------------
PERSON_CLASS_ID = 0  # COCO class id for "person"
VIEWS = list(VIEW_OFFSETS.keys())

DETECTIONS_DIR = DATA_DIR / "detections"
MANIFEST_PATH = (
    Path(__file__).resolve().parent.parent / "data_manifests" / "detections_summary.csv"
)


def load_model(model_name: str) -> YOLO:
    """Load YOLOv8 weights, caching the download under CHECKPOINTS."""
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    weights_path = CHECKPOINTS / model_name

    model = YOLO(str(weights_path) if weights_path.exists() else model_name)

    if not weights_path.exists():
        # ultralytics downloads to the current working directory by default;
        # relocate it into CHECKPOINTS so weights live with other bulky artifacts
        downloaded = Path(model_name)
        if downloaded.exists():
            shutil.move(str(downloaded), str(weights_path))
        print(f"[INFO] Downloaded {model_name} -> {weights_path}")
    else:
        print(f"[INFO] Loaded cached weights from {weights_path}")
    
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(str(weights_path))
    model.to(device)
    print(f"[INFO] Using device: {device}")

    return model


def detect_view(view_name: str, model: YOLO, force: bool = False) -> int:
    """
    Run detection over every frame in a view, writing one CSV row per
    detected person. Skips the view if output already exists, unless
    force=True. Returns the number of frames processed.
    """
    frames_dir = DATA_DIR / view_name
    out_path = DETECTIONS_DIR / f"{view_name}.csv"

    if out_path.exists() and not force:
        frame_count = len(list(frames_dir.glob("frame_*.jpg")))
        print(f"[SKIP] {view_name}: detections already present at {out_path} "
              f"(use --force to re-run)")
        return frame_count  

    frame_paths = sorted(frames_dir.glob("frame_*.jpg"))
    if not frame_paths:
        print(f"[ERROR] No frames found in {frames_dir}", file=sys.stderr)
        sys.exit(1)

    DETECTIONS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    print(f"[INFO] Running detection on {view_name} ({len(frame_paths)} frames)")
    for i, frame_path in enumerate(frame_paths):
        frame_idx = int(frame_path.stem.split("_")[1])
        results = model.predict(
            source=str(frame_path),
            conf=DETECTION_CONFIDENCE,
            imgsz=DETECTION_IMG_SIZE,
            classes=[PERSON_CLASS_ID],
            verbose=False,
        )
        for box in results[0].boxes:
            x1, y1, x2, y2 = [round(v, 2) for v in box.xyxy[0].tolist()]
            rows.append({
                "frame_idx": frame_idx,
                "view": view_name,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "confidence": round(float(box.conf[0]), 4),
                "class_id": PERSON_CLASS_ID,
            })
        
        if (i + 1) % 200 == 0 or (i + 1) == len(frame_paths):
            print(f"[INFO] {view_name}: {i + 1}/{len(frame_paths)} frames processed")

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["frame_idx", "view", "x1", "y1", "x2", "y2",
                        "confidence", "class_id"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"[INFO] {view_name}: {len(rows)} detections written to {out_path}")
    return len(frame_paths)


def write_manifest(summary_rows: list[dict]) -> Path:
    """
    Write a lightweight per-view run summary to data_manifests/detections_summary.csv.

    Merges with any existing manifest, keyed by view, so that separate
    per-view invocations (e.g. --view view1 then --view view2) accumulate
    into one combined manifest instead of each run overwriting the last.
    """
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    existing_rows = {}
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH) as f:
            for row in csv.DictReader(f):
                existing_rows[row["view"]] = row

    for row in summary_rows:
        existing_rows[row["view"]] = row

    fieldnames = ["view", "frames_processed", "model", "confidence_threshold"]
    with open(MANIFEST_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
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
    args = parser.parse_args()

    model = load_model(DETECTION_MODEL)
    views_to_run = [args.view] if args.view else VIEWS

    summary_rows = []
    for view_name in views_to_run:
        frames_processed = detect_view(view_name, model, force=args.force)
        summary_rows.append({
            "view": view_name,
            "frames_processed": frames_processed,
            "model": DETECTION_MODEL,
            "confidence_threshold": DETECTION_CONFIDENCE,
        })

    write_manifest(summary_rows)


if __name__ == "__main__":
    main()