"""
extract_frames.py

Extracts frames from the 3 raw CCTV views, applying per-view frame offsets
so that frame index N corresponds to the same real-world instant across
all views.

Offsets determined manually (see project_status.md, Section 6):
    view2 : 0  frames (reference)
    view3 : +1 frame
    view1 : +3 frames

All videos are 25.00 fps, same resolution, same duration -> no FPS
resampling needed, only frame-accurate trimming via ffmpeg's `select`
filter (chosen over -ss for frame-exact accuracy at 25fps).

Usage:
    python src/extract_frames.py
"""

import subprocess
import csv
import sys
from pathlib import Path

from config import RAW_VIDEOS, DATA_DIR, LOGS_DIR

# --- Config: per-view source filename and frame offset ---------------------
VIEWS = {
    "view1": {"source": "view1.avi", "offset": 3},
    "view2": {"source": "view2.avi", "offset": 0},
    "view3": {"source": "view3.avi", "offset": 1},
}

FPS = 25.00
RESOLUTION = None  # filled in from ffprobe below


def probe_resolution(video_path: Path) -> str:
    """Return 'WIDTHxHEIGHT' for a video using ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "csv=s=x:p=0",
            str(video_path),
        ],
        capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


def extract_view(view_name: str, source_filename: str, offset: int, force: bool = False) -> int:
    """
    Extract frames for one view, skipping `offset` leading frames.
    Returns the frame count extracted (or already present, if skipped).
    """
    src = RAW_VIDEOS / source_filename
    out_dir = DATA_DIR / view_name
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = len(list(out_dir.glob("frame_*.jpg")))
    if existing > 0 and not force:
        print(f"[SKIP] {view_name}: {existing} frames already present "
              f"(use --force to re-extract)")
        return existing

    if not src.exists():
        print(f"[ERROR] Source video not found: {src}", file=sys.stderr)
        sys.exit(1)

    if offset == 0:
        vf_arg = None
    else:
        vf_arg = f"select='gte(n\\,{offset})'"

    cmd = ["ffmpeg", "-y", "-i", str(src)]
    if vf_arg:
        cmd += ["-vf", vf_arg]
    cmd += ["-vsync", "0", "-start_number", "0",
            str(out_dir / "frame_%04d.jpg")]

    print(f"[INFO] Extracting {view_name} (offset={offset}) -> {out_dir}")
    subprocess.run(cmd, check=True)

    frame_count = len(list(out_dir.glob("frame_*.jpg")))
    print(f"[INFO] {view_name}: {frame_count} frames written")
    return frame_count


def write_metadata(rows: list[dict], resolution: str) -> Path:
    # metadata.csv is lightweight and version-controlled, so it lives in the
    # repo's data_manifests/ folder (git-tracked) rather than DATA_DIR
    # (Drive-synced, gitignored, holds only bulky frame/video data).
    manifests_dir = Path(__file__).resolve().parent.parent / "data_manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = manifests_dir / "metadata.csv"
    with open(metadata_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["view", "fps", "frame_offset", "time_offset_sec",
                        "resolution", "frame_count", "notes"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"[INFO] Metadata written to {metadata_path}")
    return metadata_path


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true",
                         help="Re-extract frames even if output already exists")
    parser.add_argument("--metadata-only", action="store_true",
                         help="Skip extraction entirely; regenerate "
                              "data_manifests/metadata.csv from existing "
                              "frame counts on disk")
    args = parser.parse_args()

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    resolution = probe_resolution(RAW_VIDEOS / VIEWS["view2"]["source"])

    rows = []
    for view_name, info in VIEWS.items():
        offset = info["offset"]
        if args.metadata_only:
            out_dir = DATA_DIR / view_name
            frame_count = len(list(out_dir.glob("frame_*.jpg")))
            print(f"[INFO] {view_name}: found {frame_count} existing frames")
        else:
            frame_count = extract_view(view_name, info["source"], offset,
                                        force=args.force)
        rows.append({
            "view": view_name,
            "fps": FPS,
            "frame_offset": offset,
            "time_offset_sec": round(offset / FPS, 4),
            "resolution": resolution,
            "frame_count": frame_count,
            "notes": "Reference view" if offset == 0
                     else f"Aligned to view2 (reference); offset +{offset} frames",
        })

    write_metadata(rows, resolution)
    if not args.metadata_only:
        print("[DONE] Frame extraction complete. Manually verify cross-view "
              "correspondence before proceeding (see requirements doc, Step 4).")


if __name__ == "__main__":
    main()