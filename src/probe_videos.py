"""
probe_videos.py

Formal Step 1 inspection (per requirements doc, Section 3.2, Step 1):
logs duration, FPS, resolution, and codec for each raw video into a
repo-tracked manifest (data_manifests/video_inspection.csv).

This is read-only inspection — no frames are extracted or modified.

Usage:
    python -m src.probe_videos
"""

import csv
import subprocess
from pathlib import Path

from config import RAW_VIDEOS

SOURCES = {
    "view1": "view1.avi",
    "view2": "view2.avi",
    "view3": "view3.avi",
}


def probe(video_path: Path) -> dict:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries",
            "stream=codec_name,width,height,r_frame_rate,duration,nb_frames",
            "-of", "default=noprint_wrappers=1",
            str(video_path),
        ],
        capture_output=True, text=True, check=True,
    )
    fields = {}
    for line in result.stdout.strip().splitlines():
        key, _, value = line.partition("=")
        fields[key] = value
    return fields


def main():
    manifests_dir = Path(__file__).resolve().parent.parent / "data_manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    out_path = manifests_dir / "video_inspection.csv"

    rows = []
    for view_name, filename in SOURCES.items():
        video_path = RAW_VIDEOS / filename
        print(f"[INFO] Probing {view_name} ({video_path})")
        fields = probe(video_path)
        rows.append({
            "view": view_name,
            "filename": filename,
            "codec_name": fields.get("codec_name", ""),
            "width": fields.get("width", ""),
            "height": fields.get("height", ""),
            "r_frame_rate": fields.get("r_frame_rate", ""),
            "duration_sec": fields.get("duration", ""),
            "nb_frames": fields.get("nb_frames", ""),
        })

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["view", "filename", "codec_name", "width", "height",
                        "r_frame_rate", "duration_sec", "nb_frames"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"[DONE] Wrote inspection results to {out_path}")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()