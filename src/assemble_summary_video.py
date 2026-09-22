"""
assemble_summary_video.py

Produces a playable video file of the final summary by stitching together
the actual frames from each event sequence. Overlays text to indicate the 
view and sequence order.

Usage:
    python -m src.assemble_summary_video
"""

import argparse
import csv
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2

from config import DATA_DIR, OUTPUTS_DIR

# --- Module-Level Constants ---
MANIFESTS_DIR = Path(__file__).resolve().parent.parent / "data_manifests"
FPS = 25.0


def assemble_video(input_csv: str, output_mp4: str) -> None:
    """
    Reads the summary CSV, collects all frames for each shot,
    overlays a label, and encodes the sequence into an MP4 using ffmpeg.
    """
    csv_path = Path(input_csv)
    if not csv_path.exists():
        print(f"[ERROR] Required input file missing: {csv_path}", file=sys.stderr)
        sys.exit(1)

    shots = []
    with open(csv_path, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            shots.append(row)

    if not shots:
        print("[INFO] No shots in summary to assemble.")
        return

    # Sort by sequence_order if it exists, otherwise process in order
    if all("sequence_order" in s for s in shots):
        shots.sort(key=lambda x: int(x["sequence_order"]))

    # Step 2 & 3: Collect ordered list of frame file paths
    print(f"[INFO] Collecting frames for {len(shots)} summary events...")
    frame_sequence = []
    for i, shot in enumerate(shots):
        view = shot["view"]
        start_f = int(shot["window_start_frame"])
        end_f = int(shot["window_end_frame"])
        seq_order = shot.get("sequence_order", str(i+1))

        for frame_idx in range(start_f, end_f + 1):
            img_filename = f"frame_{frame_idx:04d}.jpg"
            img_path = DATA_DIR / view / img_filename
            
            if not img_path.exists():
                print(f"[ERROR] Missing frame image: {img_path}", file=sys.stderr)
                sys.exit(1)  # Halt on missing frame
            
            frame_sequence.append({
                "path": img_path,
                "view": view,
                "seq_order": seq_order,
                "frame_idx": frame_idx
            })

    total_frames = len(frame_sequence)
    if total_frames == 0:
        print("[INFO] Frame sequence is empty.")
        return

    print(f"[INFO] Total frames to process: {total_frames}. Preparing video assembly...")

    out_video = Path(output_mp4)
    out_video.parent.mkdir(parents=True, exist_ok=True)

    # Step 4: Write labeled frames to a temporary directory for ffmpeg
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        for i, item in enumerate(frame_sequence):
            frame = cv2.imread(str(item["path"]))
            if frame is None:
                print(f"[ERROR] Failed to read frame: {item['path']}", file=sys.stderr)
                sys.exit(1)
                
            # Overlay on-screen label
            label = f"Event {item['seq_order']} | {item['view']} | Frame {item['frame_idx']}"
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.6
            thick = 2
            
            # Draw black background badge for text readability
            (tw, th), baseline = cv2.getTextSize(label, font, scale, thick)
            x, y = 20, frame.shape[0] - 30
            
            cv2.rectangle(frame, (x - 4, y - th - 4), (x + tw + 4, y + baseline + 4), (0, 0, 0), -1)
            cv2.putText(frame, label, (x, y), font, scale, (0, 255, 255), thick, cv2.LINE_AA)
            
            # Save to temporary directory
            out_frame_path = temp_path / f"frame_{i:06d}.jpg"
            cv2.imwrite(str(out_frame_path), frame)
            
            if (i + 1) % 500 == 0:
                print(f"[INFO] Processed {i + 1}/{total_frames} frames...")

        # Step 5 & 6: Encode with ffmpeg
        print(f"[INFO] Encoding MP4 video with ffmpeg...")
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",  # Overwrite
            "-framerate", str(FPS),
            "-i", str(temp_path / "frame_%06d.jpg"),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            str(out_video)
        ]
        
        try:
            subprocess.run(ffmpeg_cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] ffmpeg encoding failed: {e.stderr}", file=sys.stderr)
            sys.exit(1)

    # Step 7: Print [DONE] summary
    duration = total_frames / FPS
    print(f"[DONE] Rendered final summary video to {out_video.resolve()}")
    print(f"       Total frames: {total_frames}")
    print(f"       Duration: {duration:.2f} seconds")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=str, default=str(MANIFESTS_DIR / "final_summary.csv"), help="Input CSV summary file")
    parser.add_argument("--output", type=str, default=str(MANIFESTS_DIR / "final_summary_video.mp4"), help="Output MP4 video file")
    args = parser.parse_args()
    assemble_video(args.input, args.output)


if __name__ == "__main__":
    main()
