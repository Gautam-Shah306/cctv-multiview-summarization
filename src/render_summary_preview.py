"""
render_summary_preview.py

Generates a visual grid preview of the final multi-view summarization output
by compositing the keyframe images for each fused event cluster.

Usage:
    python -m src.render_summary_preview
"""

import csv
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from config import DATA_DIR

# --- Module-Level Constants ---
MANIFESTS_DIR = Path(__file__).resolve().parent.parent / "data_manifests"
FPS = 25.0  # Known camera framerate (from 250 frame / 10 sec window size)


def render_preview() -> None:
    """
    Reads the final_summary.csv and renders a grid image of the keyframes
    with sequence and timestamp labels.
    """
    csv_path = MANIFESTS_DIR / "final_summary.csv"
    if not csv_path.exists():
        print(f"[ERROR] Required input file missing: {csv_path}", file=sys.stderr)
        sys.exit(1)
        
    rows = []
    with open(csv_path, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
            
    if not rows:
        print("[INFO] No shots in summary to render.")
        return
        
    print(f"[INFO] Rendering preview for {len(rows)} summary events...")
    
    # Grid layout: max 4 columns
    cols = min(4, len(rows))
    if cols == 0:
        cols = 1
    rows_count = (len(rows) + cols - 1) // cols
    
    cell_w, cell_h = 360, 288
    padding_x, padding_y = 20, 70
    
    grid_w = cols * cell_w + (cols + 1) * padding_x
    grid_h = rows_count * cell_h + (rows_count + 1) * padding_y
    
    grid_img = Image.new("RGB", (grid_w, grid_h), color=(30, 30, 30))
    draw = ImageDraw.Draw(grid_img)
    
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except IOError:
        font = ImageFont.load_default()
        
    for i, shot in enumerate(rows):
        col = i % cols
        row_idx = i // cols
        
        view = shot["view"]
        frame_idx = int(shot["keyframe_frame_idx"])
        seq_order = shot["sequence_order"]
        start_f = int(shot["window_start_frame"])
        end_f = int(shot["window_end_frame"])
        
        start_sec = start_f / FPS
        end_sec = end_f / FPS
        
        img_filename = f"frame_{frame_idx:04d}.jpg"
        img_path = DATA_DIR / view / img_filename
        
        x = padding_x + col * (cell_w + padding_x)
        y = padding_y + row_idx * (cell_h + padding_y)
        
        if not img_path.exists():
            print(f"[ERROR] Missing frame image: {img_path}")
            draw.rectangle([x, y, x + cell_w, y + cell_h], outline=(255, 50, 50), width=2)
            draw.text((x + 10, y + cell_h // 2), "IMAGE NOT FOUND", fill=(255, 50, 50), font=font)
        else:
            try:
                img = Image.open(img_path)
                img = img.resize((cell_w, cell_h))
                grid_img.paste(img, (x, y))
                img.close()
            except Exception as e:
                print(f"[ERROR] Failed to load {img_path}: {e}")
                draw.rectangle([x, y, x + cell_w, y + cell_h], outline=(255, 50, 50), width=2)
                draw.text((x + 10, y + cell_h // 2), "LOAD ERROR", fill=(255, 50, 50), font=font)
                
        # Draw labels underneath
        label1 = f"Seq: {seq_order} | {view} | keyframe: {frame_idx}"
        label2 = f"Frames {start_f}-{end_f} ({start_sec:.1f}s - {end_sec:.1f}s)"
        
        draw.text((x, y + cell_h + 10), label1, fill=(220, 220, 220), font=font)
        draw.text((x, y + cell_h + 30), label2, fill=(180, 180, 180), font=font)
        
    out_path = MANIFESTS_DIR / "summary_preview.jpg"
    grid_img.save(out_path, quality=90)
    print(f"[DONE] Saved summary preview grid to {out_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()
    
    render_preview()


if __name__ == "__main__":
    main()
