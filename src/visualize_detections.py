"""
visualize_detections.py

Plots sample frames or video clips with detection bounding boxes,
track IDs, and confidence scores overlaid.

Usage:
    # 1. Grid of sampled frames for a view
    python -m src.visualize_detections --view view1 --out detection_grid_view1.png

    # 2. Multi-view synchronized comparison (view1, view2, view3 at same instants)
    python -m src.visualize_detections --multiview --out multiview_comparison.png

    # 3. Generate annotated MP4 video clip with bounding boxes & track IDs
    python -m src.visualize_detections --view view1 --video --out-video view1_tracked.mp4 --start-frame 200 --num-frames 300
"""

import argparse
import csv
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from config import DATA_DIR

# Distinct color palette for track IDs
COLORS = [
    (0, 255, 127),   # Spring Green
    (255, 69, 0),    # Orange Red
    (30, 144, 255),  # Dodger Blue
    (255, 215, 0),   # Gold
    (186, 85, 211),  # Medium Orchid
    (0, 255, 255),   # Cyan
    (255, 105, 180), # Hot Pink
    (50, 205, 50),   # Lime Green
    (255, 140, 0),   # Dark Orange
    (138, 43, 226),  # Blue Violet
]


def get_track_color(track_id: int) -> tuple[int, int, int]:
    """Return a consistent (R, G, B) color for a given track ID."""
    if track_id < 0:
        return (0, 255, 0)  # Default green for untracked detections
    return COLORS[track_id % len(COLORS)]


def load_detections(view: str) -> dict[int, list[dict]]:
    """Load detections for a view from CSV, keyed by frame_idx."""
    detections_csv = DATA_DIR / "detections" / f"{view}.csv"
    if not detections_csv.exists():
        print(f"[ERROR] Detections file not found: {detections_csv}")
        return {}

    detections_by_frame = {}
    with open(detections_csv) as f:
        for row in csv.DictReader(f):
            idx = int(row["frame_idx"])
            track_id = int(row.get("track_id", -1))
            box = (
                int(float(row["x1"])),
                int(float(row["y1"])),
                int(float(row["x2"])),
                int(float(row["y2"])),
            )
            conf = float(row.get("confidence", 0.0))
            detections_by_frame.setdefault(idx, []).append({
                "box": box,
                "track_id": track_id,
                "conf": conf,
            })
    return detections_by_frame


def draw_box(frame: np.ndarray, det: dict, rgb: bool = True) -> None:
    """Draw a bounding box with track ID and confidence badge on frame."""
    x1, y1, x2, y2 = det["box"]
    track_id = det["track_id"]
    conf = det["conf"]
    color = get_track_color(track_id)
    # If drawing in BGR for OpenCV video, swap color order
    cv_color = color if rgb else (color[2], color[1], color[0])

    # Bounding rectangle
    cv2.rectangle(frame, (x1, y1), (x2, y2), cv_color, 2)

    # Label text: e.g. "ID:1 | 0.85" or "0.85"
    label = f"ID:{track_id} {conf:.2f}" if track_id >= 0 else f"{conf:.2f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.4
    thick = 1
    (tw, th), baseline = cv2.getTextSize(label, font, scale, thick)

    # Draw label background badge
    ty1 = max(0, y1 - th - 4)
    cv2.rectangle(frame, (x1, ty1), (x1 + tw + 4, ty1 + th + 4), cv_color, -1)
    cv2.putText(frame, label, (x1 + 2, ty1 + th + 1), font, scale, (0, 0, 0), thick, cv2.LINE_AA)


def generate_grid(
    view: str = "view1",
    step: int = 100,
    rows: int = 4,
    cols: int = 5,
    out_path: str | None = "detection_grid.png",
    show: bool = False,
):
    """Plot a grid of sampled frames with bounding boxes and track IDs."""
    frames_dir = DATA_DIR / view
    detections_by_frame = load_detections(view)
    if not detections_by_frame:
        return

    all_indices = sorted(
        int(p.stem.split("_")[1]) for p in frames_dir.glob("frame_*.jpg")
    )[::step][: rows * cols]

    if not all_indices:
        print(f"[ERROR] No frames found in {frames_dir}")
        return

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axes = axes.flatten() if rows * cols > 1 else [axes]

    for ax, idx in zip(axes, all_indices):
        frame_file = frames_dir / f"frame_{idx:04d}.jpg"
        frame = cv2.imread(str(frame_file))
        if frame is None:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        dets = detections_by_frame.get(idx, [])
        for det in dets:
            draw_box(frame, det, rgb=True)

        ax.imshow(frame)
        ax.set_title(f"frame_{idx:04d} ({len(dets)} det)", fontsize=9)
        ax.axis("off")

    for ax in axes[len(all_indices):]:
        ax.axis("off")

    plt.tight_layout()

    if out_path:
        out_file = Path(out_path)
        plt.savefig(out_file, bbox_inches="tight", dpi=150)
        print(f"[DONE] Saved visualization grid to {out_file.resolve()}")

    if show:
        plt.show()
    plt.close()


def generate_multiview_grid(
    step: int = 200,
    samples: int = 5,
    out_path: str = "multiview_comparison.png",
):
    """Plot synchronized frames across view1, view2, view3 in rows."""
    views = ["view1", "view2", "view3"]
    detections = {v: load_detections(v) for v in views}

    frames_dirs = {v: DATA_DIR / v for v in views}
    ref_indices = sorted(
        int(p.stem.split("_")[1]) for p in frames_dirs["view1"].glob("frame_*.jpg")
    )[::step][:samples]

    fig, axes = plt.subplots(len(views), samples, figsize=(samples * 4, len(views) * 3))

    for r, v in enumerate(views):
        dets_map = detections[v]
        for c, idx in enumerate(ref_indices):
            ax = axes[r, c] if len(views) > 1 else axes[c]
            frame_file = frames_dirs[v] / f"frame_{idx:04d}.jpg"
            frame = cv2.imread(str(frame_file))
            if frame is not None:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                dets = dets_map.get(idx, [])
                for det in dets:
                    draw_box(frame, det, rgb=True)
                ax.imshow(frame)
                ax.set_title(f"{v} | frame_{idx:04d} ({len(dets)} det)", fontsize=8)
            ax.axis("off")

    plt.tight_layout()
    out_file = Path(out_path)
    plt.savefig(out_file, bbox_inches="tight", dpi=150)
    print(f"[DONE] Saved multi-view comparison grid to {out_file.resolve()}")
    plt.close()


def export_video(
    view: str = "view1",
    start_frame: int = 0,
    num_frames: int = 300,
    fps: float = 25.0,
    out_video: str = "annotated_detection.mp4",
):
    """Render an annotated MP4 video with bounding boxes and track IDs."""
    frames_dir = DATA_DIR / view
    detections_by_frame = load_detections(view)
    if not detections_by_frame:
        return

    frame_paths = sorted(frames_dir.glob("frame_*.jpg"))
    if not frame_paths:
        print(f"[ERROR] No frames found in {frames_dir}")
        return

    # Filter frame range
    selected_paths = [
        p for p in frame_paths
        if start_frame <= int(p.stem.split("_")[1]) < (start_frame + num_frames)
    ]

    if not selected_paths:
        print(f"[ERROR] No frames in range [{start_frame}, {start_frame + num_frames})")
        return

    sample_img = cv2.imread(str(selected_paths[0]))
    h, w, _ = sample_img.shape

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(out_video, fourcc, fps, (w, h))

    print(f"[INFO] Rendering video: {len(selected_paths)} frames -> {out_video}")
    for p in selected_paths:
        frame = cv2.imread(str(p))
        if frame is None:
            continue
        idx = int(p.stem.split("_")[1])
        dets = detections_by_frame.get(idx, [])
        for det in dets:
            draw_box(frame, det, rgb=False)

        cv2.putText(
            frame,
            f"{view} | frame_{idx:04d} | det:{len(dets)}",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
        out_writer.write(frame)

    out_writer.release()
    print(f"[DONE] Rendered video successfully saved to {Path(out_video).resolve()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view", default="view1", help="Camera view to visualize (default: view1)")
    parser.add_argument("--step", type=int, default=100, help="Frame step interval (default: 100)")
    parser.add_argument("--rows", type=int, default=4, help="Grid rows (default: 4)")
    parser.add_argument("--cols", type=int, default=5, help="Grid columns (default: 5)")
    parser.add_argument("--out", default="detection_grid.png", help="Output image path (default: detection_grid.png)")
    parser.add_argument("--show", action="store_true", help="Display GUI window with matplotlib")
    parser.add_argument("--multiview", action="store_true", help="Generate multi-view synchronized comparison grid")
    parser.add_argument("--video", action="store_true", help="Export an annotated MP4 video clip")
    parser.add_argument("--out-video", default="annotated_detection.mp4", help="Output video path (default: annotated_detection.mp4)")
    parser.add_argument("--start-frame", type=int, default=0, help="Start frame for video rendering (default: 0)")
    parser.add_argument("--num-frames", type=int, default=300, help="Number of frames to render into video (default: 300)")
    args = parser.parse_args()

    if args.multiview:
        generate_multiview_grid(step=args.step, samples=args.cols, out_path=args.out)
    elif args.video:
        export_video(
            view=args.view,
            start_frame=args.start_frame,
            num_frames=args.num_frames,
            out_video=args.out_video,
        )
    else:
        generate_grid(
            view=args.view,
            step=args.step,
            rows=args.rows,
            cols=args.cols,
            out_path=args.out,
            show=args.show,
        )


if __name__ == "__main__":
    main()

