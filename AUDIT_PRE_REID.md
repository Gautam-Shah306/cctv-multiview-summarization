# Audit of Codebase Changes (Pre-ReID / Stage 1a Baseline)

**Project:** Multi-View CCTV Video Summarization  
**Scope:** Audit of all file additions, modifications, architectural optimizations, and git commits made from initial project setup through the completion of Stage 1a (Frame Alignment & YOLO Object Detection Pipeline) before Stage 1b ReID.

---

## 1. Executive Summary & Evolution Timeline

```
Phase 1: Environment & Multi-View Sync ──▶ Phase 2: YOLOv8 Baseline Pipeline ──▶ Phase 3: YOLO26 Upgrade & I/O Fixes
(Commits 2bda1d4 → eea5ce1)               (Commits 9e78052 → 9d4b09e)            (Commits 6add445 → 1117883)
• Multi-user .env & config.py             • detect_objects.py (YOLOv8m)          • Switch model to yolo26m.pt
• extract_frames.py (frame offsets)       • Google Colab GPU execution           • Local frame caching (~1000x I/O speedup)
• probe_videos.py & data_manifests/       • Cross-device move fix                • Detections regenerated across all views
```

---

## 2. Comprehensive File-by-File Audit

### A. Environment, Configuration & Data Management

| File | Type | Key Changes & Description |
| :--- | :--- | :--- |
| [`.env.example`](file:///.env.example) | Added | Template environment file specifying the `DRIVE_ROOT` path for collaborator onboarding without committing sensitive or machine-specific paths. |
| [`.gitignore`](file:///.gitignore) | Modified | Configured to strictly ignore machine-specific `.env`, virtual environments (`venv/`, `cctv-mvs/`), checkpoints, and bulky data directories (`data/`, `frames/`, `*.avi`, `*.mp4`). |
| [`requirements.txt`](file:///requirements.txt) | Added | Dependency snapshot for local CPU/GPU development: `opencv-python`, `matplotlib`, `numpy`, `pandas`, `python-dotenv`, `pillow`, `kiwisolver`, `fonttools`. |
| [`requirements-colab.txt`](file:///requirements-colab.txt) | Modified | Cloud GPU dependency snapshot: initialized with `ultralytics>=8.2`, later bumped to `ultralytics>=8.4.0` for YOLO26 support. |
| [`config.py`](file:///config.py) | Modified | • Centralized paths (`RAW_VIDEOS`, `DATA_DIR`, `LOGS_DIR`, `CHECKPOINTS`) derived dynamically from `DRIVE_ROOT`.<br>• Defined detection parameters: `DETECTION_MODEL = "yolo26m.pt"` (upgraded from `yolov8m.pt`), `DETECTION_CONFIDENCE = 0.2`, `DETECTION_IMG_SIZE = 640`, `DETECTION_BATCH_SIZE = 32`, `DETECTION_HALF = True`. |

---

### B. Video Inspection & Temporal Multi-View Alignment (Phase 1)

| File | Type | Key Changes & Description |
| :--- | :--- | :--- |
| [`src/probe_videos.py`](file:///src/probe_videos.py) | Added | Read-only inspection script utilizing `ffprobe` to verify video duration, resolution (`360x288`), FPS (`25.00`), frame counts, and codec (`mpeg4`). |
| [`src/extract_frames.py`](file:///src/extract_frames.py) | Added | Extracts synchronized frames using ffmpeg `select` filter with calibrated frame offsets:<br>• `view2`: 0 frames (reference)<br>• `view3`: +1 frame offset<br>• `view1`: +3 frames offset<br>Ensures frame index $N$ corresponds to the exact same physical instant across all cameras. |
| [`data_manifests/video_inspection.csv`](file:///data_manifests/video_inspection.csv) | Added | Repo-tracked manifest recording raw video metadata for `view1.avi`, `view2.avi`, and `view3.avi`. |
| [`data_manifests/metadata.csv`](file:///data_manifests/metadata.csv) | Added | Repo-tracked manifest documenting aligned frame counts (`view1: 3912`, `view2: 3915`, `view3: 3914`), FPS, and time offsets. |

---

### C. Stage 1a Object Detection & YOLO Optimizations (Phase 2 & Phase 3)

| File | Type | Key Changes & Description |
| :--- | :--- | :--- |
| [`src/detect_objects.py`](file:///src/detect_objects.py) | Added / Modified | **1. Core Detection Engine:** Pretrained YOLO detector filtered for person class (`class_id = 0`), outputting to `DATA_DIR/detections/<view>.csv`.<br>**2. Cross-Device Checkpoint Fix:** Replaced `Path.rename` with `shutil.move` to fix `OSError: [Errno 18] Invalid cross-device link` when downloading weights in Google Drive mounts.<br>**3. Progress & Device Logging:** Added device identification (`cuda` / `cpu`) and interval logging every 200 frames.<br>**4. Local Frame Caching (Critical I/O Optimization):** Staged frames from Drive into local fast storage (`tempfile.gettempdir() / "frame_cache"`), reducing per-frame read latency from $\sim450\text{ ms} \to \sim0.4\text{ ms}$ (over $1000\times$ speedup).<br>**5. YOLO26 & Batching Upgrade:** Upgraded model architecture to YOLO26 (`yolo26m.pt`) with mini-batch parallel GPU inference and FP16 half-precision. |
| [`src/visualize_detections.py`](file:///src/visualize_detections.py) | Added | Inspection and visualization tool providing:<br>• Synchronized multi-view frame grid comparison (`--multiview`)<br>• Single-view sample detection grid with bounding boxes (`--view`, `--step`)<br>• Annotated MP4 video clip rendering (`--video`). |
| [`data_manifests/detections_summary.csv`](file:///data_manifests/detections_summary.csv) | Added / Modified | Lightweight manifest tracking processed frame counts, model version (`yolo26m.pt`), and confidence thresholds for `view1`, `view2`, and `view3`. |

---

### D. Notebooks & Documentation

| File | Type | Key Changes & Description |
| :--- | :--- | :--- |
| [`notebooks/colab_detect_objects.ipynb`](file:///notebooks/colab_detect_objects.ipynb) | Added / Modified | Interactive Google Colab notebook for GPU execution, repository synchronization, Drive mounting, I/O speed benchmarking, and visual verification. |
| [`notebooks/colab_detect_objects.py`](file:///notebooks/colab_detect_objects.py) | Added | Script counterpart of the Colab notebook for headless or scripted execution. |
| [`CODE_STYLE.md`](file:///CODE_STYLE.md) | Added | Comprehensive 700+ line repository guide documenting coding style, imports, PEP 8 standards, naming conventions, and the bulky-vs-lightweight storage separation architecture. |

---

## 3. Chronological Git Commit Log

| Commit | Date | Author | Commit Message & Summary |
| :--- | :--- | :--- | :--- |
| `2bda1d4` | 2026-07-16 | Shah Gautam Sameerbhai | `Initial commit` — Base repo and README.md. |
| `d471457` | 2026-07-16 | Gautam Shah | `Initial environment setup: Python, ffmpeg, opencv, requirements.txt` — Base `.gitignore` and requirements. |
| `fb7e646` | 2026-07-17 | Gautam Shah | `Populate requirements.txt` — Added standard vision/data dependencies. |
| `6f28fbc` | 2026-07-17 | Gautam Shah | `Add .env-based path config for multi-user setup` — Added `config.py` and `.env.example`. |
| `4b407ce` | 2026-07-18 | Gautam Shah | `Add metadata.csv to repo-tracked data_manifests/; decouple extraction from metadata generation` — Added `extract_frames.py`. |
| `9ee6502` | 2026-07-18 | Shah Gautam Sameerbhai | `Merge pull request #1 from Gautam-Shah306/feature/frame-extraction-alignment` |
| `eea5ce1` | 2026-07-18 | Gautam Shah | `Add formal ffprobe inspection log.` — Added `src/probe_videos.py` and `video_inspection.csv`. |
| `9e78052` | 2026-08-19 | Gautam Shah | `Add Stage 1 object detection pipeline` — Added `src/detect_objects.py`, `CODE_STYLE.md`, and Colab notebook. |
| `bd7c359` | 2026-08-19 | Gautam Shah | `Fix Colab model checkpoint handling` — Replaced rename with `shutil.move` for cross-device links. |
| `1117446` | 2026-08-19 | Gautam Shah | `Add device and per-view progress logging to detect_objects.py` — Logged CUDA device and interval counts. |
| `2ae521b` | 2026-08-19 | Gautam Shah | `Fix manifest overwrite and skipped-view frame count in detect_objects.py` |
| `9d4b09e` | 2026-08-19 | Gautam Shah | `Add detection manifest for view1 and view2` — Initial detection records. |
| `cb7f7af` | 2026-08-19 | Gautam Shah | `Update Colab notebook with detection visualization and inspection code` |
| `677a926` | 2026-08-19 | Gautam Shah | `Update to certain key factors` — Cleaned credentials from notebook. |
| `6add445` | 2026-08-21 | Gautam Shah | `Switch detection model to YOLO26` — Upgraded config to `yolo26m.pt` and `ultralytics>=8.4.0`. |
| `dc4137d` | 2026-08-23 | Gautam Shah | `upgraded the YOLO version` — Implemented local disk caching for small frame reads. |
| `d5e9bdf` | 2026-08-23 | Gautam Shah | `Regenerate detections under YOLO26 for view1` |
| `1117883` | 2026-08-23 | Gautam Shah | `Regenerate detections under YOLO26 for view2 and view3` |
