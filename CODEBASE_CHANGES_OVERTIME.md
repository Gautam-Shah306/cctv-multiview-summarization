# Codebase & Architecture Evolution (Post-Clone to Current)

**Project:** Multi-View CCTV Video Summarization  
**Purpose:** Comprehensive chronology of all code modifications, architecture design decisions, structural changes, and bug fixes made from project setup to current state for presentation & documentation.

---

## 1. High-Level Evolution Timeline

```
Phase 1: Environment, Multi-View Sync & Data Design
(Commits 2bda1d4 → eea5ce1)
 ├── Multi-user .env configuration
 ├── Video inspection (probe_videos.py)
 └── Calibrated frame extraction (extract_frames.py)
         │
         ▼
Phase 2: Stage 1a Baseline Object Detection
(Commits 9e78052 → 9d4b09e)
 ├── YOLOv8m baseline pipeline
 ├── Cloud Drive cross-device move fix (shutil.move)
 └── Initial detection manifests
         │
         ▼
Phase 3: Stage 1a Optimizations & Architecture Upgrade
(Commits 6add445 → 1117883)
 ├── Upgrade backbone to YOLO26m
 ├── Local fast disk staging (~1000x I/O speedup)
 └── Motion gating (background frame delta skipping)
         │
         ▼
Phase 4: Stage 1b ReID, Tracking Fix & GPU Pipeline
(Commits dd4c8a1 → 9c70e16)
 ├── ByteTrack sequential tracking loop fix (persistent track_ids)
 ├── Omni-Scale Network (OSNet) 512-d ReID embedding model
 ├── Cross-camera retrieval & similarity matrix visualizers
 └── Colab GPU execution pipeline script & documentation
```

---

## 2. Directory Structure: Initial Baseline vs. Current

### Initial State (After Initial Clone):
```
cctv-multiview-summarization/
├── .gitignore
├── README.md
└── requirements.txt
```

### Current Production State:
```
cctv-multiview-summarization/
├── .env.example                      # Template for multi-user path config
├── config.py                         # Centralized hyperparameters & paths
├── requirements.txt                  # Local dependencies
├── requirements-colab.txt            # Cloud GPU dependencies (ultralytics, lap)
│
├── data_manifests/                   # [GIT-TRACKED] Lightweight metadata & summaries
│   ├── metadata.csv                  # Frame counts, FPS, calibrated camera offsets
│   ├── video_inspection.csv          # Raw ffprobe stream diagnostics
│   ├── detections_summary.csv        # Summary of Stage 1a detection runs
│   └── reid_summary.csv              # Summary of Stage 1b ReID embeddings
│
├── notebooks/                        # Colab GPU execution notebooks & scripts
│   ├── colab_detect_objects.py       # Headless detection script
│   ├── colab_extract_reid.py         # Headless ReID extraction script
│   └── colab_stage1_gpu.py           # End-to-end GPU execution pipeline
│
├── src/                              # Production Source Code
│   ├── __init__.py
│   ├── config.py -> ../config.py
│   ├── probe_videos.py               # Video inspection & ffprobe verification
│   ├── extract_frames.py             # Temporal multi-view synchronization
│   ├── detect_objects.py             # YOLO26 + ByteTrack (sequential & batch)
│   ├── extract_reid_features.py      # OSNet 512-d person ReID feature extractor
│   ├── visualize_detections.py       # Detection bounding boxes & MP4 video exporter
│   ├── visualize_reid.py             # Cross-camera retrieval & similarity heatmaps
│   └── models/                       # Deep Learning Model Architectures
│       ├── __init__.py
│       └── reid_model.py             # OSNet architecture & weights downloader
│
├── data_storage/                     # [GITIGNORED] Bulky data storage (Local / Drive)
│   ├── raw_videos/                   # Original view1.avi, view2.avi, view3.avi
│   ├── checkpoints/                  # yolo26m.pt, osnet_x1_0_market1501.pth
│   └── data/                         # Extracted frames, detections CSVs, npz embeddings
│
└── Documentation Files:
    ├── CODE_STYLE.md                 # Architecture & repository standards
    ├── STAGE1A_OBJECT_DETECTION.md   # Stage 1a technical deep-dive
    ├── AUDIT_PRE_REID.md             # Commit audit prior to ReID
    └── CHALLENGES_AND_LIMITATIONS.md # Technical challenges & solutions for PPT
```

---

## 3. Detailed Component-by-Component Code Changes

### A. Data Architecture & Multi-User Configuration
* **Added [`.env.example`](file:///.env.example) & [`config.py`](file:///config.py):**
  * Solved machine-specific hardcoded paths by deriving all bulky directories (`DATA_DIR`, `RAW_VIDEOS`, `CHECKPOINTS`, `LOGS_DIR`) from `DRIVE_ROOT`.
  * Allows seamless switching between Google Colab (`/content/drive/...`) and Local Development without modifying code.
* **Separation of Bulky vs. Lightweight Data:**
  * Strict `.gitignore` policy keeping massive frame folders and `.npz` arrays out of Git history.
  * Created `data_manifests/` storing lightweight audit CSVs (`metadata.csv`, `detections_summary.csv`, `reid_summary.csv`).

---

### B. Video Ingestion & Multi-Camera Temporal Alignment (Phase 1)
* **Added [`src/probe_videos.py`](file:///src/probe_videos.py):**
  * Uses `ffprobe` to verify codec (`mpeg4`), resolution (`360x288`), and frame rate (`25.00 fps`).
* **Added [`src/extract_frames.py`](file:///src/extract_frames.py):**
  * Solved inter-camera physical timeline drift using calibrated frame extraction offsets:
    * `view2`: Reference base (0 offset)
    * `view3`: +1 frame offset
    * `view1`: +3 frame offset
  * Ensures frame index $t$ corresponds to the identical physical instant across all cameras.

---

### C. Stage 1a: Detection, Tracking & Architectural Optimizations (Phase 2 & 3)
* **Backbone Upgrade:**
  * Upgraded from standard `yolov8m.pt` to **`yolo26m.pt`** with multi-scale feature pyramids for detecting small/distant pedestrians.
* **Local Fast Caching (1000× Speedup):**
  * Added `tempfile.gettempdir() / "frame_cache"` staging. Replaced direct Google Drive reads (~450ms/frame) with local NVMe caching (~0.4ms/frame).
* **Cross-Device Checkpoint Fix:**
  * Replaced `Path.rename` with `shutil.move` in model loaders to prevent `OSError: [Errno 18] Invalid cross-device link` on Google Drive FUSE mounts.
* **Motion Gating:**
  * Added `check_motion()` background frame differencing. Automatically skips deep neural passes during static surveillance periods (<500 pixel delta).

---

### D. ByteTrack Sequential Association Fix (Phase 4)
* **Problem in Earlier Version:**
  * Passing batches of 32 frame paths to `model.track()` broke temporal continuity, resulting in `track_id = -1`.
* **Code Refactoring in [`src/detect_objects.py`](file:///src/detect_objects.py):**
  * Split execution into two dedicated pipelines:
    1. **Tracking Mode (`--track`):** Feeds frames strictly one-by-one in sequential order with `persist=True`, preserving Kalman filter trajectories across time.
    2. **Detection-Only Mode:** Retains parallel batched inference (`batch_size=32`) for maximum throughput.
  * Added automated `lap` linear assignment solver dependency management.

---

### E. Stage 1b: Person Re-Identification (ReID) Pipeline (Phase 4)
* **Added [`src/models/reid_model.py`](file:///src/models/reid_model.py):**
  * Implemented the **Omni-Scale Network (OSNet)** architecture with multi-scale residual streams and dynamic channel gating.
  * Added automatic checkpoint downloaders and state dict loaders for Market-1501 weights.
* **Added [`src/extract_reid_features.py`](file:///src/extract_reid_features.py):**
  * Crops detected person bounding boxes, resizes to $(256, 128)$, applies ImageNet standardization, and extracts **512-dimensional L2-normalized visual embeddings** ($\|f\|_2 = 1.0$).
  * Saves per-view embedding archives to `DATA_DIR/embeddings/<view>_embeddings.npz`.
* **Added [`src/visualize_reid.py`](file:///src/visualize_reid.py):**
  * Visualizes cross-camera retrieval (query in View 1 $\to$ top-5 cosine similarity matches in View 2/3).
  * Generates cosine similarity heatmaps to evaluate intra-camera and cross-camera embedding consistency.

---

### F. Google Colab GPU Pipeline
* **Added [`notebooks/colab_stage1_gpu.py`](file:///notebooks/colab_stage1_gpu.py):**
  * Complete turnkey GPU script for running tracking, ReID extraction, and verification plots end-to-end on Colab T4 GPU in **~10–15 minutes**.
