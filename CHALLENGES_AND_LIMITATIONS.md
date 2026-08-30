# Engineering Challenges, Limitations & Key Bottlenecks (Stage 1)

**Project:** Multi-View CCTV Video Summarization  
**Purpose:** Presentation slides content summarizing technical challenges, bottlenecks encountered, and how they were solved.

---

## 1. Executive Summary: Problem vs. Solution Matrix

| Category | Technical Challenge Encountered | Root Cause / Impact | Implemented Solution |
| :--- | :--- | :--- | :--- |
| **I/O & Cloud Storage** | Severe read latency (~450 ms/frame) across >11,700 frames | FUSE mount network round-trips on Google Drive / cloud storage | **Local Fast Frame Staging (`tempfile`)** dropping latency to ~0.4 ms/frame (>1000× speedup). |
| **Tracking State Loss** | ByteTrack lost object trajectories (`track_id = -1`) | Passing mini-batches of frames into tracker broke temporal continuity and Kalman filter state | **Sequential Frame Feeding Architecture** for tracking mode while retaining batch inference for detection. |
| **Compute Hardware** | High inference time on CPU (~5–6 hours for full multi-view sequence) | High-resolution YOLO forward passes + 33,000+ ReID crop extractions | **GPU Acceleration (Colab/CUDA)** with FP16 Half-Precision reducing execution to ~10–15 minutes. |
| **Dependency & Assigner** | Tracking crash due to missing Linear Assignment Problem solver | ByteTrack bipartite matching requires native C-extensions (`lap`) | Automated installation & dependency locking (`lap>=0.5.12` in Colab requirements). |
| **Multi-Camera Alignment** | Disconnected physical timeline between cameras | Frame index $N$ in View 1 did not correspond to physical time in View 2/3 | **Calibrated Multi-View Frame Offset Mapping** (View 2 as base, View 3: +1, View 1: +3). |
| **Data Separation** | Git repo bloat from bulky video/embedding artifacts | Storing multi-gigabyte video frames and numpy tensors inside Git | **Strict Gitignore + Lightweight Manifest Design** (`data_manifests/` CSV summaries vs `data_storage/`). |

---

## 2. Detailed Challenges & Limitations (Slide-by-Slide Content)

### Slide 1: Cloud Storage & I/O Bottleneck
* **Problem:** Direct frame reading from network-mounted drives (Google Drive / OneDrive via FUSE) took ~450ms per JPEG. Across 3 views and ~11,700 frames, pure disk I/O alone took >1.5 hours before neural inference even started.
* **Limitation:** Cloud-mounted filesystems have high metadata lookup overhead for thousands of small individual image files.
* **Resolution:** Implemented local disk caching into VM temporary NVMe/SSD storage (`tempfile.gettempdir() / "frame_cache"`), reducing per-frame read time to **~0.4ms** (>1000× speedup).

---

### Slide 2: The Multi-Object Tracking State Loss (`track_id = -1`)
* **Problem:** Initial detection runs yielded valid bounding boxes but failed to track identities across time (`track_id = -1`).
* **Root Cause:** To maximize GPU throughput, frames were passed in parallel mini-batches (`batch_size=32`). However, ByteTrack's Kalman filter and trajectory association algorithm require **strictly sequential, frame-by-frame temporal continuity**.
* **Resolution:** Decoupled the architecture:
  * **Detection Mode:** Uses parallel batched tensor inference.
  * **Tracking Mode:** Uses sequential frame-by-frame processing with `persist=True` to preserve trajectory history.

---

### Slide 3: Hardware Limitations (CPU vs. GPU Compute)
* **Problem:** Running YOLO26m and OSNet person feature extraction locally on CPU took ~5–6 hours for the full dataset (33,145 bounding box crops across 3 cameras).
* **Limitation:** Deep neural network convolution layers and multi-scale attention mechanisms are highly memory-bandwidth and FLOPs bound on standard CPUs.
* **Resolution:** Transitioned the pipeline to NVIDIA GPU execution with CUDA and FP16 half-precision, turning multi-hour runs into **~10–15 minutes**.

---

### Slide 4: CCTV Video Artifacts & Low Resolution
* **Problem:** Surveillance cameras provide native low-resolution footage ($360 \times 288$) with heavy compression, lens distortion, and motion blur.
* **Limitation:** Pedestrians in the background occupy very few pixels ($<20 \times 40$ px), causing standard detectors to miss small subjects or produce noisy ReID embeddings.
* **Resolution:**
  * Upgraded detector backbone from standard YOLOv8 to **YOLO26m** with multi-scale feature pyramids.
  * Integrated **OSNet (Omni-Scale Network)** which captures feature scales from fine-grained details to global spatial context.

---

### Slide 5: Temporal Multi-View Synchronization
* **Problem:** Multi-view cameras recorded asynchronously, meaning camera timestamps had small drifts and offsets.
* **Limitation:** Naive frame index matching ($t=100$ in View 1 vs $t=100$ in View 2) caused people to appear in conflicting spatial positions across views.
* **Resolution:** Performed calibrated frame-level sync based on visual cues and duration probing (`probe_videos.py` + calibrated offsets in `extract_frames.py`).

---

## 3. Key Takeaways & Recommendations for Future Stages

1. **Tracklet-Level Pooling for ReID:** Instead of relying on noisy single-frame bounding box crops, aggregate and average ReID embeddings across entire ByteTrack trajectories.
2. **Dynamic Motion Gating:** Skip neural network forward passes during inactive/static surveillance periods using frame differencing to save significant compute.
3. **Graph-Based Cross-Camera Clustering (Stage 2):** Use the extracted 512-d L2-normalized embeddings with spatial-temporal constraints to associate identities across non-overlapping camera fields of view.
