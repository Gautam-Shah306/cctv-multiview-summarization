# CCTV Multi-View Summarization Pipeline: Technical Inventory

This document provides a comprehensive, code-grounded technical inventory of the entire multi-view summarization pipeline, reflecting the **actual current state** of the codebase in `src/`.

## Pipeline Overview: Canonical vs Experimental
- **Canonical / Production Pipeline (Rule-Based, 250-frame windows)**:
  - Uses `KEYFRAME_SHOT_WINDOW_SEC = 10.0` (250 frames at 25 FPS) by default.
  - Relies on heuristics, camera calibration (VQ scores), and hard thresholds to merge tracklets and build cross-view events.
  - Final output is typically `data_manifests/final_summary_v2.csv`.
- **Experimental Pipeline (Graph Transformer, 150-frame sparse windows)**:
  - Uses a shorter 150-frame window (via `_w150` suffix).
  - Replaces the rule-based event fusion with a learned PyTorch Geometric Graph Attention Network (GAT) using sparsely sampled representations.
  - Generates `training_pairs_w150.csv` and outputs `final_summary_graph_transformer.csv`.

---

## 1. `src/extract_frames.py`
**Purpose:** Extracts frames from the 3 raw CCTV videos, applying manual per-view frame offsets so that frame index `N` corresponds to the exact same real-world instant across all views.
**Exact Inputs:** 
- `DRIVE_ROOT/raw_videos/view1.avi`
- `DRIVE_ROOT/raw_videos/view2.avi`
- `DRIVE_ROOT/raw_videos/view3.avi`
**Exact Outputs:** 
- `DRIVE_ROOT/data/<view_name>/frame_%04d.jpg`
- `data_manifests/metadata.csv`
**Key Parameters/Constants:**
- `FPS = 25.00`
- `view1` offset: `+3` frames
- `view2` offset: `0` frames (reference)
- `view3` offset: `+1` frame
**Core Algorithm:** Uses `ffmpeg` with the `select='gte(n\,{offset})'` video filter to skip the specified number of leading frames before extracting at 25 fps.
**Deviations Noted:** Offsets were manually determined rather than auto-synced (as per project status).

## 2. `src/detect_objects.py`
**Purpose:** Runs per-view person detection and tracking over the extracted, synced frame sequences using a pretrained YOLOv8 model and ByteTrack.
**Exact Inputs:**
- Extracted frames: `DRIVE_ROOT/data/<view_name>/frame_*.jpg`
- Pretrained weights: `yolo26m.pt` (cached to `DRIVE_ROOT/checkpoints/`)
**Exact Outputs:**
- `DRIVE_ROOT/data/detections/<view_name>.csv`
- `data_manifests/detections_summary.csv`
**Key Parameters/Constants:**
- `PERSON_CLASS_ID = 0` (COCO "person")
- `DETECTION_MODEL = "yolo26m.pt"`
- `DETECTION_CONFIDENCE = 0.2`
- `DETECTION_IMG_SIZE = 640`
- `DETECTION_BATCH_SIZE = 32`
- `MOTION_GATING_THRESHOLD = 500`
**Core Algorithm:** Runs YOLOv8 inference. If tracking is enabled, applies ByteTrack to associate boxes across frames. If motion gating is enabled, compares grayscale frames using `cv2.absdiff(curr, prev_gray)`; if the number of pixels with absolute difference > 25 is less than or equal to `MOTION_GATING_THRESHOLD` (500), the frame is skipped.
**Deviations Noted:** Implements a fast sequential ByteTrack mode that can run on precomputed batched detections if a GPU is unavailable for full tracker execution.

## 3. `src/extract_reid_features.py`
**Purpose:** Extracts deep Person Re-Identification (ReID) visual feature embeddings for all person bounding boxes detected in the previous stage using a pretrained OSNet.
**Exact Inputs:**
- Detections: `DRIVE_ROOT/data/detections/<view_name>.csv`
- Frames: `DRIVE_ROOT/data/<view_name>/frame_*.jpg`
**Exact Outputs:**
- `DRIVE_ROOT/data/embeddings/<view_name>_embeddings.npz` (contains `embeddings`, `frame_indices`, `track_ids`, `boxes`, `confidences`)
- `data_manifests/reid_summary.csv`
**Key Parameters/Constants:**
- `REID_MODEL = "osnet_x1_0"`
- `REID_WEIGHTS = "osnet_x1_0_market1501.pth"`
- `REID_EMBEDDING_DIM = 512`
- `REID_IMAGE_SIZE = (256, 128)` (H, W)
- ImageNet Norm: Mean `[0.485, 0.456, 0.406]`, Std `[0.229, 0.224, 0.225]`
- `REID_BATCH_SIZE = 64`
**Core Algorithm:** Crops detected bounding boxes from frames, resizes to 256x128 (bicubic), converts to RGB, and normalizes using ImageNet stats. Runs batched inference on OSNet, returning L2-normalized 512-dimensional feature vectors (`normalize=True` in model call).

## 4. `src/extract_dinov3_features.py`
**Purpose:** Extracts frozen, self-supervised Vision Transformer (DINO) embeddings for every full frame across views to capture global scene layout and background context.
**Exact Inputs:**
- Frames: `DRIVE_ROOT/data/<view_name>/frame_*.jpg`
**Exact Outputs:**
- `DRIVE_ROOT/data/dinov3_embeddings/<view_name>_dinov3.npz` (contains `embeddings`, `frame_indices`)
**Key Parameters/Constants:**
- Model: `facebook/dinov2-small` (ViT-Small/14, 22M parameters, 384-d output)
- `IMAGE_SIZE = (224, 224)`
- `DEFAULT_BATCH_SIZE = 32`
**Core Algorithm:** Resizes full frames to 224x224 (bicubic), applies ImageNet normalization. Feeds into frozen `AutoModel.from_pretrained`. Extracts the CLS token embedding from the last hidden state (`outputs.last_hidden_state[:, 0, :]`) and normalizes it to a unit L2 norm using `torch.nn.functional.normalize(..., p=2, dim=-1)`.
**Deviations Noted:** Despite the file name `dinov3`, the default model used is `facebook/dinov2-small` as it is ungated and lightweight.

## 5. `src/pool_tracklet_features.py`
**Purpose:** Gathers frame-level Person ReID embeddings belonging to each tracking trajectory (tracklet), computes their mean embedding, and re-normalizes the result.
**Exact Inputs:**
- Frame embeddings: `DRIVE_ROOT/data/embeddings/<view>_embeddings.npz`
- Detections: `DRIVE_ROOT/data/detections/<view>.csv`
**Exact Outputs:**
- `data_manifests/tracklet_features.csv`
- `DRIVE_ROOT/data/tracklet_embeddings.npz` (dict keyed by `view_trackid`)
**Key Parameters/Constants:**
- `LOW_CONFIDENCE_THRESHOLD = 3` (tracklets with fewer detections are flagged)
- `DEFAULT_FPS = 25.0`
**Core Algorithm:** Matches YOLO detections to OSNet embeddings using a custom score metric: `score = compute_iou(box1, box2) - 10.0 * abs(conf1 - conf2)`. Embeddings matched to the same `track_id` are grouped, averaged (`np.mean`), and explicitly L2-normalized (`vector / np.linalg.norm(vector)`).

## 6. `src/stitch_tracklets.py`
**Purpose:** A Stage 1c healing script to fix broken tracklets (fragmentation) within the same view prior to cross-view clustering by merging tracklets that are adjacent in time and visually identical.
**Exact Inputs:**
- `data_manifests/tracklet_features.csv`
- `DRIVE_ROOT/data/tracklet_embeddings.npz`
**Exact Outputs:**
- `data_manifests/tracklet_features_stitched.csv`
- `DRIVE_ROOT/data/tracklet_embeddings_stitched.npz`
**Key Parameters/Constants:**
- `MAX_GAP_FRAMES = 15`
- `SIMILARITY_THRESHOLD = 0.85`
- `MAX_STITCHED_SPAN_FRAMES = 750`
**Core Algorithm:** Sorts tracklets by start frame. Greedily chains tracklets forward if the temporal gap is `0 <= gap <= 15` frames, the Cosine Similarity between their ReID embeddings is `> 0.85`, and the resulting chained tracklet would not exceed `750` total frames. Averages and L2-normalizes the embeddings of the merged chain.
**Deviations Noted:** Enforces a hard maximum span limit (`MAX_STITCHED_SPAN_FRAMES`) to prevent runaway transitive merging across huge time gaps.

## 7. `src/associate_cross_view.py`
**Purpose:** Constructs an affinity graph across overlapping views using pairwise cosine similarity between stitched 512-d ReID embeddings, enforces temporal constraints, and resolves global identities using hierarchical clustering.
**Exact Inputs:**
- `data_manifests/tracklet_features_stitched.csv`
- `DRIVE_ROOT/data/tracklet_embeddings_stitched.npz`
**Exact Outputs:**
- `data_manifests/cross_view_graph.csv`
- `data_manifests/global_identities_v2.csv` (default output manifest)
**Key Parameters/Constants:**
- `CROSS_VIEW_MAX_TIME_GAP = 2.0` (seconds)
- `CROSS_VIEW_MIN_SIMILARITY = 0.980`
- `CROSS_VIEW_MIN_DETECTIONS = 3`
- `linkage = "complete"`
**Core Algorithm:** 
1. Filters out tracklets with `< 3` detections as singletons.
2. For candidate pairs across different views, computes physical time gap and unit-norm dot product (Cosine Similarity).
3. If `time_gap <= 2.0s`, calculates distance as `max(0.0, 1.0 - sim)`. 
4. Performs `AgglomerativeClustering(metric="precomputed", linkage="complete", distance_threshold=1.0 - 0.980)`. 
5. Resolves same-view conflicts within clusters by keeping the tracklet with the highest average similarity to anchors from other views, evicting the rest to singletons.
**Deviations Noted:** Operates on the stitched (de-fragmented) tracklet features by default. The similarity threshold was empirically increased from 0.5 to 0.980 to maximize discrimination.

## 8. `src/camera_calibration.py`
**Purpose:** Provides forward pinhole camera models based on assumed room geometry and computes View Quality (VQ) scores for every detection based on its 3D spatial relationship to the camera.
**Exact Inputs:**
- `DRIVE_ROOT/data/detections/<view>.csv`
**Exact Outputs:**
- `data_manifests/vq_scores_<view>.csv`
**Key Parameters/Constants:**
- Room: 7m x 7m, camera height 1.7m.
- View 1: `pos=[0,0,1.7]`, `target=[4.26, 3.84, 0.0]`, `fov=60.0`
- View 2: `pos=[0,7,1.7]`, `target=[5.61, 3.47, 0.0]`, `fov=60.0`
- View 3: `pos=[7,7,1.7]`, `target=[3.52, 2.10, 0.0]`, `fov=79.26`
- `FRAME_WIDTH = 360`, `FRAME_HEIGHT = 288`
**Core Algorithm:** Ray-casts the bottom-center of bounding boxes to the `Z=0` floor plane.
- $Q_i$: Confidence penalized by 0.5 if near the frame edge (`< 5` or `> MAX - 5`).
- $D_i$ (Distance): Euclidean distance from camera XY to floor XY. Score $D = \max(0, 1 - |D_i - 2.5| / 2.5)$.
- $\phi_i$ (Centering): Angle to optical axis. Score $\phi = \max(0, \cos(\phi_i))$.
- $\theta_i$ (Orientation): Angle between movement direction (based on history $\le 15$ frames ago) and vector to camera. Score $\theta = \max(0, \cos(\theta_i))$.
- Final VQ = $Q_i \times (\frac{1}{3} score_\theta + \frac{1}{3} score_\phi + \frac{1}{3} score_D)$.
**Deviations Noted:** 
1. The distance formula is triangular peaking at $2.5m$, a deliberate deviation from the paper's linear form to penalize both extreme close-ups and far distances.
2. 60-80% of detections lack movement history and fall back to $\theta_i=0$.

## 9. `src/keyframe_selection.py`
**Purpose:** Replaces heuristic pixel-level difference thresholding with adaptive keyframe selection using fused full-frame Vision Transformer (DINO) and fine-grained pedestrian (OSNet ReID) embeddings.
**Exact Inputs:**
- `DRIVE_ROOT/data/dinov3_embeddings/<view>_dinov3.npz`
- `DRIVE_ROOT/data/embeddings/<view>_embeddings.npz`
**Exact Outputs:**
- `data_manifests/keyframe_decisions_{view}[_{suffix}].csv`
**Key Parameters/Constants:**
- `KEYFRAME_SHOT_WINDOW_SEC = 10.0` (default 250 frames)
- `KEYFRAME_ALPHA = 0.5`
- `KEYFRAME_K = 1.0`
- `KEYFRAME_DEFAULT_EPSILON = 0.85`
- `reid_matching = "greedy"`
- `reid_aggregation = "mean"`
**Core Algorithm:** Frames are segmented into static temporal shots (250 frames). The seed frame (first frame) is unconditionally accepted. A candidate frame $t$ calculates redundancy against all accepted keyframes $j$ in the shot:
$s\_redund(t, j) = \alpha \times s\_D(d_t, d_j) + (1 - \alpha) \times s\_R(r_t, r_j)$ (using unit-norm dot product for $s\_D$ and greedy max-similarity matching for sets in $s\_R$).
Frame $t$ is accepted iff $\max_j(s\_redund(t, j)) < \epsilon_{shot}$, where $\epsilon_{shot} = \mu - k \times \sigma$ of the observed scores within the shot.

## 10. `src/build_summary.py`
**Purpose:** Reconciles per-view keyframe shots, global identities, and VQ scores to build event clusters and output a final multi-view summary (Canonical Pipeline).
**Exact Inputs:**
- `data_manifests/keyframe_decisions_{view}[_{suffix}].csv`
- `data_manifests/tracklet_features_stitched.csv`
- `data_manifests/global_identities_v2.csv`
- `data_manifests/vq_scores_{view}.csv`
**Exact Outputs:**
- `data_manifests/final_summary_v2.csv`
**Key Parameters/Constants:**
- `CLUSTER_TOLERANCE_FRAMES = 5`
- `MIN_OVERLAP_FRAC = 0.25`
- `BROAD_ID_THRESHOLD = 0.15`
- `max_slot_span_frames = 750`
**Core Algorithm:** 
1. Reconstructs shot boundaries (`min` to `max` accepted frame_idx).
2. Inherits global identities if a tracklet overlaps $\ge 25\%$ of the shot's frames. Excludes IDs spanning $> 15\%$ of total shots.
3. Assigns `vq_score` per shot using the `max` VQ of its contained detections.
4. Two-Phase Clustering:
   - Phase 1: Buckets shots overlapping in time (tolerance 5 frames) into slots spanning $\le 750$ frames.
   - Phase 2: Within each slot, clusters shots that share a global ID (strict dominant match for same-view, looser shared-ID for cross-view).
5. Chooses the shot with the highest `vq_score` as the cluster representative.

## 11. `src/generate_training_pairs.py`
**Purpose:** Generates labeled positive and negative training pairs of shots for training the Graph Transformer and extracts their DINOv3 and ReID features.
**Exact Inputs:**
- Same CSVs as `build_summary.py` + `DRIVE_ROOT/data/dinov3_embeddings/<view>_dinov3.npz` and `DRIVE_ROOT/data/tracklet_embeddings_stitched.npz`
**Exact Outputs:**
- `data_manifests/training_pairs[_{suffix}].csv`
- `data_manifests/training_features_dino[_{suffix}].npz`
- `data_manifests/training_features_reid[_{suffix}].npz`
**Key Parameters/Constants:**
- `CLUSTER_TOLERANCE_FRAMES = 5`
- `sample_window` dynamically calculated from the first shot length (e.g. 150).
- `hard_neg_threshold = sample_window * 2`.
**Core Algorithm:** Reconstructs and links shots. For every pair $(A, B)$:
- Positive: Shared IDs and gap $\le 5$ frames (cross-view) or $\le$ `hard_neg_threshold` (intra-view).
- Hard Negative: Different views, NO shared IDs, gap $\le$ `hard_neg_threshold`.
- Easy Negative: No shared IDs, gap $>$ `hard_neg_threshold` (or same view).
Balances data (50/50 split of the positive count for hard/easy negatives). Dumps keyframe DINO and overlapping stitched ReID embeddings for all unique shots.
**Deviations Noted:** Tightened logic ensures hard negatives must be $\le 2\times$ shot window apart.

## 12. `src/build_graph_w150_sparse.py`
**Purpose:** Builds a PyTorch Geometric (PyG) graph representation of the dataset from the training pairs CSV, reducing dimensionality via PCA (Sparse w150 variant).
**Exact Inputs:**
- `data_manifests/training_pairs_w150.csv`
- `DRIVE_ROOT/data_manifests/training_features_dino_w150_sparse.npz`
- `DRIVE_ROOT/data_manifests/training_features_reid_w150_sparse.npz`
**Exact Outputs:** 
- Returns PyG `Data` object (used in memory).
**Key Parameters/Constants:**
- PCA `n_components = 64` for both DINO and ReID.
**Core Algorithm:** Maps unique shots to nodes. For ReID features, pools multiple tracks per shot via `np.mean(axis=0)`. Applies PCA dimensionality reduction (384 $\to$ 64 for DINO, 512 $\to$ 64 for ReID). Node feature is 133-dimensional: `[DINO (64), VQ (1), Start Time Scaled (1), View One-Hot (3), ReID (64)]`. Builds undirected edges using the binary labels (0 or 1).
**Deviations Noted:** "Sparsely sampled" features generated via Colab are loaded. ReID representations are mean-pooled down to a single vector per shot.

## 13. `src/train_graph_transformer_w150_sparse.py`
**Purpose:** Trains a 2-layer Graph Attention Network (GAT) model on the sparse-sampled w150 dataset to predict cross-view cluster redundancy (binary edge classification).
**Exact Inputs:**
- PyG Data object from `build_graph_w150_sparse.py`.
**Exact Outputs:**
- Model weights: `models/graph_transformer_w150_sparse_foldX.pt`
**Key Parameters/Constants:**
- GAT config: `in_channels=133`, `hidden_channels=64`, `out_channels=64`, `heads=2`, `concat=False`, `dropout=0.4`.
- Edge Classifier: `Sequential(Linear(128, 32), ReLU, Dropout(0.4), Linear(32, 1))`
- Optimizer: Adam, `lr=1e-3`, `weight_decay=1e-4`, `BCEWithLogitsLoss`.
- Training: `epochs=300`, `patience=30`.
**Core Algorithm:** Uses 5-Fold Stratified Cross Validation. Nodes pass through a linear projection layer and two GAT layers. To predict an edge between $u$ and $v$, concatenates their node embeddings $[u, v]$ into a 128-d vector and passes it through the MLP edge classifier. Evaluates predictions via a 0.5 threshold on the sigmoid output.

## 14. `src/assemble_summary_video.py`
**Purpose:** Produces a playable MP4 video of the final summary by stitching together frames for each event sequence and overlaying text indicating the view and chronological order.
**Exact Inputs:**
- Summary CSV (e.g. `data_manifests/final_summary.csv` or similar)
- Frames: `DRIVE_ROOT/data/<view>/frame_*.jpg`
**Exact Outputs:**
- `data_manifests/final_summary_video.mp4`
**Key Parameters/Constants:**
- `FPS = 25.0`
**Core Algorithm:** Reads summary shots sequentially (sorted by `sequence_order`). Collects target frame paths. Uses OpenCV (`cv2.putText`, `cv2.rectangle`) to draw a black background badge and yellow text label (`Event {seq} | {view} | Frame {idx}`). Saves annotated frames to a temporary directory. Runs `ffmpeg -i ... -c:v libx264` to encode the frames into an H.264 MP4.
