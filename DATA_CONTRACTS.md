# Data Contracts: Multi-View CCTV Summarization

This document outlines the exact, verified data contracts based on a physical inspection of the actual output files currently on disk/Drive.

## 1. Frame Extraction Output
- **File Pattern**: `G:\My Drive\CCTV-Multiview-Project\data\viewX\frame_YYYY.jpg` (e.g., `frame_0000.jpg`)
- **Format**: JPEG Images
- **Current Data Status**:
  - `view1`: 3912 files
  - `view2`: 3915 files
  - `view3`: 3914 files

## 2. Object Detection + Tracking Output
- **File Path**: `G:\My Drive\CCTV-Multiview-Project\data\detections\viewX.csv`
- **Format**: CSV
- **Current Data Status (view1)**: 8293 rows
- **Schema & Dtypes**:
  - `frame_idx` (int64)
  - `view` (string/object, e.g., 'view1')
  - `x1`, `y1`, `x2`, `y2` (float64, bounding box coordinates)
  - `confidence` (float64)
  - `class_id` (int64)
- **CRITICAL FINDING**: The `track_id` column is **completely missing** from the actual CSV on disk.

## 3. ReID Embedding Output
- **File Path**: Intended to be `G:\My Drive\CCTV-Multiview-Project\data\embeddings\viewX_embeddings.npz`
- **CRITICAL FINDING**: **MISSING**. No `.npz` embedding files exist anywhere on the Drive or in the local repository. 

## 4. Tracklet Pooling Output
- **Manifest Path**: `d:\Semester_7\RMS\cctv-multiview-summarization\data_manifests\tracklet_features.csv`
- **Format**: CSV
- **Current Data Status**: 255 rows
- **Schema & Dtypes**:
  - `view` (string)
  - `track_id` (int64) - This is the primary identifier.
  - `start_frame`, `end_frame` (int64)
  - `start_time`, `end_time` (float64)
  - `num_detections` (int64)
- **Embeddings Path**: Intended to be `G:\My Drive\CCTV-Multiview-Project\data\tracklet_embeddings.npz`
- **CRITICAL FINDING**: The `tracklet_embeddings.npz` file is **MISSING** from disk.

## 5. Cross-View Identity Association (MCMT) Output
### A. Global Identities
- **File Path**: `d:\Semester_7\RMS\cctv-multiview-summarization\data_manifests\global_identities.csv`
- **Format**: CSV
- **Current Data Status**: 255 rows
- **Schema & Join Key**:
  - `view` (string)
  - `track_id` (int64)
  - `global_id` (int64)
  - *Join Key*: A global identity is linked back to a tracklet using the composite key `(view, track_id)`.

### B. Cross-View Graph
- **File Path**: `d:\Semester_7\RMS\cctv-multiview-summarization\data_manifests\cross_view_graph.csv`
- **Format**: CSV
- **Current Data Status**: 13651 rows
- **Schema**:
  - `view_a` (string), `track_id_a` (int64)
  - `view_b` (string), `track_id_b` (int64)
  - `cosine_similarity` (float64), `time_gap` (float64)

## 6. Keyframe Selection Output
- **File Path**: `d:\Semester_7\RMS\cctv-multiview-summarization\data_manifests\keyframe_decisions_viewX.csv`
- **Format**: CSV
- **Current Data Status (view1)**: 3912 rows
- **Schema & Dtypes**:
  - `view` (string)
  - `frame_idx` (int64)
  - `shot_id` (int64)
  - `accepted` (int64, 1 or 0)
  - `max_redund_score` (float64)
  - `epsilon_used` (float64)
- **CRITICAL FINDING**: 
  - There are **no window start/end frame indices** stored; it only stores the abstract `shot_id`.
  - Embeddings (DINOv3/ReID) are **not stored alongside** the keyframe decisions, nor are they referenced directly in the CSV. The expected `viewX_dinov3.npz` files are entirely **MISSING** from disk.

## General Observations
- **Data Loaders**: There is no central utility/helper function for loading these datasets. Existing scripts (e.g., `associate_cross_view.py`) use standard `csv.DictReader` and `np.load` locally where needed.
- **Naming Consistency**: The naming convention is very consistent. `view1`, `view2`, `view3` are used universally across folders and data columns (no instances of `cam1`, etc.).
- **GPU Requirements**: None of this generated output requires a GPU to read or process. Everything exists as structured CSV data or plain JPEG images, which can be easily processed via standard CPU file I/O (Pandas, Numpy, PIL).
