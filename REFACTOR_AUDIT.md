# REFACTOR AUDIT

This document provides a comprehensive planning audit for codebase stabilization prior to introducing new experimental tracks (ReID re-ranking, DINOv3, unsupervised clustering, contrastive learning) and the "people-only summary" requirement.

## 1. NEAR-DUPLICATE / VARIANT SCRIPTS

The following script families contain duplicated logic due to quick iteration on the `w150` (150-frame window) and `sparse` tracks:

*   **`src/build_graph.py` vs `src/build_graph_w150.py` vs `src/build_graph_w150_sparse.py`**
    *   **Differences:** Purely parameter-driven. They differ only in assumed window size, the input feature `.npz` files they load (dense vs sparse), and the output CSV names.
    *   **Consolidation:** Can be merged into a single `src/build_graph.py` using argparse flags like `--window-size` and `--embedding-type sparse|dense`.
*   **`src/train_graph_transformer.py` vs `src/train_graph_transformer_w150.py` vs `src/train_graph_transformer_w150_sparse.py`**
    *   **Differences:** Input dataset paths, feature dimensionality (dense tracklet embeddings vs sparse spatial-temporal embeddings), and labeling logic.
    *   **Consolidation:** Highly compressible into a single script parameterized by a `--dataset-suffix` flag and dynamically reading feature dimensions from the input data.

## 2. DATA ARTIFACT VERSIONING

Several historical, version-stamped data manifests exist alongside current canonical files in `data_manifests/`:

*   **Summaries:** `final_summary.csv` is current (content matches `_v3`). `final_summary_v1_deprecated.csv`, `final_summary_v2.csv`, `final_summary_v2_deprecated.csv`, and `final_summary_v3.csv` are purely historical.
*   **Identities:** `global_identities_v2.csv` is current. `global_identities.csv` is historical.
*   **Cross-View Graphs:** `cross_view_graph_v2.csv` is current. `cross_view_graph.csv` is historical.
*   **Keyframes:** `keyframe_decisions_view*_w150.csv` and base `keyframe_decisions_view*.csv` are canonical for their respective tracks. The `_setbased_k0.5` and `_setbased_k1.0` variants are historical tuning runs.
*   **Recommendation:** All historical/tuning variants are safe to archive off the main repository working directory (they are already persisted on Google Drive). Only the canonical files should remain tracked in git.

## 3. SCRATCH SCRIPT INVENTORY

The `scratch/` directory and repo root contain numerous one-off analysis scripts:

*   **Notebook Generators:** `scratch_build_ipynb.py`, `scratch_build_ipynb_sparse.py`, `scratch_build_ipynb_w150.py` represent manual scaffolding for notebooks.
*   **Threshold & Analytical Tuning:** `analyze_degree.py`, `analyze_threshold.py`, `analyze_w150.py`, `check_sim.py`, `find_crossview_thresh.py`, `tune_threshold.py`, `validate_separation.py` were used to empirically derive constants (like $K=0.5$). Their findings are codified in `src/`.
*   **Experimental Runners:** `infer_and_evaluate_sparse.py`, `train_weighted.py`.
*   **Recommendation:** All of these represent past exploration phases. Because they are safely backed up to Google Drive (via previous persistence tasks), they are **safe to archive/remove** from the main repository to reduce clutter.

## 4. CONFIG / CONSTANTS AUDIT

While file paths are well-centralized in `config.py`, algorithmic design parameters are currently hardcoded across individual modules:

*   **`src/keyframe_selection.py`**: `WINDOW_SIZE` (250 or 150), `K_THRESH = 0.5`.
*   **`src/associate_cross_view.py`**: `CROSS_VIEW_MAX_TIME_GAP` (250 or 150), `CROSS_VIEW_MIN_SIMILARITY = 0.85`, `CROSS_VIEW_MIN_DETECTIONS = 3`.
*   **`src/build_summary.py`**: Multi-objective weights (VQ vs temporal gaps).
*   **`src/camera_calibration.py`**: `FRAME_WIDTH`, `FRAME_HEIGHT`, and fixed camera definitions (pose, FOV).
*   **Recommendation:** Move algorithmic hyper-parameters into a `HYPERPARAMS` or `PIPELINE_CONFIG` dictionary inside `config.py`. As we introduce DINOv3 and contrastive learning, these thresholds will need rapid toggling from a single source of truth.

## 5. CODE_STYLE.md COMPLIANCE CHECK

Spot-checked 5 scripts: `camera_calibration.py`, `keyframe_selection.py`, `build_graph.py`, `associate_cross_view.py`, `train_graph_transformer.py`.

**Systematic Drift Identified:**
1.  **Missing Headers:** `build_graph.py` and `train_graph_transformer.py` completely lack the mandated triple-quoted module docstring.
2.  **Usage Blocks:** Even where module docstrings exist (`camera_calibration.py`, `keyframe_selection.py`), the explicitly required `Usage:` block is missing.
3.  **Type Annotations:** Missing on newer functions (e.g., `def build_graph():` lacks a return type).
4.  **Import Ordering:** `train_graph_transformer.py` mixes PyTorch/sklearn (third-party) with `sys`/`pathlib` (standard library) without the required PEP-8 blank-line separation.

## 6. "PEOPLE-ONLY SUMMARY" IMPACT ANALYSIS

The current pipeline explicitly designs *around* empty frames rather than discarding them. Shifting to a "people-only summary" will require logic changes at the following locations:

1.  **`src/keyframe_selection.py` (Redundancy Scoring):** 
    *   **Impact:** Lines 194-235 implement a specific `fallback` flag. If a frame has 0 ReID detections, it falls back to purely visual DINO scene similarity (`fallback_triggered = True`). This actively allows empty frames to be selected as keyframes.
2.  **`src/extract_sparse_embeddings.py`:**
    *   **Impact:** Implements a "ReID Fallback Search" to look at adjacent frames if a target frame has no detections. If it totally fails, it pads with `np.zeros(512)`.
3.  **`src/generate_training_pairs.py`:**
    *   **Impact:** Logs `[WARNING] No ReID embeddings found... Using zeros` and pads empty frames with zero-vectors to keep the shot valid in the dataset.
4.  **`src/build_summary.py` / `src/camera_calibration.py`:**
    *   **Impact:** Empty frames may receive a VQ score of 0.0, but can still be forced into the final summary due to temporal gap spacing rules (rule-based track) or graph topology (Graph Transformer track). 

To enforce a people-only summary, we must filter out zero-detection frames *before* these fallback and padding mechanisms are invoked.
