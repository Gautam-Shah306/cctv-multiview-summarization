# %% [markdown]
# # Regenerate and Persist Missing Embeddings
# This notebook connects to Drive, verifies GPU, and regenerates missing ReID, DINOv3, and tracklet embeddings.
# 
# To run this in Colab, upload this file, ensure the runtime is set to GPU, and run all cells.

# %%
import os
import sys
from pathlib import Path
from google.colab import drive
import torch

# %%
# ==============================================================================
# Step a: Mount Drive and set environment variable
# ==============================================================================
print("[STEP A] Mounting Google Drive...")
drive.mount('/content/drive')
os.environ["DRIVE_ROOT"] = "/content/drive/MyDrive/CCTV-Multiview-Project"
DRIVE_ROOT = Path(os.environ["DRIVE_ROOT"])

# %%
# ==============================================================================
# Step b: Verify GPU
# ==============================================================================
print("\n[STEP B] Verifying GPU...")
if not torch.cuda.is_available():
    print("[FAIL] GPU is not available! Please change the runtime type to T4/A100 GPU and restart.")
    sys.exit(1)
print(f"[PASS] GPU detected: {torch.cuda.get_device_name(0)}")

# %%
# ==============================================================================
# Setup Environment
# ==============================================================================
repo_dir = "/content/cctv-multiview-summarization"
if not os.path.exists(repo_dir):
    print(f"[INFO] Cloning repository to {repo_dir}...")
    !git clone https://github.com/Gautam-Shah306/cctv-multiview-summarization.git {repo_dir}

os.chdir(repo_dir)
!git fetch origin
!git checkout feature/stage1-object-detection
!git pull origin feature/stage1-object-detection
!pip install -q -r requirements-colab.txt

def verify_file(path_str):
    p = Path(path_str)
    if p.exists() and p.stat().st_size > 0:
        return True, p.stat().st_size
    return False, 0

# %%
# ==============================================================================
# Step c: Run ReID Feature Extraction
# ==============================================================================
print("\n" + "="*50)
print("[STEP C] Running ReID Feature Extraction...")
print("="*50)
!python -m src.extract_reid_features --force

print("\nVerifying ReID Outputs...")
reid_files = [
    DRIVE_ROOT / "data" / "embeddings" / "view1_embeddings.npz",
    DRIVE_ROOT / "data" / "embeddings" / "view2_embeddings.npz",
    DRIVE_ROOT / "data" / "embeddings" / "view3_embeddings.npz",
]

reid_passed = True
reid_results = {}
for f in reid_files:
    exists, size = verify_file(f)
    if exists:
        print(f"[PASS] {f.name} exists with size {size} bytes.")
        reid_results[f.name] = size
    else:
        print(f"[FAIL] {f.name} is missing or empty.")
        reid_passed = False

if not reid_passed:
    print("[FAIL] Step C failed verification. Halting.")
    sys.exit(1)

# %%
# ==============================================================================
# Step d: Run DINOv3 Feature Extraction
# ==============================================================================
print("\n" + "="*50)
print("[STEP D] Running DINOv3 Feature Extraction...")
print("="*50)
!python -m src.extract_dinov3_features --view all --force

print("\nVerifying DINOv3 Outputs...")
dino_files = [
    DRIVE_ROOT / "data" / "dinov3_embeddings" / "view1_dinov3.npz",
    DRIVE_ROOT / "data" / "dinov3_embeddings" / "view2_dinov3.npz",
    DRIVE_ROOT / "data" / "dinov3_embeddings" / "view3_dinov3.npz",
]

dino_passed = True
dino_results = {}
for f in dino_files:
    exists, size = verify_file(f)
    if exists:
        print(f"[PASS] {f.name} exists with size {size} bytes.")
        dino_results[f.name] = size
    else:
        print(f"[FAIL] {f.name} is missing or empty.")
        dino_passed = False

if not dino_passed:
    print("[FAIL] Step D failed verification. Halting.")
    sys.exit(1)

# %%
# ==============================================================================
# Step e: Run Tracklet Feature Pooling
# ==============================================================================
print("\n" + "="*50)
print("[STEP E] Running Tracklet Feature Pooling...")
print("="*50)
!python -m src.pool_tracklet_features --force

print("\nVerifying Tracklet Pooling Output...")
tracklet_file = DRIVE_ROOT / "data" / "tracklet_embeddings.npz"
tracklet_passed = True
tracklet_results = {}
exists, size = verify_file(tracklet_file)
if exists:
    print(f"[PASS] {tracklet_file.name} exists with size {size} bytes.")
    tracklet_results[tracklet_file.name] = size
else:
    print(f"[FAIL] {tracklet_file.name} is missing or empty.")
    tracklet_passed = False

if not tracklet_passed:
    print("[FAIL] Step E failed verification. Halting.")
    sys.exit(1)

# %%
# ==============================================================================
# Step f: Final Consolidated Report
# ==============================================================================
print("\n" + "="*50)
print("FINAL CONSOLIDATED REPORT")
print("="*50)
print("[SUCCESS] All 3 embedding types generated and verified on Drive.")
for k, v in reid_results.items():
    print(f"- {k}: {v} bytes \t(Drive: data/embeddings/{k})")
for k, v in dino_results.items():
    print(f"- {k}: {v} bytes \t(Drive: data/dinov3_embeddings/{k})")
for k, v in tracklet_results.items():
    print(f"- {k}: {v} bytes \t(Drive: data/{k})")
print("="*50)
