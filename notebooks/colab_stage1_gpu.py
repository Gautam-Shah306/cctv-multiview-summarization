# %% [markdown]
# # Stage 1: End-to-End Object Detection, ByteTracking & ReID (GPU Accelerated)
# 
# This notebook runs Stage 1a (YOLO + ByteTrack) and Stage 1b (OSNet ReID) using Google Colab T4/V100/A100 GPU acceleration.
#
# **Key advantages on GPU:**
# - Full-sequence ByteTrack object detection across all 3 views runs in ~10–15 minutes (vs >3–4 hours on CPU).
# - OSNet ReID extraction runs in FP16 with batch size 64 in <5 minutes (vs ~2 hours on CPU).

# %% [markdown]
# ### 1. Mount Google Drive & Environment Setup

# %%
from google.colab import drive
drive.mount('/content/drive')

# %%
import os
os.environ["DRIVE_ROOT"] = "/content/drive/MyDrive/CCTV-Multiview-Project"

# %% [markdown]
# ### 2. Verify GPU Acceleration & PyTorch CUDA Setup

# %%
import torch
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device Name:", torch.cuda.get_device_name(0))
    print("VRAM Allocated:", round(torch.cuda.memory_allocated(0)/1024**3, 1), "GB")
    !nvidia-smi
else:
    print("WARNING: GPU not detected! Go to Runtime -> Change runtime type -> Select T4 GPU")

# %% [markdown]
# ### 3. Clone / Update Repository

# %%
# If repository not yet cloned in /content:
import os
if not os.path.exists('/content/cctv-multiview-summarization'):
    from getpass import getpass
    github_token = getpass("Enter GitHub Personal Access Token (or press enter if repo is public): ")
    if github_token.strip():
        repo_url = f"https://{github_token.strip()}@github.com/Gautam-Shah306/cctv-multiview-summarization.git"
    else:
        repo_url = "https://github.com/Gautam-Shah306/cctv-multiview-summarization.git"
    !git clone {repo_url} /content/cctv-multiview-summarization

# %%
%cd /content/cctv-multiview-summarization
!git fetch origin
!git checkout feature/stage1-object-detection
!git pull origin feature/stage1-object-detection

# %% [markdown]
# ### 4. Install Dependencies

# %%
!pip install -q -r requirements-colab.txt lap>=0.5.12

# %% [markdown]
# ### 5. Stage 1a: Run YOLO + ByteTrack across All Views (GPU)
# This generates `data/detections/<view>.csv` with persistent `track_id`s.

# %%
!python -m src.detect_objects --track --force

# %% [markdown]
# ### 6. Stage 1b: Run OSNet Person ReID Embeddings across All Views (GPU)
# Extracts 512-d visual embeddings pooled by tracklets/boxes into `data/embeddings/<view>_embeddings.npz`.

# %%
!python -m src.extract_reid_features --force --batch-size 64

# %% [markdown]
# ### 7. Visual Quality Assurance: Verification Plots

# %%
!python -m src.visualize_reid --query-view view1 --gallery-view view2 --out reid_retrieval_v1_v2.png
!python -m src.visualize_reid --query-view view1 --gallery-view view3 --out reid_retrieval_v1_v3.png
!python -m src.visualize_reid --similarity-matrix --view view1 --out reid_similarity_matrix_view1.png

# %%
from IPython.display import Image, display
print("Cross-Camera Retrieval View 1 -> View 2:")
display(Image("reid_retrieval_v1_v2.png"))

print("Cross-Camera Retrieval View 1 -> View 3:")
display(Image("reid_retrieval_v1_v3.png"))

print("Similarity Matrix (View 1):")
display(Image("reid_similarity_matrix_view1.png"))

# %% [markdown]
# ### 8. Commit Updated Summary Manifests to GitHub

# %%
!git add data_manifests/detections_summary.csv data_manifests/reid_summary.csv
!git commit -m "Update Stage 1a detections (with ByteTrack IDs) and Stage 1b ReID summary manifests"
!git push
