# %% [markdown]
# # Stage 1b: Person Re-Identification (ReID) Feature Extraction
# 
# This notebook runs Stage 1b of the multi-view CCTV summarization pipeline:
# 1. Loads person detections from Stage 1a (`data/detections/<view>.csv`).
# 2. Crops person patches from synchronized multi-view frames.
# 3. Extracts 512-d visual embeddings using pretrained OSNet (Market-1501).
# 4. Saves compressed embedding archives (`data/embeddings/<view>_embeddings.npz`).
# 5. Visualizes cross-camera person retrieval and similarity matrices.

# %%
from google.colab import drive
drive.mount('/content/drive')

# %%
import os
os.environ["DRIVE_ROOT"] = "/content/drive/MyDrive/CCTV-Multiview-Project"

# %%
import torch
print("CUDA available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

# %%
%cd /content/cctv-multiview-summarization
!git pull

# %%
# Install required dependencies
!pip install -q -r requirements-colab.txt

# %% [markdown]
# ### Run ReID Feature Extraction across All Views

# %%
!python -m src.extract_reid_features --view view1 --batch-size 64

# %%
!python -m src.extract_reid_features --view view2 --batch-size 64

# %%
!python -m src.extract_reid_features --view view3 --batch-size 64

# %% [markdown]
# ### Visual Verification: Cross-Camera Person Matching

# %%
!python -m src.visualize_reid --query-view view1 --gallery-view view2 --out reid_retrieval_v1_v2.png

# %%
from IPython.display import Image
Image("reid_retrieval_v1_v2.png")

# %% [markdown]
# ### Visual Verification: Similarity Matrix Heatmap

# %%
!python -m src.visualize_reid --similarity-matrix --view view1 --num-samples 25 --out reid_matrix_view1.png
Image("reid_matrix_view1.png")

# %% [markdown]
# ### Commit Summary Manifest to GitHub

# %%
!git status
!git add data_manifests/reid_summary.csv
!git commit -m "Add Stage 1b ReID embeddings summary manifest"
!git push
