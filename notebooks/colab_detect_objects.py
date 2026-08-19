# %%
from google.colab import drive
drive.mount('/content/drive')

# %%
from getpass import getpass

github_token = getpass("Enter your GitHub token: ")

# %%
import subprocess

repo_url = f"https://{github_token}@github.com/Gautam-Shah306/cctv-multiview-summarization.git"

subprocess.run(["git", "clone", repo_url], check=True)

# %%
%cd cctv-multiview-summarization

# %%
!git fetch origin

# %%
!git branch -a

# %%
!git checkout feature/stage1-object-detection

# %%
!pip install -q -r requirements-colab.txt

# %%
import os

os.environ["DRIVE_ROOT"] = "/content/drive/MyDrive/CCTV-Multiview-Project"

# %%
import torch
print("CUDA available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

# %%
from config import DATA_DIR

print("DATA_DIR:", DATA_DIR)
print("Exists:", DATA_DIR.exists())
print("View1 exists:", (DATA_DIR / "view1").exists())

# %%
!python -m src.detect_objects --view view1

# %%
!git pull

# %%
!python -m src.detect_objects --view view1

# %%
!nvidia-smi

# %%
!git pull

# %%
from config import DATA_DIR

frame_path = DATA_DIR / "view1" / "frame_000100.jpg"

print("Frame:", frame_path)
print("Exists:", frame_path.exists())

# %%
list((DATA_DIR / "view1").glob("frame_*.jpg"))[:5]

# %%
import csv, cv2
import matplotlib.pyplot as plt
from config import DATA_DIR

frame_idx_to_check = 3000
frame = cv2.imread(str(DATA_DIR / "view1" / f"frame_{frame_idx_to_check:04d}.jpg"))
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

with open(DATA_DIR / "detections" / "view1.csv") as f:
    for row in csv.DictReader(f):
        if int(row["frame_idx"]) == frame_idx_to_check:
            x1, y1, x2, y2 = (int(float(row[k])) for k in ("x1", "y1", "x2", "y2"))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

plt.figure(figsize=(10, 6))
plt.imshow(frame)
plt.axis("off")
plt.show()

# %%
!git pull

# %%
!python -m src.detect_objects --view view2

# %%
!git status

# %%
!git status
!git add data_manifests/detections_summary.csv
!git commit -m "Add detection manifest for view1 and view2"

# %%
!git add data_manifests/detections_summary.csv

# %%
!git status

# %%
!git commit -m "Add detection manifest for view1 and view2"

# %%
!git config --global user.email "gtmshh306@gmail.com"
!git config --global user.name "Gautam-Shah306"

# %%
!git commit -m "Add detection manifest for view1 and view2"

# %%
!git status


# %%
!git push


