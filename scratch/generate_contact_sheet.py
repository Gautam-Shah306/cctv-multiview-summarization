import csv
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import random

DATA_DIR = Path("data_storage/data")

def get_first_crop(view, tid):
    with open(DATA_DIR / "detections" / f"{view}.csv") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        if int(r["track_id"]) == tid:
            frame = int(r["frame_idx"])
            img_path = DATA_DIR / view / f"frame_{frame:04d}.jpg"
            img = cv2.imread(str(img_path))
            if img is None: continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]
            x1 = max(0, min(int(round(float(r["x1"]))), w - 1))
            y1 = max(0, min(int(round(float(r["y1"]))), h - 1))
            x2 = max(x1 + 1, min(int(round(float(r["x2"]))), w))
            y2 = max(y1 + 1, min(int(round(float(r["y2"]))), h))
            return img[y1:y2, x1:x2]
    return None

random.seed(42)

for view in ["view1", "view2", "view3"]:
    with open(DATA_DIR / "detections" / f"{view}.csv") as f:
        rows = list(csv.DictReader(f))
    tids = list(set(int(r["track_id"]) for r in rows if int(r["track_id"]) != -1))
    random.shuffle(tids)
    
    fig, axes = plt.subplots(4, 8, figsize=(16, 12))
    fig.suptitle(f"{view} Tracklets", fontsize=16)
    for i, ax in enumerate(axes.flatten()):
        if i < len(tids):
            tid = tids[i]
            crop = get_first_crop(view, tid)
            if crop is not None:
                ax.imshow(crop)
                ax.set_title(f"T{tid}")
            ax.axis('off')
    plt.tight_layout()
    plt.savefig(f"scratch/{view}_contact.png")
