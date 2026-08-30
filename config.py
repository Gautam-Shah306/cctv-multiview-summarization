from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

DRIVE_ROOT = Path(os.environ["DRIVE_ROOT"])

RAW_VIDEOS = DRIVE_ROOT / "raw_videos"
DATA_DIR = DRIVE_ROOT / "data"
LOGS_DIR = DRIVE_ROOT / "logs"
CHECKPOINTS = DRIVE_ROOT / "checkpoints"

# --- Config: Stage 1a object detection -------------------------------------
DETECTION_MODEL = "yolo26m.pt"   # was yolov8m.pt
DETECTION_CONFIDENCE = 0.2
DETECTION_IMG_SIZE   = 640     # optimized for native 360x288 CCTV resolution (multiple of 32)
DETECTION_BATCH_SIZE = 32        # batch size for parallel inference
DETECTION_HALF       = True      # use FP16 half-precision on GPU
DETECTION_TRACKER    = "bytetrack.yaml"
MOTION_GATING_THRESHOLD = 500   # minimum pixel delta count to consider a frame active

# --- Config: Stage 1b Person ReID ------------------------------------------
REID_MODEL = "osnet_x1_0"
REID_WEIGHTS = "osnet_x1_0_market1501.pth"
REID_EMBEDDING_DIM = 512
REID_IMAGE_SIZE = (256, 128)     # (height, width) standard for person ReID
REID_BATCH_SIZE = 64
REID_HALF = True