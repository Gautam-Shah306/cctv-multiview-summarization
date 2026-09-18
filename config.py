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

# --- Config: Stage 2 Cross-View Association --------------------------------
OUTPUTS_DIR = DRIVE_ROOT / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
CROSS_VIEW_MAX_TIME_GAP = 2.0      # max physical time difference (seconds) for overlapping FOVs
# Empirically derived threshold: ReID backbone features (pre-projection) exhibit same-person similarities 
# ~0.75-0.95 and diff-person similarities clustered < 0.75. 0.75 maximizes discrimination over default 0.5.
CROSS_VIEW_MIN_SIMILARITY = 0.980
CROSS_VIEW_MIN_DETECTIONS = 3      # tracklets with fewer detections are treated as singletons

# --- Config: Novelty #1 Learned Keyframe Selection (DINO + ReID) ------------
DINOV3_MODEL = "facebook/dinov2-small"  # ViT-S encoder (22M params, 384-d)
DINOV3_EMBEDDINGS_DIR = DATA_DIR / "dinov3_embeddings"
DINOV3_EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
KEYFRAME_SHOT_WINDOW_SEC = 10.0         # temporal window per shot in seconds
KEYFRAME_ALPHA = 0.5                   # fusion weight for DINO similarity vs ReID similarity
KEYFRAME_K = 1.0                       # std multiplier for adaptive threshold epsilon = mu - k * sigma
KEYFRAME_DEFAULT_EPSILON = 0.85        # initial fallback epsilon when within-shot history < 2
