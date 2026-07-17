from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

DRIVE_ROOT = Path(os.environ["DRIVE_ROOT"])

RAW_VIDEOS = DRIVE_ROOT / "raw_videos"
DATA_DIR = DRIVE_ROOT / "data"
LOGS_DIR = DRIVE_ROOT / "logs"
CHECKPOINTS = DRIVE_ROOT / "checkpoints"