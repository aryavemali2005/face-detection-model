"""
Central configuration for the Smart Hygiene & Intrusion Monitoring System.
Edit the values here to tune behaviour without touching detection code.
"""

from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()  # allows overriding any setting via a .env file

# ── Project root (one level above src/) ──────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent

# ── Data paths ────────────────────────────────────────────────────────────────
STAFF_PHOTOS_DIR = ROOT_DIR / "data" / "staff_photos"
RAW_VIDEOS_DIR   = ROOT_DIR / "data" / "raw_videos"
RAW_IMAGES_DIR   = ROOT_DIR / "data" / "raw_images"
ANNOTATED_DIR    = ROOT_DIR / "data" / "annotated"

# ── Model paths ───────────────────────────────────────────────────────────────
MODELS_DIR          = ROOT_DIR / "models"
PPE_MODEL_PATH      = MODELS_DIR / "ppe_yolov8.pt"   # trained PPE detector (added later)
FACE_ENCODINGS_PATH = MODELS_DIR / "face_encodings.pkl"  # pre-computed staff encodings

# ── Output paths ──────────────────────────────────────────────────────────────
OUTPUTS_DIR          = ROOT_DIR / "outputs"
LOG_PATH             = OUTPUTS_DIR / "events.log"
ALERT_RECORDINGS_DIR = OUTPUTS_DIR / "alert_clips"
LOGS_DIR             = ROOT_DIR / "data" / "logs"

# ── Video source ──────────────────────────────────────────────────────────────
# 0 = default webcam.  Replace with a file path string or RTSP URL for IP cams.
VIDEO_SOURCE = int(os.getenv("VIDEO_SOURCE", 0))

# ── Face recognition ─────────────────────────────────────────────────────────
# Lower tolerance → stricter match (fewer false positives).
# Typical range: 0.45 (very strict) – 0.65 (lenient).
FACE_MATCH_TOLERANCE: float = float(os.getenv("FACE_MATCH_TOLERANCE", 0.55))

# ── PPE detection ─────────────────────────────────────────────────────────────
# Minimum YOLO confidence score (0.0–1.0) to count a PPE item as detected.
PPE_CONFIDENCE_THRESHOLD: float = float(os.getenv("PPE_CONFIDENCE_THRESHOLD", 0.60))

# PPE classes the model must detect for a person to be considered compliant.
REQUIRED_PPE_CLASSES: list[str] = ["head_cap", "gloves", "apron", "slip_ins"]

# ── Alert cooldown ────────────────────────────────────────────────────────────
# Seconds to wait before repeating the same audio alert for the same person.
# Prevents the TTS engine from spamming the same message on every frame.
ALERT_COOLDOWN_SECONDS: int = int(os.getenv("ALERT_COOLDOWN_SECONDS", 10))

# ── Performance ───────────────────────────────────────────────────────────────
# Face recognition is slow (~200 ms/frame). Running it every N frames keeps
# the video smooth while still catching new arrivals quickly.
FACE_DETECTION_INTERVAL: int = int(os.getenv("FACE_DETECTION_INTERVAL", 5))

# Scale factor applied before face detection (smaller = faster, less accurate).
FRAME_SCALE_FACTOR: float = float(os.getenv("FRAME_SCALE_FACTOR", 0.25))

# ── Productivity tracking ─────────────────────────────────────────────────────
# Weight given to movement (0.0–1.0); the rest goes to presence time.
# 0.4 means: 60% presence time + 40% activity level = combined score.
ACTIVITY_WEIGHT: float = float(os.getenv("ACTIVITY_WEIGHT", 0.4))

# ── Room geometry (informational, used for future zone mapping) ───────────────
ROOM_WIDTH_FT  = 25
ROOM_HEIGHT_FT = 25
