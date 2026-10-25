import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

IMAGE_ROOT = Path(os.getenv("IMAGE_ROOT", "data")).resolve()

ALLOWED_ORIGINS = [
    o.strip()
    for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:5173,http://localhost:3000").split(",")
    if o.strip()
]

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "10"))

REVIEWER_API_KEY = os.getenv("REVIEWER_API_KEY")

DISCLAIMER = (
    "Educational prototype. This is not a medical diagnosis. "
    "See a dermatologist about any skin concern."
)