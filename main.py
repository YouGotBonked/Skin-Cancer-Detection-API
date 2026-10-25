"""Cancer Detection API

Run:   uvicorn main:app --reload
Documentation:  http://localhost:8000/docs

Endpoints:
    GET  /health     is the API up, is a model loaded?
    POST /predict    upload an image (+ consent) -> class probabilities
    POST /feedback   reviewer-only: attach a confirmed label to a prediction
"""
import hashlib
import hmac
import io
import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

import config
from db import SessionLocal, get_db
from inference import DEFAULT_CLASSES, Classifier
from models import Feedback, ImageRecord, Lesion, ModelVersion, Prediction

log = logging.getLogger("api")
Image.MAX_IMAGE_PIXELS = 50_000_000
MAX_BYTES = config.MAX_UPLOAD_MB * 1024 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.classifier = None
    app.state.model_version_id = None
    app.state.model_name = None
    try:
        # Dynamically find the absolute path of the root directory on Render
        BASE_DIR = Path(__file__).resolve().parent
        fixed_weights_path = BASE_DIR / "models" / "model.pt"

        with SessionLocal() as db:
            mv = db.scalars(select(ModelVersion).where(ModelVersion.is_active).limit(1)).first()
            if mv:
                if fixed_weights_path.exists():
                    classes = (mv.metrics or {}).get("classes", DEFAULT_CLASSES)
                    # Pass the absolute server path directly, bypassing the broken DB path
                    app.state.classifier = Classifier(str(fixed_weights_path), classes)
                    app.state.model_version_id = mv.id
                    app.state.model_name = mv.name
                    log.info("Loaded model %s from %s", mv.name, fixed_weights_path)
                else:
                    log.warning("Active model found in DB, but file is physically missing at: %s", fixed_weights_path)
            else:
                log.warning("No active model record found in the database. /predict will return 503.")
    except Exception:
        log.exception("Startup could not load a model (are the tables created?)")
    yield


app = FastAPI(title="Skin Lesion Classifier API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictionOut(BaseModel):
    prediction_id: int
    top_class: str
    probs: dict[str, float]
    model_version: str
    disclaimer: str


class FeedbackIn(BaseModel):
    prediction_id: int
    confirmed_label: str
    confirmed_by: Optional[str] = None


@app.get("/health")
def health(request: Request):
    return {
        "status": "ok",
        "model_loaded": request.app.state.classifier is not None,
        "model_version": request.app.state.model_name,
    }


@app.post("/predict", response_model=PredictionOut)
def predict(
    request: Request,
    file: UploadFile = File(...),
    consent: bool = Form(False),
    db: Session = Depends(get_db),
):
    if not consent:
        raise HTTPException(400, "You must consent to the image being stored before uploading.")

    clf: Optional[Classifier] = request.app.state.classifier
    if clf is None:
        raise HTTPException(503, "No model is loaded yet. Train and register one first.")

    raw = file.file.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise HTTPException(413, f"Image is larger than {config.MAX_UPLOAD_MB} MB.")
    try:
        Image.open(io.BytesIO(raw)).verify()
        img = Image.open(io.BytesIO(raw))
        img = ImageOps.exif_transpose(img).convert("RGB")
    except Exception:
        raise HTTPException(400, "That file isn't a readable image (use JPEG, PNG or WebP).")

    img.thumbnail((1600, 1600))

    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=92)
    data = buf.getvalue()
    digest = hashlib.sha256(data).hexdigest()

    image_row = db.scalars(select(ImageRecord).where(ImageRecord.sha256 == digest)).first()
    if image_row is None:
        rel_path = f"uploads/{digest}.jpg"
        out_path = config.IMAGE_ROOT / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(data)

        lesion = Lesion(
            source_dataset="user_upload",
            external_lesion_id=uuid.uuid4().hex,
            split="unassigned",
        )
        db.add(lesion)
        db.flush()
        image_row = ImageRecord(
            lesion_id=lesion.id,
            file_path=rel_path,
            source="user_upload",
            label_source="none",
            sha256=digest,
            width=img.width,
            height=img.height,
        )
        db.add(image_row)
        db.flush()

    probs = clf.predict(img)
    top_class = max(probs, key=probs.get)
    pred = Prediction(
        image_id=image_row.id,
        model_version_id=request.app.state.model_version_id,
        probs=probs,
        top_class=top_class,
        consent_given=True,
    )
    db.add(pred)
    db.commit()

    return PredictionOut(
        prediction_id=pred.id,
        top_class=top_class,
        probs=probs,
        model_version=request.app.state.model_name,
        disclaimer=config.DISCLAIMER,
    )


@app.post("/feedback")
def feedback(
    body: FeedbackIn,
    request: Request,
    x_api_key: Optional[str] = Header(default=None),
    db: Session = Depends(get_db),
):
    if not config.REVIEWER_API_KEY:
        raise HTTPException(503, "Feedback is disabled (REVIEWER_API_KEY is not set).")
    if not hmac.compare_digest(x_api_key or "", config.REVIEWER_API_KEY):
        raise HTTPException(401, "Invalid API key.")

    clf: Optional[Classifier] = request.app.state.classifier
    valid = clf.classes if clf else DEFAULT_CLASSES
    if body.confirmed_label not in valid:
        raise HTTPException(400, f"confirmed_label must be one of {valid}")

    pred = db.get(Prediction, body.prediction_id)
    if pred is None:
        raise HTTPException(404, "Prediction not found.")

    db.add(
        Feedback(
            prediction_id=pred.id,
            confirmed_label=body.confirmed_label,
            confirmed_by=body.confirmed_by,
        )
    )
    if pred.image.source == "user_upload":
        pred.image.dx_label = body.confirmed_label
        pred.image.label_source = "expert"
    db.commit()
    return {"status": "saved"}
