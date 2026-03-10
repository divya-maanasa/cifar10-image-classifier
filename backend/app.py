"""
app.py
------
FastAPI backend for the CIFAR-10 image classification service.

Endpoints
---------
  GET  /              — Health check
  GET  /classes       — List of CIFAR-10 class names
  POST /predict       — Upload an image, receive top-k predictions
  GET  /model/info    — Model architecture summary & metrics

Run
---
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload
"""

import io
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent
MODEL_DIR  = BASE_DIR.parent / "model" / "saved_model"
MODEL_PATH = MODEL_DIR / "cifar10_model.keras"
STATS_PATH = MODEL_DIR / "norm_stats.json"
METRICS_PATH = MODEL_DIR / "metrics.json"

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]
IMG_SIZE = (32, 32)
MAX_FILE_SIZE_MB = 10

# ── FastAPI App ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="CIFAR-10 Image Classifier",
    description="Classify images into 10 categories using a CNN trained on CIFAR-10.",
    version="1.0.0",
)

# Allow all origins in development; restrict in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Globals (loaded once at startup) ─────────────────────────────────────────
_model = None
_mean  = None
_std   = None
_metrics = {}


@app.on_event("startup")
async def load_model():
    global _model, _mean, _std, _metrics

    if not MODEL_PATH.exists():
        logger.warning(
            f"Model not found at {MODEL_PATH}. "
            "Train the model first: cd model && python train.py"
        )
        return

    logger.info(f"Loading model from {MODEL_PATH} …")
    t0 = time.time()
    _model = tf.keras.models.load_model(str(MODEL_PATH))
    logger.info(f"Model loaded in {time.time() - t0:.2f}s")

    if STATS_PATH.exists():
        with open(STATS_PATH) as f:
            stats = json.load(f)
        _mean = np.array(stats["mean"], dtype=np.float32)
        _std  = np.array(stats["std"],  dtype=np.float32)
        logger.info("Normalization stats loaded.")

    if METRICS_PATH.exists():
        with open(METRICS_PATH) as f:
            _metrics = json.load(f)


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class Prediction(BaseModel):
    label: str
    confidence: float
    class_index: int


class PredictionResponse(BaseModel):
    prediction: str          # top-1 label
    confidence: float        # top-1 confidence
    top_k: List[Prediction]  # top-5 predictions
    inference_time_ms: float


class ClassesResponse(BaseModel):
    classes: List[str]
    count: int


class ModelInfoResponse(BaseModel):
    name: str
    input_shape: List[int]
    output_classes: int
    total_params: int
    metrics: dict


# ── Helpers ───────────────────────────────────────────────────────────────────

def preprocess(image_bytes: bytes) -> np.ndarray:
    """Decode, resize, normalize → (1, 32, 32, 3) float32."""
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400,
                            detail=f"Cannot decode image: {exc}")

    img = img.resize(IMG_SIZE, Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0   # [0, 1]

    if _mean is not None and _std is not None:
        arr = (arr - _mean) / (_std + 1e-7)

    return np.expand_dims(arr, axis=0)               # (1, 32, 32, 3)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", summary="Health check")
async def root():
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "message": "CIFAR-10 Image Classifier API",
    }


@app.get("/classes", response_model=ClassesResponse, summary="List CIFAR-10 classes")
async def get_classes():
    return ClassesResponse(classes=CLASS_NAMES, count=len(CLASS_NAMES))


@app.get("/model/info", response_model=ModelInfoResponse, summary="Model metadata")
async def model_info():
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    total_params = _model.count_params()
    input_shape  = list(_model.input_shape[1:])  # drop batch dim

    return ModelInfoResponse(
        name=_model.name,
        input_shape=input_shape,
        output_classes=len(CLASS_NAMES),
        total_params=total_params,
        metrics=_metrics,
    )


@app.post("/predict", response_model=PredictionResponse, summary="Classify an image")
async def predict(file: UploadFile = File(..., description="Image file (JPEG/PNG/WEBP)")):
    # ── Validate ──────────────────────────────────────────────────────────────
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train the model first.",
        )

    allowed_types = {"image/jpeg", "image/png", "image/webp", "image/gif"}
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported media type '{file.content_type}'. "
                   f"Allowed: {allowed_types}",
        )

    image_bytes = await file.read()
    if len(image_bytes) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {MAX_FILE_SIZE_MB} MB.",
        )

    # ── Inference ─────────────────────────────────────────────────────────────
    try:
        t0    = time.perf_counter()
        arr   = preprocess(image_bytes)
        probs = _model.predict(arr, verbose=0)[0]           # (10,)
        elapsed_ms = (time.perf_counter() - t0) * 1000
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}")

    # ── Build response ────────────────────────────────────────────────────────
    top5_indices = np.argsort(probs)[::-1][:5]
    top_k = [
        Prediction(
            label=CLASS_NAMES[i],
            confidence=round(float(probs[i]), 4),
            class_index=int(i),
        )
        for i in top5_indices
    ]

    best = top_k[0]
    logger.info(
        f"Prediction: {best.label} ({best.confidence*100:.1f}%) "
        f"in {elapsed_ms:.1f} ms — file={file.filename}"
    )

    return PredictionResponse(
        prediction=best.label,
        confidence=round(best.confidence, 4),
        top_k=top_k,
        inference_time_ms=round(elapsed_ms, 2),
    )
