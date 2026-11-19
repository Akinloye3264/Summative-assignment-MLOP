"""FastAPI application for Taro Leaf Blight model inference and monitoring."""

from __future__ import annotations

import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status

from api.schemas import (
    DatasetUploadResponse,
    HealthResponse,
    PredictionResponse,
    RetrainingJobResponse,
)
from src.dataset_store import DatasetStore, DatasetStoreError
from src.prediction import PredictionService
from src.preprocessing import InvalidImageError, MAX_IMAGE_BYTES
from src.retraining import RetrainingError, RetrainingManager

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(
    os.getenv(
        "MODEL_PATH",
        PROJECT_ROOT / "models" / "taro_leaf_blight_model.h5",
    )
)
CLASS_NAMES_PATH = Path(
    os.getenv(
        "CLASS_NAMES_PATH",
        PROJECT_ROOT / "models" / "class_names.json",
    )
)
METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"
METRICS_PATH = PROJECT_ROOT / "models" / "metrics.json"
TRAINING_MODEL_PATH = Path(
    os.getenv(
        "TRAINING_MODEL_PATH",
        PROJECT_ROOT / "models" / "taro_leaf_blight_model.keras",
    )
)
UPLOAD_DIR = Path(
    os.getenv("UPLOAD_DIR", PROJECT_ROOT / "data" / "uploaded")
)
RETRAINING_DIR = Path(
    os.getenv("RETRAINING_DIR", PROJECT_ROOT / "data" / "retraining")
)
PROMOTED_MODEL_PATH = Path(
    os.getenv(
        "PROMOTED_MODEL_PATH",
        RETRAINING_DIR / "current_model.keras",
    )
)
RETRAIN_MIN_IMAGES = int(os.getenv("RETRAIN_MIN_IMAGES", "8"))
RETRAIN_MIN_PER_CLASS = int(os.getenv("RETRAIN_MIN_PER_CLASS", "2"))
RETRAIN_EPOCHS = int(os.getenv("RETRAIN_EPOCHS", "3"))
MAX_UPLOAD_FILES = int(os.getenv("MAX_UPLOAD_FILES", "50"))

with METADATA_PATH.open(encoding="utf-8") as file:
    MODEL_METADATA = json.load(file)
with METRICS_PATH.open(encoding="utf-8") as file:
    MODEL_METRICS = json.load(file)

MODEL_VERSION = str(MODEL_METADATA["model_version"])
STARTED_AT = time.time()
CLASS_NAMES = list(MODEL_METADATA["class_names"])
predictor = PredictionService(MODEL_PATH, CLASS_NAMES_PATH, MODEL_VERSION)
dataset_store = DatasetStore(UPLOAD_DIR, CLASS_NAMES)


def activate_promoted_model(model_path: Path, model_version: str) -> None:
    predictor.load(model_path=model_path, model_version=model_version)


retraining_manager = RetrainingManager(
    dataset_store=dataset_store,
    jobs_dir=RETRAINING_DIR / "jobs",
    training_model_path=TRAINING_MODEL_PATH,
    promoted_model_path=PROMOTED_MODEL_PATH,
    minimum_images=RETRAIN_MIN_IMAGES,
    minimum_per_class=RETRAIN_MIN_PER_CLASS,
    epochs=RETRAIN_EPOCHS,
    on_promoted=activate_promoted_model,
)


@asynccontextmanager
async def lifespan(_: FastAPI):
    if PROMOTED_MODEL_PATH.is_file():
        predictor.load(
            model_path=PROMOTED_MODEL_PATH,
            model_version=f"1.0.{PROMOTED_MODEL_PATH.stat().st_mtime_ns}",
        )
    else:
        predictor.load()
    yield


def refresh_promoted_model() -> None:
    """Keep every horizontally scaled API replica on the promoted model."""

    predictor.load_if_changed(PROMOTED_MODEL_PATH)


app = FastAPI(
    title="Taro Leaf Blight Classification API",
    description=(
        "Predict Healthy, Early Blight, Mid Blight, or Late Blight from one "
        "taro-leaf image."
    ),
    version=MODEL_VERSION,
    lifespan=lifespan,
)
@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health() -> HealthResponse:
    refresh_promoted_model()
    return HealthResponse(
        status="healthy" if predictor.is_loaded else "starting",
        model_loaded=predictor.is_loaded,
        model_version=predictor.model_version,
        uptime_seconds=round(time.time() - STARTED_AT, 2),
    )


@app.get("/model/info", tags=["Monitoring"])
def model_info() -> dict:
    refresh_promoted_model()
    return {
        **MODEL_METADATA,
        "model_version": predictor.model_version,
        "active_model_path": predictor.model_path.name,
        "loaded_at": predictor.loaded_at,
    }


@app.get("/metrics", tags=["Monitoring"])
def metrics() -> dict:
    return MODEL_METRICS


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    refresh_promoted_model()
    if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Upload a JPEG, PNG, or WebP image.",
        )

    image_bytes = await file.read(MAX_IMAGE_BYTES + 1)
    await file.close()
    try:
        result = predictor.predict(image_bytes)
    except InvalidImageError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    return PredictionResponse(
        **result,
        model_version=predictor.model_version,
    )


@app.get("/datasets/summary", tags=["Dataset"])
def dataset_summary() -> dict:
    counts = dataset_store.class_counts()
    return {
        "class_names": CLASS_NAMES,
        "class_counts": counts,
        "total_images": sum(counts.values()),
        "minimum_images_for_retraining": RETRAIN_MIN_IMAGES,
        "minimum_images_per_class": RETRAIN_MIN_PER_CLASS,
    }


@app.post(
    "/datasets/upload",
    response_model=DatasetUploadResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Dataset"],
)
async def upload_labelled_images(
    label: str = Form(...),
    files: list[UploadFile] = File(...),
) -> DatasetUploadResponse:
    if len(files) > MAX_UPLOAD_FILES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Upload at most {MAX_UPLOAD_FILES} files per request.",
        )

    uploads: list[tuple[str, bytes]] = []
    try:
        for upload in files:
            if upload.content_type not in {
                "image/jpeg",
                "image/png",
                "image/webp",
            }:
                raise HTTPException(
                    status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                    detail=f"'{upload.filename}' must be JPEG, PNG, or WebP.",
                )
            content = await upload.read(MAX_IMAGE_BYTES + 1)
            uploads.append((upload.filename or "image", content))
    finally:
        for upload in files:
            await upload.close()

    try:
        result = dataset_store.add_batch(label, uploads)
    except (DatasetStoreError, InvalidImageError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    return DatasetUploadResponse(**result)


@app.post(
    "/retrain",
    response_model=RetrainingJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["Retraining"],
)
def start_retraining() -> RetrainingJobResponse:
    try:
        job = retraining_manager.start()
    except RetrainingError as exc:
        response_status = (
            status.HTTP_409_CONFLICT
            if "already active" in str(exc)
            else status.HTTP_422_UNPROCESSABLE_ENTITY
        )
        raise HTTPException(status_code=response_status, detail=str(exc)) from exc
    return RetrainingJobResponse(**job)


@app.get(
    "/retrain/{job_id}",
    response_model=RetrainingJobResponse,
    tags=["Retraining"],
)
def retraining_status(job_id: str) -> RetrainingJobResponse:
    try:
        job = retraining_manager.get(job_id)
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Retraining job not found.",
        ) from exc
    return RetrainingJobResponse(**job)
