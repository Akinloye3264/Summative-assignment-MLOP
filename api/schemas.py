"""Response schemas for the public inference API."""

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
    uptime_seconds: float


class PredictionResponse(BaseModel):
    predicted_class: str
    display_class: str
    confidence: float = Field(ge=0.0, le=1.0)
    probabilities: dict[str, float]
    inference_ms: float
    model_version: str


class DatasetUploadResponse(BaseModel):
    label: str
    received: int
    stored: int
    duplicates: int
    stored_files: list[str]
    class_counts: dict[str, int]
    total_images: int


class RetrainingJobResponse(BaseModel):
    job_id: str
    status: str
    created_at: str
    started_at: str | None = None
    completed_at: str | None = None
    dataset_images: int
    class_counts: dict[str, int]
    epochs: int
    model_version: str | None = None
    metrics: dict[str, float] | None = None
    error: str | None = None
