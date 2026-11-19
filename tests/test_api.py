from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import api.main as main


def create_upload(color: tuple[int, int, int] = (45, 130, 55)) -> BytesIO:
    image = Image.new("RGB", (320, 240), color=color)
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    buffer.seek(0)
    return buffer


@pytest.fixture(scope="module")
def client() -> TestClient:
    with TestClient(main.app) as test_client:
        yield test_client


@pytest.fixture
def isolated_upload_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "uploaded"
    monkeypatch.setattr(main.dataset_store, "root", root)
    for class_name in main.CLASS_NAMES:
        (root / class_name).mkdir(parents=True)
    return root


def test_health_and_prediction(client: TestClient) -> None:
    health_response = client.get("/health")
    assert health_response.status_code == 200
    assert health_response.json()["model_loaded"] is True

    prediction_response = client.post(
        "/predict",
        files={"file": ("leaf.jpg", create_upload(), "image/jpeg")},
    )
    assert prediction_response.status_code == 200
    result = prediction_response.json()
    assert result["predicted_class"] in result["probabilities"]
    assert 0 <= result["confidence"] <= 1
    assert len(result["probabilities"]) == 4


def test_prediction_rejects_text_file(client: TestClient) -> None:
    response = client.post(
        "/predict",
        files={"file": ("notes.txt", b"not an image", "text/plain")},
    )
    assert response.status_code == 415


def test_upload_and_deduplicate_labelled_images(
    client: TestClient, isolated_upload_store: Path
) -> None:
    response = client.post(
        "/datasets/upload",
        data={"label": "Healthy"},
        files=[
            ("files", ("one.jpg", create_upload(), "image/jpeg")),
            ("files", ("two.jpg", create_upload(), "image/jpeg")),
        ],
    )
    assert response.status_code == 201
    body = response.json()
    assert body["received"] == 2
    assert body["stored"] == 1
    assert body["duplicates"] == 1
    assert body["class_counts"]["Healthy"] == 1

    duplicate = client.post(
        "/datasets/upload",
        data={"label": "Healthy"},
        files={"files": ("again.jpg", create_upload(), "image/jpeg")},
    )
    assert duplicate.status_code == 201
    assert duplicate.json()["stored"] == 0
    assert duplicate.json()["duplicates"] == 1

    summary = client.get("/datasets/summary")
    assert summary.status_code == 200
    assert summary.json()["total_images"] == 1


def test_upload_rejects_invalid_label_and_image(
    client: TestClient, isolated_upload_store: Path
) -> None:
    bad_label = client.post(
        "/datasets/upload",
        data={"label": "Unknown"},
        files={"files": ("leaf.jpg", create_upload(), "image/jpeg")},
    )
    assert bad_label.status_code == 422

    bad_image = client.post(
        "/datasets/upload",
        data={"label": "Healthy"},
        files={"files": ("fake.jpg", b"not an image", "image/jpeg")},
    )
    assert bad_image.status_code == 422


def test_retraining_rejects_an_insufficient_dataset(
    client: TestClient, isolated_upload_store: Path
) -> None:
    response = client.post("/retrain")
    assert response.status_code == 422
    assert "required" in response.json()["detail"]


def test_retraining_job_contract(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = {
        "job_id": "job-123",
        "status": "queued",
        "created_at": "2026-11-17T12:00:00+00:00",
        "started_at": None,
        "completed_at": None,
        "dataset_images": 8,
        "class_counts": {name: 2 for name in main.CLASS_NAMES},
        "epochs": 3,
        "model_version": None,
        "metrics": None,
        "error": None,
    }
    monkeypatch.setattr(main.retraining_manager, "start", lambda: job)
    monkeypatch.setattr(main.retraining_manager, "get", lambda job_id: job)

    started = client.post("/retrain")
    assert started.status_code == 202
    assert started.json()["job_id"] == "job-123"

    status_response = client.get("/retrain/job-123")
    assert status_response.status_code == 200
    assert status_response.json()["status"] == "queued"


def test_unknown_retraining_job_returns_404(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def missing(_: str) -> dict:
        raise KeyError("missing")

    monkeypatch.setattr(main.retraining_manager, "get", missing)
    response = client.get("/retrain/missing")
    assert response.status_code == 404
