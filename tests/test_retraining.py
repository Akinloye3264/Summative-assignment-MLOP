from __future__ import annotations

import time
from io import BytesIO
from pathlib import Path

from PIL import Image

from src.dataset_store import DatasetStore
from src.retraining import RetrainingManager


CLASSES = ["Early_Blight", "Healthy", "Late_Blight", "Mid_Blight"]


def image_bytes(color: tuple[int, int, int]) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (32, 32), color).save(buffer, format="PNG")
    return buffer.getvalue()


def test_background_job_persists_completed_status(tmp_path: Path) -> None:
    store = DatasetStore(tmp_path / "uploads", CLASSES)
    for class_index, class_name in enumerate(CLASSES):
        for image_index in range(2):
            value = 30 + class_index * 40 + image_index
            store.add_batch(
                class_name,
                [(f"{image_index}.png", image_bytes((value, 80, 120)))],
            )

    manager = RetrainingManager(
        dataset_store=store,
        jobs_dir=tmp_path / "jobs",
        training_model_path=tmp_path / "base.keras",
        promoted_model_path=tmp_path / "current.keras",
        minimum_images=8,
        minimum_per_class=2,
        epochs=1,
    )
    manager._train = lambda job_id: ({"accuracy": 0.8}, "1.0.test")  # type: ignore[method-assign]

    started = manager.start()
    deadline = time.time() + 5
    status = manager.get(started["job_id"])
    while status["status"] in {"queued", "running"} and time.time() < deadline:
        time.sleep(0.02)
        status = manager.get(started["job_id"])

    manager.close()
    assert status["status"] == "completed"
    assert status["metrics"] == {"accuracy": 0.8}
    assert status["model_version"] == "1.0.test"
    assert (tmp_path / "jobs" / f"{started['job_id']}.json").is_file()
