"""Background fine-tuning jobs for newly labelled taro-leaf images."""

from __future__ import annotations

import json
import os
import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from src.dataset_store import DatasetStore


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RetrainingError(RuntimeError):
    """Raised when a retraining job cannot be accepted."""


class RetrainingManager:
    def __init__(
        self,
        *,
        dataset_store: DatasetStore,
        jobs_dir: Path,
        training_model_path: Path,
        promoted_model_path: Path,
        minimum_images: int,
        minimum_per_class: int,
        epochs: int,
        on_promoted: Callable[[Path, str], None] | None = None,
    ) -> None:
        self.dataset_store = dataset_store
        self.jobs_dir = jobs_dir
        self.training_model_path = training_model_path
        self.promoted_model_path = promoted_model_path
        self.minimum_images = minimum_images
        self.minimum_per_class = minimum_per_class
        self.epochs = epochs
        self.on_promoted = on_promoted
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.active_lock_path = self.jobs_dir / "active.lock"
        self.promoted_model_path.parent.mkdir(parents=True, exist_ok=True)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="retrain")
        self._lock = threading.Lock()
        self._active_job_id: str | None = None

    def close(self) -> None:
        """Stop accepting work and wait for an active training job to finish."""

        self._executor.shutdown(wait=True, cancel_futures=False)

    def start(self) -> dict:
        counts = self.dataset_store.class_counts()
        total = sum(counts.values())
        if total < self.minimum_images:
            raise RetrainingError(
                f"At least {self.minimum_images} uploaded images are required; "
                f"currently {total}."
            )
        insufficient = {
            name: count
            for name, count in counts.items()
            if count < self.minimum_per_class
        }
        if insufficient:
            details = ", ".join(f"{name}={count}" for name, count in insufficient.items())
            raise RetrainingError(
                f"Each class requires at least {self.minimum_per_class} images. "
                f"Insufficient: {details}."
            )

        with self._lock:
            if self._active_job_id is not None:
                active = self.get(self._active_job_id)
                if active["status"] in {"queued", "running"}:
                    raise RetrainingError(
                        f"Retraining job {self._active_job_id} is already active."
                    )
            job_id = uuid.uuid4().hex
            self._claim_shared_lock(job_id)
            job = {
                "job_id": job_id,
                "status": "queued",
                "created_at": utc_now(),
                "started_at": None,
                "completed_at": None,
                "dataset_images": total,
                "class_counts": counts,
                "epochs": self.epochs,
                "model_version": None,
                "metrics": None,
                "error": None,
            }
            try:
                self._write(job)
                self._active_job_id = job_id
                self._executor.submit(self._run, job_id)
            except Exception:
                self._release_shared_lock(job_id)
                raise
            return job

    def get(self, job_id: str) -> dict:
        path = self.jobs_dir / f"{job_id}.json"
        if not path.is_file():
            raise KeyError(job_id)
        with path.open(encoding="utf-8") as file:
            return json.load(file)

    def _write(self, job: dict) -> None:
        target = self.jobs_dir / f"{job['job_id']}.json"
        temporary = target.with_suffix(".json.tmp")
        with temporary.open("w", encoding="utf-8") as file:
            json.dump(job, file, indent=2)
        os.replace(temporary, target)

    def _claim_shared_lock(self, job_id: str) -> None:
        """Claim one retraining slot shared by all container replicas."""

        for _ in range(2):
            try:
                descriptor = os.open(
                    self.active_lock_path,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                )
                with os.fdopen(descriptor, "w", encoding="utf-8") as file:
                    file.write(job_id)
                return
            except FileExistsError:
                existing_id = self.active_lock_path.read_text(
                    encoding="utf-8"
                ).strip()
                if not existing_id:
                    raise RetrainingError(
                        "Another replica is acquiring the retraining job lock."
                    )
                try:
                    existing = self.get(existing_id)
                except KeyError:
                    existing = {"status": "stale"}
                if existing.get("status") in {"queued", "running"}:
                    raise RetrainingError(
                        f"Retraining job {existing_id} is already active."
                    )
                self.active_lock_path.unlink(missing_ok=True)
        raise RetrainingError("Could not acquire the retraining job lock.")

    def _release_shared_lock(self, job_id: str) -> None:
        try:
            current_id = self.active_lock_path.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            return
        if current_id == job_id:
            self.active_lock_path.unlink(missing_ok=True)

    def _run(self, job_id: str) -> None:
        job = self.get(job_id)
        job.update(status="running", started_at=utc_now())
        self._write(job)
        try:
            metrics, model_version = self._train(job_id)
            job.update(
                status="completed",
                completed_at=utc_now(),
                metrics=metrics,
                model_version=model_version,
            )
        except Exception as exc:  # status must capture background failures
            job.update(
                status="failed",
                completed_at=utc_now(),
                error=f"{type(exc).__name__}: {exc}",
            )
            traceback.print_exc()
        finally:
            self._write(job)
            with self._lock:
                if self._active_job_id == job_id:
                    self._active_job_id = None
            self._release_shared_lock(job_id)

    def _train(self, job_id: str) -> tuple[dict, str]:
        import tensorflow as tf

        if not self.training_model_path.is_file():
            raise FileNotFoundError(
                f"Training model not found: {self.training_model_path}"
            )

        model = tf.keras.models.load_model(
            str(self.training_model_path), compile=False
        )
        base_model = model.get_layer("mobilenetv2_1.00_224")
        base_model.trainable = True
        for layer in base_model.layers[:-20]:
            layer.trainable = False

        dataset = tf.keras.utils.image_dataset_from_directory(
            self.dataset_store.root,
            labels="inferred",
            label_mode="int",
            class_names=self.dataset_store.class_names,
            image_size=(224, 224),
            batch_size=16,
            shuffle=True,
            seed=42,
        ).prefetch(tf.data.AUTOTUNE)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"],
        )
        history = model.fit(dataset, epochs=self.epochs, verbose=2)

        temporary_model = self.promoted_model_path.with_name(
            f".{job_id}.keras"
        )
        model.save(str(temporary_model))
        os.replace(temporary_model, self.promoted_model_path)

        model_version = f"1.0.{self.promoted_model_path.stat().st_mtime_ns}"
        if self.on_promoted is not None:
            self.on_promoted(self.promoted_model_path, model_version)

        metrics = {
            name: round(float(values[-1]), 6)
            for name, values in history.history.items()
        }
        return metrics, model_version
