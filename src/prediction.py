"""Model loading and prediction logic for the Taro Leaf Blight classifier."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.preprocessing import preprocess_image_bytes


class PredictionService:
    """Load the verified inference model once and serve thread-safe predictions."""

    def __init__(
        self,
        model_path: Path,
        class_names_path: Path,
        model_version: str = "unknown",
    ) -> None:
        self.model_path = model_path
        self.class_names_path = class_names_path
        self.model: Any | None = None
        self.class_names: list[str] = []
        self.loaded_at: float | None = None
        self.model_version = model_version
        self._lock = threading.Lock()
        self._load_lock = threading.Lock()
        self._source_mtime_ns: int | None = None

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def load(
        self,
        model_path: Path | None = None,
        model_version: str | None = None,
    ) -> None:
        selected_path = model_path or self.model_path
        if not selected_path.is_file():
            raise FileNotFoundError(f"Model not found: {selected_path}")
        if not self.class_names_path.is_file():
            raise FileNotFoundError(
                f"Class names not found: {self.class_names_path}"
            )

        with self.class_names_path.open(encoding="utf-8") as file:
            class_names = json.load(file)
        if not isinstance(class_names, list) or not all(
            isinstance(name, str) for name in class_names
        ):
            raise ValueError("class_names.json must contain a list of strings.")

        with self._load_lock:
            import tensorflow as tf

            model = tf.keras.models.load_model(
                str(selected_path),
                compile=False,
            )
            output_classes = int(model.output_shape[-1])
            if output_classes != len(class_names):
                raise ValueError(
                    "Model output count does not match class_names.json."
                )

            model.predict(
                np.zeros((1, 224, 224, 3), dtype=np.float32), verbose=0
            )
            with self._lock:
                self.model = model
                self.class_names = class_names
                self.model_path = selected_path
                if model_version is not None:
                    self.model_version = model_version
                self.loaded_at = time.time()
                self._source_mtime_ns = selected_path.stat().st_mtime_ns

    def load_if_changed(self, model_path: Path) -> bool:
        """Load a promoted shared model once per replica when it changes."""

        if not model_path.is_file():
            return False
        modified = model_path.stat().st_mtime_ns
        if self.model_path == model_path and self._source_mtime_ns == modified:
            return False
        version = f"1.0.{modified}"
        self.load(model_path=model_path, model_version=version)
        return True

    def predict(self, image_bytes: bytes) -> dict[str, Any]:
        if self.model is None:
            raise RuntimeError("The prediction model is not loaded.")

        batch = preprocess_image_bytes(image_bytes)
        started = time.perf_counter()
        with self._lock:
            raw_output = self.model.predict(batch, verbose=0)[0]
        inference_ms = (time.perf_counter() - started) * 1000

        probabilities = np.asarray(raw_output, dtype=np.float64)
        predicted_index = int(np.argmax(probabilities))
        probability_map = {
            name: round(float(probability), 6)
            for name, probability in zip(self.class_names, probabilities)
        }

        return {
            "predicted_class": self.class_names[predicted_index],
            "display_class": self.class_names[predicted_index].replace("_", " "),
            "confidence": round(float(probabilities[predicted_index]), 6),
            "probabilities": probability_map,
            "inference_ms": round(inference_ms, 2),
        }
