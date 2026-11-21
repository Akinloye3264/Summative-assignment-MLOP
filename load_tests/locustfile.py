"""Measure multipart prediction requests against the running taro API."""

import os
from itertools import cycle
from pathlib import Path

from locust import HttpUser, between, task


IMAGE_TYPES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}
sample_override = os.getenv("LOCUST_IMAGE")
sample_paths = (
    [Path(sample_override)]
    if sample_override
    else sorted(Path(__file__).with_name("samples").glob("*/*.jpg"))
)
if not sample_paths:
    raise FileNotFoundError("No benchmark images found in load_tests/samples.")

samples = cycle(
    [(path.name, path.read_bytes(), IMAGE_TYPES[path.suffix.lower()]) for path in sample_paths]
)


class PredictionUser(HttpUser):
    wait_time = between(0.25, 0.75)

    @task
    def predict_leaf(self):
        files = {"file": next(samples)}
        with self.client.post("/predict", files=files, timeout=60, catch_response=True) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if not isinstance(data, dict) or not data.get("predicted_class"):
                        response.failure("Response is missing a predicted class")
                except ValueError:
                    response.failure("Response is not JSON")
