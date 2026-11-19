from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from src.preprocessing import InvalidImageError, preprocess_image_bytes


def create_image_bytes(size: tuple[int, int] = (400, 200)) -> bytes:
    image = Image.new("RGB", size, color=(20, 140, 60))
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def test_preprocess_returns_expected_batch() -> None:
    batch = preprocess_image_bytes(create_image_bytes())

    assert batch.shape == (1, 224, 224, 3)
    assert batch.dtype == np.float32
    assert 0 <= batch.min() <= batch.max() <= 255


def test_preprocess_rejects_non_image() -> None:
    with pytest.raises(InvalidImageError):
        preprocess_image_bytes(b"not an image")
