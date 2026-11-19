"""Image validation and preprocessing shared by API inference and retraining."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
MAX_IMAGE_BYTES = 10 * 1024 * 1024
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}


class InvalidImageError(ValueError):
    """Raised when an uploaded file is not a supported, decodable image."""


def _open_image(image_bytes: bytes) -> Image.Image:
    if not image_bytes:
        raise InvalidImageError("The uploaded image is empty.")
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise InvalidImageError("The image exceeds the 10 MB upload limit.")

    try:
        with Image.open(BytesIO(image_bytes)) as source:
            if source.format not in ALLOWED_FORMATS:
                supported = ", ".join(sorted(ALLOWED_FORMATS))
                raise InvalidImageError(
                    f"Unsupported image format. Use one of: {supported}."
                )
            source.verify()

        with Image.open(BytesIO(image_bytes)) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            image.load()
            return image.copy()
    except InvalidImageError:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InvalidImageError("The uploaded file is not a valid image.") from exc


def resize_with_padding(
    image: Image.Image,
    size: tuple[int, int] = (IMAGE_WIDTH, IMAGE_HEIGHT),
) -> Image.Image:
    """Resize while preserving aspect ratio, then center on a black canvas."""

    contained = ImageOps.contain(image, size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, color=(0, 0, 0))
    left = (size[0] - contained.width) // 2
    top = (size[1] - contained.height) // 2
    canvas.paste(contained, (left, top))
    return canvas


def preprocess_image_bytes(image_bytes: bytes) -> np.ndarray:
    """Return one float32 RGB batch; normalization is embedded in the model."""

    image = resize_with_padding(_open_image(image_bytes))
    array = np.asarray(image, dtype=np.float32)
    return np.expand_dims(array, axis=0)


def validate_image_bytes(image_bytes: bytes) -> str:
    """Validate an upload and return its detected lowercase file extension."""

    image = _open_image(image_bytes)
    detected_format = image.format
    # ``convert`` in _open_image removes format metadata, so inspect the source.
    with Image.open(BytesIO(image_bytes)) as source:
        detected_format = source.format
    extensions = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}
    return extensions[str(detected_format)]


def preprocess_image_file(path: str | Path) -> np.ndarray:
    """Preprocess an image stored on disk."""

    return preprocess_image_bytes(Path(path).read_bytes())
