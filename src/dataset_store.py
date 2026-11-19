"""Validated, deduplicated storage for newly labelled training images."""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path

from src.preprocessing import validate_image_bytes


class DatasetStoreError(ValueError):
    """Raised when an uploaded labelled batch cannot be stored safely."""


class DatasetStore:
    def __init__(self, root: Path, class_names: list[str]) -> None:
        self.root = root
        self.class_names = class_names
        self.root.mkdir(parents=True, exist_ok=True)
        for class_name in class_names:
            (self.root / class_name).mkdir(parents=True, exist_ok=True)

    def validate_label(self, label: str) -> None:
        if label not in self.class_names:
            allowed = ", ".join(self.class_names)
            raise DatasetStoreError(f"Unknown label '{label}'. Allowed: {allowed}.")

    def add_batch(self, label: str, uploads: list[tuple[str, bytes]]) -> dict:
        """Validate the entire batch, then atomically store unique images."""

        self.validate_label(label)
        if not uploads:
            raise DatasetStoreError("Upload at least one image.")

        validated: list[tuple[str, bytes, str, str]] = []
        seen_in_batch: set[str] = set()
        for original_name, content in uploads:
            extension = validate_image_bytes(content)
            digest = hashlib.sha256(content).hexdigest()
            if digest in seen_in_batch:
                continue
            seen_in_batch.add(digest)
            validated.append((original_name, content, extension, digest))

        class_dir = self.root / label
        # Check all label conflicts before writing anything, so a rejected batch
        # cannot leave a partial set of files behind.
        for original_name, _, _, digest in validated:
            conflicting = [
                path
                for other_label in self.class_names
                if other_label != label
                for path in (self.root / other_label).glob(f"{digest}.*")
            ]
            if conflicting:
                raise DatasetStoreError(
                    f"'{original_name}' was previously uploaded with label "
                    f"'{conflicting[0].parent.name}'."
                )

        stored: list[str] = []
        duplicates: list[str] = []
        for original_name, content, extension, digest in validated:
            existing = list(class_dir.glob(f"{digest}.*"))
            if existing:
                duplicates.append(existing[0].name)
                continue

            target = class_dir / f"{digest}{extension}"
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{digest}-", dir=class_dir
            )
            try:
                with os.fdopen(descriptor, "wb") as temporary_file:
                    temporary_file.write(content)
                    temporary_file.flush()
                    os.fsync(temporary_file.fileno())
                os.replace(temporary_name, target)
            except Exception:
                Path(temporary_name).unlink(missing_ok=True)
                raise
            stored.append(target.name)

        return {
            "label": label,
            "received": len(uploads),
            "stored": len(stored),
            "duplicates": len(uploads) - len(stored),
            "stored_files": stored,
            "class_counts": self.class_counts(),
            "total_images": self.total_images(),
        }

    def class_counts(self) -> dict[str, int]:
        extensions = {".jpg", ".jpeg", ".png", ".webp"}
        return {
            class_name: sum(
                1
                for path in (self.root / class_name).iterdir()
                if path.is_file() and path.suffix.lower() in extensions
            )
            for class_name in self.class_names
        }

    def total_images(self) -> int:
        return sum(self.class_counts().values())
