"""Validated, deduplicated SQLite storage for labelled training images."""

from __future__ import annotations

import hashlib
import sqlite3
from contextlib import closing
from pathlib import Path

from src.preprocessing import validate_image_bytes


class DatasetStoreError(ValueError):
    """Raised when an uploaded labelled batch cannot be stored safely."""


class DatasetStore:
    def __init__(self, root: Path, class_names: list[str]) -> None:
        self.root = root
        self.class_names = class_names
        root.mkdir(parents=True, exist_ok=True)
        self.database_path = root / "uploads.sqlite3"
        with closing(self._connect()) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """CREATE TABLE IF NOT EXISTS images (
                    digest TEXT PRIMARY KEY,
                    label TEXT NOT NULL,
                    extension TEXT NOT NULL,
                    content BLOB NOT NULL
                )"""
            )
            if connection.execute("PRAGMA user_version").fetchone()[0] == 0:
                for label in self.class_names:
                    for path in sorted((root / label).glob("*")):
                        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                            content = path.read_bytes()
                            extension = validate_image_bytes(content)
                            digest = hashlib.sha256(content).hexdigest()
                            self._store_image(connection, label, path.name, content, extension, digest)
                connection.execute("PRAGMA user_version = 1")

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.database_path, timeout=30)

    def _store_image(
        self, connection: sqlite3.Connection, label: str, original_name: str,
        content: bytes, extension: str, digest: str,
    ) -> bool:
        existing = connection.execute(
            "SELECT label FROM images WHERE digest = ?", (digest,)
        ).fetchone()
        if existing:
            if existing[0] != label:
                raise DatasetStoreError(
                    f"'{original_name}' was previously uploaded with label '{existing[0]}'."
                )
            return False
        connection.execute(
            "INSERT INTO images (digest, label, extension, content) VALUES (?, ?, ?, ?)",
            (digest, label, extension, content),
        )
        return True

    def validate_label(self, label: str) -> None:
        if label not in self.class_names:
            allowed = ", ".join(self.class_names)
            raise DatasetStoreError(f"Unknown label '{label}'. Allowed: {allowed}.")

    def add_batch(self, label: str, uploads: list[tuple[str, bytes]]) -> dict:
        """Validate the entire batch, then atomically store unique images."""

        self.validate_label(label)
        if not uploads:
            raise DatasetStoreError("Upload at least one image.")

        validated: dict[str, tuple[str, bytes, str]] = {}
        for original_name, content in uploads:
            extension = validate_image_bytes(content)
            digest = hashlib.sha256(content).hexdigest()
            validated.setdefault(digest, (original_name, content, extension))

        stored = []
        with closing(self._connect()) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            for digest, (name, content, extension) in validated.items():
                if self._store_image(connection, label, name, content, extension, digest):
                    stored.append(f"{digest}{extension}")
            counts = self._class_counts(connection)

        return {
            "label": label,
            "received": len(uploads),
            "stored": len(stored),
            "duplicates": len(uploads) - len(stored),
            "stored_files": stored,
            "class_counts": counts,
            "total_images": sum(counts.values()),
        }

    def image_ids(self) -> dict[str, list[str]]:
        images: dict[str, list[str]] = {name: [] for name in self.class_names}
        with closing(self._connect()) as connection:
            for digest, label in connection.execute("SELECT digest, label FROM images ORDER BY digest"):
                images[label].append(digest)
        return images

    def image_bytes(self, digest: str) -> bytes:
        with closing(self._connect()) as connection:
            record = connection.execute(
                "SELECT content FROM images WHERE digest = ?", (digest,)
            ).fetchone()
        if record is None:
            raise KeyError(digest)
        return record[0]

    def _class_counts(self, connection: sqlite3.Connection) -> dict[str, int]:
        counts = dict(connection.execute("SELECT label, COUNT(*) FROM images GROUP BY label"))
        return {name: counts.get(name, 0) for name in self.class_names}

    def class_counts(self) -> dict[str, int]:
        with closing(self._connect()) as connection:
            return self._class_counts(connection)
