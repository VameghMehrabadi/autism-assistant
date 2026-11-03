"""Disk-backed embedding cache for the semantic labeling engine.

Design
------
- One cache file pair per embedding model: ``<slug>.npy`` (float32 matrix,
  one row per cached text) and ``<slug>.index.json`` (row order + metadata).
- Cache key = SHA-256 of (schema version, model name, prefix mode, role,
  max length, normalize flag, normalized text). Role matters because E5
  ``query:``/``passage:`` prefixes produce different vectors for the same
  text.
- Writes are atomic (temp file + ``os.replace``) and guarded by a lock, so
  Streamlit threads sharing one labeler cannot corrupt the cache.
- Any read failure (corrupt/truncated file, stale schema, dimension
  mismatch) degrades transparently: the corrupt files are set aside with a
  ``.corrupt`` suffix, the cache starts empty, and a warning is printed.
  The labeling pipeline never crashes because of the cache.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any

import numpy as np

import config

SCHEMA_VERSION = "emb-cache-v1"


class EmbeddingCache:
    """Content-addressed disk cache mapping texts to embedding vectors."""

    def __init__(
        self,
        model_name: str,
        prefix_mode: str,
        max_length: int = 0,
        normalize: bool = True,
        cache_dir: str | Path | None = None,
        enabled: bool = True,
    ):
        self.model_name = model_name
        self.prefix_mode = prefix_mode
        self.max_length = int(max_length or 0)
        self.normalize = bool(normalize)
        self.enabled = bool(enabled)
        self.cache_dir = Path(cache_dir) if cache_dir else config.EMBED_CACHE_DIR
        slug = "".join(c if c.isalnum() else "-" for c in model_name).strip("-")
        self._base = self.cache_dir / slug
        self._matrix_path = self._base.with_suffix(".npy")
        self._index_path = self._base.with_suffix(".index.json")
        self._lock = threading.RLock()
        self._hash_to_row: dict[str, int] = {}
        self._matrix: np.ndarray | None = None
        if self.enabled:
            self._load()

    # ------------------------------------------------------------------
    # Keying
    # ------------------------------------------------------------------

    def _hash(self, text: str, role: str) -> str:
        payload = json.dumps(
            [
                SCHEMA_VERSION,
                self.model_name,
                self.prefix_mode,
                role,
                self.max_length,
                self.normalize,
                text,
            ],
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        try:
            if not (self._matrix_path.exists() and self._index_path.exists()):
                return
            index: dict[str, Any] = json.loads(
                self._index_path.read_text(encoding="utf-8")
            )
            if (
                index.get("schema") != SCHEMA_VERSION
                or index.get("model") != self.model_name
            ):
                return  # stale cache from a different schema/model: rebuild
            matrix = np.load(self._matrix_path)
            hashes: list[str] = list(index["hashes"])
            if matrix.ndim != 2 or matrix.shape[0] != len(hashes):
                return
            self._matrix = matrix.astype(np.float32, copy=False)
            self._hash_to_row = {h: i for i, h in enumerate(hashes)}
        except Exception as exc:  # cache must never break the pipeline
            self._set_aside_corrupt(exc)

    def _set_aside_corrupt(self, exc: Exception) -> None:
        self._hash_to_row = {}
        self._matrix = None
        try:
            for path in (self._matrix_path, self._index_path):
                if path.exists():
                    path.rename(path.with_name(path.name + ".corrupt"))
        except OSError:
            pass
        print(f"[embed_cache] WARNING: cache unreadable ({exc}); rebuilding.")

    def _save(self) -> None:
        self._base.parent.mkdir(parents=True, exist_ok=True)
        tmp_matrix = self._matrix_path.with_name(self._matrix_path.name + ".tmp")
        tmp_index = self._index_path.with_name(self._index_path.name + ".tmp")
        with open(tmp_matrix, "wb") as handle:
            np.save(handle, self._matrix)
        tmp_index.write_text(
            json.dumps(
                {
                    "schema": SCHEMA_VERSION,
                    "model": self.model_name,
                    "prefix_mode": self.prefix_mode,
                    "max_length": self.max_length,
                    "normalize": self.normalize,
                    "dim": int(self._matrix.shape[1]) if self._matrix is not None else 0,
                    "hashes": list(self._hash_to_row.keys()),
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        os.replace(tmp_matrix, self._matrix_path)
        os.replace(tmp_index, self._index_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def lookup(
        self, texts: list[str], role: str
    ) -> tuple[list[int], np.ndarray | None, list[int]]:
        """Look up cached vectors for ``texts``.

        Returns ``(hit_positions, hit_matrix, missing_positions)`` where
        ``hit_matrix[i]`` is the vector for text ``hit_positions[i]``.
        With the cache disabled or empty, everything is reported missing.
        """
        if not self.enabled or not texts:
            return [], None, list(range(len(texts)))
        with self._lock:
            hits: list[tuple[int, int]] = []
            missing: list[int] = []
            for i, text in enumerate(texts):
                row = self._hash_to_row.get(self._hash(text, role))
                if row is None:
                    missing.append(i)
                else:
                    hits.append((i, row))
        if not hits:
            return [], None, list(range(len(texts)))
        hit_positions = [i for i, _ in hits]
        hit_matrix = self._matrix[[row for _, row in hits]].copy()
        return hit_positions, hit_matrix, missing

    def store(self, texts: list[str], vectors: np.ndarray, role: str) -> None:
        """Append vectors for ``texts`` to the cache and persist atomically."""
        if not self.enabled or not texts:
            return
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(texts):
            raise ValueError(
                f"vectors shape {vectors.shape} does not match {len(texts)} texts"
            )
        with self._lock:
            if self._matrix is None:
                self._matrix = np.zeros((0, vectors.shape[1]), dtype=np.float32)
            if vectors.shape[1] != self._matrix.shape[1]:
                raise ValueError(
                    "embedding dimension mismatch: cache has "
                    f"{self._matrix.shape[1]}, new vectors have {vectors.shape[1]}"
                )
            new_rows: list[tuple[str, np.ndarray]] = []
            for text, vector in zip(texts, vectors):
                key = self._hash(text, role)
                if key not in self._hash_to_row:
                    new_rows.append((key, vector))
            if not new_rows:
                return
            start = self._matrix.shape[0]
            self._matrix = np.vstack(
                [self._matrix, np.stack([v for _, v in new_rows])]
            )
            for offset, (key, _) in enumerate(new_rows):
                self._hash_to_row[key] = start + offset
            self._save()

    @property
    def size(self) -> int:
        """Number of cached vectors (0 when disabled)."""
        return len(self._hash_to_row) if self.enabled else 0
