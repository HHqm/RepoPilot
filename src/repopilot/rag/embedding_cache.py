"""Content-addressed embedding cache shared by every repository index.

Chunk IDs change whenever a file is edited (they encode position), so the
vector store alone cannot tell that most chunks of an edited file still have
identical text. Keying embeddings by ``sha256(model ids + chunk text)`` lets an
incremental re-index pay the model cost only for genuinely new text, and lets
a rebuilt collection (schema bump, repository moved, collection lost) reuse
every vector it has already computed.
"""

from __future__ import annotations

import sqlite3
import time
from array import array
from collections.abc import Sequence
from hashlib import sha256
from pathlib import Path
from threading import Lock

from .embeddings import HybridEmbeddingProvider
from .types import HybridEmbeddingBatch, SparseEmbedding


class EmbeddingCache:
    """SQLite store of float32 dense + sparse vectors with LRU-style pruning."""

    def __init__(self, path: Path, *, max_entries: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.max_entries = max(1, max_entries)
        self._lock = Lock()
        self._connection = sqlite3.connect(
            path,
            timeout=30,
            check_same_thread=False,
        )
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=NORMAL")
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                key TEXT PRIMARY KEY,
                dense BLOB NOT NULL,
                sparse_indices BLOB NOT NULL,
                sparse_values BLOB NOT NULL,
                used_at INTEGER NOT NULL
            )
            """
        )
        self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_embeddings_used_at "
            "ON embeddings (used_at)"
        )
        self._connection.commit()
        # Recency is a logical clock, not wall time: Windows clocks can tick
        # in milliseconds, and equal timestamps would make eviction arbitrary.
        (latest,) = self._connection.execute(
            "SELECT COALESCE(MAX(used_at), 0) FROM embeddings"
        ).fetchone()
        self._clock = int(latest)

    def _tick(self) -> int:
        self._clock = max(self._clock + 1, time.time_ns())
        return self._clock

    def get_many(
        self, keys: Sequence[str]
    ) -> dict[str, tuple[list[float], SparseEmbedding]]:
        if not keys:
            return {}
        found: dict[str, tuple[list[float], SparseEmbedding]] = {}
        with self._lock:
            # SQLite caps bound parameters; 500 stays well below every default.
            for start in range(0, len(keys), 500):
                batch = list(keys[start : start + 500])
                placeholders = ",".join("?" * len(batch))
                rows = self._connection.execute(
                    "SELECT key, dense, sparse_indices, sparse_values "
                    f"FROM embeddings WHERE key IN ({placeholders})",
                    batch,
                ).fetchall()
                for key, dense, indices, values in rows:
                    found[key] = (
                        array("f", dense).tolist(),
                        SparseEmbedding(
                            indices=array("I", indices).tolist(),
                            values=array("f", values).tolist(),
                        ),
                    )
                if rows:
                    self._connection.execute(
                        "UPDATE embeddings SET used_at = ? "
                        f"WHERE key IN ({','.join('?' * len(rows))})",
                        [self._tick(), *(row[0] for row in rows)],
                    )
            self._connection.commit()
        return found

    def put_many(
        self,
        items: Sequence[tuple[str, list[float], SparseEmbedding]],
    ) -> None:
        if not items:
            return
        with self._lock:
            now = self._tick()
            self._connection.executemany(
                """
                INSERT OR REPLACE INTO embeddings
                    (key, dense, sparse_indices, sparse_values, used_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (
                        key,
                        array("f", dense).tobytes(),
                        array("I", sparse.indices).tobytes(),
                        array("f", sparse.values).tobytes(),
                        now,
                    )
                    for key, dense, sparse in items
                ],
            )
            (count,) = self._connection.execute(
                "SELECT COUNT(*) FROM embeddings"
            ).fetchone()
            overflow = int(count) - self.max_entries
            if overflow > 0:
                self._connection.execute(
                    """
                    DELETE FROM embeddings WHERE key IN (
                        SELECT key FROM embeddings ORDER BY used_at LIMIT ?
                    )
                    """,
                    (overflow,),
                )
            self._connection.commit()

    def __len__(self) -> int:
        with self._lock:
            (count,) = self._connection.execute(
                "SELECT COUNT(*) FROM embeddings"
            ).fetchone()
        return int(count)

    def close(self) -> None:
        with self._lock:
            self._connection.close()


class CachedHybridEmbedding:
    """Decorator over a ``HybridEmbeddingProvider`` that skips cached texts.

    Queries are never cached: they are cheap single encodes and caching them
    would retain user questions on disk.
    """

    def __init__(
        self,
        inner: HybridEmbeddingProvider,
        cache: EmbeddingCache,
    ) -> None:
        self.inner = inner
        self.cache = cache
        self.model_id = inner.model_id
        self.sparse_model_id = inner.sparse_model_id
        self.dimension = inner.dimension
        self.hits = 0
        self.misses = 0
        self._counter_lock = Lock()

    def _key(self, text: str) -> str:
        return sha256(
            f"{self.model_id}\0{self.sparse_model_id}\0{text}".encode("utf-8")
        ).hexdigest()

    def encode_documents(self, texts: Sequence[str]) -> HybridEmbeddingBatch:
        keys = [self._key(text) for text in texts]
        cached = self.cache.get_many(list(dict.fromkeys(keys)))
        missing: dict[str, str] = {}
        for key, text in zip(keys, texts, strict=True):
            if key not in cached:
                missing.setdefault(key, text)
        if missing:
            fresh = self.inner.encode_documents(list(missing.values()))
            new_items = list(
                zip(missing, fresh.dense, fresh.sparse, strict=True)
            )
            self.cache.put_many(new_items)
            cached.update(
                {key: (dense, sparse) for key, dense, sparse in new_items}
            )
        with self._counter_lock:
            self.misses += len(missing)
            self.hits += len(texts) - len(missing)
        return HybridEmbeddingBatch(
            dense=[cached[key][0] for key in keys],
            sparse=[cached[key][1] for key in keys],
        )

    def encode_query(self, text: str) -> tuple[list[float], SparseEmbedding]:
        return self.inner.encode_query(text)

    def close(self) -> None:
        self.inner.close()
        self.cache.close()
