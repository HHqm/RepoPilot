"""Production-oriented retrieval for RepoPilot.

The public surface is intentionally small: ``HybridRAG`` owns ingestion,
incremental indexing, Qdrant storage, hybrid retrieval, and reranking.
"""

from .pipeline import HybridRAG
from .types import (
    DocumentChunk,
    IndexedFile,
    IndexedRepository,
    RetrievalStats,
    SearchOptions,
)


__all__ = [
    "DocumentChunk",
    "HybridRAG",
    "IndexedFile",
    "IndexedRepository",
    "RetrievalStats",
    "SearchOptions",
]
