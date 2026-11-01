from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


RAG_SCHEMA_VERSION = 5


@dataclass(frozen=True)
class IndexedFile:
    path: str
    document_id: str
    sha256: str
    size: int
    language: str
    source_type: str
    parser: str
    symbols: tuple[str, ...] = ()
    chunk_count: int = 0
    redaction_count: int = 0
    # Last-modified time captured before hashing, or 0 when the timestamp was
    # too recent to trust (see HybridRAG._trusted_mtime_ns).
    mtime_ns: int = 0

    def to_manifest(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "document_id": self.document_id,
            "sha256": self.sha256,
            "size": self.size,
            "mtime_ns": self.mtime_ns,
            "language": self.language,
            "source_type": self.source_type,
            "parser": self.parser,
            "symbols": list(self.symbols),
            "chunk_count": self.chunk_count,
            "redaction_count": self.redaction_count,
        }

    @classmethod
    def from_manifest(cls, value: dict[str, Any]) -> IndexedFile:
        return cls(
            path=str(value["path"]),
            document_id=str(value["document_id"]),
            sha256=str(value["sha256"]),
            size=int(value["size"]),
            language=str(value["language"]),
            source_type=str(value["source_type"]),
            parser=str(value["parser"]),
            symbols=tuple(str(item) for item in value.get("symbols", [])),
            chunk_count=int(value.get("chunk_count", 0)),
            redaction_count=int(value.get("redaction_count", 0)),
            mtime_ns=int(value.get("mtime_ns", 0)),
        )


@dataclass(frozen=True)
class ParsedBlock:
    text: str
    start_line: int | None = None
    end_line: int | None = None
    page_start: int | None = None
    page_end: int | None = None
    heading_path: tuple[str, ...] = ()
    symbol: str | None = None


@dataclass(frozen=True)
class ParsedDocument:
    parser: str
    source_type: str
    language: str
    blocks: tuple[ParsedBlock, ...]
    symbols: tuple[str, ...] = ()


@dataclass(frozen=True)
class DocumentChunk:
    chunk_id: str
    document_id: str
    repo_id: str
    path: str
    text: str
    contextualized_text: str
    content_hash: str
    document_version: str
    source_type: str
    language: str
    parser: str
    chunk_index: int
    start_line: int | None = None
    end_line: int | None = None
    page_start: int | None = None
    page_end: int | None = None
    heading_path: tuple[str, ...] = ()
    symbol: str | None = None

    def payload(self) -> dict[str, Any]:
        normalized_path = self.path.replace("\\", "/").casefold()
        if normalized_path.startswith("client/"):
            trust_level = "untrusted_client"
        elif normalized_path.startswith("internal/"):
            trust_level = "workflow_generated"
        else:
            trust_level = "repository_source"
        return {
            "schema_version": RAG_SCHEMA_VERSION,
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "repo_id": self.repo_id,
            "path": self.path,
            "text": self.text,
            "content_hash": self.content_hash,
            "document_version": self.document_version,
            "trust_level": trust_level,
            "source_type": self.source_type,
            "language": self.language,
            "parser": self.parser,
            "chunk_index": self.chunk_index,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "heading_path": list(self.heading_path),
            "symbol": self.symbol,
        }


@dataclass(frozen=True)
class SparseEmbedding:
    indices: list[int]
    values: list[float]


@dataclass(frozen=True)
class HybridEmbeddingBatch:
    dense: list[list[float]]
    sparse: list[SparseEmbedding]


@dataclass(frozen=True)
class SearchCandidate:
    point_id: str
    fusion_score: float
    payload: dict[str, Any]


FUSION_METHODS = ("rrf", "dbsf")


@dataclass(frozen=True)
class SearchOptions:
    """Switches for each retrieval stage, used by production and ablations.

    The default is the production pipeline: dense + sparse recall, Reciprocal
    Rank Fusion, cross-encoder reranking and per-document diversification.
    """

    dense: bool = True
    sparse: bool = True
    fusion: str = "rrf"
    rerank: bool = True
    diversify: bool = True

    def __post_init__(self) -> None:
        if not (self.dense or self.sparse):
            raise ValueError("SearchOptions needs dense and/or sparse recall")
        if self.fusion not in FUSION_METHODS:
            raise ValueError(f"fusion must be one of {FUSION_METHODS}")

    @property
    def label(self) -> str:
        if self.dense and self.sparse:
            recall = f"hybrid_{self.fusion}"
        else:
            recall = "dense" if self.dense else "sparse"
        stages = [recall]
        if self.rerank:
            stages.append("rerank")
        if self.diversify:
            stages.append("diversify")
        return "+".join(stages)


@dataclass(frozen=True)
class RetrievalStats:
    """Per-query stage timings and funnel sizes for traces and evaluation."""

    options: str
    embed_ms: float
    recall_ms: float
    rerank_ms: float
    total_ms: float
    candidates: int
    returned: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "options": self.options,
            "embed_ms": round(self.embed_ms, 2),
            "recall_ms": round(self.recall_ms, 2),
            "rerank_ms": round(self.rerank_ms, 2),
            "total_ms": round(self.total_ms, 2),
            "candidates": self.candidates,
            "returned": self.returned,
        }


@dataclass
class IndexedRepository:
    root: Path
    repo_id: str
    collection_name: str
    files: dict[str, IndexedFile]
    chunk_count: int
    indexed_chunks: int
    reused_files: int
    deleted_files: int
    skipped_files: dict[str, str] = field(default_factory=dict)
    # Subset of reused_files proven unchanged by (size, mtime) without reading.
    stat_reused_files: int = 0
    rebuilt: bool = False

    def repo_map(self) -> list[dict[str, Any]]:
        return [
            {
                "path": item.path,
                "document_id": item.document_id,
                "sha256": item.sha256,
                "size": item.size,
                "language": item.language,
                "source_type": item.source_type,
                "parser": item.parser,
                "symbols": list(item.symbols),
                "chunks": item.chunk_count,
                "redactions": item.redaction_count,
            }
            for item in self.files.values()
        ]
