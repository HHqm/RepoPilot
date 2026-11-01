from __future__ import annotations

import json
import os
import time
import weakref
from collections.abc import Iterable
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from threading import RLock
from typing import Any

from ..config import Settings
from ..models import Evidence
from .chunking import StructureAwareChunker
from .embedding_cache import CachedHybridEmbedding, EmbeddingCache
from .embeddings import (
    BgeCrossEncoderReranker,
    BgeM3HybridEmbedding,
    DeterministicHybridEmbedding,
    DeterministicReranker,
)
from .parsers import (
    ALLOWED_SUFFIXES,
    CODE_SUFFIXES,
    RICH_DOCUMENT_SUFFIXES,
    DocumentLoader,
    DocumentParsingError,
    generic_code_symbols,
    redact_document_secrets,
)
from .store import QdrantVectorStore
from .types import (
    RAG_SCHEMA_VERSION,
    IndexedFile,
    IndexedRepository,
    RetrievalStats,
    SearchOptions,
)


IGNORED_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    "dist",
    "build",
    "coverage",
    ".repopilot",
}
SENSITIVE_NAMES = {
    ".env",
    ".env.local",
    ".env.production",
    "id_rsa",
    "id_ed25519",
    "credentials",
    "credentials.json",
    "secrets.json",
}


class HybridRAG:
    """End-to-end incremental hybrid RAG with a persistent Qdrant index."""

    def __init__(
        self,
        *,
        settings: Settings,
        store: QdrantVectorStore,
        loader: DocumentLoader,
        chunker: StructureAwareChunker,
    ):
        self.settings = settings
        self.store = store
        self.loader = loader
        self.chunker = chunker
        self._manifest_dir = settings.runtime_dir / "rag" / "manifests"
        self._manifest_dir.mkdir(parents=True, exist_ok=True)
        self._lock_registry_guard = RLock()
        self._repository_locks: weakref.WeakValueDictionary[str, RLock] = (
            weakref.WeakValueDictionary()
        )
        self._embedding_lock = RLock()
        # Repositories whose manifest has been reconciled with the vector
        # store since this process started (see _index_locked).
        self._verified_repositories: set[str] = set()

    @classmethod
    def from_settings(cls, settings: Settings) -> HybridRAG:
        if settings.rag_embedding_provider == "deterministic":
            embedding = DeterministicHybridEmbedding()
        elif settings.rag_embedding_provider == "bge-m3":
            embedding = BgeM3HybridEmbedding(
                settings.rag_embedding_model,
                device=settings.rag_device,
                batch_size=settings.rag_embedding_batch_size,
                cache_dir=str(settings.rag_model_cache)
                if settings.rag_model_cache
                else None,
            )
            if settings.rag_embedding_cache_entries > 0:
                # The deterministic test provider is cheaper than a cache read.
                embedding = CachedHybridEmbedding(
                    embedding,
                    EmbeddingCache(
                        settings.runtime_dir / "rag" / "embedding_cache.sqlite3",
                        max_entries=settings.rag_embedding_cache_entries,
                    ),
                )
        else:
            raise ValueError(
                "REPOPILOT_RAG_EMBEDDING_PROVIDER must be "
                "'bge-m3' or 'deterministic'"
            )

        if settings.rag_reranker_provider == "deterministic":
            reranker = DeterministicReranker()
        elif settings.rag_reranker_provider == "bge":
            reranker = BgeCrossEncoderReranker(
                settings.rag_reranker_model,
                device=settings.rag_device,
                batch_size=settings.rag_reranker_batch_size,
                max_length=settings.rag_reranker_max_length,
                cache_dir=str(settings.rag_model_cache)
                if settings.rag_model_cache
                else None,
            )
        else:
            raise ValueError(
                "REPOPILOT_RAG_RERANKER_PROVIDER must be "
                "'bge' or 'deterministic'"
            )

        local_path = settings.rag_qdrant_path
        if local_path is None and not settings.rag_qdrant_url:
            local_path = settings.runtime_dir / "rag" / "qdrant"
        store = QdrantVectorStore(
            embedding=embedding,
            reranker=reranker,
            url=settings.rag_qdrant_url,
            api_key=settings.rag_qdrant_api_key,
            local_path=local_path,
            timeout_seconds=settings.rag_qdrant_timeout_seconds,
            candidate_k=settings.rag_candidate_k,
            score_threshold=settings.rag_score_threshold,
            max_chunks_per_document=settings.rag_max_chunks_per_document,
        )
        return cls(
            settings=settings,
            store=store,
            loader=DocumentLoader(
                settings.rag_document_parser,
                settings.rag_docling_artifacts_path,
            ),
            chunker=StructureAwareChunker(
                settings.rag_chunk_tokens,
                settings.rag_chunk_overlap_tokens,
            ),
        )

    def index_repository(self, repo_path: str | Path) -> IndexedRepository:
        root = Path(repo_path).resolve()
        if not root.is_dir():
            raise ValueError(f"Repository path is not a directory: {root}")
        repo_id = sha256(str(root).casefold().encode("utf-8")).hexdigest()[:24]
        with self._lock_registry_guard:
            repository_lock = self._repository_locks.setdefault(
                repo_id, RLock()
            )
        with repository_lock:
            return self._index_locked(root, repo_id)

    def _index_locked(self, root: Path, repo_id: str) -> IndexedRepository:
        manifest_path = self._manifest_dir / f"{repo_id}.json"
        previous = self._read_manifest(manifest_path)
        rebuilt = False
        if (
            previous.get("schema_version") != RAG_SCHEMA_VERSION
            or previous.get("collection_name") != self.store.collection_name
            or not self._manifest_matches_store(repo_id, previous)
        ):
            rebuilt = bool(previous)
            previous = {}
            self.store.delete_repository(repo_id)
        previous_files = dict(previous.get("files") or {})

        def invalidate_previous(relative_path: str) -> None:
            previous_file = previous_files.get(relative_path)
            if previous_file:
                self.store.delete_document(
                    repo_id,
                    str(previous_file["document_id"]),
                )

        candidates = sorted(
            self._candidate_files(root),
            key=self._candidate_priority,
        )
        skipped_files: dict[str, str] = {}
        if len(candidates) > self.settings.max_files:
            skipped_files["<file-limit>"] = (
                f"Indexed the first {self.settings.max_files} safe files; "
                f"skipped {len(candidates) - self.settings.max_files}."
            )
            candidates = candidates[: self.settings.max_files]

        files: dict[str, IndexedFile] = {}
        changed_chunks = []
        changed_versions: list[tuple[str, str]] = []
        reused_files = 0
        stat_reused_files = 0
        seen_paths: set[str] = set()
        for path in candidates:
            relative = path.relative_to(root).as_posix()
            seen_paths.add(relative)
            file_limit = (
                self.settings.rag_max_document_bytes
                if path.suffix.lower() in RICH_DOCUMENT_SUFFIXES
                else self.settings.max_file_bytes
            )
            previous_file = previous_files.get(relative)
            try:
                stat = path.stat()
                if stat.st_size > file_limit:
                    skipped_files[relative] = (
                        f"file exceeds {file_limit} bytes"
                    )
                    invalidate_previous(relative)
                    continue
                if (
                    previous_file
                    and int(previous_file.get("mtime_ns", 0)) != 0
                    and int(previous_file["mtime_ns"]) == stat.st_mtime_ns
                    and int(previous_file.get("size", -1)) == stat.st_size
                ):
                    # Same size and a trusted, unchanged mtime: skip reading
                    # and hashing, like git's stat-based index check.
                    files[relative] = IndexedFile.from_manifest(previous_file)
                    reused_files += 1
                    stat_reused_files += 1
                    continue
                mtime_ns = self._trusted_mtime_ns(stat.st_mtime_ns)
                raw = path.read_bytes()
            except OSError as exc:
                skipped_files[relative] = f"read failed: {exc}"
                invalidate_previous(relative)
                continue
            if len(raw) > file_limit:
                skipped_files[relative] = (
                    f"file exceeds {file_limit} bytes"
                )
                invalidate_previous(relative)
                continue
            digest = sha256(raw).hexdigest()
            if previous_file and previous_file.get("sha256") == digest:
                files[relative] = replace(
                    IndexedFile.from_manifest(previous_file),
                    mtime_ns=mtime_ns,
                )
                reused_files += 1
                continue

            document_id = sha256(
                f"{repo_id}:{relative}".encode("utf-8")
            ).hexdigest()[:32]
            try:
                parsed = self.loader.parse(
                    path,
                    raw,
                    file_limit,
                    self.settings.rag_max_document_pages,
                )
            except DocumentParsingError as exc:
                skipped_files[relative] = str(exc)
                invalidate_previous(relative)
                continue
            parsed, redaction_count = redact_document_secrets(parsed)
            symbols = parsed.symbols
            if not symbols and path.suffix.lower() in CODE_SUFFIXES:
                symbols = generic_code_symbols(parsed.blocks[0].text)
            indexed_file = IndexedFile(
                path=relative,
                document_id=document_id,
                sha256=digest,
                size=len(raw),
                language=parsed.language,
                source_type=parsed.source_type,
                parser=parsed.parser,
                symbols=symbols,
                redaction_count=redaction_count,
                mtime_ns=mtime_ns,
            )
            chunks = self.chunker.chunk(
                repo_id=repo_id,
                indexed_file=indexed_file,
                parsed=parsed,
            )
            if not chunks:
                skipped_files[relative] = "parser produced no non-empty chunks"
                invalidate_previous(relative)
                continue
            indexed_file = replace(indexed_file, chunk_count=len(chunks))
            files[relative] = indexed_file
            changed_chunks.extend(chunks)
            changed_versions.append((document_id, digest))

        deleted_paths = set(previous_files) - seen_paths
        for relative in deleted_paths:
            self.store.delete_document(
                repo_id,
                str(previous_files[relative]["document_id"]),
            )
        if changed_chunks:
            # Parsing and manifest work may proceed per repository; only the
            # shared embedding model is serialized because GPU/model runtimes
            # are commonly not re-entrant.
            with self._embedding_lock:
                self.store.upsert(
                    changed_chunks,
                    batch_size=self.settings.rag_embedding_batch_size,
                )
            for document_id, current_version in changed_versions:
                self.store.delete_old_document_versions(
                    repo_id,
                    document_id,
                    current_version,
                )
        if not files:
            detail = "; ".join(
                f"{path}: {reason}" for path, reason in skipped_files.items()
            )
            raise ValueError(
                "No indexable repository content was found"
                + (f" ({detail})" if detail else "")
            )

        manifest = {
            "schema_version": RAG_SCHEMA_VERSION,
            "repo_id": repo_id,
            "root": str(root),
            "collection_name": self.store.collection_name,
            "embedding_model": self.store.embedding.model_id,
            "sparse_model": self.store.embedding.sparse_model_id,
            "reranker_model": self.store.reranker.model_id,
            "files": {
                path: item.to_manifest() for path, item in files.items()
            },
        }
        self._write_manifest(manifest_path, manifest)
        self._verified_repositories.add(repo_id)
        return IndexedRepository(
            root=root,
            repo_id=repo_id,
            collection_name=self.store.collection_name,
            files=files,
            chunk_count=sum(item.chunk_count for item in files.values()),
            indexed_chunks=len(changed_chunks),
            reused_files=reused_files,
            deleted_files=len(deleted_paths),
            skipped_files=skipped_files,
            stat_reused_files=stat_reused_files,
            rebuilt=rebuilt,
        )

    # A file modified within this window of being hashed may change again
    # without moving its mtime (coarse filesystem timestamps), so its stat is
    # not trusted for the next run. Mirrors git's "racy clean" handling.
    RACY_MTIME_WINDOW_NS = 2_000_000_000

    @classmethod
    def _trusted_mtime_ns(cls, mtime_ns: int) -> int:
        return (
            mtime_ns
            if mtime_ns < time.time_ns() - cls.RACY_MTIME_WINDOW_NS
            else 0
        )

    def _manifest_matches_store(
        self, repo_id: str, manifest: dict[str, Any]
    ) -> bool:
        """Reconcile a manifest with Qdrant once per repository per process.

        A manifest can outlive its vectors (collection recreated, storage
        restored from an older backup, crash between upsert and manifest
        write). Trusting it would silently return empty search results, so
        the first index in each process compares chunk counts and rebuilds
        the repository on any mismatch.
        """

        if not manifest or repo_id in self._verified_repositories:
            return True
        expected = sum(
            int(item.get("chunk_count", 0))
            for item in (manifest.get("files") or {}).values()
        )
        return self.store.count_repository(repo_id) == expected

    def search(
        self,
        index: IndexedRepository,
        query: str,
        *,
        top_k: int,
        source_types: list[str] | None = None,
        options: SearchOptions | None = None,
    ) -> list[Evidence]:
        evidence, _ = self.search_with_stats(
            index,
            query,
            top_k=top_k,
            source_types=source_types,
            options=options,
        )
        return evidence

    def search_with_stats(
        self,
        index: IndexedRepository,
        query: str,
        *,
        top_k: int,
        source_types: list[str] | None = None,
        options: SearchOptions | None = None,
    ) -> tuple[list[Evidence], RetrievalStats]:
        if not query.strip():
            raise ValueError("RAG query must not be blank")
        with self._embedding_lock:
            ranked, stats = self.store.search(
                repo_id=index.repo_id,
                query=query,
                top_k=max(1, top_k),
                source_types=source_types,
                options=options,
            )
        evidence = []
        for rank, (candidate, rerank_score) in enumerate(ranked, start=1):
            payload = candidate.payload
            evidence.append(
                Evidence(
                    evidence_id=f"E{rank}",
                    chunk_id=str(payload["chunk_id"]),
                    document_id=str(payload["document_id"]),
                    path=str(payload["path"]),
                    text=str(payload["text"]),
                    source_type=str(payload.get("source_type", "document")),
                    trust_level=str(
                        payload.get("trust_level", "repository_source")
                    ),
                    document_version=(
                        str(payload["document_version"])
                        if payload.get("document_version")
                        else None
                    ),
                    language=str(payload.get("language", "text")),
                    start_line=self._optional_int(payload.get("start_line")),
                    end_line=self._optional_int(payload.get("end_line")),
                    page_start=self._optional_int(payload.get("page_start")),
                    page_end=self._optional_int(payload.get("page_end")),
                    heading_path=[
                        str(item) for item in payload.get("heading_path") or []
                    ],
                    symbol=(
                        str(payload["symbol"]) if payload.get("symbol") else None
                    ),
                    fusion_score=round(candidate.fusion_score, 6),
                    rerank_score=round(rerank_score, 6),
                    score=round(rerank_score, 6),
                )
            )
        return evidence, stats

    def close(self) -> None:
        self.store.close()

    @staticmethod
    def _optional_int(value: Any) -> int | None:
        return int(value) if value is not None else None

    @staticmethod
    def _read_manifest(path: Path) -> dict[str, Any]:
        if not path.is_file():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return {}

    @staticmethod
    def _write_manifest(path: Path, value: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(path)

    @staticmethod
    def _is_ignored_name(name: str) -> bool:
        folded = name.casefold()
        return folded in IGNORED_DIRS or folded.startswith(".repopilot")

    @classmethod
    def _candidate_files(cls, root: Path) -> Iterable[Path]:
        # Prune ignored directories while walking. ``rglob`` would descend
        # into node_modules/.git/.venv first and only then discard their
        # files, which dominates indexing time on real repositories. Symlinked
        # directories are not followed, so aliases cannot double-index files.
        for directory, dirnames, filenames in os.walk(root):
            dirnames[:] = [
                name for name in dirnames if not cls._is_ignored_name(name)
            ]
            parent = Path(directory)
            for filename in filenames:
                lowered = filename.lower()
                if (
                    cls._is_ignored_name(filename)
                    or lowered in SENSITIVE_NAMES
                    or lowered.startswith(".env.")
                ):
                    continue
                path = parent / filename
                if path.suffix.lower() not in ALLOWED_SUFFIXES:
                    continue
                try:
                    if not path.is_file():
                        continue
                    path.resolve().relative_to(root)
                except (OSError, ValueError):
                    continue
                yield path

    @staticmethod
    def _candidate_priority(path: Path) -> tuple[int, str]:
        suffix = path.suffix.lower()
        if suffix in CODE_SUFFIXES:
            priority = 0
        elif suffix in RICH_DOCUMENT_SUFFIXES:
            priority = 2
        else:
            priority = 1
        return priority, path.as_posix().casefold()
