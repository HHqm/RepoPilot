from __future__ import annotations

import time
from collections.abc import Sequence
from hashlib import sha256
from pathlib import Path
from threading import RLock
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient, models

from .embeddings import HybridEmbeddingProvider, Reranker
from .types import (
    RAG_SCHEMA_VERSION,
    DocumentChunk,
    RetrievalStats,
    SearchCandidate,
    SearchOptions,
)


_FUSIONS = {"rrf": models.Fusion.RRF, "dbsf": models.Fusion.DBSF}


class QdrantVectorStore:
    DENSE_VECTOR = "dense"
    SPARSE_VECTOR = "sparse"

    def __init__(
        self,
        *,
        embedding: HybridEmbeddingProvider,
        reranker: Reranker,
        url: str | None = None,
        api_key: str | None = None,
        local_path: Path | None = None,
        timeout_seconds: int = 30,
        candidate_k: int = 32,
        score_threshold: float = 0.0,
        max_chunks_per_document: int = 3,
    ):
        self.embedding = embedding
        self.reranker = reranker
        self.candidate_k = max(2, candidate_k)
        self.score_threshold = score_threshold
        self.max_chunks_per_document = max(1, max_chunks_per_document)
        self._lock = RLock()
        if url:
            self.client = QdrantClient(
                url=url,
                api_key=api_key,
                timeout=timeout_seconds,
                check_compatibility=True,
            )
            self.local_mode = False
        elif local_path is None:
            self.client = QdrantClient(":memory:")
            self.local_mode = True
        else:
            local_path.mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(
                path=str(local_path),
                force_disable_check_same_thread=True,
            )
            self.local_mode = True
        schema_fingerprint = sha256(
            (
                f"v{RAG_SCHEMA_VERSION}:{embedding.model_id}:"
                f"{embedding.sparse_model_id}:"
                f"{embedding.dimension}"
            ).encode("utf-8")
        ).hexdigest()[:12]
        self.collection_name = f"repopilot_chunks_{schema_fingerprint}"
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        with self._lock:
            if not self.client.collection_exists(self.collection_name):
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config={
                        self.DENSE_VECTOR: models.VectorParams(
                            size=self.embedding.dimension,
                            distance=models.Distance.COSINE,
                        )
                    },
                    sparse_vectors_config={
                        self.SPARSE_VECTOR: models.SparseVectorParams(
                            modifier=models.Modifier.IDF,
                        )
                    },
                    on_disk_payload=not self.local_mode,
                )
            if not self.local_mode:
                for field_name in (
                    "repo_id",
                    "document_id",
                    "document_version",
                    "source_type",
                    "language",
                    "trust_level",
                ):
                    self.client.create_payload_index(
                        collection_name=self.collection_name,
                        field_name=field_name,
                        field_schema=models.PayloadSchemaType.KEYWORD,
                        wait=True,
                    )

    def upsert(self, chunks: Sequence[DocumentChunk], batch_size: int = 16) -> None:
        for start in range(0, len(chunks), max(1, batch_size)):
            batch = chunks[start : start + batch_size]
            embeddings = self.embedding.encode_documents(
                [chunk.contextualized_text for chunk in batch]
            )
            points = []
            for chunk, dense, sparse in zip(
                batch,
                embeddings.dense,
                embeddings.sparse,
                strict=True,
            ):
                point_id = str(uuid5(NAMESPACE_URL, chunk.chunk_id))
                points.append(
                    models.PointStruct(
                        id=point_id,
                        vector={
                            self.DENSE_VECTOR: dense,
                            self.SPARSE_VECTOR: models.SparseVector(
                                indices=sparse.indices,
                                values=sparse.values,
                            ),
                        },
                        payload=chunk.payload(),
                    )
                )
            with self._lock:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points,
                    wait=True,
                )

    def delete_document(self, repo_id: str, document_id: str) -> None:
        self._delete_by_filter(
            models.Filter(
                must=[
                    self._match("repo_id", repo_id),
                    self._match("document_id", document_id),
                ]
            )
        )

    def delete_old_document_versions(
        self,
        repo_id: str,
        document_id: str,
        current_version: str,
    ) -> None:
        self._delete_by_filter(
            models.Filter(
                must=[
                    self._match("repo_id", repo_id),
                    self._match("document_id", document_id),
                ],
                must_not=[
                    self._match("document_version", current_version),
                ],
            )
        )

    def delete_repository(self, repo_id: str) -> None:
        self._delete_by_filter(
            models.Filter(must=[self._match("repo_id", repo_id)])
        )

    def _delete_by_filter(self, query_filter: models.Filter) -> None:
        with self._lock:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=query_filter,
                wait=True,
            )

    def count_repository(self, repo_id: str) -> int:
        with self._lock:
            return int(
                self.client.count(
                    collection_name=self.collection_name,
                    count_filter=models.Filter(
                        must=[self._match("repo_id", repo_id)]
                    ),
                    exact=True,
                ).count
            )

    def search(
        self,
        *,
        repo_id: str,
        query: str,
        top_k: int,
        source_types: Sequence[str] | None = None,
        options: SearchOptions | None = None,
    ) -> tuple[list[tuple[SearchCandidate, float]], RetrievalStats]:
        """Recall → (fusion) → rerank → diversify, timing every stage.

        Without reranking the fused (or single-retriever) score ranks results,
        and the reranker probability threshold no longer applies because the
        scores live on a different scale.
        """

        options = options or SearchOptions()
        started = time.perf_counter()
        dense, sparse = self.embedding.encode_query(query)
        embedded = time.perf_counter()
        conditions = [self._match("repo_id", repo_id)]
        if source_types:
            conditions.append(
                models.FieldCondition(
                    key="source_type",
                    match=models.MatchAny(any=list(source_types)),
                )
            )
        repository_filter = models.Filter(must=conditions)
        candidate_limit = max(top_k, self.candidate_k)
        recall_queries: list[tuple[str, object]] = []
        if options.dense:
            recall_queries.append((self.DENSE_VECTOR, dense))
        if options.sparse:
            recall_queries.append(
                (
                    self.SPARSE_VECTOR,
                    models.SparseVector(
                        indices=sparse.indices,
                        values=sparse.values,
                    ),
                )
            )
        with self._lock:
            if len(recall_queries) == 1:
                using, vector = recall_queries[0]
                response = self.client.query_points(
                    collection_name=self.collection_name,
                    query=vector,
                    using=using,
                    query_filter=repository_filter,
                    limit=candidate_limit,
                    with_payload=True,
                    with_vectors=False,
                )
            else:
                response = self.client.query_points(
                    collection_name=self.collection_name,
                    prefetch=[
                        models.Prefetch(
                            query=vector,
                            using=using,
                            filter=repository_filter,
                            limit=candidate_limit,
                        )
                        for using, vector in recall_queries
                    ],
                    query=models.FusionQuery(fusion=_FUSIONS[options.fusion]),
                    query_filter=repository_filter,
                    limit=candidate_limit,
                    with_payload=True,
                    with_vectors=False,
                )
        recalled = time.perf_counter()
        candidates = [
            SearchCandidate(
                point_id=str(point.id),
                fusion_score=float(point.score),
                payload=dict(point.payload or {}),
            )
            for point in response.points
        ]
        if options.rerank:
            scores = self.reranker.score(
                query,
                [
                    self._contextualized_payload(candidate.payload)
                    for candidate in candidates
                ],
            )
            threshold = self.score_threshold
            reranked = time.perf_counter()
        else:
            scores = [candidate.fusion_score for candidate in candidates]
            threshold = float("-inf")
            reranked = recalled
        ranked = sorted(
            zip(candidates, scores, strict=True),
            key=lambda item: (item[1], item[0].fusion_score),
            reverse=True,
        )
        per_document_limit = (
            self.max_chunks_per_document if options.diversify else top_k
        )
        selected: list[tuple[SearchCandidate, float]] = []
        document_counts: dict[str, int] = {}
        for item in ranked:
            candidate, score = item
            if score < threshold:
                continue
            document_id = str(candidate.payload.get("document_id", ""))
            count = document_counts.get(document_id, 0)
            if count >= per_document_limit:
                continue
            document_counts[document_id] = count + 1
            selected.append(item)
            if len(selected) >= top_k:
                break
        finished = time.perf_counter()
        stats = RetrievalStats(
            options=options.label,
            embed_ms=(embedded - started) * 1000,
            recall_ms=(recalled - embedded) * 1000,
            rerank_ms=(reranked - recalled) * 1000,
            total_ms=(finished - started) * 1000,
            candidates=len(candidates),
            returned=len(selected),
        )
        return selected, stats

    @staticmethod
    def _contextualized_payload(payload: dict) -> str:
        prefix = [f"path: {payload.get('path', '')}"]
        headings = payload.get("heading_path") or []
        if headings:
            prefix.append("section: " + " > ".join(str(item) for item in headings))
        if payload.get("symbol"):
            prefix.append(f"symbol: {payload['symbol']}")
        return "\n".join([*prefix, "", str(payload.get("text", ""))])

    @staticmethod
    def _match(key: str, value: str) -> models.FieldCondition:
        return models.FieldCondition(
            key=key,
            match=models.MatchValue(value=value),
        )

    def close(self) -> None:
        self.reranker.close()
        self.embedding.close()
        self.client.close()
