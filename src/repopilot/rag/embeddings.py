from __future__ import annotations

import math
import re
from collections import Counter
from hashlib import blake2b
from threading import RLock
from typing import Protocol, Sequence

from .types import HybridEmbeddingBatch, SparseEmbedding


TOKEN_PATTERN = re.compile(
    r"[A-Za-z_][A-Za-z0-9_]*|\d+(?:\.\d+)?|[\u4e00-\u9fff]"
)


def multilingual_tokens(text: str) -> list[str]:
    """Tokenize identifiers, numbers, and CJK text without language guessing."""

    lowered = text.lower()
    tokens = TOKEN_PATTERN.findall(lowered)
    chinese_runs = re.findall(r"[\u4e00-\u9fff]+", lowered)
    for run in chinese_runs:
        tokens.extend(run[index : index + 2] for index in range(len(run) - 1))
    return tokens


def _stable_index(token: str) -> int:
    return int.from_bytes(
        blake2b(token.encode("utf-8"), digest_size=4).digest(), "big"
    ) & 0x7FFFFFFF


class HybridEmbeddingProvider(Protocol):
    model_id: str
    sparse_model_id: str
    dimension: int

    def encode_documents(self, texts: Sequence[str]) -> HybridEmbeddingBatch: ...

    def encode_query(self, text: str) -> tuple[list[float], SparseEmbedding]: ...

    def close(self) -> None: ...


class Reranker(Protocol):
    model_id: str

    def score(self, query: str, passages: Sequence[str]) -> list[float]: ...

    def close(self) -> None: ...


class MultilingualSparseEncoder:
    """Qdrant-IDF-compatible sparse encoder for code and multilingual text.

    Stable token hashes make vectors reproducible across processes. Qdrant's
    ``Modifier.IDF`` supplies corpus-level inverse document frequency.
    """

    model_id = "repopilot/multilingual-code-sparse-v2"

    @staticmethod
    def encode(text: str) -> SparseEmbedding:
        frequencies = Counter(multilingual_tokens(text))
        by_index: dict[int, float] = {}
        for token, frequency in frequencies.items():
            index = _stable_index(token)
            by_index[index] = by_index.get(index, 0.0) + (
                1.0 + math.log(float(frequency))
            )
        indices = sorted(by_index)
        return SparseEmbedding(
            indices=indices,
            values=[by_index[index] for index in indices],
        )


class DeterministicHybridEmbedding:
    """Small deterministic provider reserved for tests and offline CI."""

    model_id = "repopilot/deterministic-dense-v2"
    sparse_model_id = MultilingualSparseEncoder.model_id
    dimension = 384

    @staticmethod
    def _dense(text: str) -> list[float]:
        values = [0.0] * DeterministicHybridEmbedding.dimension
        for token in multilingual_tokens(text):
            digest = blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % len(values)
            sign = 1.0 if digest[4] & 1 else -1.0
            values[index] += sign
        norm = math.sqrt(sum(value * value for value in values))
        if norm:
            values = [value / norm for value in values]
        return values

    def encode_documents(self, texts: Sequence[str]) -> HybridEmbeddingBatch:
        return HybridEmbeddingBatch(
            dense=[self._dense(text) for text in texts],
            sparse=[MultilingualSparseEncoder.encode(text) for text in texts],
        )

    def encode_query(self, text: str) -> tuple[list[float], SparseEmbedding]:
        return self._dense(text), MultilingualSparseEncoder.encode(text)

    def close(self) -> None:
        return None


class BgeM3HybridEmbedding:
    """BGE-M3 dense embeddings plus multilingual sparse lexical vectors."""

    sparse_model_id = MultilingualSparseEncoder.model_id
    dimension = 1024

    def __init__(
        self,
        model_name: str,
        *,
        device: str = "auto",
        batch_size: int = 8,
        cache_dir: str | None = None,
    ):
        self.model_id = model_name
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.cache_dir = cache_dir
        self._model = None
        self._lock = RLock()

    def _load(self):
        with self._lock:
            if self._model is not None:
                return self._model
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "BGE-M3 requires sentence-transformers. "
                    "Run: pip install -e ."
                ) from exc
            kwargs = {}
            if self.device != "auto":
                kwargs["device"] = self.device
            if self.cache_dir:
                kwargs["cache_folder"] = self.cache_dir
            self._model = SentenceTransformer(self.model_name, **kwargs)
            get_dimension = getattr(
                self._model,
                "get_embedding_dimension",
                None,
            )
            detected = (
                get_dimension()
                if get_dimension is not None
                else self._model.get_sentence_embedding_dimension()
            )
            if detected and int(detected) != self.dimension:
                raise RuntimeError(
                    f"Embedding dimension changed for {self.model_name}: "
                    f"expected {self.dimension}, got {detected}"
                )
            return self._model

    def _encode_dense(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        with self._lock:
            vectors = self._load().encode(
                list(texts),
                batch_size=self.batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        return [[float(value) for value in row] for row in vectors]

    def encode_documents(self, texts: Sequence[str]) -> HybridEmbeddingBatch:
        return HybridEmbeddingBatch(
            dense=self._encode_dense(texts),
            sparse=[MultilingualSparseEncoder.encode(text) for text in texts],
        )

    def encode_query(self, text: str) -> tuple[list[float], SparseEmbedding]:
        return self._encode_dense([text])[0], MultilingualSparseEncoder.encode(text)

    def close(self) -> None:
        with self._lock:
            self._model = None


class DeterministicReranker:
    model_id = "repopilot/deterministic-reranker-v2"

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        query_tokens = set(multilingual_tokens(query))
        if not query_tokens:
            return [0.0 for _ in passages]
        scores = []
        for passage in passages:
            passage_tokens = set(multilingual_tokens(passage))
            overlap = len(query_tokens & passage_tokens) / len(query_tokens)
            scores.append(float(overlap))
        return scores

    def close(self) -> None:
        return None


class BgeCrossEncoderReranker:
    def __init__(
        self,
        model_name: str,
        *,
        device: str = "auto",
        batch_size: int = 4,
        max_length: int = 1024,
        cache_dir: str | None = None,
    ):
        self.model_id = model_name
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.cache_dir = cache_dir
        self._tokenizer = None
        self._model = None
        self._runtime_device = None
        self._lock = RLock()

    def _load(self):
        with self._lock:
            if self._model is not None:
                return self._tokenizer, self._model, self._runtime_device
            try:
                import torch
                from transformers import (
                    AutoModelForSequenceClassification,
                    AutoTokenizer,
                )
            except ImportError as exc:
                raise RuntimeError(
                    "BGE reranking requires torch and transformers."
                ) from exc
            cache_kwargs = {"cache_dir": self.cache_dir} if self.cache_dir else {}
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, **cache_kwargs
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, **cache_kwargs
            )
            if self.device == "auto":
                runtime_device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                runtime_device = self.device
            model.to(runtime_device)
            model.eval()
            self._tokenizer = tokenizer
            self._model = model
            self._runtime_device = runtime_device
            return tokenizer, model, runtime_device

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        if not passages:
            return []
        import torch

        with self._lock:
            tokenizer, model, device = self._load()
            scores: list[float] = []
            for start in range(0, len(passages), self.batch_size):
                batch = passages[start : start + self.batch_size]
                pairs = [[query, passage] for passage in batch]
                inputs = tokenizer(
                    pairs,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                inputs = {
                    key: value.to(device) for key, value in inputs.items()
                }
                with torch.no_grad():
                    logits = (
                        model(**inputs, return_dict=True)
                        .logits.view(-1)
                        .float()
                    )
                    probabilities = torch.sigmoid(logits).cpu().tolist()
                scores.extend(float(value) for value in probabilities)
        return scores

    def close(self) -> None:
        with self._lock:
            self._tokenizer = None
            self._model = None
            self._runtime_device = None
