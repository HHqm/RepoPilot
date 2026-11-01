import os
import tempfile
import time
import unittest
from pathlib import Path

from repopilot.config import Settings
from repopilot.rag import HybridRAG, SearchOptions
from repopilot.rag.chunking import StructureAwareChunker
from repopilot.rag.embedding_cache import CachedHybridEmbedding, EmbeddingCache
from repopilot.rag.embeddings import (
    DeterministicHybridEmbedding,
    DeterministicReranker,
    lexical_terms,
)
from repopilot.rag.parsers import DocumentLoader
from repopilot.rag.store import QdrantVectorStore


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = PROJECT_ROOT / "examples" / "buggy_calculator"


class CountingEmbedding(DeterministicHybridEmbedding):
    """Deterministic provider that records every text sent to the "model"."""

    def __init__(self):
        self.encoded: list[str] = []

    def encode_documents(self, texts):
        self.encoded.extend(texts)
        return super().encode_documents(texts)


def _settings(root: Path) -> Settings:
    return Settings(
        deepseek_api_key=None,
        deepseek_base_url="https://api.deepseek.com",
        deepseek_model="deepseek-v4-pro",
        llm_mode="mock",
        max_files=100,
        max_file_bytes=200_000,
        rag_top_k=6,
        test_timeout_seconds=30,
        runtime_dir=root / "runtime",
        rag_embedding_provider="deterministic",
        rag_reranker_provider="deterministic",
        rag_document_parser="text-only",
    )


def _write_module(path: Path, functions: int, marker: str = "") -> None:
    body = []
    for index in range(functions):
        body.append(
            f"def handler_{index}(value):\n"
            f"    '''Handle request type {index}{marker if index == 0 else ''}.'''\n"
            f"    return value * {index}\n"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n\n".join(body), encoding="utf-8")


def _age(path: Path, seconds: float = 60) -> None:
    past = time.time() - seconds
    os.utime(path, (past, past))


class LexicalTermsTests(unittest.TestCase):
    def test_identifiers_are_split_and_kept_whole(self):
        terms = lexical_terms("safe_divide raises ValueError")
        for term in ("safe_divide", "safe", "divide", "valueerror", "value", "error"):
            self.assertIn(term, terms)

    def test_camel_case_keeps_acronyms_and_folds_plurals(self):
        terms = lexical_terms("HTTPServerError versions retries")
        for term in ("http", "server", "error", "version", "retry"):
            self.assertIn(term, terms)
        self.assertNotIn("versio", terms)

    def test_cjk_bigrams_are_preserved(self):
        terms = lexical_terms("删除旧版本")
        self.assertIn("旧版", terms)
        self.assertIn("版本", terms)


class SearchOptionsTests(unittest.TestCase):
    def test_options_reject_empty_recall_and_unknown_fusion(self):
        with self.assertRaises(ValueError):
            SearchOptions(dense=False, sparse=False)
        with self.assertRaises(ValueError):
            SearchOptions(fusion="borda")

    def test_labels_describe_the_pipeline(self):
        self.assertEqual(SearchOptions().label, "hybrid_rrf+rerank+diversify")
        self.assertEqual(
            SearchOptions(dense=False, rerank=False, diversify=False).label,
            "sparse",
        )


class RagOptimizationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.settings = _settings(self.root)
        self.rag = HybridRAG.from_settings(self.settings)

    def tearDown(self):
        self.rag.close()
        self.temporary.cleanup()

    def test_every_ablation_configuration_returns_stats(self):
        index = self.rag.index_repository(EXAMPLE)
        for options in (
            SearchOptions(),
            SearchOptions(sparse=False, rerank=False, diversify=False),
            SearchOptions(dense=False, rerank=False, diversify=False),
            SearchOptions(fusion="dbsf", rerank=False),
        ):
            evidence, stats = self.rag.search_with_stats(
                index, "safe_divide zero divisor", top_k=3, options=options
            )
            self.assertTrue(evidence, options.label)
            self.assertEqual(stats.options, options.label)
            self.assertEqual(stats.returned, len(evidence))
            self.assertGreaterEqual(stats.candidates, stats.returned)
            self.assertGreaterEqual(stats.total_ms, stats.recall_ms)
            if not options.rerank:
                self.assertEqual(stats.rerank_ms, 0.0)

    def test_natural_language_matches_snake_case_identifier(self):
        repository = self.root / "identifiers"
        (repository / "billing.py").parent.mkdir(parents=True)
        (repository / "billing.py").write_text(
            "def apply_refund_policy(order):\n    return order.total\n",
            encoding="utf-8",
        )
        (repository / "shipping.py").write_text(
            "def schedule_courier(order):\n    return order.address\n",
            encoding="utf-8",
        )
        index = self.rag.index_repository(repository)
        evidence = self.rag.search(
            index,
            "refund policy",
            top_k=1,
            options=SearchOptions(dense=False, rerank=False),
        )
        self.assertEqual(evidence[0].path, "billing.py")

    def test_unchanged_files_are_reused_from_stat_without_reading(self):
        repository = self.root / "stat-repo"
        for name in ("a.py", "b.py", "c.py"):
            _write_module(repository / name, 2)
            _age(repository / name)
        first = self.rag.index_repository(repository)
        self.assertEqual(first.stat_reused_files, 0)
        second = self.rag.index_repository(repository)
        self.assertEqual(second.stat_reused_files, 3)
        self.assertEqual(second.indexed_chunks, 0)

        # A real edit moves the mtime, so the file is re-read and re-indexed.
        _write_module(repository / "b.py", 2, marker=" (edited)")
        _age(repository / "b.py", seconds=30)
        third = self.rag.index_repository(repository)
        self.assertEqual(third.stat_reused_files, 2)
        self.assertGreater(third.indexed_chunks, 0)

    def test_recently_modified_files_are_never_trusted_by_stat(self):
        repository = self.root / "racy-repo"
        _write_module(repository / "fresh.py", 1)
        self.rag.index_repository(repository)
        again = self.rag.index_repository(repository)
        self.assertEqual(again.stat_reused_files, 0)
        self.assertEqual(again.reused_files, 1)

    def test_lost_vectors_are_detected_and_rebuilt_after_restart(self):
        first = self.rag.index_repository(EXAMPLE)
        self.rag.store.delete_repository(first.repo_id)
        self.rag.close()
        self.rag = HybridRAG.from_settings(self.settings)
        healed = self.rag.index_repository(EXAMPLE)
        self.assertTrue(healed.rebuilt)
        self.assertEqual(healed.indexed_chunks, first.chunk_count)
        results = self.rag.search(healed, "safe_divide", top_k=3)
        self.assertIn("src/calculator.py", {item.path for item in results})


class EmbeddingCacheTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def cached(self, max_entries: int = 1000):
        inner = CountingEmbedding()
        cache = EmbeddingCache(self.root / "cache.sqlite3", max_entries=max_entries)
        return inner, CachedHybridEmbedding(inner, cache)

    def test_repeated_texts_skip_the_model_and_round_trip_vectors(self):
        inner, cached = self.cached()
        try:
            first = cached.encode_documents(["alpha beta", "gamma"])
            second = cached.encode_documents(["gamma", "alpha beta", "delta"])
        finally:
            cached.close()
        self.assertEqual(inner.encoded, ["alpha beta", "gamma", "delta"])
        self.assertEqual((cached.hits, cached.misses), (2, 3))
        for left, right in zip(first.dense[0], second.dense[1], strict=True):
            self.assertAlmostEqual(left, right, places=6)
        self.assertEqual(first.sparse[1].indices, second.sparse[0].indices)

    def test_cache_is_bounded_by_least_recent_use(self):
        inner, cached = self.cached(max_entries=2)
        try:
            cached.encode_documents(["one", "two"])
            cached.encode_documents(["one"])  # refresh "one"
            cached.encode_documents(["three"])  # evicts "two"
            self.assertEqual(len(cached.cache), 2)
            inner.encoded.clear()
            cached.encode_documents(["one", "two"])
        finally:
            cached.close()
        self.assertEqual(inner.encoded, ["two"])

    def test_editing_one_function_only_re_embeds_changed_chunks(self):
        settings = _settings(self.root)
        inner, cached = self.cached()
        store = QdrantVectorStore(
            embedding=cached,
            reranker=DeterministicReranker(),
            local_path=self.root / "qdrant",
        )
        rag = HybridRAG(
            settings=settings,
            store=store,
            loader=DocumentLoader("text-only", None),
            chunker=StructureAwareChunker(64, 8),
        )
        repository = self.root / "repo"
        _write_module(repository / "service.py", 12)
        try:
            first = rag.index_repository(repository)
            self.assertGreaterEqual(first.indexed_chunks, 12)
            inner.encoded.clear()
            _write_module(repository / "service.py", 12, marker=" (edited)")
            second = rag.index_repository(repository)
        finally:
            rag.close()
        # Every chunk of the edited file is upserted again (chunk IDs encode
        # position), but only the one with new text reaches the model.
        self.assertEqual(second.indexed_chunks, first.indexed_chunks)
        self.assertEqual(len(inner.encoded), 1)
        self.assertIn("(edited)", inner.encoded[0])


if __name__ == "__main__":
    unittest.main()
