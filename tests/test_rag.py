import hashlib
import shutil
import tempfile
import unittest
from itertools import pairwise
from pathlib import Path

from qdrant_client import models

from repopilot.config import Settings
from repopilot.rag import HybridRAG


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = PROJECT_ROOT / "examples" / "buggy_calculator"


def write_text_pdf(path: Path, text: str) -> None:
    escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream = f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET".encode("ascii")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>"
        ),
        (
            b"<< /Length "
            + str(len(stream)).encode("ascii")
            + b" >>\nstream\n"
            + stream
            + b"\nendstream"
        ),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    data = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"
    offsets = [0]
    for number, body in enumerate(objects, start=1):
        offsets.append(len(data))
        data += (
            f"{number} 0 obj\n".encode("ascii")
            + body
            + b"\nendobj\n"
        )
    xref_offset = len(data)
    data += f"xref\n0 {len(objects) + 1}\n".encode("ascii")
    data += b"0000000000 65535 f \n"
    for offset in offsets[1:]:
        data += f"{offset:010d} 00000 n \n".encode("ascii")
    data += (
        b"trailer\n<< /Size "
        + str(len(objects) + 1).encode("ascii")
        + b" /Root 1 0 R >>\nstartxref\n"
        + str(xref_offset).encode("ascii")
        + b"\n%%EOF\n"
    )
    path.write_bytes(data)


class HybridRagTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.settings = Settings(
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
        self.rag = HybridRAG.from_settings(self.settings)

    def tearDown(self):
        self.rag.close()
        self.temporary.cleanup()

    def test_retrieves_relevant_code_with_qdrant_rrf_and_reranking(self):
        index = self.rag.index_repository(EXAMPLE)
        results = self.rag.search(
            index,
            "safe_divide zero divisor ValueError",
            top_k=3,
        )
        self.assertTrue(results)
        self.assertEqual(results[0].path, "src/calculator.py")
        self.assertEqual(results[0].source_type, "code")
        self.assertGreaterEqual(results[0].start_line or 0, 1)
        self.assertTrue(results[0].chunk_id)
        self.assertGreaterEqual(results[0].fusion_score, 0)
        self.assertGreaterEqual(results[0].rerank_score, 0)

    def test_search_never_leaks_chunks_from_another_repository(self):
        first = Path(self.temporary.name) / "first-repository"
        second = Path(self.temporary.name) / "second-repository"
        first.mkdir()
        second.mkdir()
        (first / "only_first.py").write_text(
            "alpha_isolation_marker = 'belongs only to first repository'\n",
            encoding="utf-8",
        )
        (second / "only_second.py").write_text(
            "beta_isolation_marker = 'belongs only to second repository'\n",
            encoding="utf-8",
        )
        self.rag.index_repository(first)
        second_index = self.rag.index_repository(second)

        results = self.rag.search(
            second_index,
            "alpha_isolation_marker",
            top_k=6,
        )

        self.assertTrue(results)
        self.assertEqual(
            {item.path for item in results},
            {"only_second.py"},
        )

    def test_sensitive_files_are_not_indexed(self):
        index = self.rag.index_repository(EXAMPLE)
        self.assertNotIn(".env", index.files)

    def test_runtime_directories_are_never_indexed(self):
        root = Path(self.temporary.name) / "runtime-pollution"
        root.mkdir()
        (root / "README.md").write_text(
            "public scheduling documentation",
            encoding="utf-8",
        )
        hidden = root / ".repopilot-eval" / "rag" / "manifests"
        hidden.mkdir(parents=True)
        (hidden / "leak.json").write_text(
            '{"secret_benchmark_marker": true}',
            encoding="utf-8",
        )
        index = self.rag.index_repository(root)
        self.assertIn("README.md", index.files)
        self.assertFalse(
            any(path.startswith(".repopilot") for path in index.files)
        )

    def test_dependency_directories_are_pruned_at_any_depth(self):
        root = Path(self.temporary.name) / "dependency-pollution"
        (root / "src").mkdir(parents=True)
        (root / "src" / "app.py").write_text(
            "def handler():\n    return 'ok'\n",
            encoding="utf-8",
        )
        for ignored in (
            Path("node_modules") / "left-pad",
            Path("packages") / "web" / "node_modules" / "react",
            Path(".venv") / "lib",
            Path(".git") / "hooks",
        ):
            (root / ignored).mkdir(parents=True)
            (root / ignored / "index.js").write_text(
                "module.exports = 1;\n",
                encoding="utf-8",
            )
        index = self.rag.index_repository(root)
        self.assertEqual(sorted(index.files), ["src/app.py"])

    def test_chunk_windows_respect_budget_and_cover_every_line(self):
        from repopilot.rag.chunking import StructureAwareChunker
        from repopilot.rag.embeddings import multilingual_tokens

        chunker = StructureAwareChunker(max_tokens=64, overlap_tokens=16)
        lines = [
            f"value_{index} = compute(alpha, beta, gamma)  # 第{index}行"
            for index in range(200)
        ]
        blocks = chunker._split_lines(lines, 1)
        covered: set[int] = set()
        previous_start = 0
        for block in blocks:
            self.assertGreater(block.start_line, previous_start)
            previous_start = block.start_line
            block_lines = block.text.splitlines()
            covered.update(range(block.start_line, block.end_line + 1))
            tokens = sum(
                max(1, len(multilingual_tokens(line))) for line in block_lines
            )
            if len(block_lines) > 1:
                self.assertLessEqual(tokens, 64)
        self.assertEqual(covered, set(range(1, len(lines) + 1)))
        self.assertTrue(
            any(
                current.start_line <= previous.end_line
                for previous, current in pairwise(blocks)
            ),
            "adjacent windows should overlap",
        )

    def test_file_that_becomes_oversized_drops_old_vectors(self):
        root = Path(self.temporary.name) / "stale-vector"
        root.mkdir()
        target = root / "knowledge.txt"
        target.write_text("unique_stale_vector_marker", encoding="utf-8")
        first = self.rag.index_repository(root)
        self.assertTrue(
            self.rag.search(first, "unique_stale_vector_marker", top_k=3)
        )
        target.write_text(
            "unique_stale_vector_marker " * 20_000,
            encoding="utf-8",
        )
        with self.assertRaises(ValueError):
            self.rag.index_repository(root)
        self.assertEqual(
            self.rag.search(first, "unique_stale_vector_marker", top_k=3),
            [],
        )

    def test_second_index_is_incremental_and_reuses_unchanged_files(self):
        first = self.rag.index_repository(EXAMPLE)
        second = self.rag.index_repository(EXAMPLE)
        self.assertGreater(first.indexed_chunks, 0)
        self.assertEqual(second.indexed_chunks, 0)
        self.assertEqual(second.reused_files, len(second.files))
        self.assertEqual(first.chunk_count, second.chunk_count)

    def test_python_chunks_preserve_symbol_metadata(self):
        index = self.rag.index_repository(EXAMPLE)
        results = self.rag.search(index, "safe_divide", top_k=3)
        symbols = {item.symbol for item in results}
        self.assertIn("safe_divide", symbols)

    def test_persistent_qdrant_reuses_manifest_after_restart(self):
        first = self.rag.index_repository(EXAMPLE)
        self.rag.close()
        self.rag = HybridRAG.from_settings(self.settings)
        second = self.rag.index_repository(EXAMPLE)
        self.assertEqual(second.indexed_chunks, 0)
        self.assertEqual(second.reused_files, len(first.files))
        results = self.rag.search(second, "safe_divide", top_k=5)
        self.assertIn("src/calculator.py", {item.path for item in results})

    def test_text_pdf_is_ingested_with_page_provenance(self):
        repository = Path(self.temporary.name) / "pdf-repository"
        repository.mkdir()
        write_text_pdf(
            repository / "runbook.pdf",
            "Vector database incident recovery runbook",
        )
        index = self.rag.index_repository(repository)
        results = self.rag.search(
            index,
            "vector database incident recovery",
            top_k=1,
        )
        self.assertEqual(results[0].path, "runbook.pdf")
        self.assertEqual(results[0].page_start, 1)
        self.assertEqual(results[0].page_end, 1)
        self.assertEqual(results[0].source_type, "document")

    def test_document_size_limit_is_separate_from_source_code_limit(self):
        repository = Path(self.temporary.name) / "large-pdf-repository"
        repository.mkdir()
        target = repository / "handbook.pdf"
        write_text_pdf(target, "Enterprise retrieval operations handbook")
        target.write_bytes(
            target.read_bytes()
            + b"\n%"
            + b"padding" * 35_000
        )
        self.assertGreater(target.stat().st_size, self.settings.max_file_bytes)
        index = self.rag.index_repository(repository)
        self.assertIn("handbook.pdf", index.files)

    def test_changed_file_replaces_old_document_version(self):
        repository = Path(self.temporary.name) / "changed-repository"
        shutil.copytree(EXAMPLE, repository)
        first = self.rag.index_repository(repository)
        target = repository / "src" / "calculator.py"
        target.write_text(
            target.read_text(encoding="utf-8")
            + "\n# vector_version_marker\n",
            encoding="utf-8",
        )
        second = self.rag.index_repository(repository)
        document = second.files["src/calculator.py"]
        points, _ = self.rag.store.client.scroll(
            collection_name=second.collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="repo_id",
                        match=models.MatchValue(value=second.repo_id),
                    ),
                    models.FieldCondition(
                        key="document_id",
                        match=models.MatchValue(value=document.document_id),
                    ),
                ]
            ),
            limit=100,
            with_payload=True,
            with_vectors=False,
        )
        versions = {
            str(point.payload["document_version"])
            for point in points
        }
        expected = hashlib.sha256(target.read_bytes()).hexdigest()
        self.assertEqual(versions, {expected})
        self.assertGreater(second.indexed_chunks, 0)
        self.assertLess(second.reused_files, len(first.files))

    def test_secret_like_content_is_redacted_before_vector_storage(self):
        repository = Path(self.temporary.name) / "redaction-repository"
        repository.mkdir()
        (repository / "notes.md").write_text(
            "# Provider\n\napi_key=abcdefghijklmnop123456\n",
            encoding="utf-8",
        )
        index = self.rag.index_repository(repository)
        document = index.files["notes.md"]
        points, _ = self.rag.store.client.scroll(
            collection_name=index.collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id",
                        match=models.MatchValue(value=document.document_id),
                    )
                ]
            ),
            limit=10,
            with_payload=True,
            with_vectors=False,
        )
        stored_text = "\n".join(str(point.payload["text"]) for point in points)
        self.assertNotIn("abcdefghijklmnop123456", stored_text)
        self.assertIn("[REDACTED]", stored_text)
        self.assertEqual(document.redaction_count, 1)


if __name__ == "__main__":
    unittest.main()
