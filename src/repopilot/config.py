from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    deepseek_api_key: str | None
    deepseek_base_url: str
    deepseek_model: str
    llm_mode: str
    max_files: int
    max_file_bytes: int
    rag_top_k: int
    test_timeout_seconds: int
    runtime_dir: Path
    max_cached_indices: int = 32
    max_workers: int = 4
    max_pending_runs: int = 16
    max_cached_sessions: int = 128
    allow_host_test_execution: bool = False
    allow_private_model_endpoints: bool = False
    allowed_workspace_roots: tuple[Path, ...] = ()
    api_token: str | None = None
    llm_max_output_tokens: int = 12_000
    rag_embedding_provider: str = "deterministic"
    rag_embedding_model: str = "BAAI/bge-m3"
    rag_reranker_provider: str = "deterministic"
    rag_reranker_model: str = "BAAI/bge-reranker-v2-m3"
    rag_device: str = "auto"
    rag_model_cache: Path | None = None
    rag_qdrant_url: str | None = None
    rag_qdrant_api_key: str | None = None
    rag_qdrant_path: Path | None = None
    rag_qdrant_timeout_seconds: int = 30
    rag_candidate_k: int = 32
    rag_score_threshold: float = 0.0
    rag_max_chunks_per_document: int = 3
    rag_embedding_batch_size: int = 8
    rag_reranker_batch_size: int = 4
    rag_reranker_max_length: int = 1024
    rag_chunk_tokens: int = 512
    rag_chunk_overlap_tokens: int = 64
    rag_document_parser: str = "auto"
    rag_docling_artifacts_path: Path | None = None
    rag_max_document_bytes: int = 25_000_000
    rag_max_document_pages: int = 200
    context_max_input_tokens: int = 64_000
    context_planner_tokens: int = 28_000
    context_coder_tokens: int = 64_000
    context_reviewer_tokens: int = 48_000
    context_reflection_tokens: int = 32_000
    context_max_evidence_item_tokens: int = 2_000
    context_max_file_item_tokens: int = 12_000
    context_max_diff_tokens: int = 24_000
    context_max_test_output_tokens: int = 8_000

    def redact_secrets(self, text: str) -> str:
        redacted = text
        for secret in (self.deepseek_api_key, self.rag_qdrant_api_key):
            if secret:
                redacted = redacted.replace(secret, "[REDACTED]")
        return redacted

    @classmethod
    def from_env(cls, runtime_dir: str | Path | None = None) -> "Settings":
        load_dotenv(override=False)
        root = Path(runtime_dir or os.getenv("REPOPILOT_RUNTIME_DIR", ".repopilot"))
        mode = os.getenv("REPOPILOT_LLM_MODE", "mock").strip().lower()
        if mode not in {"mock", "deepseek"}:
            raise ValueError("REPOPILOT_LLM_MODE must be 'mock' or 'deepseek'")
        embedding_provider = os.getenv(
            "REPOPILOT_RAG_EMBEDDING_PROVIDER", "bge-m3"
        ).strip().lower()
        reranker_provider = os.getenv(
            "REPOPILOT_RAG_RERANKER_PROVIDER", "bge"
        ).strip().lower()
        qdrant_path_value = os.getenv("REPOPILOT_QDRANT_PATH", "").strip()
        model_cache_value = os.getenv("REPOPILOT_RAG_MODEL_CACHE", "").strip()
        docling_artifacts_value = os.getenv(
            "REPOPILOT_DOCLING_ARTIFACTS_PATH",
            os.getenv("DOCLING_ARTIFACTS_PATH", ""),
        ).strip()
        default_docling_artifacts = (
            Path.home() / ".cache" / "docling" / "models"
        )
        workspace_value = os.getenv("REPOPILOT_WORKSPACE_ROOTS", "").strip()
        workspace_roots = tuple(
            Path(value).expanduser().resolve()
            for value in workspace_value.split(os.pathsep)
            if value.strip()
        ) or (Path.cwd().resolve(),)
        return cls(
            deepseek_api_key=os.getenv("DEEPSEEK_API_KEY") or None,
            deepseek_base_url=os.getenv(
                "DEEPSEEK_BASE_URL", "https://api.deepseek.com"
            ).rstrip("/"),
            deepseek_model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro"),
            llm_mode=mode,
            max_files=int(os.getenv("REPOPILOT_MAX_FILES", "400")),
            max_file_bytes=int(os.getenv("REPOPILOT_MAX_FILE_BYTES", "200000")),
            rag_top_k=int(os.getenv("REPOPILOT_RAG_TOP_K", "8")),
            test_timeout_seconds=int(
                os.getenv("REPOPILOT_TEST_TIMEOUT_SECONDS", "120")
            ),
            runtime_dir=root.resolve(),
            max_cached_indices=max(
                1, int(os.getenv("REPOPILOT_MAX_CACHED_INDICES", "32"))
            ),
            max_workers=max(1, int(os.getenv("REPOPILOT_MAX_WORKERS", "4"))),
            max_pending_runs=max(
                1, int(os.getenv("REPOPILOT_MAX_PENDING_RUNS", "16"))
            ),
            max_cached_sessions=max(
                8, int(os.getenv("REPOPILOT_MAX_CACHED_SESSIONS", "128"))
            ),
            allow_host_test_execution=(
                os.getenv("REPOPILOT_ALLOW_HOST_TEST_EXECUTION", "0")
                .strip()
                .lower()
                in {"1", "true", "yes", "on"}
            ),
            allow_private_model_endpoints=(
                os.getenv("REPOPILOT_ALLOW_PRIVATE_MODEL_ENDPOINTS", "0")
                .strip()
                .lower()
                in {"1", "true", "yes", "on"}
            ),
            allowed_workspace_roots=workspace_roots,
            api_token=os.getenv("REPOPILOT_API_TOKEN") or None,
            llm_max_output_tokens=max(
                512, int(os.getenv("REPOPILOT_LLM_MAX_OUTPUT_TOKENS", "12000"))
            ),
            rag_embedding_provider=embedding_provider,
            rag_embedding_model=os.getenv(
                "REPOPILOT_RAG_EMBEDDING_MODEL", "BAAI/bge-m3"
            ).strip(),
            rag_reranker_provider=reranker_provider,
            rag_reranker_model=os.getenv(
                "REPOPILOT_RAG_RERANKER_MODEL",
                "BAAI/bge-reranker-v2-m3",
            ).strip(),
            rag_device=os.getenv("REPOPILOT_RAG_DEVICE", "auto").strip(),
            rag_model_cache=(
                Path(model_cache_value).resolve()
                if model_cache_value
                else (root.resolve() / "rag" / "models")
            ),
            rag_qdrant_url=(
                os.getenv("REPOPILOT_QDRANT_URL", "").strip().rstrip("/") or None
            ),
            rag_qdrant_api_key=os.getenv("REPOPILOT_QDRANT_API_KEY") or None,
            rag_qdrant_path=(
                Path(qdrant_path_value).resolve() if qdrant_path_value else None
            ),
            rag_qdrant_timeout_seconds=max(
                1, int(os.getenv("REPOPILOT_QDRANT_TIMEOUT_SECONDS", "30"))
            ),
            rag_candidate_k=max(
                2, int(os.getenv("REPOPILOT_RAG_CANDIDATE_K", "32"))
            ),
            rag_score_threshold=float(
                os.getenv("REPOPILOT_RAG_SCORE_THRESHOLD", "0.05")
            ),
            rag_max_chunks_per_document=max(
                1,
                int(
                    os.getenv(
                        "REPOPILOT_RAG_MAX_CHUNKS_PER_DOCUMENT",
                        "3",
                    )
                ),
            ),
            rag_embedding_batch_size=max(
                1, int(os.getenv("REPOPILOT_RAG_EMBEDDING_BATCH_SIZE", "8"))
            ),
            rag_reranker_batch_size=max(
                1, int(os.getenv("REPOPILOT_RAG_RERANKER_BATCH_SIZE", "4"))
            ),
            rag_reranker_max_length=max(
                128, int(os.getenv("REPOPILOT_RAG_RERANKER_MAX_LENGTH", "1024"))
            ),
            rag_chunk_tokens=max(
                64, int(os.getenv("REPOPILOT_RAG_CHUNK_TOKENS", "512"))
            ),
            rag_chunk_overlap_tokens=max(
                0, int(os.getenv("REPOPILOT_RAG_CHUNK_OVERLAP_TOKENS", "64"))
            ),
            rag_document_parser=os.getenv(
                "REPOPILOT_RAG_DOCUMENT_PARSER", "auto"
            )
            .strip()
            .lower(),
            rag_docling_artifacts_path=(
                Path(docling_artifacts_value).expanduser().resolve()
                if docling_artifacts_value
                else (
                    default_docling_artifacts.resolve()
                    if default_docling_artifacts.is_dir()
                    else None
                )
            ),
            rag_max_document_bytes=max(
                1_000_000,
                int(
                    os.getenv(
                        "REPOPILOT_RAG_MAX_DOCUMENT_BYTES",
                        "25000000",
                    )
                ),
            ),
            rag_max_document_pages=max(
                1,
                int(
                    os.getenv(
                        "REPOPILOT_RAG_MAX_DOCUMENT_PAGES",
                        "200",
                    )
                ),
            ),
            context_max_input_tokens=max(
                2_048,
                int(os.getenv("REPOPILOT_CONTEXT_MAX_INPUT_TOKENS", "64000")),
            ),
            context_planner_tokens=max(
                1_024,
                int(os.getenv("REPOPILOT_CONTEXT_PLANNER_TOKENS", "28000")),
            ),
            context_coder_tokens=max(
                1_024,
                int(os.getenv("REPOPILOT_CONTEXT_CODER_TOKENS", "64000")),
            ),
            context_reviewer_tokens=max(
                1_024,
                int(os.getenv("REPOPILOT_CONTEXT_REVIEWER_TOKENS", "48000")),
            ),
            context_reflection_tokens=max(
                1_024,
                int(
                    os.getenv(
                        "REPOPILOT_CONTEXT_REFLECTION_TOKENS",
                        "32000",
                    )
                ),
            ),
            context_max_evidence_item_tokens=max(
                128,
                int(
                    os.getenv(
                        "REPOPILOT_CONTEXT_MAX_EVIDENCE_ITEM_TOKENS",
                        "2000",
                    )
                ),
            ),
            context_max_file_item_tokens=max(
                256,
                int(
                    os.getenv(
                        "REPOPILOT_CONTEXT_MAX_FILE_ITEM_TOKENS",
                        "12000",
                    )
                ),
            ),
            context_max_diff_tokens=max(
                512,
                int(
                    os.getenv(
                        "REPOPILOT_CONTEXT_MAX_DIFF_TOKENS",
                        "24000",
                    )
                ),
            ),
            context_max_test_output_tokens=max(
                256,
                int(
                    os.getenv(
                        "REPOPILOT_CONTEXT_MAX_TEST_OUTPUT_TOKENS",
                        "8000",
                    )
                ),
            ),
        )
