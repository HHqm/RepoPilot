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
    max_review_revisions: int = 1
    max_repair_attempts: int = 2
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
    rag_embedding_cache_entries: int = 50_000
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
    def from_env(cls, runtime_dir: str | Path | None = None) -> Settings:
        load_dotenv(override=False)
        root = Path(
            runtime_dir or os.getenv("REPOPILOT_RUNTIME_DIR", ".repopilot")
        ).resolve()
        mode = _env_str("REPOPILOT_LLM_MODE", "mock").lower()
        if mode not in {"mock", "deepseek"}:
            raise ValueError("REPOPILOT_LLM_MODE must be 'mock' or 'deepseek'")
        qdrant_path_value = _env_str("REPOPILOT_QDRANT_PATH")
        model_cache_value = _env_str("REPOPILOT_RAG_MODEL_CACHE")
        docling_artifacts_value = _env_str(
            "REPOPILOT_DOCLING_ARTIFACTS_PATH",
            os.getenv("DOCLING_ARTIFACTS_PATH", ""),
        )
        default_docling_artifacts = (
            Path.home() / ".cache" / "docling" / "models"
        )
        workspace_roots = tuple(
            Path(value).expanduser().resolve()
            for value in _env_str("REPOPILOT_WORKSPACE_ROOTS").split(os.pathsep)
            if value.strip()
        ) or (Path.cwd().resolve(),)
        return cls(
            deepseek_api_key=os.getenv("DEEPSEEK_API_KEY") or None,
            deepseek_base_url=os.getenv(
                "DEEPSEEK_BASE_URL", "https://api.deepseek.com"
            ).rstrip("/"),
            deepseek_model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro"),
            llm_mode=mode,
            max_files=_env_int("REPOPILOT_MAX_FILES", 400),
            max_file_bytes=_env_int("REPOPILOT_MAX_FILE_BYTES", 200_000),
            rag_top_k=_env_int("REPOPILOT_RAG_TOP_K", 8),
            test_timeout_seconds=_env_int(
                "REPOPILOT_TEST_TIMEOUT_SECONDS", 120
            ),
            runtime_dir=root,
            max_cached_indices=_env_int(
                "REPOPILOT_MAX_CACHED_INDICES", 32, minimum=1
            ),
            max_workers=_env_int("REPOPILOT_MAX_WORKERS", 4, minimum=1),
            max_pending_runs=_env_int(
                "REPOPILOT_MAX_PENDING_RUNS", 16, minimum=1
            ),
            max_cached_sessions=_env_int(
                "REPOPILOT_MAX_CACHED_SESSIONS", 128, minimum=8
            ),
            allow_host_test_execution=_env_bool(
                "REPOPILOT_ALLOW_HOST_TEST_EXECUTION"
            ),
            allow_private_model_endpoints=_env_bool(
                "REPOPILOT_ALLOW_PRIVATE_MODEL_ENDPOINTS"
            ),
            allowed_workspace_roots=workspace_roots,
            api_token=os.getenv("REPOPILOT_API_TOKEN") or None,
            llm_max_output_tokens=_env_int(
                "REPOPILOT_LLM_MAX_OUTPUT_TOKENS", 12_000, minimum=512
            ),
            max_review_revisions=_env_int(
                "REPOPILOT_MAX_REVIEW_REVISIONS", 1, minimum=0
            ),
            max_repair_attempts=_env_int(
                "REPOPILOT_MAX_REPAIR_ATTEMPTS", 2, minimum=0
            ),
            rag_embedding_provider=_env_str(
                "REPOPILOT_RAG_EMBEDDING_PROVIDER", "bge-m3"
            ).lower(),
            rag_embedding_model=_env_str(
                "REPOPILOT_RAG_EMBEDDING_MODEL", "BAAI/bge-m3"
            ),
            rag_reranker_provider=_env_str(
                "REPOPILOT_RAG_RERANKER_PROVIDER", "bge"
            ).lower(),
            rag_reranker_model=_env_str(
                "REPOPILOT_RAG_RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"
            ),
            rag_device=_env_str("REPOPILOT_RAG_DEVICE", "auto"),
            rag_model_cache=(
                Path(model_cache_value).resolve()
                if model_cache_value
                else root / "rag" / "models"
            ),
            rag_qdrant_url=_env_str("REPOPILOT_QDRANT_URL").rstrip("/") or None,
            rag_qdrant_api_key=os.getenv("REPOPILOT_QDRANT_API_KEY") or None,
            rag_qdrant_path=(
                Path(qdrant_path_value).resolve() if qdrant_path_value else None
            ),
            rag_qdrant_timeout_seconds=_env_int(
                "REPOPILOT_QDRANT_TIMEOUT_SECONDS", 30, minimum=1
            ),
            rag_candidate_k=_env_int(
                "REPOPILOT_RAG_CANDIDATE_K", 32, minimum=2
            ),
            rag_score_threshold=_env_float(
                "REPOPILOT_RAG_SCORE_THRESHOLD", 0.05
            ),
            rag_max_chunks_per_document=_env_int(
                "REPOPILOT_RAG_MAX_CHUNKS_PER_DOCUMENT", 3, minimum=1
            ),
            rag_embedding_batch_size=_env_int(
                "REPOPILOT_RAG_EMBEDDING_BATCH_SIZE", 8, minimum=1
            ),
            rag_embedding_cache_entries=_env_int(
                "REPOPILOT_RAG_EMBEDDING_CACHE_ENTRIES", 50_000, minimum=0
            ),
            rag_reranker_batch_size=_env_int(
                "REPOPILOT_RAG_RERANKER_BATCH_SIZE", 4, minimum=1
            ),
            rag_reranker_max_length=_env_int(
                "REPOPILOT_RAG_RERANKER_MAX_LENGTH", 1024, minimum=128
            ),
            rag_chunk_tokens=_env_int(
                "REPOPILOT_RAG_CHUNK_TOKENS", 512, minimum=64
            ),
            rag_chunk_overlap_tokens=_env_int(
                "REPOPILOT_RAG_CHUNK_OVERLAP_TOKENS", 64, minimum=0
            ),
            rag_document_parser=_env_str(
                "REPOPILOT_RAG_DOCUMENT_PARSER", "auto"
            ).lower(),
            rag_docling_artifacts_path=(
                Path(docling_artifacts_value).expanduser().resolve()
                if docling_artifacts_value
                else (
                    default_docling_artifacts.resolve()
                    if default_docling_artifacts.is_dir()
                    else None
                )
            ),
            rag_max_document_bytes=_env_int(
                "REPOPILOT_RAG_MAX_DOCUMENT_BYTES",
                25_000_000,
                minimum=1_000_000,
            ),
            rag_max_document_pages=_env_int(
                "REPOPILOT_RAG_MAX_DOCUMENT_PAGES", 200, minimum=1
            ),
            context_max_input_tokens=_env_int(
                "REPOPILOT_CONTEXT_MAX_INPUT_TOKENS", 64_000, minimum=2_048
            ),
            context_planner_tokens=_env_int(
                "REPOPILOT_CONTEXT_PLANNER_TOKENS", 28_000, minimum=1_024
            ),
            context_coder_tokens=_env_int(
                "REPOPILOT_CONTEXT_CODER_TOKENS", 64_000, minimum=1_024
            ),
            context_reviewer_tokens=_env_int(
                "REPOPILOT_CONTEXT_REVIEWER_TOKENS", 48_000, minimum=1_024
            ),
            context_reflection_tokens=_env_int(
                "REPOPILOT_CONTEXT_REFLECTION_TOKENS", 32_000, minimum=1_024
            ),
            context_max_evidence_item_tokens=_env_int(
                "REPOPILOT_CONTEXT_MAX_EVIDENCE_ITEM_TOKENS",
                2_000,
                minimum=128,
            ),
            context_max_file_item_tokens=_env_int(
                "REPOPILOT_CONTEXT_MAX_FILE_ITEM_TOKENS", 12_000, minimum=256
            ),
            context_max_diff_tokens=_env_int(
                "REPOPILOT_CONTEXT_MAX_DIFF_TOKENS", 24_000, minimum=512
            ),
            context_max_test_output_tokens=_env_int(
                "REPOPILOT_CONTEXT_MAX_TEST_OUTPUT_TOKENS", 8_000, minimum=256
            ),
        )


_TRUE_VALUES = {"1", "true", "yes", "on"}


def _env_str(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _env_bool(name: str) -> bool:
    return _env_str(name, "0").lower() in _TRUE_VALUES


def _env_int(name: str, default: int, *, minimum: int | None = None) -> int:
    """Parse an integer setting, naming the variable when it is malformed."""

    raw = _env_str(name)
    try:
        value = int(raw) if raw else default
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    return value if minimum is None else max(minimum, value)


def _env_float(name: str, default: float) -> float:
    raw = _env_str(name)
    try:
        return float(raw) if raw else default
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc
