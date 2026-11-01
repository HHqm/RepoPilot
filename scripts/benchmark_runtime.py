"""Micro-benchmarks for RepoPilot's runtime hot paths.

Runs fully offline (mock LLM, deterministic embeddings) inside a temporary
runtime directory, so it never touches a live service or model API:

    python scripts/benchmark_runtime.py
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from collections.abc import Callable
from pathlib import Path


def _isolated_environment(root: Path) -> None:
    os.environ.update(
        {
            "REPOPILOT_RUNTIME_DIR": str(root / "runtime"),
            "REPOPILOT_QDRANT_PATH": str(root / "qdrant"),
            "REPOPILOT_LLM_MODE": "mock",
            "REPOPILOT_RAG_EMBEDDING_PROVIDER": "deterministic",
            "REPOPILOT_RAG_RERANKER_PROVIDER": "deterministic",
            "REPOPILOT_API_TOKEN": "",
        }
    )


def _mean_ms(function: Callable[[], object], repeat: int) -> float:
    function()  # warm-up
    started = time.perf_counter()
    for _ in range(repeat):
        function()
    return (time.perf_counter() - started) / repeat * 1000


def _synthetic_repository(root: Path, sources: int, vendored: int) -> Path:
    for index in range(sources):
        path = root / "src" / f"module_{index}.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"def handler_{index}():\n    return {index}\n")
    for index in range(vendored):
        path = root / "node_modules" / f"pkg{index % 150}" / f"f{index}.js"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("module.exports = 1;\n")
    return root.resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=int, default=500)
    parser.add_argument("--vendored-files", type=int, default=3000)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="repopilot-bench-") as directory:
        root = Path(directory)
        _isolated_environment(root)
        from repopilot.rag.chunking import StructureAwareChunker
        from repopilot.rag.pipeline import HybridRAG
        from repopilot.service import RepoPilotService, RunRuntime

        results: dict[str, float] = {}
        with RepoPilotService() as service:
            results["checkpoint_lookup_ms"] = _mean_ms(
                lambda: service._checkpoint_response("0" * 32), 30
            )
            runtime = RunRuntime(run_id="b" * 32, status="running", state={})
            started = time.perf_counter()
            for index in range(args.events):
                service._publish(
                    runtime,
                    "agent_trace",
                    {"summary": "x" * 200, "index": index},
                )
            results["publish_event_ms"] = (
                (time.perf_counter() - started) / args.events * 1000
            )

        repository = _synthetic_repository(
            root / "repo", sources=40, vendored=args.vendored_files
        )
        results["candidate_walk_ms"] = _mean_ms(
            lambda: list(HybridRAG._candidate_files(repository)), 5
        )
        chunker = StructureAwareChunker(512, 64)
        lines = [
            f"result_{index} = service.compute(alpha, beta, gamma)  # 第{index}行"
            for index in range(20_000)
        ]
        results["chunk_20k_lines_ms"] = _mean_ms(
            lambda: chunker._split_lines(lines, 1), 3
        )

    print(
        json.dumps(
            {key: round(value, 3) for key, value in results.items()},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
