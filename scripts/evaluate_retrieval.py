from __future__ import annotations

import argparse
import json
import math
import tempfile
from dataclasses import replace
from pathlib import Path

from repopilot.config import Settings
from repopilot.rag import HybridRAG


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def reciprocal_rank(paths: list[str], relevant: set[str]) -> float:
    for rank, path in enumerate(paths, start=1):
        if path in relevant:
            return 1.0 / rank
    return 0.0


def ndcg(paths: list[str], relevant: set[str]) -> float:
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, path in enumerate(paths, start=1)
        if path in relevant
    )
    ideal_hits = min(len(relevant), len(paths))
    ideal = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, ideal_hits + 1)
    )
    return dcg / ideal if ideal else 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cases",
        default=str(PROJECT_ROOT / "evals" / "retrieval_cases.jsonl"),
    )
    parser.add_argument(
        "--runtime-dir",
        default=None,
        help=(
            "Optional isolated benchmark directory. When omitted, a unique "
            "temporary directory is created and removed automatically."
        ),
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--production-models",
        action="store_true",
        help="Use configured BGE models instead of deterministic CI models.",
    )
    args = parser.parse_args()

    runtime_owner = (
        None
        if args.runtime_dir
        else tempfile.TemporaryDirectory(prefix="repopilot-retrieval-eval-")
    )
    runtime_path = Path(
        args.runtime_dir if args.runtime_dir else runtime_owner.name
    ).resolve()
    settings = replace(
        Settings.from_env(runtime_path),
        # An eval must never attach to the live service's embedded Qdrant
        # directory or mutate a configured remote production collection.
        rag_qdrant_url=None,
        rag_qdrant_api_key=None,
        rag_qdrant_path=runtime_path / "qdrant",
    )
    if not args.production_models:
        settings = replace(
            settings,
            rag_embedding_provider="deterministic",
            rag_reranker_provider="deterministic",
            rag_score_threshold=0.0,
        )
    cases = [
        json.loads(line)
        for line in Path(args.cases).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rag = None
    results = []
    try:
        rag = HybridRAG.from_settings(settings)
        indices = {}
        for case in cases:
            repo = str((PROJECT_ROOT / case["repository"]).resolve())
            if repo not in indices:
                indices[repo] = rag.index_repository(repo)
            evidence = rag.search(
                indices[repo],
                case["query"],
                top_k=args.top_k,
            )
            paths = list(dict.fromkeys(item.path for item in evidence))
            relevant = set(case["relevant_paths"])
            results.append(
                {
                    "case_id": case["case_id"],
                    "paths": paths,
                    "recall_at_k": len(set(paths) & relevant) / len(relevant),
                    "reciprocal_rank": reciprocal_rank(paths, relevant),
                    "ndcg_at_k": ndcg(paths, relevant),
                }
            )
    finally:
        if rag is not None:
            rag.close()
        if runtime_owner is not None:
            runtime_owner.cleanup()

    count = max(1, len(results))
    summary = {
        "top_k": args.top_k,
        "cases": results,
        "mean_recall_at_k": sum(
            item["recall_at_k"] for item in results
        )
        / count,
        "mrr": sum(item["reciprocal_rank"] for item in results) / count,
        "mean_ndcg_at_k": sum(item["ndcg_at_k"] for item in results) / count,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
