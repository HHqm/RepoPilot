"""Retrieval benchmark and ablation study for RepoPilot's hybrid RAG.

Default run (CI gate) evaluates the production pipeline with deterministic,
offline models:

    python scripts/evaluate_retrieval.py --min-recall 0.9 --min-mrr 0.7

Ablation over every retrieval stage, with real BGE-M3 + reranker weights:

    python scripts/evaluate_retrieval.py --ablation --production-models \
        --report docs/RAG_EVALUATION.md

Each repository is indexed once; every configuration then queries the same
collection, so differences come only from the retrieval stages themselves.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import sys
import tempfile
import time
from collections import defaultdict
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from repopilot.config import Settings
from repopilot.rag import HybridRAG, SearchOptions


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ABLATIONS: dict[str, SearchOptions] = {
    "dense": SearchOptions(sparse=False, rerank=False, diversify=False),
    "sparse": SearchOptions(dense=False, rerank=False, diversify=False),
    "hybrid_rrf": SearchOptions(rerank=False, diversify=False),
    "hybrid_dbsf": SearchOptions(fusion="dbsf", rerank=False, diversify=False),
    "hybrid_rrf+rerank": SearchOptions(diversify=False),
    "production": SearchOptions(),
}


def reciprocal_rank(paths: list[str], relevant: set[str]) -> float:
    for rank, path in enumerate(paths, start=1):
        if path in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(paths: list[str], relevant: set[str], k: int) -> float:
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, path in enumerate(paths[:k], start=1)
        if path in relevant
    )
    # The ideal ranking places every relevant document first within k slots,
    # independent of how many results this configuration happened to return.
    ideal = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, min(len(relevant), k) + 1)
    )
    return dcg / ideal if ideal else 0.0


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def evaluate_configuration(
    rag: HybridRAG,
    indices: dict[str, Any],
    cases: list[dict[str, Any]],
    options: SearchOptions,
    top_k: int,
) -> dict[str, Any]:
    results = []
    latencies: dict[str, list[float]] = defaultdict(list)
    for case in cases:
        evidence, stats = rag.search_with_stats(
            indices[case["repository"]],
            case["query"],
            top_k=top_k,
            options=options,
        )
        for key in ("embed_ms", "recall_ms", "rerank_ms", "total_ms"):
            latencies[key].append(getattr(stats, key))
        paths = list(dict.fromkeys(item.path for item in evidence))
        relevant = set(case["relevant_paths"])
        results.append(
            {
                "case_id": case["case_id"],
                "category": case.get("category", "uncategorized"),
                "paths": paths,
                "hit_at_1": float(bool(paths) and paths[0] in relevant),
                "recall_at_k": len(set(paths[:top_k]) & relevant) / len(relevant),
                "reciprocal_rank": reciprocal_rank(paths, relevant),
                "ndcg_at_k": ndcg_at_k(paths, relevant, top_k),
            }
        )
    count = max(1, len(results))

    def mean(key: str, rows: list[dict[str, Any]] = results) -> float:
        return sum(row[key] for row in rows) / max(1, len(rows))

    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        by_category[row["category"]].append(row)
    return {
        "options": options.label,
        "cases": results,
        "hit_at_1": mean("hit_at_1"),
        "mean_recall_at_k": mean("recall_at_k"),
        "mrr": mean("reciprocal_rank"),
        "mean_ndcg_at_k": mean("ndcg_at_k"),
        "categories": {
            category: {
                "cases": len(rows),
                "mrr": mean("reciprocal_rank", rows),
                "recall_at_k": mean("recall_at_k", rows),
            }
            for category, rows in sorted(by_category.items())
        },
        "latency_ms": {
            "p50_total": statistics.median(latencies["total_ms"]) if count else 0,
            "p95_total": percentile(latencies["total_ms"], 0.95),
            "mean_embed": statistics.fmean(latencies["embed_ms"]),
            "mean_recall": statistics.fmean(latencies["recall_ms"]),
            "mean_rerank": statistics.fmean(latencies["rerank_ms"]),
        },
    }


def render_report(summary: dict[str, Any]) -> str:
    configs = summary["configurations"]
    lines = [
        "# RAG 检索评测报告",
        "",
        "> 由 `scripts/evaluate_retrieval.py --ablation --report ...` 自动生成，请勿手改。",
        "",
        "## 实验设置",
        "",
        f"- 生成时间：{summary['generated_at']}",
        f"- 模型：Embedding `{summary['embedding_model']}`，Sparse "
        f"`{summary['sparse_model']}`，Reranker `{summary['reranker_model']}`",
        f"- 运行环境：{summary['environment']}",
        f"- 用例：{summary['case_count']} 条，覆盖 {len(summary['categories'])} 类查询"
        f"（{', '.join(summary['categories'])}），语料 {summary['corpus_chunks']} 个知识块",
        f"- 指标：Top-{summary['top_k']} 文件级 Hit@1、Recall@k、MRR、nDCG@k；"
        "延迟为单次查询端到端耗时",
        "- 用例文件：`evals/retrieval_cases.jsonl`（不进入被检索语料，避免泄漏）",
        "",
        "## 消融结果",
        "",
        "| 配置 | Hit@1 | Recall@k | MRR | nDCG@k | p50 ms | p95 ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, result in configs.items():
        latency = result["latency_ms"]
        lines.append(
            f"| `{name}` | {result['hit_at_1']:.3f} | "
            f"{result['mean_recall_at_k']:.3f} | {result['mrr']:.3f} | "
            f"{result['mean_ndcg_at_k']:.3f} | {latency['p50_total']:.1f} | "
            f"{latency['p95_total']:.1f} |"
        )
    categories = summary["categories"]
    lines += [
        "",
        "## 分查询类型 MRR",
        "",
        "| 配置 | " + " | ".join(f"`{item}`" for item in categories) + " |",
        "| --- |" + " ---: |" * len(categories),
    ]
    for name, result in configs.items():
        cells = [
            f"{result['categories'].get(item, {}).get('mrr', 0.0):.3f}"
            for item in categories
        ]
        lines.append(f"| `{name}` | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## 各阶段平均耗时（ms）",
        "",
        "| 配置 | 查询编码 | 向量召回/融合 | 重排 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, result in configs.items():
        latency = result["latency_ms"]
        lines.append(
            f"| `{name}` | {latency['mean_embed']:.1f} | "
            f"{latency['mean_recall']:.1f} | {latency['mean_rerank']:.1f} |"
        )
    production = configs.get("production")
    if production:
        misses = [
            row["case_id"]
            for row in production["cases"]
            if row["reciprocal_rank"] < 1.0
        ]
        lines += [
            "",
            "## 生产配置未排第一的用例",
            "",
            ", ".join(f"`{item}`" for item in misses) if misses else "无",
        ]
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
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
    parser.add_argument(
        "--corpus-root",
        default=str(PROJECT_ROOT),
        help=(
            "Directory that case repositories are resolved against. Point it "
            "at a frozen snapshot to compare code changes on an identical corpus."
        ),
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--production-models",
        action="store_true",
        help="Use configured BGE models instead of deterministic CI models.",
    )
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="Evaluate every retrieval-stage configuration, not only production.",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Write a Markdown report to this path.",
    )
    parser.add_argument(
        "--min-recall",
        type=float,
        default=0.0,
        help="Exit non-zero when production mean Recall@k drops below this.",
    )
    parser.add_argument(
        "--min-mrr",
        type=float,
        default=0.0,
        help="Exit non-zero when production MRR drops below this value.",
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
    configurations = ABLATIONS if args.ablation else {
        "production": ABLATIONS["production"]
    }
    rag = None
    try:
        rag = HybridRAG.from_settings(settings)
        indices = {}
        index_seconds = 0.0
        for case in cases:
            repository = case["repository"]
            if repository not in indices:
                started = time.perf_counter()
                indices[repository] = rag.index_repository(
                    (Path(args.corpus_root) / repository).resolve()
                )
                index_seconds += time.perf_counter() - started
        results = {
            name: evaluate_configuration(rag, indices, cases, options, args.top_k)
            for name, options in configurations.items()
        }
        summary = {
            "generated_at": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
            "environment": (
                f"{platform.platform(terse=True)}, "
                f"Python {platform.python_version()}, "
                f"device={settings.rag_device}"
            ),
            "embedding_model": rag.store.embedding.model_id,
            "sparse_model": rag.store.embedding.sparse_model_id,
            "reranker_model": rag.store.reranker.model_id,
            "top_k": args.top_k,
            "case_count": len(cases),
            "categories": sorted(
                {case.get("category", "uncategorized") for case in cases}
            ),
            "corpus_chunks": sum(item.chunk_count for item in indices.values()),
            "index_seconds": round(index_seconds, 2),
            "configurations": results,
        }
    finally:
        if rag is not None:
            rag.close()
        if runtime_owner is not None:
            runtime_owner.cleanup()

    printable = {
        **summary,
        "configurations": {
            name: {key: value for key, value in result.items() if key != "cases"}
            for name, result in results.items()
        },
    }
    print(json.dumps(printable, ensure_ascii=False, indent=2))
    if args.report:
        Path(args.report).write_text(render_report(summary), encoding="utf-8")

    production = results["production"]
    failures = []
    if production["mean_recall_at_k"] < args.min_recall:
        failures.append(
            f"mean Recall@{args.top_k} {production['mean_recall_at_k']:.3f} "
            f"< {args.min_recall:.3f}"
        )
    if production["mrr"] < args.min_mrr:
        failures.append(f"MRR {production['mrr']:.3f} < {args.min_mrr:.3f}")
    for failure in failures:
        print(f"retrieval quality gate failed: {failure}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
