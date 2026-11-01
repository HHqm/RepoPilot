# RAG 检索评测报告

> 由 `scripts/evaluate_retrieval.py --ablation --report ...` 自动生成，请勿手改。

## 实验设置

- 生成时间：2026-11-01 10:41 UTC
- 模型：Embedding `BAAI/bge-m3`，Sparse `repopilot/multilingual-code-sparse-v3`，Reranker `BAAI/bge-reranker-v2-m3`
- 运行环境：Windows-10, Python 3.11.15, device=cpu
- 用例：34 条，覆盖 5 类查询（doc_en, doc_zh, identifier, natural_en, natural_zh），语料 543 个知识块
- 指标：Top-5 文件级 Hit@1、Recall@k、MRR、nDCG@k；延迟为单次查询端到端耗时
- 用例文件：`evals/retrieval_cases.jsonl`（不进入被检索语料，避免泄漏）

## 消融结果

| 配置 | Hit@1 | Recall@k | MRR | nDCG@k | p50 ms | p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `dense` | 0.706 | 0.863 | 0.784 | 0.794 | 124.8 | 138.3 |
| `sparse` | 0.618 | 0.706 | 0.657 | 0.669 | 159.7 | 184.2 |
| `hybrid_rrf` | 0.735 | 0.863 | 0.804 | 0.811 | 172.6 | 189.3 |
| `hybrid_dbsf` | 0.765 | 0.833 | 0.800 | 0.804 | 170.7 | 207.6 |
| `hybrid_rrf+rerank` | 0.853 | 0.897 | 0.882 | 0.879 | 44391.6 | 50500.3 |
| `production` | 0.853 | 0.897 | 0.882 | 0.879 | 44948.5 | 49894.3 |

## 分查询类型 MRR

| 配置 | `doc_en` | `doc_zh` | `identifier` | `natural_en` | `natural_zh` |
| --- | ---: | ---: | ---: | ---: | ---: |
| `dense` | 1.000 | 1.000 | 0.817 | 0.625 | 0.667 |
| `sparse` | 1.000 | 0.905 | 1.000 | 0.500 | 0.250 |
| `hybrid_rrf` | 1.000 | 1.000 | 0.950 | 0.750 | 0.569 |
| `hybrid_dbsf` | 1.000 | 1.000 | 1.000 | 0.750 | 0.517 |
| `hybrid_rrf+rerank` | 1.000 | 1.000 | 0.950 | 0.750 | 0.792 |
| `production` | 1.000 | 1.000 | 0.950 | 0.750 | 0.792 |

## 各阶段平均耗时（ms）

| 配置 | 查询编码 | 向量召回/融合 | 重排 |
| --- | ---: | ---: | ---: |
| `dense` | 110.0 | 12.6 | 0.0 |
| `sparse` | 112.8 | 46.2 | 0.0 |
| `hybrid_rrf` | 112.1 | 59.6 | 0.0 |
| `hybrid_dbsf` | 110.8 | 61.2 | 0.0 |
| `hybrid_rrf+rerank` | 119.6 | 59.9 | 33908.8 |
| `production` | 121.0 | 60.3 | 33562.0 |

## 生产配置未排第一的用例

`qdrant-version-cleanup`, `stale-approval-digest`, `rollback-on-failure`, `secret-redaction`, `phase-labels`
