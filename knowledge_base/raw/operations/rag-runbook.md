# RAG 运维手册

## 模型首次启动

BGE-M3 和 BGE Reranker 会在首次生产检索时下载。模型缓存目录由
`REPOPILOT_RAG_MODEL_CACHE` 控制。离线 CI 必须显式使用 deterministic provider，
不能把测试替身描述为生产 Embedding。

## 文档解析失败

普通文本 PDF 可以由 pypdf 降级解析。扫描件、复杂表格、PPTX、DOCX 和 XLSX
需要安装 `documents` extra，并由 Docling 恢复版面、表格和阅读顺序。解析失败
会写入索引报告的 `skipped_files`，不能静默忽略。

## Qdrant

本地开发可以使用持久化 Local Mode。Docker 和多进程部署使用独立 Qdrant
服务。生产环境必须启用认证、TLS、备份和基于 `repo_id`、`document_id` 的
Payload 索引。
