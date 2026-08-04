# RepoPilot Retrieval Architecture

## Retrieval stages

RepoPilot uses a two-stage retrieval pipeline. Qdrant first performs dense semantic
retrieval and sparse lexical retrieval, then combines both rankings with Reciprocal
Rank Fusion. A multilingual cross-encoder reranks the fused candidates before the
Planner Agent receives evidence.

## Persistence and updates

Each repository has a stable repository identifier and a versioned manifest. Files
are keyed by SHA-256. Unchanged files reuse their existing vectors; changed and
deleted documents are updated through document-level Qdrant filters.

## Evidence contract

Every result contains a document ID, chunk ID, source path, content hash, parser,
language, RRF score, reranker score, and either line or page provenance. Planning
may only cite evidence IDs returned by the Retrieval Agent.
