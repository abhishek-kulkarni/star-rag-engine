# STAR RAG Engine: Principal-Level Architectural Rules

This document defines the mandatory architectural standards for the STAR RAG Engine. All development must adhere to these rules to ensure the system is production-ready, resilient, secure, and enterprise-grade.

## 1. Strict Multi-Tenancy
* **Database Isolation**: Every query involving `Document`, `IngestionJob`, or `DocumentChunk` MUST include a `user_id` filter. Never assume a global scope.
* **Storage Isolation**: All raw files in MinIO MUST be prefixed with `{user_id}/`.
* **Vector Isolation**: Similarity searches MUST be scoped to the `user_id` via PostgreSQL partial indices or metadata filtering.

## 2. AI Precision & Prompt Engineering
* **Split-Prompt Strategy**: Never use a single string for LLM prompts. Use `system_instruction` for behavior/constraints and a `user_message` for data/context.
* **Citation Grounding**: Context chunks provided to the LLM must be wrapped in explicit `--- CHUNK ID: {id} ---` headers to ensure verifiable citations in the Pydantic response.
* **Semantic Chunking**: Do not use simple sliding windows for text splitting. Use **Recursive Character Splitting** that respects paragraph (`\n\n`), sentence (`. `), and word (` `) boundaries.
* **Responsible AI (RAI)**: Every system prompt must include anti-bias guardrails to prevent demographic markers from influencing behavioral evaluations.
* **PII Sanitization**: All text extracted from standard documents MUST pass through a sanitization middleware (e.g., Microsoft Presidio) to mask sensitive entities (names, emails) before vectorization.
* **Deterministic Structured Output**: Do not rely on prompt engineering to parse JSON. You MUST pass the `STARAnswerResponse` Pydantic model directly into the Gemini API client's `response_schema` configuration to guarantee strict type compliance.

## 3. Resiliency & Background Processing
* **Idempotency**: All Celery tasks (especially ingestion) must be idempotent. Re-running a task for the same `document_id` should not create duplicate chunks.
* **State Transparency**: Every stage of the ingestion pipeline (Parsing, Chunking, Embedding) must update the `IngestionJob` status in the database with timestamps.
* **Orphan Management**: A "Sweeper" task must run periodically to detect and fail jobs that have been stuck in a non-terminal state longer than the defined timeout.
* **Dual-Write Mitigation**: API routes triggering background tasks MUST wrap the broker call in a `try/except` block catching connection errors (e.g., `kombu.exceptions.OperationalError`). If the broker is down, rollback the database state to `FAILED` and return a 503 error.
* **Dead Letter Queues (DLQ)**: Celery tasks must not drop exhausted messages. Implement a custom `on_failure` hook to route poisoned payloads (e.g., MinIO URIs) to a dedicated Redis DLQ list.

## 4. Performance & Scalability
* **Vector Indexing**: Always use HNSW indexing for `pgvector` columns. Tuning parameters (`m`, `ef_construction`) must be optimized for Gemini’s 768-dimensional vectors.
* **Batch Operations**: Database inserts for `DocumentChunk` and requests to Gemini Embeddings should be performed in batches to minimize latency and API overhead.
* **Memory Safety**: Ingestion tasks must use streaming or buffered reads for large documents to prevent OOM (Out of Memory) errors in worker nodes.
* **Zero-Downtime Index Maintenance**: To prevent HNSW graph degradation from frequent document deletions, implement a scheduled Celery Beat task that executes `REINDEX INDEX CONCURRENTLY` on partitioned tables.

## 5. Engineering Excellence
* **TDD Enforcement**: 95%+ code coverage is the baseline. Every feature must have unit tests (mocking external APIs) and integration tests (verifying DB/Celery flow).
* **Type Integrity**: Use strict Python type hints and Pydantic V2 for all data boundaries (API inputs, LLM outputs, Internal services).
* **Documentation**: Every class and public method must have a docstring explaining the **Architectural Role** of the component, not just the mechanical implementation.

## 6. Data Privacy & AI Evaluation
* **The Right to be Forgotten**: Document deletion MUST be implemented as a synchronous cascade. Purge the MinIO raw file, delete the Postgres relational records, and explicitly hard-delete the 768-dimensional vectors from `pgvector`.
* **Golden Dataset Evaluation**: Unit tests are insufficient for generative generation. The test suite must include an "LLM-as-a-Judge" script (`evaluate_rag.py`) that measures Context Precision and Faithfulness against a static Golden Dataset of target questions and reference answers.