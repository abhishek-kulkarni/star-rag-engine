# STAR RAG Engine

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/abhishek-kulkarni/star-rag-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/abhishek-kulkarni/star-rag-engine/actions/workflows/ci.yml)
[![Python 3.14+](https://img.shields.io/badge/python-3.14+-blue.svg)](https://www.python.org/)
[![Coverage: 100%](https://img.shields.io/badge/coverage-100%25-brightgreen.svg)]()

An asynchronous, multi-tenant Retrieval-Augmented Generation (RAG) system designed for generating context-aware technical materials (resumes, interview prep) using the Gemini API.

## 🚀 Overview

The STAR RAG Engine allows users to upload technical documents and highly structured **Plan Artifacts**. It processes these asynchronously to provide a high-performance, grounded AI experience that enforces strict business logic and formatting rules.

### Key Features
- **Multi-Tenant Isolation**: Every document, vector, and job is strictly bound to a `user_id` at the database and vector store levels.
- **Asynchronous Ingestion**: Chained Celery tasks (Parse → Chunk → Embed) ensure the API remains responsive during heavy processing.
- **Plan Artifact Enforcement**: Special rule-based documents are injected into LLM System Instructions to guarantee deterministic behavior.
- **STAR-Format Generation**: Synchronous RAG queries return structured JSON matching the Situation, Task, Action, Result framework.
- **Enterprise Resiliency**: Includes Dead Letter Queues (DLQ), "The Sweeper" for orphaned jobs, and HNSW vector index maintenance.

## 🛠 Tech Stack
- **API**: FastAPI, Pydantic, Uvicorn
- **Async Workers**: Celery, Redis
- **LLM**: Google Gemini API (`gemini-2.5-flash`, `gemini-embedding-2`)
- **Persistence**: PostgreSQL + `pgvector`, SQLAlchemy 2.0
- **Storage**: MinIO (S3-compatible)
- **Security**: Microsoft Presidio (PII Sanitization)
- **Observability**: Prometheus, Celery telemetry
- **Environment**: Docker Compose, Python 3.14+

## 🏗 Architectural Principles
We strictly follow **Clean Architecture** boundaries:
- **API**: Stateless route handlers and dependency injection.
- **Services**: Pure business logic and LLM orchestration.
- **Models**: SQLAlchemy domain models and Pydantic schemas.
- **Workers**: Discrete, idempotent tasks for data processing.

## 🚦 Getting Started

### Prerequisites
- **Python 3.14+** (Managed via `uv`)
- **Docker & Docker Compose**
- **Google Gemini API Key** — get one at [aistudio.google.com](https://aistudio.google.com/app/apikey)

### 📄 Documentation
- **System Design**: [STAR RAG Engine - System Design.pdf](design/STAR%20RAG%20Engine%20-%20System%20Design.pdf)
- **Architecture Diagrams**: [Ingestion](design/STAR%20RAG%20Engine%20-%20Ingestion%20Sequence%20Diagram.png), [Query/RAG](design/STAR%20RAG%20Engine%20-%20Query:RAG%20Sequence%20Diagram.png)

### Setup & Run

> [!IMPORTANT]
> `docker-compose.yml` uses **required** environment variable substitution. `POSTGRES_PASSWORD`, `MINIO_ROOT_PASSWORD`, and `GEMINI_API_KEY` **must** be set — the stack will refuse to start without them to prevent insecure deployments.

1. **Configure environment**:
   ```bash
   cp .env.example .env
   # Open .env and fill in POSTGRES_PASSWORD, MINIO_ROOT_PASSWORD, GEMINI_API_KEY
   ```
2. **Launch all services** (Postgres, Redis, MinIO, API, Celery workers, Prometheus):
   ```bash
   docker-compose up --build
   ```
3. **Access the UI**: Navigate to `http://localhost:8000` after startup.

#### Running services individually (for development)

1. **Start infrastructure only**:
   ```bash
   docker-compose up -d postgres redis minio
   ```
2. **Install dependencies**:
   ```bash
   uv sync
   ```
3. **Run API**:
   ```bash
   uv run fastapi dev app/main.py
   ```
4. **Run Workers** (in a separate terminal):
   ```bash
   uv run celery -A app.workers.celery_app worker --loglevel=info -Q ingestion,celery
   ```

## 📄 Plan Artifacts
A **Plan Artifact** is a special configuration document (typically a `.txt` or `.md` file) that defines the "rules of engagement" for the AI. Unlike standard documents, Plan Artifacts are:
- **Immutable Rubrics**: Injected directly into the LLM's system instructions on every query.
- **Global Context**: Enforce specific business logic, interview rubrics (e.g., Amazon Leadership Principles), or tone rules.
- **Fast Path**: Stored directly in MinIO — no parsing/vectorization pipeline is triggered.

**Example**: Upload an `amazon_leadership_principles.txt` file as a Plan Artifact. Every subsequent STAR answer will be grounded in those principles without further configuration. A sample artifact is provided in `data/` (not committed — see `.gitignore`).

## 🎨 UI Interaction
The engine includes a premium, glassmorphic dashboard for real-time interaction.
1. **Access**: Navigate to `http://localhost:8000` after starting the API.
2. **Document Ingestion**:
   - **Standard Document** (default): Upload PDFs, DOCX, or TXT files to be parsed and vectorized for semantic search.
   - **Plan Artifact**: Upload a rubric/rules `.txt` file. It skips vectorization and instead grounds the AI's behavior on every query.
   - **Action**: Drag & drop files into the upload zone. Monitor real-time job status (Pending → Parsing → Chunking → Embedding → Completed).
3. **Semantic Query**: Use the "Semantic Retrieval" box to ask technical questions. The engine automatically injects your active **Plan Artifacts** as system rubrics and retrieves the top-5 relevant document chunks to generate a grounded **STAR (Situation, Task, Action, Result)** answer.

## 🧪 Verification
- **Unit & Integration Tests**:
  ```bash
  uv run pytest tests/unit
  ```
- **RAG Evaluation**:
  > [!WARNING]
  > The `scripts/evaluate_rag.py` script has been **disabled** with a safety gate to prevent accidental execution and quota drainage. To enable it for a specific run, comment out the `sys.exit(1)` block at the top of the file.

  ```bash
  uv run python scripts/evaluate_rag.py
  ```

## 🔒 Security Notes
- **Secrets**: Never commit `.env`. The `docker-compose.yml` uses `:?err` substitution — the stack will **fail to start** if required secrets are missing.
- **PII**: All ingested document text is sanitized via Microsoft Presidio before storage and embedding (Names, Emails, Phone Numbers, URLs).
- **Multi-Tenancy**: Every database query and vector search is filtered by `user_id` at the partition level — cross-user data leakage is structurally impossible.
- **Data**: Files in `data/` are excluded from version control via `.gitignore`. Do not commit real resumes or personal information.

## 📝 License
MIT — see [LICENSE](LICENSE).
