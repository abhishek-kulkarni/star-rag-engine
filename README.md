# STAR RAG Engine

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
- **LLM**: Google Gemini API (`gemini-1.5-flash`, `text-embedding-004`)
- **Persistence**: PostgreSQL + `pgvector`, SQLAlchemy 2.0
- **Storage**: MinIO (S3-compatible)
- **Environment**: Docker Compose, Python 3.14+

## 🏗 Architectural Principles
We strictly follow **Clean Architecture** boundaries:
- **API**: Stateless route handlers and dependency injection.
- **Services**: Pure business logic and LLM orchestration.
- **Models**: SQLAlchemy domain models and Pydantic schemas.
- **Workers**: Discrete, idempotent tasks for data processing.

## 🚦 Getting Started

### Prerequisites
- Docker & Docker Compose
- Google Gemini API Key

### Setup
1. Clone the repository.
2. Create a `.env` file with your credentials (see `app/config/settings.py`).
3. Launch the infrastructure:
   ```bash
   docker-compose up --build
   ```

## 🧪 Verification
- **Automated**: Unit, Integration, and E2E tests using `pytest`.
- **AI Evaluation**: "LLM-as-a-Judge" script measuring Context Precision, Faithfulness, and Answer Relevance.

## 📝 License
Proprietary. © 2026 Abhishek Kulkarni.
