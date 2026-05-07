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
- **Python 3.14+** (Managed via `uv`)
- **Docker & Docker Compose**
- **Google Gemini API Key**

### 📄 Documentation
- **System Design**: [STAR RAG Engine - System Design.pdf](design/STAR%20RAG%20Engine%20-%20System%20Design.pdf)
- **Architecture Diagrams**: [Ingestion](design/STAR%20RAG%20Engine%20-%20Ingestion%20Sequence%20Diagram.png), [Query/RAG](design/STAR%20RAG%20Engine%20-%20Query:RAG%20Sequence%20Diagram.png)

### Setup & Run
1. **Infrastructure**: Launch the core services (Postgres, Redis, MinIO):
   ```bash
   docker-compose up -d
   ```
2. **Environment**: Use `uv` to manage dependencies and virtual environment:
   ```bash
   # Install dependencies
   uv sync
   
   # Setup environment variables
   cp .env.example .env  # Update with your GEMINI_API_KEY
   ```
3. **Run API**:
   ```bash
   uv run fastapi dev app/main.py
   ```
4. **Run Workers**: In a separate terminal:
   ```bash
   uv run celery -A app.workers.celery_app worker --loglevel=info -Q ingestion,celery
   ```

## 🎨 UI Interaction
The engine includes a premium, glassmorphic dashboard for real-time interaction.
1. **Access**: Navigate to `http://localhost:8000` after starting the API.
2. **Ingestion**: Drag & drop files (**PDF, DOCX, TXT**) into the upload zone. Monitor real-time status (Pending → Parsing → Chunking → Embedding → Completed).
3. **Semantic Query**: Use the "Semantic Retrieval" box to ask technical questions. The engine will return grounded answers in the **STAR (Situation, Task, Action, Result)** format.

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
- **Automated**: Unit, Integration, and E2E tests using `pytest`.
- **AI Evaluation**: "LLM-as-a-Judge" architecture measuring Context Precision, Faithfulness, and Answer Relevance.

## 📝 License
Proprietary. © 2026 Abhishek Kulkarni.
