from datetime import datetime

from app.models.document import JobStatus
from app.models.schemas import IngestionJobResponse, RAGQueryRequest, STARAnswerResponse


def test_rag_query_request_validation():
    req = RAGQueryRequest(query="What is STAR?", include_plan_artifacts=True)
    assert req.query == "What is STAR?"


def test_star_answer_response_schema():
    data = {
        "situation": "S",
        "task": "T",
        "action": "A",
        "result": "R",
        "citations": [1, 2, 3],
    }
    resp = STARAnswerResponse(**data)
    assert resp.situation == "S"
    assert resp.citations == [1, 2, 3]


def test_ingestion_job_response():
    data = {
        "id": 1,
        "status": JobStatus.COMPLETED,
        "error_message": None,
        "created_at": datetime.now(),
        "parsed_at": datetime.now(),
        "chunked_at": datetime.now(),
        "completed_at": datetime.now(),
    }
    resp = IngestionJobResponse(**data)
    assert resp.status == JobStatus.COMPLETED
