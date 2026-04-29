from app.models.document import (
    Document,
    DocumentChunk,
    DocumentType,
    IngestionJob,
    JobStatus,
)


def test_models_exist():
    # Verify enums exist
    assert DocumentType.STANDARD_DOC == "STANDARD_DOC"
    assert JobStatus.PENDING == "PENDING"

    doc = Document(
        user_id="user_123",
        filename="test.pdf",
        doc_type=DocumentType.STANDARD_DOC,
        minio_raw_uri="s3://test",
    )
    assert doc.user_id == "user_123"

    job = IngestionJob(document=doc, status=JobStatus.PENDING)
    assert job.status == JobStatus.PENDING

    chunk = DocumentChunk(
        document=doc, chunk_index=0, text_content="hello", embedding=[0.1] * 768
    )
    assert len(chunk.embedding) == 768
