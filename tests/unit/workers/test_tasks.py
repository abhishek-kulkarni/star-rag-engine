from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

# We expect these tasks to exist in app.workers.tasks
from app.workers.tasks import chunk_task, parse_task


@pytest.fixture
def mock_db():
    return MagicMock()


@pytest.mark.asyncio
async def test_parse_task_success():
    """Verify that parse_task extracts text and updates job status."""
    job_id = str(uuid4())
    content = b"PDF Content"

    with patch("app.services.parser_service.parser_service.parse_pdf") as mock_parse:
        mock_parse.return_value = "Extracted Text"

        # This will fail because app.workers.tasks doesn't exist yet
        result = parse_task.delay(job_id, content)
        assert result is not None


@pytest.mark.asyncio
async def test_chunk_task_semantic_integrity():
    """Verify that chunk_task correctly invokes ParserService.split_text."""
    text = "Some long text to be chunked."

    with patch("app.services.parser_service.parser_service.split_text") as mock_split:
        mock_split.return_value = ["chunk 1", "chunk 2"]

        result = chunk_task.delay("job_123", text)
        assert result is not None
