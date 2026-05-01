from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.models.schemas import STARAnswerResponse
from app.services.llm_service import LLMService


@pytest.fixture
def llm_service():
    with patch("google.genai.Client"):
        service = LLMService()
        return service


@pytest.mark.asyncio
async def test_get_embeddings(llm_service):
    # Mock the new SDK response structure in the .aio namespace
    mock_response = MagicMock()
    mock_response.embeddings = [MagicMock(values=[0.1, 0.2, 0.3])]
    # Mock usage metadata for telemetry
    mock_response.usage_metadata.prompt_token_count = 10
    llm_service.client.aio.models.embed_content = AsyncMock(return_value=mock_response)

    embedding = await llm_service.get_embeddings("test text")

    assert embedding == [0.1, 0.2, 0.3]
    llm_service.client.aio.models.embed_content.assert_called_once()


def test_get_embeddings_batch_sync(llm_service):
    """Verify synchronous LLM service handles batch requests."""
    mock_response = MagicMock()
    mock_response.embeddings = [
        MagicMock(values=[0.1]),
        MagicMock(values=[0.2]),
    ]
    # Mock usage metadata for telemetry
    mock_response.usage_metadata.prompt_token_count = 20
    llm_service.client.models.embed_content.return_value = mock_response

    embeddings = llm_service.get_embeddings_batch_sync(["text1", "text2"])

    assert embeddings == [[0.1], [0.2]]
    llm_service.client.models.embed_content.assert_called_once()


@pytest.mark.asyncio
async def test_generate_star_answer(llm_service):
    # Mock the new SDK response structure in the .aio namespace
    mock_response = MagicMock()
    mock_parsed = STARAnswerResponse(
        situation="Situation",
        task="Task",
        action="Action",
        result="Result",
        citations=[1],
    )
    mock_response.parsed = mock_parsed

    # Mock usage metadata for telemetry
    mock_response.usage_metadata = MagicMock()
    mock_response.usage_metadata.prompt_token_count = 100
    mock_response.usage_metadata.candidates_token_count = 50

    llm_service.client.aio.models.generate_content = AsyncMock(
        return_value=mock_response
    )

    chunks = [{"id": 1, "text_content": "Context content"}]
    answer = await llm_service.generate_star_answer("test query", chunks)

    assert isinstance(answer, STARAnswerResponse)
    assert answer.situation == "Situation"
    assert answer.citations == [1]
    llm_service.client.aio.models.generate_content.assert_called_once()
