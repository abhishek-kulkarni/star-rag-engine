from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.models.schemas import STARAnswerResponse
from app.services.llm_service import LLMService


@pytest.mark.asyncio
async def test_generate_star_answer_retry_on_429():
    """Verify that generate_star_answer retries on 429 errors."""
    service = LLMService()

    # Mock response object
    mock_response = AsyncMock()
    mock_response.parsed = STARAnswerResponse(
        situation="s", task="t", action="a", result="r", citations=[]
    )
    mock_response.usage_metadata = MagicMock()
    mock_response.usage_metadata.prompt_token_count = 10
    mock_response.usage_metadata.candidates_token_count = 5

    # We want it to fail with 429 twice, then succeed on the 3rd attempt
    side_effect = [
        Exception("429 Resource Exhausted"),
        Exception("429 Resource Exhausted"),
        mock_response,
    ]

    with patch.object(
        service.client.aio.models, "generate_content", side_effect=side_effect
    ) as mock_gen:
        # Patch asyncio.sleep to speed up tests
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            with patch("app.services.llm_service.telemetry") as mock_telemetry:
                result = await service.generate_star_answer("query", [])

                assert result.situation == "s"
                assert mock_gen.call_count == 3
                assert mock_sleep.call_count == 2
                # Verify token tracking
                mock_telemetry.track_tokens.assert_called_once_with(
                    prompt=10, completion=5
                )


@pytest.mark.asyncio
async def test_get_embeddings():
    """Verify single embedding retrieval."""
    service = LLMService()
    mock_response = AsyncMock()
    mock_response.embeddings = [MagicMock(values=[0.1, 0.2])]
    mock_response.usage_metadata.prompt_token_count = 5

    with patch.object(
        service.client.aio.models, "embed_content", return_value=mock_response
    ):
        res = await service.get_embeddings("hello")
        assert res == [0.1, 0.2]


def test_get_embeddings_batch_sync():
    """Verify batch embedding retrieval."""
    service = LLMService()
    mock_response = MagicMock()
    mock_response.embeddings = [MagicMock(values=[0.1]), MagicMock(values=[0.2])]
    mock_response.usage_metadata.prompt_token_count = 10

    with patch.object(
        service.client.models, "embed_content", return_value=mock_response
    ):
        res = service.get_embeddings_batch_sync(["a", "b"])
        assert res == [[0.1], [0.2]]


@pytest.mark.asyncio
async def test_generate_star_answer_failure_telemetry():
    """Verify that telemetry is tracked on final failure."""
    service = LLMService()

    # Always fail
    side_effect = Exception("Permanent Error")

    with (
        patch.object(
            service.client.aio.models, "generate_content", side_effect=side_effect
        ),
        patch("app.services.llm_service.telemetry") as mock_telemetry,
    ):
        with pytest.raises(Exception) as exc:
            await service.generate_star_answer("query", [])

        assert "Permanent Error" in str(exc.value)
        # Verify telemetry error counter was incremented
        mock_telemetry.llm_errors.labels.return_value.inc.assert_called_once()


@pytest.mark.asyncio
async def test_get_embeddings_retry_on_429():
    """Verify that get_embeddings retries on 429 errors."""
    service = LLMService()

    mock_response = AsyncMock()
    mock_response.embeddings = [MagicMock(values=[0.1, 0.2])]

    # Fail twice, succeed on 3rd
    side_effect = [
        Exception("429 Resource Exhausted"),
        Exception("RESOURCE_EXHAUSTED"),
        mock_response,
    ]

    with patch.object(
        service.client.aio.models, "embed_content", side_effect=side_effect
    ) as mock_embed:
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            res = await service.get_embeddings("hello")
            assert res == [0.1, 0.2]
            assert mock_embed.call_count == 3
            assert mock_sleep.call_count == 2


@pytest.mark.asyncio
async def test_get_embeddings_failure_telemetry():
    """Verify embedding failure telemetry."""
    service = LLMService()

    with (
        patch.object(
            service.client.aio.models,
            "embed_content",
            side_effect=Exception("Embed Fail"),
        ),
        patch("app.services.llm_service.telemetry") as mock_telemetry,
    ):
        with pytest.raises(Exception, match="Embed Fail"):
            await service.get_embeddings("hello")

        mock_telemetry.llm_errors.labels.assert_called_with(model="embed")
        mock_telemetry.llm_errors.labels.return_value.inc.assert_called_once()
