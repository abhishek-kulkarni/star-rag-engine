import asyncio
import logging

from google import genai
from google.genai import types

from app.config.prompts import get_system_instructions, get_user_prompt
from app.config.settings import settings
from app.core.logging import telemetry
from app.models.schemas import STARAnswerResponse


class LLMService:
    """
    Handles interactions with the modern Google GenAI SDK.
    Uses gemini-embedding-2 for vectors and gemini-2.0-flash for generation.
    """

    def __init__(self):
        # The new SDK uses a unified Client object
        self.client = genai.Client(api_key=settings.GEMINI_API_KEY)
        self.embedding_model = "gemini-embedding-2"
        self.generation_model = "gemini-2.5-flash"

    async def get_embeddings(self, text: str) -> list[float]:
        """
        Generates 768-dimensional embeddings for a given text.
        Includes resilient retry logic for 429 Rate Limits.
        """
        max_retries = 5
        for attempt in range(max_retries):
            try:
                # Using .aio namespace for true non-blocking execution
                response = await self.client.aio.models.embed_content(
                    model=self.embedding_model,
                    contents=text,
                    config=types.EmbedContentConfig(
                        task_type="RETRIEVAL_DOCUMENT", output_dimensionality=768
                    ),
                )
                # Track Telemetry (Estimation)
                # Gemini embeddings don't return usage_metadata, so we estimate
                estimated_tokens = max(1, len(text) // 4)
                telemetry.track_tokens(prompt=estimated_tokens, completion=0)

                return response.embeddings[0].values
            except Exception as e:
                logging.warning(
                    f"[LLMService] Embedding attempt {attempt} failed: {str(e)}"
                )
                if (
                    "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)
                ) and attempt < max_retries - 1:
                    import random

                    # Exponential Backoff + Jitter
                    wait_time = (2**attempt) * 2 + random.uniform(0, 0.5)
                    await asyncio.sleep(wait_time)
                    continue
                telemetry.llm_errors.labels(model="embed").inc()
                raise

        raise RuntimeError("get_embeddings failed")  # pragma: no cover

    def get_embeddings_batch_sync(self, texts: list[str]) -> list[list[float]]:
        logging.info(f"[LLMService] Embedding batch of {len(texts)} chunks")
        """
        Synchronous version of batch embedding for Celery workers.
        """
        # Explicitly wrap strings in Content objects to ensure the API
        # returns one embedding per item, rather than aggregating them.
        contents = [types.Content(parts=[types.Part.from_text(text=t)]) for t in texts]

        response = self.client.models.embed_content(
            model=self.embedding_model,
            contents=contents,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT", output_dimensionality=768
            ),
        )

        # Track Telemetry (Estimation)
        # Gemini embeddings don't return usage_metadata, so we estimate
        # based on character counts (Avg ~4 chars per token).
        total_chars = sum(len(t) for t in texts)
        estimated_tokens = max(1, total_chars // 4)

        telemetry.track_tokens(prompt=estimated_tokens, completion=0)

        return [emb.values for emb in response.embeddings]

    async def generate_star_answer(
        self,
        query: str,
        retrieved_chunks: list[dict],
        plan_artifacts_text: str | None = None,
    ) -> STARAnswerResponse:
        """
        Generates a structured STAR answer grounded in the provided contexts.
        Uses a dynamic system instruction to inject rubrics and anti-bias guardrails.
        Includes basic retry logic for 429 Rate Limits.
        """
        system_instruction = get_system_instructions(plan_artifacts_text)
        user_prompt = get_user_prompt(query, retrieved_chunks)

        # 4. Generate Answer with Resilient Retry Logic (Exponential Backoff + Jitter)
        max_retries = 5
        response = None
        for attempt in range(max_retries):
            try:
                # Using .aio.models.generate_content for asynchronous generation
                response = await self.client.aio.models.generate_content(
                    model=self.generation_model,
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        response_mime_type="application/json",
                        response_schema=STARAnswerResponse,
                        temperature=0.0,
                    ),
                )
                break  # Success!
            except Exception as e:
                logging.warning(
                    f"[LLMService] Retry attempt {attempt} failed: {str(e)}"
                )
                if (
                    "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)
                ) and attempt < max_retries - 1:
                    import random

                    # Exponential Backoff: 2, 4, 8, 16...
                    # Jitter: +/- up to 500ms
                    wait_time = (2**attempt) * 2 + random.uniform(0, 0.5)
                    await asyncio.sleep(wait_time)
                    continue
                telemetry.llm_errors.labels(model="generate").inc()
                raise

        # 5. Track Telemetry (Tokens)
        if response and response.usage_metadata:
            telemetry.track_tokens(
                prompt=response.usage_metadata.prompt_token_count,
                completion=response.usage_metadata.candidates_token_count,
            )

        if response:
            return response.parsed

        raise RuntimeError("generate_star_answer failed")  # pragma: no cover


llm_service = LLMService()
