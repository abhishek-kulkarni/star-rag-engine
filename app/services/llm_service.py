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
        self.generation_model = "gemini-2.0-flash"

    async def get_embeddings(self, text: str) -> list[float]:
        """
        Generates 768-dimensional embeddings for a given text.
        """
        # Using .aio namespace for true non-blocking execution
        response = await self.client.aio.models.embed_content(
            model=self.embedding_model,
            contents=text,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT", output_dimensionality=768
            ),
        )

        # Track Telemetry (Tokens)
        usage = getattr(response, "usage_metadata", None)
        if usage:
            telemetry.track_tokens(prompt=usage.prompt_token_count, completion=0)

        return response.embeddings[0].values

    def get_embeddings_batch_sync(self, texts: list[str]) -> list[list[float]]:
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

        # Track Telemetry (Tokens)
        usage = getattr(response, "usage_metadata", None)
        if usage:
            telemetry.track_tokens(prompt=usage.prompt_token_count, completion=0)

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
        """
        system_instruction = get_system_instructions(plan_artifacts_text)
        user_prompt = get_user_prompt(query, retrieved_chunks)

        # Using .aio.models.generate_content for asynchronous generation
        response = await self.client.aio.models.generate_content(
            model=self.generation_model,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                response_schema=STARAnswerResponse,
                temperature=0.0,  # Ensure maximum determinism for Principal-level RAG
            ),
        )

        # 5. Track Telemetry (Tokens)
        if response.usage_metadata:
            telemetry.track_tokens(
                prompt=response.usage_metadata.prompt_token_count,
                completion=response.usage_metadata.candidates_token_count,
            )

        return response.parsed


llm_service = LLMService()
