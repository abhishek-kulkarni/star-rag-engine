from google import genai
from google.genai import types

from app.config.prompts import get_system_instructions, get_user_prompt
from app.config.settings import settings
from app.models.schemas import STARAnswerResponse


class LLMService:
    """
    Handles interactions with the modern Google GenAI SDK.
    Uses text-embedding-004 for vectors and gemini-2.0-flash for generation.
    """

    def __init__(self):
        # The new SDK uses a unified Client object
        self.client = genai.Client(api_key=settings.GEMINI_API_KEY)
        self.embedding_model = "text-embedding-004"
        self.generation_model = "gemini-2.0-flash"

    async def get_embeddings(self, text: str) -> list[float]:
        """
        Generates 768-dimensional embeddings for a given text.
        """
        # Using .aio namespace for true non-blocking execution
        response = await self.client.aio.models.embed_content(
            model=self.embedding_model,
            contents=text,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
        )
        return response.embeddings[0].values

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

        # The SDK automatically parses the JSON into the response_schema
        # (Pydantic model)
        return response.parsed
