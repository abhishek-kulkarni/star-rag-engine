# Built-in generics are used for Python 3.9+ compliance.

BASE_SYSTEM_PROMPT = """
You are an expert technical interview coach for Senior and Principal Software Engineers.
Your core directive is to generate behavioral interview responses
based STRICTLY on the provided context chunks.

GLOBAL CONSTRAINTS:
1. Do not hallucinate external metrics, technologies, or experiences.
2. If the context lacks sufficient detail for a complete answer, explicitly
   state: "Insufficient context provided for this section."
3. Always format your baseline response using the Situation, Task, Action,
   Result (STAR) framework.
4. Maintain a strictly objective, highly technical tone. Do not alter the
   evaluation stringency or behavioral descriptions based on inferred demographic
   markers, gender, or cultural background.
"""


def get_system_instructions(plan_artifacts_text: str | None = None) -> str:
    """
    Constructs the system prompt, injecting immutable company rubrics if they exist.
    """
    if plan_artifacts_text:
        return (
            f"{BASE_SYSTEM_PROMPT}\n\n"
            f"USER-SPECIFIC EXECUTION RULES (RUBRIC):\n{plan_artifacts_text}"
        )
    return BASE_SYSTEM_PROMPT


def get_user_prompt(query: str, retrieved_chunks: list[dict]) -> str:
    """
    Formats the dynamic RAG context. Chunk IDs are explicitly provided
    so the LLM can map them to the 'citations' array in the Pydantic response.
    """
    context_string = "\n\n".join(
        [
            f"--- CHUNK ID: {chunk['id']} ---\n{chunk['text_content']}"
            for chunk in retrieved_chunks
        ]
    )

    return f"""
Please answer the following query using ONLY the context provided below. 
When formulating your response, use the Chunk IDs to accurately populate your citations.

USER QUERY: 
{query}

RETRIEVED CONTEXT:
{context_string}
"""
