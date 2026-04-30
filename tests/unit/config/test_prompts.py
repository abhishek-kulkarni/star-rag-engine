from app.config.prompts import (
    BASE_SYSTEM_PROMPT,
    get_system_instructions,
    get_user_prompt,
)


def test_get_system_instructions_no_artifacts():
    """Verify system prompt without plan artifacts."""
    instructions = get_system_instructions(None)
    assert instructions == BASE_SYSTEM_PROMPT


def test_get_system_instructions_with_artifacts():
    """Verify system prompt with injected plan artifacts (Rule 2.1)."""
    rubric = "Always use Amazon Leadership Principles."
    instructions = get_system_instructions(rubric)

    assert BASE_SYSTEM_PROMPT in instructions
    assert "USER-SPECIFIC EXECUTION RULES (RUBRIC):" in instructions
    assert rubric in instructions


def test_get_user_prompt_formatting():
    """Verify dynamic RAG context formatting with Chunk IDs."""
    query = "How do you handle scaling?"
    chunks = [
        {"id": 1, "text_content": "We used Kubernetes."},
        {"id": 2, "text_content": "Auto-scaling was enabled."},
    ]

    prompt = get_user_prompt(query, chunks)

    assert query in prompt
    assert "--- CHUNK ID: 1 ---" in prompt
    assert "We used Kubernetes." in prompt
    assert "--- CHUNK ID: 2 ---" in prompt
    assert "Auto-scaling was enabled." in prompt
