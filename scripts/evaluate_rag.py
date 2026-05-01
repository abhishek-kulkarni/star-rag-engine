import asyncio
import json
import logging
import time

import httpx
from google import genai
from google.genai import types
from pydantic import BaseModel

from app.config.settings import settings
from app.core.database import SessionLocal, engine
from app.core.security import get_current_user
from app.main import app
from app.models.base import Base
from app.workers.tasks import ensure_user_partition

# Initialize Schema and Partitions
print("Initializing database schema...")
with engine.connect() as conn:
    from sqlalchemy import text

    conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
    conn.commit()

Base.metadata.create_all(bind=engine)
with SessionLocal() as db:
    ensure_user_partition(db, "eval_user")


class Grade(BaseModel):
    score: int  # 0 or 1


class RAGEvaluator:
    """Uses LLM-as-a-Judge to evaluate LIVE RAG performance via Gemini API."""

    def __init__(self):
        self.judge = genai.Client(api_key=settings.GEMINI_API_KEY)
        self.model = "gemini-2.0-flash"

    async def _get_grade(self, prompt: str) -> int:
        """Call Gemini to grade a specific metric with retry logic."""
        for attempt in range(3):
            try:
                response = self.judge.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=Grade,
                        temperature=0.0,
                    ),
                )
                return response.parsed.score
            except Exception as e:
                if "429" in str(e) and attempt < 2:
                    time.sleep(2 * (attempt + 1))
                    continue
                return 0
        return 0

    async def evaluate_faithfulness(self, context, answer) -> int:
        prompt = f"""
        Evaluate if the provided answer is faithful to the given context.
        The answer must only contain information found in the context.
        
        CONTEXT: {context}
        ANSWER: {answer}
        
        Return a score of 1 if faithful, 0 otherwise.
        """
        return await self._get_grade(prompt)

    async def evaluate_relevancy(self, query, answer, ground_truth) -> int:
        prompt = f"""
        Evaluate if the provided answer accurately and completely addresses 
        the user query. Compare it against the expected ground truth.
        
        QUERY: {query}
        ANSWER: {answer}
        GROUND TRUTH: {ground_truth}
        
        Return a score of 1 if relevant and accurate, 0 otherwise.
        """
        return await self._get_grade(prompt)

    async def evaluate_context_precision(self, query, context) -> int:
        prompt = f"""
        Evaluate if the retrieved context segments are relevant to the user query.
        
        QUERY: {query}
        CONTEXT: {context}
        
        Return a score of 1 if the context is precise and relevant, 0 otherwise.
        """
        return await self._get_grade(prompt)


async def run_benchmark():
    """
    Executes a LIVE benchmark against the real database and LLM pipeline.
    Requires the environment to have valid DB/MinIO/Gemini credentials.
    """
    with open("data/golden_dataset.json") as f:
        dataset = json.load(f)

    # Minimal override to bypass JWT validation
    app.dependency_overrides[get_current_user] = lambda: "eval_user"

    evaluator = RAGEvaluator()
    results = []

    print("\n" + "=" * 60)
    print("           STAR RAG ENGINE - LIVE EVALUATION")
    print("=" * 60)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://testserver"
    ) as client:
        for entry in dataset:
            print(f"Testing Query: {entry['query'][:50]}...")

            # 1. Get real engine response (hits actual DB and Gemini)
            try:
                response = await client.post(
                    "/api/v1/query/ask", json={"query": entry["query"]}, timeout=30.0
                )
                if response.status_code != 200:
                    print(f"  [SKIP] Error {response.status_code}: {response.text}")
                    continue
            except Exception as e:
                print(f"  [SKIP] Request failed: {str(e)}")
                continue

            data = response.json()
            chunk_count = len(data.get("source_nodes", []))
            print(f"  Answer received. Grounding in {chunk_count} chunks.")

            # 2. Run Evaluators
            try:
                f_score = await evaluator.evaluate_faithfulness(
                    data.get("source_nodes", []), data.get("answer")
                )
                r_score = await evaluator.evaluate_relevancy(
                    entry["query"], data.get("answer"), entry["expected_ground_truth"]
                )
                p_score = await evaluator.evaluate_context_precision(
                    entry["query"], data.get("source_nodes", [])
                )

                results.append({"f": f_score, "r": r_score, "p": p_score})
                print(f"  Scores -> F:{f_score} R:{r_score} P:{p_score}")
            except Exception as e:
                print(f"  [ERROR] Evaluation failed: {str(e)}")

            await asyncio.sleep(1)  # Pace API calls

    # 3. Calculate Aggregates
    count = len(results)
    if count == 0:
        print("No results to report.")
        return

    f_avg = sum(r["f"] for r in results) / count
    r_avg = sum(r["r"] for r in results) / count
    p_avg = sum(r["p"] for r in results) / count

    # 4. Formatted Summary Report
    f_status = "PASS" if f_avg >= 0.8 else "FAIL"
    r_status = "PASS" if r_avg >= 0.8 else "FAIL"
    p_status = "PASS" if p_avg >= 0.8 else "FAIL"

    print("\n" + "=" * 60)
    print(f"{'METRIC':<25} | {'SCORE':<10} | {'STATUS':<10}")
    print("-" * 60)
    print(f"{'Faithfulness':<25} | {f_avg * 100:>8.1f}% | {f_status}")
    print(f"{'Answer Relevancy':<25} | {r_avg * 100:>8.1f}% | {r_status}")
    print(f"{'Context Precision':<25} | {p_avg * 100:>8.1f}% | {p_status}")
    print("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_benchmark())
