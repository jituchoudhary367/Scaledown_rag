"""
LLM-as-Judge quality evaluator.
Returns a quality score on 0-1 scale.
"""
import json
from langchain_openai import ChatOpenAI
from src.config import get_llm_kwargs

JUDGE_PROMPT = """
You are an impartial evaluator.

Question:
{question}

Answer:
{answer}

Score the answer from 1 to 5 on each criterion:

1. Factual Correctness
2. Completeness
3. Relevance
4. No Hallucinations

Return ONLY a JSON object:
{{
 "factual": score,
 "completeness": score,
 "relevance": score,
 "hallucination": score,
 "overall": average_score
}}
"""


def evaluate_answer(question, answer):
    """Evaluate answer quality. Returns dict with 'overall' key on 0-1 scale."""
    judge_llm = ChatOpenAI(**get_llm_kwargs())
    prompt = JUDGE_PROMPT.format(question=question, answer=answer)
    result = judge_llm.invoke(prompt).content

    try:
        scores = json.loads(result)
        overall = scores.get("overall", 3.0)
        # Normalize to 0-1 scale (original is 1-5)
        normalized = round(overall / 5.0, 4)
        scores["overall"] = normalized
        return scores
    except Exception:
        return {"overall": 0.6}  # Default fallback