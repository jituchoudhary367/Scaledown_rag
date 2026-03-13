from langchain_openai import ChatOpenAI

judge_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

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
    prompt = JUDGE_PROMPT.format(question=question, answer=answer)
    result = judge_llm.invoke(prompt).content

    try:
        import json
        return json.loads(result)
    except:
        return {"overall": None}