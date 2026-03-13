"""Reduce stage: Combine partial summaries into a single consolidated summary."""
import time
from langchain_openai import ChatOpenAI
from src.config import get_llm_kwargs
from src.evaluation.token_tracker import count_tokens

REDUCE_PROMPT = """
Combine the following partial summaries into a coherent consolidated summary.
Preserve key findings, statistics, and conclusions.

Summaries:
{summaries}
"""


def reduce_summarize(summaries):
    """Combine summaries into one (Reduce stage)."""
    llm = ChatOpenAI(**get_llm_kwargs())
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            start = time.time()
            combined = "\n".join(summaries)
            prompt = REDUCE_PROMPT.format(summaries=combined)
            response = llm.invoke(prompt).content

            latency = time.time() - start
            tokens = count_tokens(prompt) + count_tokens(response)
            return response, tokens, latency
        except Exception as e:
            if "429" in str(e) and attempt < max_retries - 1:
                print(f"      Rate limit hit in reduce stage. Retrying in 10s... ({attempt + 1}/{max_retries})")
                time.sleep(10)
            else:
                print(f"      Error in reduce stage: {e}")
                return f"Error: {e}", 0, 0
