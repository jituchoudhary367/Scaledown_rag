"""Map stage: Summarize each retrieved chunk individually."""
import time
from langchain_openai import ChatOpenAI
from src.config import get_llm_kwargs
from src.evaluation.token_tracker import count_tokens

MAP_PROMPT = """
Summarize the following document chunk concisely while preserving key facts,
figures, and technical details.

Chunk:
{chunk}
"""


def map_summarize(chunks):
    """Summarize each chunk individually (Map stage)."""
    llm = ChatOpenAI(**get_llm_kwargs())
    summaries = []
    total_tokens = 0
    total_time = 0

    for i, doc in enumerate(chunks):
        # Throttle to avoid per-minute limits
        if i > 0:
            time.sleep(3)
            
        max_retries = 3
        for attempt in range(max_retries):
            try:
                start = time.time()
                prompt = MAP_PROMPT.format(chunk=doc.page_content)
                response = llm.invoke(prompt).content
                latency = time.time() - start

                tokens = count_tokens(prompt) + count_tokens(response)
                total_tokens += tokens
                total_time += latency
                summaries.append(response)
                break
            except Exception as e:
                if "429" in str(e) and attempt < max_retries - 1:
                    print(f"      Rate limit hit in map stage. Retrying in 10s... ({attempt + 1}/{max_retries})")
                    time.sleep(10)
                else:
                    print(f"      Error in map stage: {e}")
                    summaries.append(f"Error: {e}")
                    break

    return summaries, total_tokens, total_time
