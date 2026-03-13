from langchain_openai import ChatOpenAI
from src.evaluation.token_tracker import count_tokens
import time

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

MAP_PROMPT = """
Summarize the following document chunk concisely while preserving key facts,
figures, and technical details.

Chunk:
{chunk}
"""

def map_summarize(chunks):
    summaries = []
    total_tokens = 0
    total_time = 0

    for doc in chunks:
        start = time.time()
        prompt = MAP_PROMPT.format(chunk=doc.page_content)
        response = llm.invoke(prompt).content
        latency = time.time() - start

        tokens = count_tokens(prompt) + count_tokens(response)
        total_tokens += tokens
        total_time += latency

        summaries.append(response)

    return summaries, total_tokens, total_time