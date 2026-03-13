from langchain_openai import ChatOpenAI
from src.evaluation.token_tracker import count_tokens
import time

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

REDUCE_PROMPT = """
Combine the following partial summaries into a coherent consolidated summary.
Preserve key findings, statistics, and conclusions.

Summaries:
{summaries}
"""

def reduce_summarize(summaries):
    start = time.time()

    combined = "\n".join(summaries)
    prompt = REDUCE_PROMPT.format(summaries=combined)
    response = llm.invoke(prompt).content

    latency = time.time() - start
    tokens = count_tokens(prompt) + count_tokens(response)

    return response, tokens, latency