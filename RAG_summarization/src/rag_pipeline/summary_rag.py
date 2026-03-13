"""
RAG + Summarization Pipeline - Core RAG logic
Flow: Query → Retriever → Chunk summaries → Combine summaries → LLM answer
"""
import time
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from src.config import get_llm_kwargs
from src.summarization.summary_pipeline import build_summary_context
from src.evaluation.token_tracker import count_tokens


SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the question based ONLY on the "
    "provided summarized context. If the answer is not in the context, say so."
)


def run_summary_rag(query, retriever):
    """
    Execute summarization RAG pipeline and return standardized metrics.
    
    Returns:
        tuple: (answer, latency_dict, token_dict)
    """
    llm = ChatOpenAI(**get_llm_kwargs())

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ])

    # ── Retrieval ──
    t_ret_start = time.time()
    docs = retriever.invoke(query)
    retrieval_time = time.time() - t_ret_start

    # Original context tokens (before summarization)
    raw_context = "\n\n".join(d.page_content for d in docs)
    original_tokens = count_tokens(raw_context)

    # ── Map-Reduce Summarization (compression step) ──
    t_comp_start = time.time()
    summary_context, summary_stats = build_summary_context(docs)
    compression_time = time.time() - t_comp_start

    # Compressed context tokens
    optimized_tokens = count_tokens(summary_context)
    compression_ratio = round(original_tokens / max(optimized_tokens, 1), 2)

    # ── Generation ──
    t_gen_start = time.time()
    messages = prompt_template.format_messages(context=summary_context, question=query)
    response = llm.invoke(messages)
    generation_time = time.time() - t_gen_start

    answer = response.content
    total_latency = retrieval_time + generation_time  # Summarization time excluded per user request

    # ── Token counts ──
    query_tokens = count_tokens(query)
    answer_tokens = count_tokens(answer)
    input_tokens = query_tokens + optimized_tokens
    total_tokens = input_tokens + answer_tokens + summary_stats["total_summary_tokens"]

    latency = {
        "retrieval_time": round(retrieval_time, 4),
        "compression_time": round(compression_time, 4),
        "generation_time": round(generation_time, 4),
        "total_latency": round(total_latency, 4),
    }

    tokens = {
        "input_tokens": input_tokens,
        "output_tokens": answer_tokens,
        "total_tokens": total_tokens,
        "original_tokens": original_tokens,
        "optimized_tokens": optimized_tokens,
        "compression_ratio": compression_ratio,
    }

    return answer, latency, tokens