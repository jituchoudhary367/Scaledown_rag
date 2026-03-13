"""
Classic RAG Pipeline - Core RAG logic
Flow: Query → Retriever → Top-K chunks → LLM answer (no context processing)
"""
import time
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from src.config import get_llm_kwargs
from src.evaluation.token_tracker import count_tokens


SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the question based ONLY on the "
    "provided context. If the answer is not in the context, say so."
)


def run_classic_rag(query, retriever):
    """
    Execute classic RAG pipeline and return standardized metrics.
    
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

    # Build context
    context_text = "\n\n".join(d.page_content for d in docs)

    # ── Generation ──
    t_gen_start = time.time()
    messages = prompt_template.format_messages(context=context_text, question=query)
    response = llm.invoke(messages)
    generation_time = time.time() - t_gen_start

    answer = response.content
    total_latency = retrieval_time + generation_time

    # ── Token counts ──
    query_tokens = count_tokens(query)
    context_tokens = count_tokens(context_text)
    answer_tokens = count_tokens(answer)
    input_tokens = query_tokens + context_tokens
    total_tokens = input_tokens + answer_tokens

    latency = {
        "retrieval_time": round(retrieval_time, 4),
        "compression_time": 0.0,
        "generation_time": round(generation_time, 4),
        "total_latency": round(total_latency, 4),
    }

    tokens = {
        "input_tokens": input_tokens,
        "output_tokens": answer_tokens,
        "total_tokens": total_tokens,
        "original_tokens": context_tokens,
        "optimized_tokens": context_tokens,  # No compression
        "compression_ratio": 1.0,
    }

    return answer, latency, tokens