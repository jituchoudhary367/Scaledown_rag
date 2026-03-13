"""
ScaleDown RAG Pipeline - Core RAG logic
Flow: Query → Retriever → Context Compression (ScaleDown) → LLM answer
"""
import time
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from src.config import get_llm_kwargs, SCALEDOWN_TARGET_MODEL, SCALEDOWN_RATE
from src.compression.scaledown_compressor import ScaleDownContextCompressor
from src.evaluation.token_tracker import count_tokens


SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the question based ONLY on the "
    "provided context. If the answer is not in the context, say so."
)

# Initialize compressor (lazy-loads SDK)
_compressor = None


def _get_compressor():
    global _compressor
    if _compressor is None:
        _compressor = ScaleDownContextCompressor(
            target_model=SCALEDOWN_TARGET_MODEL,
            rate=SCALEDOWN_RATE,
        )
    return _compressor


def run_scaledown_rag(query, retriever):
    """
    Execute ScaleDown compression RAG pipeline.

    Returns:
        tuple: (answer, latency_dict, token_dict)
    """
    llm = ChatOpenAI(**get_llm_kwargs())
    compressor = _get_compressor()

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ])

    # ── Retrieval ──
    t_ret_start = time.time()
    docs = retriever.invoke(query)
    retrieval_time = time.time() - t_ret_start

    # Build raw context
    raw_context = "\n\n".join(d.page_content for d in docs)

    # ── Compression (ScaleDown) ──
    max_retries = 3
    for attempt in range(max_retries):
        try:
            comp_result = compressor.compress(context=raw_context, query=query)
            compressed_context = comp_result["compressed_context"]
            compression_time = comp_result["compression_time"]
            original_tokens = comp_result["original_tokens"]
            compressed_tokens = comp_result["compressed_tokens"]
            compression_ratio = comp_result["compression_ratio"]
            break
        except Exception as e:
            if "429" in str(e) and attempt < max_retries - 1:
                print(f"      Rate limit hit in compression. Retrying in 10s... ({attempt + 1}/{max_retries})")
                time.sleep(10)
            else:
                print(f"      Error in compression: {e}")
                compressed_context = raw_context # Fallback to raw
                compression_time = 0
                original_tokens = count_tokens(raw_context)
                compressed_tokens = original_tokens
                compression_ratio = 1.0
                break

    # ── Generation ──
    max_retries = 3
    for attempt in range(max_retries):
        try:
            t_gen_start = time.time()
            messages = prompt_template.format_messages(context=compressed_context, question=query)
            response = llm.invoke(messages)
            generation_time = time.time() - t_gen_start
            answer = response.content
            break
        except Exception as e:
            if "429" in str(e) and attempt < max_retries - 1:
                print(f"      Rate limit hit in generation. Retrying in 10s... ({attempt + 1}/{max_retries})")
                time.sleep(10)
            else:
                print(f"      Error in generation: {e}")
                answer = f"Error: {e}"
                generation_time = 0
                break
    total_latency = retrieval_time + generation_time  # compression_time excluded per design

    # ── Token counts ──
    query_tokens = count_tokens(query)
    answer_tokens = count_tokens(answer)
    input_tokens = query_tokens + compressed_tokens
    total_tokens = input_tokens + answer_tokens

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
        "optimized_tokens": compressed_tokens,
        "compression_ratio": compression_ratio,
    }

    return answer, latency, tokens
