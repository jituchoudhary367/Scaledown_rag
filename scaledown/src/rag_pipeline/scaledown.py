import time
from scaledown import ScaleDownCompressor
from langchain_openai import ChatOpenAI
from src.evaluation.token_tracker import count_tokens

compressor = ScaleDownCompressor(
    target_model="gpt-4o",
    rate="auto"
)

def run_scaledown_rag(query, retriever):
    start_total = time.time()

    # retrieve
    docs = retriever.get_relevant_documents(query)

    # combine context
    raw_context = "\n".join([d.page_content for d in docs])

    # compress
    start_comp = time.time()
    result = compressor.compress(context=raw_context, prompt="")
    comp_time = time.time() - start_comp

    comp_context = result.output
    comp_stats = {
        "before_tokens": result.stats.input_tokens,
        "after_tokens": result.stats.output_tokens,
        "ratio": result.stats.ratio
    }

    # generation
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = f"Context:\n{comp_context}\n\nQuestion:\n{query}"
    start_gen = time.time()
    answer = llm.invoke(prompt).content
    gen_time = time.time() - start_gen

    total_time = time.time() - start_total

    # tokens
    query_tokens = count_tokens(query)
    context_tokens = count_tokens(comp_context)
    answer_tokens = count_tokens(answer)

    token_usage = {
        "query_tokens": query_tokens,
        "context_tokens": context_tokens,
        "answer_tokens": answer_tokens,
        "compression": comp_stats,
        "total_tokens": query_tokens + context_tokens + answer_tokens + comp_stats["after_tokens"]
    }

    latency = {
        "compression_time": comp_time,
        "generation_time": gen_time,
        "total_time": total_time
    }

    return answer, latency, token_usage