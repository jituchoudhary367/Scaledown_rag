from src.summarization.map_summarizer import map_summarize
from src.summarization.reduce_summarizer import reduce_summarize

def build_summary_context(chunks):
    map_summaries, map_tokens, map_time = map_summarize(chunks)
    final_summary, reduce_tokens, reduce_time = reduce_summarize(map_summaries)

    stats = {
        "map_tokens": map_tokens,
        "reduce_tokens": reduce_tokens,
        "total_summary_tokens": map_tokens + reduce_tokens,
        "map_time": map_time,
        "reduce_time": reduce_time,
        "total_summary_time": map_time + reduce_time
    }

    return final_summary, stats