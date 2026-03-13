Build a full Python project implementing a Retrieval-Augmented Generation pipeline that uses Map-Reduce Summarization before answer generation.

OBJECTIVE:
Evaluate the impact of hierarchical semantic summarization on latency, token usage, and answer quality.

DATA SOURCE:
Same PDF corpus used in baseline RAG experiment.

SYSTEM ARCHITECTURE:

1. INGESTION & RETRIEVAL
   Identical to Classic RAG:
   • PDF loading
   • Chunking
   • Embeddings
   • FAISS vector search
   • Top-K retrieval
2. MAP-REDUCE SUMMARIZATION LAYER

MAP STAGE:
• Each retrieved chunk summarized independently via LLM
• Preserve technical details and numeric facts
• Track tokens and latency for each call

REDUCE STAGE:
• Combine all mini-summaries
• Produce a unified global summary
• Track tokens and latency

1. ANSWER GENERATION
   • Provide reduced summary as context
   • Send summary + query to LLM
   • Generate final answer
2. TOKEN ACCOUNTING
   Track tokens across ALL stages:
   • Map summaries
   • Reduce summary
   • Final generation
   • Total tokens across pipeline
3. LATENCY TRACKING
   Measure separately:
   • Retrieval time
   • Map stage time
   • Reduce stage time
   • Final generation time
   • Total pipeline time
4. QUALITY EVALUATION
   Same LLM-judge system as baseline experiment.
5. BATCH EXECUTION
   Run automatically across question dataset.
6. LOGGING FORMAT
   {
   "pipeline": "summary_rag",
   "latency": {...},
   "tokens": {
   "map_tokens": ...,
   "reduce_tokens": ...,
   "generation_tokens": ...,
   "total_tokens": ...
   },
   "quality": {...}
   }

CONSTRAINTS:
• Use hierarchical Map-Reduce summarization
• Do NOT concatenate raw chunks directly
• Multiple LLM calls allowed
• Maintain experimental parity with baseline

OUTPUT:
Provide full project code, folder structure, and dependencies.
