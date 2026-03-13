Build a Python project implementing a Retrieval-Augmented Generation pipeline that integrates ScaleDown’s official context compression framework.

OBJECTIVE:
Evaluate how algorithmic semantic compression affects latency, token efficiency, and answer quality compared to baseline RAG and summarization RAG.

DATA SOURCE:
Same PDF corpus used in previous experiments.

SYSTEM ARCHITECTURE:

1. INGESTION & RETRIEVAL
   Same as baseline:
   • PDF loading
   • Chunking
   • Embeddings
   • FAISS vector search
   • Top-K retrieval
2. SCALE-DOWN COMPRESSION LAYER
   • Concatenate retrieved chunks
   • Pass full context into ScaleDown Compressor
   • Use semantic compression optimizer
   • Automatically reduce context token count
   • Track compression ratio
3. ANSWER GENERATION
   • Provide compressed context to LLM
   • Generate answer
   • Single LLM call for answering
4. TOKEN ACCOUNTING
   Track:
   • Tokens before compression
   • Tokens after compression
   • Compression ratio
   • Query tokens
   • Answer tokens
   • Total tokens
5. LATENCY TRACKING
   Measure:
   • Retrieval time
   • Compression time
   • LLM generation time
   • Total pipeline time
6. QUALITY EVALUATION
   Same LLM-judge scoring as other experiments.
7. BATCH EXECUTION
   Run pipeline across question set automatically.
8. LOGGING FORMAT
   {
   "pipeline": "scaledown_rag",
   "latency": {...},
   "tokens": {
   "before_compression": ...,
   "after_compression": ...,
   "compression_ratio": ...,
   "total_tokens": ...
   },
   "quality": {...}
   }

CONSTRAINTS:
• Use official ScaleDown Python package
• Do NOT implement custom compression
• Only one LLM call for answering
• Preserve identical evaluation settings

OUTPUT:
Provide full implementation with modular architecture and dependency setup.

use this

pip install scaledown[semantic]

**from**scaledown**import**ScaleDownCompressor

**from**scaledown**import**ScaleDownCompressor

compressor**=**ScaleDownCompressor(
target_model**=**"gpt-4o",  **# match main answer model**
rate**=**"auto"**# let SD determine best compression rate**
)

# retrieved_context = concatenated chunks (string)

result**=**compressor**.**compress(**context**=**retrieved_context**, **prompt**=**""**)
compressed_text**=**result**.**output  **# compressed context**
compression_stats**=** {
"before_tokens": **result**.stats**.**input_tokens,
"after_tokens": **result**.stats**.**output_tokens,
"compression_ratio": **result**.stats**.**ratio
}
