"""
ScaleDown Context Compressor — wrapper around the official ScaleDown SDK.

Uses ScaleDownCompressor.compress() to reduce context tokens before LLM generation.

SDK Reference: https://github.com/scaledown-team/scaledown

Usage:
    from src.compression.scaledown_compressor import ScaleDownContextCompressor
    compressor = ScaleDownContextCompressor()
    result = compressor.compress(context="...", query="...")
"""
import os
import time
from src.evaluation.token_tracker import count_tokens


class ScaleDownContextCompressor:
    """Wrapper around the ScaleDown SDK for RAG context compression."""

    def __init__(self, target_model="gpt-4o-mini", rate="auto", api_key=None):
        self.target_model = target_model
        self.rate = rate
        self.api_key = api_key or os.getenv("SCALEDOWN_API_KEY", "")
        self._compressor = None

    def _get_compressor(self):
        """Lazy-load the ScaleDown compressor."""
        if self._compressor is None:
            try:
                from scaledown import ScaleDownCompressor
                self._compressor = ScaleDownCompressor(
                    target_model=self.target_model,
                    rate=self.rate,
                )
            except ImportError:
                raise ImportError(
                    "scaledown package not installed. "
                    "Install with: pip install scaledown"
                )
        return self._compressor

    def compress(self, context: str, query: str) -> dict:
        """
        Compress context using ScaleDown API.

        Args:
            context: The raw retrieved context to compress.
            query: The user query (used as prompt for compression).

        Returns:
            dict with keys:
                - compressed_context: The compressed text
                - original_tokens: Token count before compression
                - compressed_tokens: Token count after compression
                - compression_ratio: original / compressed
                - compression_time: Time taken for compression
        """
        original_tokens = count_tokens(context)

        try:
            compressor = self._get_compressor()

            start = time.time()
            result = compressor.compress(context=context, prompt=query)
            compression_time = time.time() - start

            # Extract compressed text from result
            compressed_context = str(result)
            compressed_tokens = count_tokens(compressed_context)

            # Try to get metrics from SDK result
            try:
                if hasattr(result, 'metrics'):
                    original_tokens = getattr(result.metrics, 'original_prompt_tokens', original_tokens)
                    compressed_tokens = getattr(result.metrics, 'compressed_prompt_tokens', compressed_tokens)
            except Exception:
                pass

        except Exception as e:
            print(f"  [ScaleDown] SDK error: {e}. Using fallback compression.")
            compressed_context, compressed_tokens, compression_time = self._fallback_compress(context, query)

        compression_ratio = round(original_tokens / max(compressed_tokens, 1), 2)

        return {
            "compressed_context": compressed_context,
            "original_tokens": original_tokens,
            "compressed_tokens": compressed_tokens,
            "compression_ratio": compression_ratio,
            "compression_time": round(compression_time, 4),
        }

    def _fallback_compress(self, context: str, query: str):
        """
        Fallback compression when SDK is unavailable.
        Uses simple extractive approach: keep sentences most relevant to query.
        """
        import re

        start = time.time()

        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', context)
        query_words = set(query.lower().split())

        # Score sentences by relevance to query
        scored = []
        for s in sentences:
            s_words = set(s.lower().split())
            overlap = len(query_words & s_words)
            scored.append((overlap, s))

        # Keep top 60% of sentences by relevance
        scored.sort(key=lambda x: x[0], reverse=True)
        keep_count = max(1, int(len(scored) * 0.6))
        kept = [s for _, s in scored[:keep_count]]

        compressed = " ".join(kept)
        compression_time = time.time() - start
        compressed_tokens = count_tokens(compressed)

        return compressed, compressed_tokens, compression_time
