"""
ScaleDown RAG Pipeline - Main entry point
Flow: PDF → Chunks → Embeddings → FAISS → Retriever → ScaleDown Compress → LLM → Metrics
"""
import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import PDF_PATH, LOG_PATH
from src.ingest.pdf_loader import load_pdf
from src.ingest.text_chunker import chunk_documents
from src.retrieval.embedder import get_embedder
from src.retrieval.vector_store import build_vector_store
from src.retrieval.retriever import get_retriever
from src.rag_pipeline.scaledown_rag import run_scaledown_rag
from src.evaluation.quality_eval import evaluate_answer
from src.evaluation.metrics_logger import log_metrics


def load_questions(path):
    if not os.path.exists(path):
        print(f"Warning: Question file not found at {path}")
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [q.strip() for q in f.readlines() if q.strip()]


def main():
    print("=" * 60)
    print("  ScaleDown Context Compression RAG Pipeline")
    print("=" * 60)

    # 1. Ingestion
    print("\n[1/4] Loading PDF...")
    docs = load_pdf(PDF_PATH)
    print(f"  Loaded {len(docs)} pages")

    # 2. Chunking
    print("[2/4] Chunking documents...")
    chunks = chunk_documents(docs)
    print(f"  Created {len(chunks)} chunks")

    # 3. Embedding & Indexing
    print("[3/4] Building vector store...")
    embedder = get_embedder()
    vs = build_vector_store(chunks, embedder)
    retriever = get_retriever(vs)
    print("  FAISS index ready")

    # 4. Run queries
    questions_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "questions.txt")
    if not os.path.exists(questions_path):
        questions_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "questions.txt")
    questions = load_questions(questions_path)[:5]  # Limited run: 5 questions only
    if not questions:
        print("No questions found. Exiting.")
        return

    print(f"[4/4] Running {len(questions)} queries...\n")

    results = []

    for i, question in enumerate(questions, 1):
        print(f"  [{i}/{len(questions)}] {question[:60]}...")

        try:
            answer, latency, tokens = run_scaledown_rag(question, retriever)
            quality = evaluate_answer(question, answer)
            quality_score = quality.get("overall", 0.0)
        except Exception as e:
            print(f"    ERROR: {e}")
            answer = f"Error: {e}"
            latency = {"retrieval_time": 0, "compression_time": 0, "generation_time": 0, "total_latency": 0}
            tokens = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "original_tokens": 0, "optimized_tokens": 0, "compression_ratio": 1.0}
            quality_score = 0.0
            quality = {"overall": 0.0}

        result = {
            "pipeline": "scaledown_rag",
            "query": question,
            "answer": answer,
            "latency": latency,
            "tokens": tokens,
            "quality": quality_score,
            "compression_ratio": tokens.get("compression_ratio", 1.0),
        }
        results.append(result)

        log_metrics({
            "pipeline": "scaledown_rag",
            "query": question,
            "answer": answer,
            "latency_sec": latency["total_latency"],
            "tokens": tokens,
            "quality": quality,
        })

        print(f"    Latency: {latency['total_latency']:.2f}s | Tokens: {tokens['total_tokens']} | Quality: {quality_score:.3f} | Compression: {tokens['compression_ratio']:.2f}x")
        time.sleep(2)

    # Save results
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    output_file = os.path.join(os.path.dirname(LOG_PATH), "scaledown_results.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"  ScaleDown RAG complete. Results → {output_file}")
    print(f"{'=' * 60}")

    return results


if __name__ == "__main__":
    main()
