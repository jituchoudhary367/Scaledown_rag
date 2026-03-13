from src.ingest.pdf_loader import load_pdf
from src.ingest.text_chunker import chunk_documents
from src.retrieval.embedder import get_embedder
from src.retrieval.vector_store import build_vector_store
from src.retrieval.retriever import get_retriever
from src.rag_pipeline.classic_rag import run_classic_rag
from src.evaluation.metrics_logger import log_metrics
from src.evaluation.quality_eval import evaluate_answer
from tqdm import tqdm
import statistics
import os
from src.config import PDF_PATH

def load_questions(path):
    if not os.path.exists(path):
        print(f"Warning: Question file not found at {path}")
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [q.strip() for q in f.readlines() if q.strip()]

def main():
    print("Loading documents...")
    docs = load_pdf(PDF_PATH)
    
    print("Chunking...")
    chunks = chunk_documents(docs)
    
    print("Embedding + Indexing...")
    embedder = get_embedder()
    vs = build_vector_store(chunks, embedder)
    retriever = get_retriever(vs)

    print("Loading questions...")
    questions_path = "data/question.txt" if os.path.exists("data/question.txt") else "data/questions.txt"
    questions = load_questions(questions_path)

    if not questions:
        print("No questions found. Exiting.")
        return

    latencies = []
    total_tokens_list = []
    quality_results = []

    print(f"\nRunning batch experiment with {len(questions)} queries...\n")

    for q in tqdm(questions):
        try:
            answer, latency, tokens = run_classic_rag(q, retriever)
            quality = evaluate_answer(q, answer)

            latencies.append(latency)
            total_tokens_list.append(tokens["total_tokens"])
            quality_results.append(quality)

            log_metrics({
                "pipeline": "classic_rag",
                "query": q,
                "answer": answer,
                "latency_sec": latency,
                "tokens": tokens,
                "quality": quality
            })
        except Exception as e:
            print(f"\nError processing query '{q}': {e}")

    print("\n===== BATCH SUMMARY =====")
    print(f"Total Queries: {len(questions)}")
    
    if latencies:
        print(f"Average Latency: {statistics.mean(latencies):.2f}s")
        # P95 calculation
        sorted_latencies = sorted(latencies)
        p95_index = int(len(sorted_latencies) * 0.95)
        print(f"P95 Latency: {sorted_latencies[min(p95_index, len(sorted_latencies)-1)]:.2f}s")
    
    if total_tokens_list:
        print(f"Average Tokens: {statistics.mean(total_tokens_list):.0f}")

    quality_scores = [
        res.get("overall") for res in quality_results if res and res.get("overall") is not None
    ]
    if quality_scores:
        print(f"Average Quality Score: {statistics.mean(quality_scores):.2f}/5")
    
    print("=========================")

if __name__ == "__main__":
    main()
