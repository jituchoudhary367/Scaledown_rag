"""
Experiment Runner — Orchestrates all three RAG pipelines and generates CSV results.

Produces:
  - logs/results.csv       (per-query results for all pipelines)
  - logs/summary_metrics.csv  (aggregated metrics per pipeline)

Usage:
  python experiments/run_all_pipelines.py
"""
import os
import sys
import csv
import json
import time
import numpy as np
from datetime import datetime

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from shared_config import load_questions, RESULTS_CSV, SUMMARY_CSV

# ── Pipeline Descriptions ──
PIPELINES = [
    {
        "name": "classic_rag",
        "label": "Classic RAG",
        "dir": os.path.join(PROJECT_ROOT, "classic"),
        "main_module": "classic.main",
        "skip_run": True,
    },
    {
        "name": "summarization_rag",
        "label": "RAG + Summarization",
        "dir": os.path.join(PROJECT_ROOT, "RAG_summarization"),
        "main_module": "RAG_summarization.main",
        "skip_run": True,
    },
    {
        "name": "scaledown_rag",
        "label": "ScaleDown RAG",
        "dir": os.path.join(PROJECT_ROOT, "scaledown_rag"),
        "main_module": "scaledown_rag.main",
        "skip_run": True,
    },
]

# CSV column definitions
RESULTS_COLUMNS = [
    "timestamp", "pipeline", "query", "answer",
    "retrieval_time", "compression_time", "generation_time", "total_latency",
    "input_tokens", "output_tokens", "total_tokens",
    "original_tokens", "optimized_tokens", "compression_ratio",
    "quality_score",
]

SUMMARY_COLUMNS = [
    "pipeline", "num_queries",
    "avg_latency", "p95_latency",
    "avg_retrieval_time", "avg_compression_time", "avg_generation_time",
    "avg_input_tokens", "avg_output_tokens", "avg_total_tokens",
    "avg_original_tokens", "avg_optimized_tokens",
    "avg_compression_ratio",
    "avg_quality",
]


import subprocess

def run_pipeline_standalone(pipeline_info):
    """Run a pipeline using subprocess to ensure environment isolation."""
    pipeline_dir = pipeline_info["dir"]
    label = pipeline_info["label"]

    print(f"\n{'='*60}")
    print(f"  Running: {label} (Isolated Process)")
    print(f"{'='*60}\n")

    try:
        skip_run = pipeline_info.get("skip_run", False)
        if not skip_run:

            result = subprocess.run(
                [sys.executable, "main.py"],
                cwd=pipeline_dir,
                capture_output=False, 
                text=True
            )
            
            if result.returncode != 0:
                print(f"\n  ✗ {label} failed with return code {result.returncode}")
                return []
        else:
            print(f"  (Skipping execution, loading existing results...)")

        pipeline_name = pipeline_info["name"]
        if "classic" in pipeline_name:
            json_file = "classic_results.json"
        elif "summarization" in pipeline_name:
            json_file = "summarization_results.json"
        elif "scaledown" in pipeline_name:
            json_file = "scaledown_results.json"
        else:
            json_file = "results.json"
            
        
        log_paths = [
            os.path.join(pipeline_dir, "logs", json_file),
            os.path.join(PROJECT_ROOT, "logs", json_file)
        ]
        
        for path in log_paths:
            if os.path.exists(path):
                with open(path, "r") as f:
                    return json.load(f)
                    
        print(f"  Warning: Results file {json_file} not found for {label}")
        return []

    except Exception as e:
        print(f"\n  ERROR running {label}: {e}")
        return []


def write_results_csv(all_results, output_path):
    """Write per-query results to CSV."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_COLUMNS)
        writer.writeheader()

        for r in all_results:
            latency = r.get("latency", {})
            tokens = r.get("tokens", {})
            row = {
                "timestamp": datetime.now().isoformat(),
                "pipeline": r.get("pipeline", ""),
                "query": r.get("query", ""),
                "answer": r.get("answer", "")[:500],  
                "retrieval_time": latency.get("retrieval_time", 0),
                "compression_time": latency.get("compression_time", 0),
                "generation_time": latency.get("generation_time", 0),
                "total_latency": latency.get("total_latency", 0),
                "input_tokens": tokens.get("input_tokens", 0),
                "output_tokens": tokens.get("output_tokens", 0),
                "total_tokens": tokens.get("total_tokens", 0),
                "original_tokens": tokens.get("original_tokens", 0),
                "optimized_tokens": tokens.get("optimized_tokens", 0),
                "compression_ratio": tokens.get("compression_ratio", 1.0),
                "quality_score": r.get("quality", 0.0),
            }
            writer.writerow(row)

    print(f"\n  Results CSV → {output_path}")


def write_summary_csv(all_results, output_path):
    """Compute and write aggregated metrics per pipeline."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    
    pipelines = {}
    for r in all_results:
        name = r.get("pipeline", "unknown")
        if name not in pipelines:
            pipelines[name] = []
        pipelines[name].append(r)

    rows = []
    for name, results in pipelines.items():
        latencies = [r["latency"]["total_latency"] for r in results if "latency" in r]
        retrieval_times = [r["latency"]["retrieval_time"] for r in results if "latency" in r]
        compression_times = [r["latency"]["compression_time"] for r in results if "latency" in r]
        generation_times = [r["latency"]["generation_time"] for r in results if "latency" in r]
        input_tokens = [r["tokens"]["input_tokens"] for r in results if "tokens" in r]
        output_tokens = [r["tokens"]["output_tokens"] for r in results if "tokens" in r]
        total_tokens = [r["tokens"]["total_tokens"] for r in results if "tokens" in r]
        original_tokens = [r["tokens"]["original_tokens"] for r in results if "tokens" in r]
        optimized_tokens = [r["tokens"]["optimized_tokens"] for r in results if "tokens" in r]
        compression_ratios = [r["tokens"]["compression_ratio"] for r in results if "tokens" in r]
        quality_scores = [r.get("quality", 0.0) for r in results]

        row = {
            "pipeline": name,
            "num_queries": len(results),
            "avg_latency": round(np.mean(latencies), 4) if latencies else 0,
            "p95_latency": round(np.percentile(latencies, 95), 4) if latencies else 0,
            "avg_retrieval_time": round(np.mean(retrieval_times), 4) if retrieval_times else 0,
            "avg_compression_time": round(np.mean(compression_times), 4) if compression_times else 0,
            "avg_generation_time": round(np.mean(generation_times), 4) if generation_times else 0,
            "avg_input_tokens": round(np.mean(input_tokens), 2) if input_tokens else 0,
            "avg_output_tokens": round(np.mean(output_tokens), 2) if output_tokens else 0,
            "avg_total_tokens": round(np.mean(total_tokens), 2) if total_tokens else 0,
            "avg_original_tokens": round(np.mean(original_tokens), 2) if original_tokens else 0,
            "avg_optimized_tokens": round(np.mean(optimized_tokens), 2) if optimized_tokens else 0,
            "avg_compression_ratio": round(np.mean(compression_ratios), 2) if compression_ratios else 1.0,
            "avg_quality": round(np.mean(quality_scores), 4) if quality_scores else 0,
        }
        rows.append(row)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"  Summary CSV → {output_path}")


def main():
    print("╔" + "═"*58 + "╗")
    print("║   RAG Architecture Comparison — Experiment Runner     ║")
    print("╚" + "═"*58 + "╝")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    all_results = []

    for pipeline_info in PIPELINES:
        start = time.time()
        results = run_pipeline_standalone(pipeline_info)
        elapsed = time.time() - start

        if results:
            all_results.extend(results)
            print(f"\n  ✓ {pipeline_info['label']}: {len(results)} queries in {elapsed:.1f}s")
        else:
            print(f"\n  ✗ {pipeline_info['label']}: No results (pipeline may have errored)")

    if all_results:
        print(f"\n{'='*60}")
        print(f"  Generating CSV reports...")
        print(f"{'='*60}")

        write_results_csv(all_results, RESULTS_CSV)
        write_summary_csv(all_results, SUMMARY_CSV)

        print(f"\n  Total: {len(all_results)} results across {len(PIPELINES)} pipelines")
    else:
        print("\n  No results to write. Check pipeline errors above.")

    print(f"\n{'='*60}")
    print(f"  Experiment complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
