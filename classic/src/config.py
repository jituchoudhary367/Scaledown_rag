"""
Classic RAG Pipeline - Configuration
Imports shared constants and provides pipeline-specific settings.
"""
import os
import sys

# Add project root to path for shared_config access
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from shared_config import (
    OPENROUTER_API_KEY,
    LLM_MODEL,
    LLM_BASE_URL,
    LLM_TEMPERATURE,
    EMBED_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K,
    get_llm_kwargs,
    load_questions,
)

# Pipeline-specific paths
PIPELINE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PIPELINE_DIR, "data")
PDF_PATH = os.path.join(DATA_DIR, "ci_climate_report.pdf")
LOG_PATH = os.path.join(PIPELINE_DIR, "logs", "results.json")
QUESTIONS_PATH = os.path.join(DATA_DIR, "question.txt")