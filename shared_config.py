"""
Shared configuration for all RAG pipelines.
Central source of truth for experiment parameters.
"""
import os
from dotenv import load_dotenv

# Load .env from project root
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

# ── API Keys ──────────────────────────────────────────────
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
SCALEDOWN_API_KEY = os.getenv("SCALEDOWN_API_KEY", "")

# ── LLM Configuration (OpenRouter) ───────────────────────
LLM_MODEL = os.getenv("LLM_MODEL", "")
LLM_BASE_URL = "https://openrouter.ai/api/v1"
LLM_TEMPERATURE = 0

# ── Embedding Configuration ──────────────────────────────
EMBED_MODEL = "all-MiniLM-L6-v2"  # HuggingFace local model (free, no API key)

# ── Chunking & Retrieval Constants ────────────────────────
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 5

# ── ScaleDown Configuration ──────────────────────────────
SCALEDOWN_TARGET_MODEL = "gpt-4o-mini"
SCALEDOWN_RATE = "auto"

# ── File Paths ────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PDF_FILENAME = "ci_climate_report.pdf"
QUESTIONS_FILE = os.path.join(PROJECT_ROOT, "data", "questions.txt")

# ── Logging ───────────────────────────────────────────────
RESULTS_CSV = os.path.join(PROJECT_ROOT, "logs", "results.csv")
SUMMARY_CSV = os.path.join(PROJECT_ROOT, "logs", "summary_metrics.csv")


def get_llm_kwargs():
    """Return kwargs dict for ChatOpenAI using OpenRouter."""
    return {
        "model": LLM_MODEL,
        "api_key": OPENROUTER_API_KEY,
        "base_url": LLM_BASE_URL,
        "temperature": LLM_TEMPERATURE,
        "timeout": 60,
    }


def load_questions(path=None):
    """Load evaluation questions from file."""
    path = path or QUESTIONS_FILE
    if not os.path.exists(path):
        print(f"Warning: Questions file not found at {path}")
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [q.strip() for q in f.readlines() if q.strip()]
