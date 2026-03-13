"""Embedder using HuggingFace sentence-transformers (free, no API key)."""
from langchain_huggingface import HuggingFaceEmbeddings
from src.config import EMBED_MODEL


def get_embedder():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)