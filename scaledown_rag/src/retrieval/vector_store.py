"""FAISS vector store builder."""
from langchain_community.vectorstores import FAISS


def build_vector_store(chunks, embedder):
    return FAISS.from_documents(chunks, embedder)
