"""Text chunker using custom splitter."""
from src.config import CHUNK_SIZE, CHUNK_OVERLAP
from langchain_core.documents import Document


class SimpleRecursiveCharacterTextSplitter:
    def __init__(self, chunk_size, chunk_overlap):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_documents(self, documents):
        chunks = []
        for doc in documents:
            text = doc.page_content
            start = 0
            while start < len(text):
                end = start + self.chunk_size
                chunk_text = text[start:end]
                chunks.append(Document(
                    page_content=chunk_text,
                    metadata=doc.metadata.copy()
                ))
                if end >= len(text):
                    break
                start += (self.chunk_size - self.chunk_overlap)
        return chunks


def chunk_documents(documents):
    splitter = SimpleRecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_documents(documents)
