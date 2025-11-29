"""FAISS index construction and persistence utilities for GDPR chunks."""
from pathlib import Path
import pickle
import faiss
import tempfile
from langchain_community.vectorstores import FAISS
from .embedding_wrapper import OpenAIEmbeddingWrapper
from langchain.schema import Document
from typing import List

# Store index in OS temp directory to avoid Windows Unicode path issues with FAISS file IO.
INDEX_DIR = Path(tempfile.gettempdir()) / "gdpr_rag_index"
INDEX_DIR.mkdir(parents=True, exist_ok=True)
FAISS_INDEX_FILE = INDEX_DIR / "faiss.index"
DOCSTORE_FILE = INDEX_DIR / "docstore.pkl"


def build_index(chunks: List[Document], model: str = "text-embedding-3-small") -> FAISS:
    embeddings = OpenAIEmbeddingWrapper(model=model)
    store = FAISS.from_documents(chunks, embeddings)
    return store


def persist(store: FAISS):
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(store.index, str(FAISS_INDEX_FILE))
    with open(DOCSTORE_FILE, "wb") as f:
        pickle.dump({"docstore": store.docstore, "index_to_docstore_id": store.index_to_docstore_id}, f)


def load(model: str = "text-embedding-3-small") -> FAISS:
    idx = faiss.read_index(str(FAISS_INDEX_FILE))
    with open(DOCSTORE_FILE, "rb") as f:
        payload = pickle.load(f)
    return FAISS(
        embedding_function=OpenAIEmbeddingWrapper(model=model),
        index=idx,
        docstore=payload["docstore"],
        index_to_docstore_id=payload["index_to_docstore_id"],
    )


def load_or_build(chunks: List[Document]) -> FAISS:
    if FAISS_INDEX_FILE.exists() and DOCSTORE_FILE.exists():
        return load()
    store = build_index(chunks)
    persist(store)
    return store
