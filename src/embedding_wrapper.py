"""OpenAI embedding wrapper compatible with FAISS usage (mimics LangChain embedding interface)."""
from __future__ import annotations
from typing import List
import os
from openai import OpenAI


class OpenAIEmbeddingWrapper:
    def __init__(self, model: str = "text-embedding-3-small"):
        self.model = model
        try:
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        except Exception:
            # Offline fallback if client initialization fails (missing key or lib issue)
            self.client = None

    # Allow wrapper to be used as a callable embedding_function by LangChain vectorstores.
    def __call__(self, text: str):  # pragma: no cover - thin delegation
        return self.embed_query(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        if not self.client or not getattr(self.client, "api_key", None):
            # Offline fallback: zero-vectors (fixed dim for small embedding model)
            return [[0.0] * 1536 for _ in texts]
        resp = self.client.embeddings.create(model=self.model, input=texts)
        return [d.embedding for d in resp.data]

    def embed_query(self, text: str) -> List[float]:
        if not self.client or not getattr(self.client, "api_key", None):
            return [0.0] * 1536
        resp = self.client.embeddings.create(model=self.model, input=[text])
        return resp.data[0].embedding
