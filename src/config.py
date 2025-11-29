"""Central configuration for GDPR RAG system parameters."""
from dataclasses import dataclass

@dataclass
class AppConfig:
    chunk_size: int = 1500
    chunk_overlap: int = 200
    retriever_k: int = 5
    temperature: float = 0.0
    max_answer_tokens: int = 800
    score_threshold: float | None = None

CONFIG = AppConfig()
