"""Agentic helper tools for GDPR RAG.

Tools implemented (first iteration without LangGraph orchestration):
1. retrieve(store, query, k): vector retrieval returning docs & lightweight metadata.
2. citation_checker(answer): evaluate presence of GDPR structural citations (Articles / pages).
3. summarizer(llm, question, docs): produce concise, source-grounded summary.
4. hallucination_score(embeddings, answer, docs): naive semantic overlap scoring to flag low-support sentences.

Design Notes:
- Citation coverage threshold heuristic triggers summarizer fallback if low.
- Hallucination scoring is illustrative: it embeds answer sentences and compares cosine similarity with concatenated retrieved chunk embeddings.
- Errors in embedding or model calls are caught and converted to graceful degradation (tool returns partial signals).
"""
from __future__ import annotations
from typing import List, Dict, Any, Tuple
import re
import math

from langchain.schema import Document
import os
from openai import OpenAI
from .embedding_wrapper import OpenAIEmbeddingWrapper

CITATION_PATTERN = re.compile(r"(Article\s+\d+|p\.\d+)")


def retrieve(store, query: str, k: int = 5) -> List[Document]:
    return store.similarity_search(query, k=k)


def citation_checker(answer: str) -> Dict[str, Any]:
    citations = CITATION_PATTERN.findall(answer)
    sentences = [s.strip() for s in re.split(r"[.!?]\s+", answer) if s.strip()]
    coverage = len(citations) / max(len(sentences), 1)
    return {
        "citations": citations,
        "sentence_count": len(sentences),
        "citation_coverage": coverage,
        "low": coverage < 0.3,  # heuristic threshold
    }


def summarizer(question: str, docs: List[Document]) -> str:
    joined = "\n\n".join(d.page_content[:800] for d in docs)
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    except Exception:
        client = type("OfflineClient", (), {"api_key": None})()
    if not getattr(client, "api_key", None):
        return "[Offline summarizer] API key missing; using truncated context.\n" + joined[:500]
    messages = [
        {"role": "system", "content": "You are a GDPR assistant. Use ONLY provided text, cite Articles or pages."},
        {"role": "user", "content": f"Question: {question}\n\nText:\n{joined}\n\nAnswer with citations:"},
    ]
    resp = client.chat.completions.create(model="gpt-4o-mini", messages=messages, temperature=0)
    return resp.choices[0].message.content


def _embed(embeddings: OpenAIEmbeddingWrapper, texts: List[str]) -> List[List[float]]:
    return embeddings.embed_documents(texts)


def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return 0.0
    num = sum(x * y for x, y in zip(a, b))
    da = math.sqrt(sum(x * x for x in a))
    db = math.sqrt(sum(y * y for y in b))
    if da == 0 or db == 0:
        return 0.0
    return num / (da * db)


def hallucination_score(answer: str, docs: List[Document], model: str = "text-embedding-3-small") -> Dict[str, Any]:
    try:
        embeddings = OpenAIEmbeddingWrapper(model=model)
    except Exception:
        return {"error": "embedding_init_failed"}
    # Candidate sentences (exclude pure citation lines)
    sentences = [s.strip() for s in re.split(r"[.!?]\s+", answer) if s.strip()]
    if not sentences:
        return {"sentences": [], "scores": [], "low_support": []}
    # Build corpus embedding from docs
    corpus_text = "\n".join(d.page_content[:1000] for d in docs)
    try:
        corpus_vec = embeddings.embed_query(corpus_text)
    except Exception:
        return {"error": "embedding_corpus_failed"}
    sent_vectors = []
    scores = []
    low_support = []
    for s in sentences[:12]:  # limit for cost
        try:
            vec = embeddings.embed_query(s)
        except Exception:
            vec = []
        sent_vectors.append(vec)
        score = _cosine(vec, corpus_vec) if vec else 0.0
        scores.append(score)
        if score < 0.55:  # heuristic threshold
            low_support.append(s)
    return {
        "sentences": sentences[:12],
        "scores": scores,
        "low_support": low_support,
    }
