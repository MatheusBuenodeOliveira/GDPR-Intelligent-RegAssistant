"""Improved hallucination detection utilities.

Approach:
1. Split answer into sentences (filter very short ones).
2. Embed each sentence and each retrieved source chunk separately.
3. For each sentence compute max cosine similarity across all chunk embeddings.
4. Classify sentences with similarity < threshold as low-support.
5. Optional regeneration: if proportion of low-support > regen_ratio threshold, build a regeneration prompt
   instructing the model to re-ground those sentences using only provided chunks.

Offline behavior: if embeddings unavailable (no API key) returns structure with 'offline': True and skips regeneration.
"""
from __future__ import annotations
from typing import List, Dict, Any
import re, math, os
from openai import OpenAI
from .embedding_wrapper import OpenAIEmbeddingWrapper


SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

def _split_sentences(text: str) -> List[str]:
    raw = [s.strip() for s in SENTENCE_SPLIT.split(text) if s.strip()]
    return [s for s in raw if len(s) > 20]  # drop ultra-short


def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return 0.0
    num = sum(x*y for x,y in zip(a,b))
    da = math.sqrt(sum(x*x for x in a))
    db = math.sqrt(sum(y*y for y in b))
    if da == 0 or db == 0:
        return 0.0
    return num/(da*db)


def analyze_support(answer: str, docs, embed_model: str = "text-embedding-3-small", threshold: float = 0.58) -> Dict[str, Any]:
    sentences = _split_sentences(answer)
    wrapper = OpenAIEmbeddingWrapper(model=embed_model)
    offline = not getattr(wrapper.client, "api_key", None)
    if offline:
        return {"offline": True, "sentences": sentences, "scores": [0.0]*len(sentences), "low_support": sentences, "threshold": threshold}
    chunk_texts = [d.page_content[:800] for d in docs]
    chunk_vecs = wrapper.embed_documents(chunk_texts)
    sent_vecs = [wrapper.embed_query(s) for s in sentences]
    scores = []
    low_support = []
    for s, vec in zip(sentences, sent_vecs):
        best = 0.0
        for c in chunk_vecs:
            sim = _cosine(vec, c)
            if sim > best:
                best = sim
        scores.append(best)
        if best < threshold:
            low_support.append(s)
    return {"offline": False, "sentences": sentences, "scores": scores, "low_support": low_support, "threshold": threshold}


def regenerate_if_needed(question: str, answer: str, docs, analysis: Dict[str, Any], regen_ratio: float = 0.4) -> Dict[str, Any]:
    """If too many sentences are low-support, regenerate a grounded variant using provided docs."""
    if analysis.get("offline"):
        return {"regenerated": False, "answer": answer, "reason": "offline"}
    sentences = analysis["sentences"]
    if not sentences:
        return {"regenerated": False, "answer": answer, "reason": "no_sentences"}
    low = analysis["low_support"]
    if len(low)/max(len(sentences),1) <= regen_ratio:
        return {"regenerated": False, "answer": answer, "reason": "ratio_ok"}
    joined = "\n\n".join(d.page_content[:1000] for d in docs)
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    if not getattr(client, "api_key", None):
        return {"regenerated": False, "answer": answer, "reason": "offline"}
    prompt = (
        "You are a GDPR assistant. Regenerate a grounded answer to the user's question. "
        "Focus on correcting the following low-support sentences (they lacked source similarity):\n" +
        "\n".join(f"- {s}" for s in low) +
        "\nUse ONLY the provided source excerpts and cite Articles/Recitals/pages. If unsure, say so.\n" +
        f"Question: {question}\nSources:\n{joined}\nGrounded Answer:"
    )
    resp = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"user", "content": prompt}], temperature=0)
    regenerated = resp.choices[0].message.content
    return {"regenerated": True, "answer": regenerated, "low_support_before": low, "reason": "regen_trigger"}
