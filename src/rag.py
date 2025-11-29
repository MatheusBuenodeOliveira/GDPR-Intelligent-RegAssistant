"""Baseline RAG using direct OpenAI client without langchain-openai dependency.

Includes optional LangSmith tracing spans (if `langsmith` and API key available)
to record retrieval inputs and model invocation metadata for auditability.
"""
from __future__ import annotations
from typing import List
import os
from openai import OpenAI
from .tracing import span
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

SYSTEM_PROMPT = (
    "You are a privacy assistant. Answer strictly based on the GDPR. Cite Articles/Recitals and page numbers. "
    "If uncertain, say so and do not fabricate citations."
)


def _format_docs(docs: List[Document]) -> str:
    parts = []
    for d in docs:
        meta = d.metadata or {}
        page = meta.get("page", meta.get("page_number", "?"))
        header = meta.get("section_header", "")
        snippet = d.page_content[:600].replace("\n", " ")
        parts.append(f"[p.{page}] {header} :: {snippet}")
    return "\n\n".join(parts)


def answer(store: FAISS, question: str, k: int = 5) -> str:
    with span("rag.answer", {"k": k, "question": question[:200]}):
        docs = store.similarity_search(question, k=k)
        context_block = _format_docs(docs)
        try:
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        except Exception:
            client = type("OfflineClient", (), {"api_key": None})()  # graceful offline placeholder
        if not getattr(client, "api_key", None):
            return "[Offline Mode] Cannot access OpenAI API. Retrieved context:\n" + context_block
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {question}\n\nContext:\n{context_block}\n\nRespond concisely and cite sources."},
        ]
        resp = client.chat.completions.create(model="gpt-4o-mini", messages=messages, temperature=0)
        return resp.choices[0].message.content
