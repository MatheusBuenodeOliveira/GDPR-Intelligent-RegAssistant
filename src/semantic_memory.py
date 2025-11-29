"""Semantic conversation memory using FAISS for similarity-based recall.

Stores embeddings of past user/assistant messages to allow semantic retrieval
of relevant turns for grounding answers when sessions grow large.

Persistence Files:
  .memory/semantic_messages.json : serialized list of {role, content}
  .memory/semantic_index.faiss / semantic_store.pkl : FAISS + mapping for reload

Strategy (MVP): rebuild index on each append. Acceptable for small conversation (< few hundred).
Can be optimized later with incremental add.
Offline fallback (no API key): retrieval returns last N messages chronologically.
"""
from __future__ import annotations
from pathlib import Path
import json, pickle, os
from typing import List, Dict, Any
import faiss
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from .embedding_wrapper import OpenAIEmbeddingWrapper

MEM_DIR = Path("g:/programação/GDPR-Intelligent-RegAssistant/.memory")
MEM_DIR.mkdir(parents=True, exist_ok=True)
MSG_FILE = MEM_DIR / "semantic_messages.json"
INDEX_FILE = MEM_DIR / "semantic_index.faiss"
STORE_FILE = MEM_DIR / "semantic_store.pkl"


def _load_messages() -> List[Dict[str, Any]]:
    if not MSG_FILE.exists():
        return []
    try:
        return json.loads(MSG_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []


def _save_messages(msgs: List[Dict[str, Any]]):
    MSG_FILE.write_text(json.dumps(msgs, ensure_ascii=False, indent=2), encoding="utf-8")


def _build_index(msgs: List[Dict[str, Any]]) -> FAISS:
    if not msgs:
        return None  # type: ignore
    texts = [m['content'] for m in msgs]
    embeddings = OpenAIEmbeddingWrapper()
    store = FAISS.from_texts(texts, embeddings, metadatas=[{"role": m['role']} for m in msgs])
    # Persist
    faiss.write_index(store.index, str(INDEX_FILE))
    with open(STORE_FILE, "wb") as f:
        pickle.dump({"docstore": store.docstore, "index_to_docstore_id": store.index_to_docstore_id}, f)
    return store


def _load_index() -> FAISS | None:
    if not INDEX_FILE.exists() or not STORE_FILE.exists():
        return None
    try:
        idx = faiss.read_index(str(INDEX_FILE))
        with open(STORE_FILE, "rb") as f:
            payload = pickle.load(f)
        return FAISS(
            embedding_function=OpenAIEmbeddingWrapper(),
            index=idx,
            docstore=payload["docstore"],
            index_to_docstore_id=payload["index_to_docstore_id"],
        )
    except Exception:
        return None


def add_message(role: str, content: str):
    msgs = _load_messages()
    msgs.append({"role": role, "content": content})
    _save_messages(msgs)
    # Rebuild index (skip if offline)
    wrapper = OpenAIEmbeddingWrapper()
    if not getattr(wrapper.client, "api_key", None):
        return  # offline; no embedding rebuild
    _build_index(msgs)


def retrieve_semantic_context(query: str, k: int = 3) -> List[str]:
    msgs = _load_messages()
    wrapper = OpenAIEmbeddingWrapper()
    if not getattr(wrapper.client, "api_key", None):
        # Offline: return last k messages (excluding very long ones)
        return [m['content'][:500] for m in msgs[-k:]]
    store = _load_index()
    if store is None:
        store = _build_index(msgs)
    if store is None:
        return []
    results = store.similarity_search(query, k=k)
    return [r.page_content[:600] for r in results]


def build_semantic_context_block(query: str, k: int = 3) -> str:
    retrieved = retrieve_semantic_context(query, k=k)
    if not retrieved:
        return ""
    return "\n--- Semantic Memory ---\n" + "\n".join(f"[mem] {t}" for t in retrieved)
