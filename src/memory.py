"""Persistent conversation memory utilities."""
from __future__ import annotations
import json
from pathlib import Path
from typing import List, Dict, Any
from .semantic_memory import add_message as add_semantic_message

MEMORY_FILE = Path("g:/programação/GDPR-Intelligent-RegAssistant/.memory/chat_history.json")
MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)

# Each entry: {"role": "user"|"assistant"|"tool", "content": "..."}

def load_history() -> List[Dict[str, Any]]:
    if not MEMORY_FILE.exists():
        return []
    try:
        return json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []


def save_history(history: List[Dict[str, Any]]):
    MEMORY_FILE.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")


def append_message(role: str, content: str):
    hist = load_history()
    hist.append({"role": role, "content": content})
    save_history(hist)
    # Also update semantic memory index (best-effort; offline safe)
    try:
        add_semantic_message(role, content)
    except Exception:
        pass


def get_context(last_n: int = 5) -> str:
    hist = load_history()[-last_n:]
    return "\n".join(f"{m['role']}: {m['content']}" for m in hist)
