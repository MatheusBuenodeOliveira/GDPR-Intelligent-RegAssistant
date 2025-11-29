"""Audit logging for queries and decisions."""
from __future__ import annotations
import json, time
from pathlib import Path
from typing import Dict, Any, List

AUDIT_DIR = Path("g:/programação/GDPR-Intelligent-RegAssistant/exports")
AUDIT_DIR.mkdir(parents=True, exist_ok=True)
SESSION_FILE = AUDIT_DIR / "session_log.json"

_DECISIONS: List[Dict[str, Any]] = []

def _append(entry: Dict[str, Any]):
    _DECISIONS.append(entry)
    SESSION_FILE.write_text(json.dumps(_DECISIONS, ensure_ascii=False, indent=2), encoding="utf-8")

def log_event(event: str, data: Dict[str, Any]):
    """Log a generic event with its data payload."""
    entry = {"ts": time.time(), "event": event, **data}
    _append(entry)


def log_retrieval(question: str, docs: list):
    """Log retrieval step with lightweight metadata about returned documents."""
    meta = []
    for d in docs[:10]:  # limit logging volume
        m = d.metadata or {}
        meta.append({
            "page": m.get("page", m.get("page_number")),
            "section_header": m.get("section_header"),
            "len": len(d.page_content),
        })
    _append({"ts": time.time(), "event": "retrieval", "question": question, "results": meta})


def log_guardrail(trigger: str, detail: Dict[str, Any]):
    """Log that a guardrail fired (adversarial, toxic, pii, etc.)."""
    _append({"ts": time.time(), "event": "guardrail", "trigger": trigger, **detail})


def get_events() -> List[Dict[str, Any]]:
    return list(_DECISIONS)
