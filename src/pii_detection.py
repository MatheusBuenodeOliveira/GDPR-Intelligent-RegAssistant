"""Advanced (lightweight) PII detection utilities.

Combines existing regex patterns with optional spaCy NER for PERSON / ORG names.
Falls back gracefully if spaCy or model not available.

NOTE: For full production GDPR compliance, more robust patterns and jurisdiction-specific IDs required.
"""
from __future__ import annotations
import re
from typing import List

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_GENERIC_ID_RE = re.compile(r"\b\d{11}\b")
_PHONE_RE = re.compile(r"\b\d{3}[- ]?\d{3}[- ]?\d{3}\b")
_NAME_HEURISTIC_RE = re.compile(r"(?=\b([A-Z][a-z]+\s[A-Z][a-z]+)\b)")  # Overlapping bigrams via lookahead
_NAME_STOP_FIRST = {"Contact", "Internal", "Give", "Explain"}

def _load_spacy_model():
    try:
        import spacy  # type: ignore
        try:
            return spacy.load("en_core_web_sm")
        except Exception:
            return None
    except Exception:
        return None


def detect_pii_advanced(text: str, enable_spacy: bool = True) -> List[str]:
    hits: List[str] = []
    hits.extend(_EMAIL_RE.findall(text))
    hits.extend(_GENERIC_ID_RE.findall(text))
    hits.extend(_PHONE_RE.findall(text))
    # Name heuristic (avoids over-triggering by limiting length)
    # Overlapping bigrams
    for m in _NAME_HEURISTIC_RE.findall(text):
        parts = m.split()
        if len(parts) == 2 and parts[0] not in _NAME_STOP_FIRST:
            hits.append(m)
    # Fallback manual sliding window to catch cases missed by regex engine nuances
    words = re.split(r"[^A-Za-z]+", text)
    for i in range(len(words)-1):
        w1, w2 = words[i], words[i+1]
        if w1 and w2 and w1[0].isupper() and w2[0].isupper() and w1.isalpha() and w2.isalpha():
            candidate = f"{w1} {w2}"
            if candidate not in hits and w1 not in _NAME_STOP_FIRST:
                hits.append(candidate)
    # Optional spaCy NER
    if enable_spacy:
        nlp = _load_spacy_model()
        if nlp:
            try:
                doc = nlp(text[:8000])  # safety cap
                for ent in doc.ents:
                    if ent.label_ in {"PERSON", "ORG"} and ent.text not in hits:
                        hits.append(ent.text)
            except Exception:
                pass
    return list(dict.fromkeys(hits))  # dedupe preserving order
