"""Simple guardrails: input/output validation and safety flags.

Responsibilities:
1. Detect adversarial prompt injection attempts.
2. Detect toxic / disallowed content and refuse.
3. Light PII pattern spotting (illustrative only, NOT production grade).
4. Enforce presence of citation markers (page or article) in model output.
5. Surface PII alerts to downstream auditing.

All messages and tags are now in English to keep system consistent.
"""
import re
from .pii_detection import detect_pii_advanced
SAFE_REFUSAL = "Sorry, I can't assist with that."
WARN_TAG = "[Safety Warning]"

"""Guardrail detection utilities.

Pattern / term lists have been expanded and lightly tuned based on synthetic
precision/recall evaluation (see responsible_ai_report.md Guardrail Evaluation Snapshot).
We prioritize high recall for clearly adversarial jailbreak attempts while
keeping the toxicity list scoped to overt slurs / abuse to reduce false positives.

If future empirical tuning shows excessive false positives, consider:
 - Introducing severity tiers (block vs warn)
 - Distance / fuzzy matching only for high‑severity subset
 - Maintaining allowlists for domain terms (e.g. academic discussion)
"""

ADVERSARIAL_PATTERNS = [
    # Core jailbreak directives
    r"ignore (?:the )?previous instructions",
    r"ignore (all|the) rules",  # legacy pattern retained for backward compatibility/tests
    r"bypass (?:all )?safety",
    r"disregard (?:all|the) instructions",
    r"override (?:the )?system",
    # Impersonation / role escalation
    r"pretend (?:to be|you're)",
    r"you are now (?:an|the)",
    # Data exfiltration / secrets
    r"extract .* secrets",
    r"leak (?:all )?data",
    r"disable (?:all )?safeguards",
]

TOXIC_TERMS = [
    # Keep list concise; high‑precision abusive terms
    "idiot", "stupid", "hate", "racist", "sexist",
    "terrorist", "genocide", "homophobic", "abuse"
]

PII_PATTERNS = [
    r"\b\d{11}\b",  # Generic 11-digit personal identifier (example)
    r"\b\d{3}[- ]?\d{3}[- ]?\d{3}\b",  # Phone-like fragments
    r"@[\w.-]+\.[a-z]{2,}",  # Email pattern (simplified)
]


def is_adversarial(text: str) -> bool:
    """Detect obvious jailbreak / instruction override attempts.

    Uses regex patterns; anchored loosely to allow phrasal variation. Future enhancement:
    incorporate embedding similarity to a curated jailbreak intent set for recall without
    over-expanding patterns.
    """
    t = text.lower()
    return any(re.search(p, t) for p in ADVERSARIAL_PATTERNS)


def is_toxic(text: str) -> bool:
    """Direct substring check for high‑precision toxicity lexicon.

    Avoid stemming/fuzzy here to reduce false positives (e.g., 'hateful' vs 'hate').
    """
    t = text.lower()
    return any(term in t for term in TOXIC_TERMS)


def detect_pii(text: str) -> list[str]:
    """Unified PII detection combining legacy regex patterns with advanced heuristics / NER."""
    hits = []
    for pattern in PII_PATTERNS:
        hits.extend(re.findall(pattern, text))
    # Extend with advanced
    try:
        adv = detect_pii_advanced(text)
        hits.extend(adv)
    except Exception:
        pass
    # Deduplicate
    return list(dict.fromkeys(hits))


def guard_input(q: str) -> str:
    """Apply input guardrails with severity tiers.

    - Toxic: hard refusal.
    - Adversarial (non-toxic): rewrite + warning tag.
    - Benign: passthrough.
    """
    if is_toxic(q):
        return SAFE_REFUSAL
    if is_adversarial(q):
        rewritten = re.sub(r"(?i)ignore.*", "", q).strip()
        return f"[Safe Rewrite] {WARN_TAG} {rewritten}"
    return q


def guard_output(answer: str) -> str:
    """Post-process model answer enforcing citation marker presence and tagging simple PII."""
    if not re.search(r"p\.\d+|Article\s+\d+", answer):
        answer += "\n\n[Note] Answer without detected citations."
    pii = detect_pii(answer)
    if pii:
        answer += f"\n[PII Alert] patterns found: {pii}"
    return answer
