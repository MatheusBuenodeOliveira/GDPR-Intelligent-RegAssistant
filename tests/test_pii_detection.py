"""Tests for enhanced PII detection (without requiring spaCy model present)."""
from src.guardrails import detect_pii


def test_basic_email_and_id_detection():
    sample = "Contact John Doe via john.doe@example.com. Internal ID 12345678901 is sensitive."  # email + 11-digit + name
    hits = detect_pii(sample)
    assert any(h.endswith("@example.com") for h in hits)
    assert any(h.isdigit() and len(h) == 11 for h in hits)
    # Name heuristic may capture John Doe even offline
    assert any("John Doe" == h for h in hits)