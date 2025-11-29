"""Tests for semantic memory offline fallback and retrieval structure."""
from src.semantic_memory import add_message, retrieve_semantic_context, build_semantic_context_block


def test_semantic_memory_offline(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    add_message("user", "What is personal data definition?")
    add_message("assistant", "Article 4 defines personal data including any identifier.")
    add_message("user", "Explain scope of GDPR.")
    ctx = retrieve_semantic_context("scope", k=2)
    assert len(ctx) <= 2 and all(isinstance(c, str) for c in ctx)
    block = build_semantic_context_block("personal data", k=2)
    assert "Semantic Memory" in block or block == ""