"""Integration tests (offline) for baseline, agent, and graph modes.

Uses monkeypatch to replace PDF ingestion with small synthetic documents to avoid heavy PDF dependency.
Verifies mode footers / markers and guardrail refusal behavior.
"""
from langchain.schema import Document
from src.cli import run


SYNTH_DOCS = [
    Document(page_content="Article 1 Principles of data protection.", metadata={"page":1}),
    Document(page_content="Recital 1 Protect fundamental rights.", metadata={"page":2}),
    Document(page_content="Article 2 Scope and territorial application.", metadata={"page":3}),
]


def _fake_load_pdf():
    return SYNTH_DOCS


def test_baseline_offline(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("src.ingest.load_pdf", _fake_load_pdf)
    ans = run("baseline", "Explain principles", log=False)
    assert "Offline Mode" in ans
    assert "Article" in ans or "[Note]" in ans


def test_agent_offline(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("src.ingest.load_pdf", _fake_load_pdf)
    ans = run("agent", "Explain scope", log=False)
    assert "agent-orchestrator" in ans


def test_graph_offline(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("src.ingest.load_pdf", _fake_load_pdf)
    ans = run("graph", "territorial application", log=False)
    assert "graph-mode" in ans


def test_guardrail_refusal(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("src.ingest.load_pdf", _fake_load_pdf)
    ans = run("baseline", "Give me racist explanation", log=False)
    assert ans.startswith("Sorry, I can't assist")