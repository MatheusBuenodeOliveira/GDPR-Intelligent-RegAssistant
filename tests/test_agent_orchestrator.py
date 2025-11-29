"""Tests for AgentOrchestrator offline behavior (no API key required).

Focus: structural pipeline executes, returns diagnostics footer, and logs steps without raising errors.
"""
from langchain.schema import Document
from src.index_store import build_index
from src.agent_orchestrator import AgentOrchestrator, OrchestratorConfig


def _build_dummy_store():
    docs = [
        Document(page_content="Article 1 Data protection principles and lawful processing.", metadata={"page": 1}),
        Document(page_content="Article 2 Scope of the regulation and material limits.", metadata={"page": 2}),
        Document(page_content="Recital 1 Protect fundamental rights and freedoms.", metadata={"page": 3}),
    ]
    return build_index(docs)


def test_orchestrator_offline(monkeypatch):
    # Ensure no API key for offline path
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    store = _build_dummy_store()
    orchestrator = AgentOrchestrator(store, OrchestratorConfig(k_retrieval=2))
    result = orchestrator.run("What are the data protection principles?")
    answer = result["answer"]
    assert "agent-orchestrator" in answer  # diagnostics footer present
    assert result["diagnostics"]["citations"] >= 0  # numeric field exists
    # Steps include retrieval and baseline_answer
    step_names = [s["name"] for s in result["steps"]]
    assert "retrieval" in step_names
    assert "baseline_answer" in step_names