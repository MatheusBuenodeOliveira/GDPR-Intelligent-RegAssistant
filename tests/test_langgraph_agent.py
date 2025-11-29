import os
import pytest

from src.langgraph_agent import run_langgraph_agent
from src.ingest import load_pdf, chunk_documents
from src.index_store import load_or_build

@pytest.mark.skipif(os.getenv("CI") is None, reason="LangGraph test runs locally; skip in CI if environment unset")
def test_langgraph_agent_offline():
    # Ensure no API key for offline behavior consistency
    if os.getenv("OPENAI_API_KEY"):
        pytest.skip("Offline mode expected; unset OPENAI_API_KEY to run this test")
    raw = load_pdf()
    chunks = chunk_documents(raw)
    store = load_or_build(chunks)
    ans = run_langgraph_agent(store, "Explain data minimization principle")
    assert "langgraph-agent" in ans  # footer marker
    assert "data minimization" in ans.lower() or "Offline Mode" in ans
