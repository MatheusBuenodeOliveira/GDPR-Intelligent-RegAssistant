import os, pytest
from src.index_store import load_or_build
from src.ingest import load_pdf, chunk_documents
from src.agent_tools import retrieve, citation_checker


def test_retrieve_and_citation_basic():
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set; skipping agent tools test.")
    raw = load_pdf()
    chunks = chunk_documents(raw)
    store = load_or_build(chunks)
    docs = retrieve(store, "data subject rights", k=3)
    assert len(docs) > 0
    fake_answer = "Article 12 describes transparent information. Article 15 covers access rights."
    c = citation_checker(fake_answer)
    assert c["citation_coverage"] > 0