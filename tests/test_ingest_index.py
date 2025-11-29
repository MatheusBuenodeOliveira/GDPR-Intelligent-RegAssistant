import pytest, os
from src.ingest import load_pdf, chunk_documents, PDF_PATH
from src.index_store import load_or_build

def test_ingest_and_index_smoke():
    if not PDF_PATH.exists():
        pytest.skip("GDPR PDF not present; skipping ingestion/index smoke test.")
    raw = load_pdf()
    assert len(raw) > 0
    chunks = chunk_documents(raw)
    assert len(chunks) > 10
    store = load_or_build(chunks)
    docs = store.similarity_search("personal data", k=3)
    assert len(docs) > 0
