"""Tests for improved hallucination support analysis (offline mode)."""
from langchain.schema import Document
from src.hallucination import analyze_support, regenerate_if_needed


def test_offline_analysis_structure(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    docs = [Document(page_content="Article 1 Principles.", metadata={"page":1}),
            Document(page_content="Article 2 Scope.", metadata={"page":2})]
    ans = "Article 1 sets out fundamental principles of data protection. It requires fairness and transparency."  # two sentences
    analysis = analyze_support(ans, docs, threshold=0.5)
    assert "sentences" in analysis and len(analysis["sentences"]) >= 1
    assert analysis.get("offline") is True
    regen = regenerate_if_needed("What are principles?", ans, docs, analysis)
    assert regen["regenerated"] is False