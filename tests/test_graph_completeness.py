"""Test graph completeness validator with partial synthetic data."""
from src.graph_rag import validate_completeness


def test_graph_completeness_partial():
    nodes = [
        {"type": "article", "number": "1"},
        {"type": "article", "number": "2"},
        {"type": "recital", "number": "1"},
        {"type": "recital", "number": "2"},
    ]
    comp = validate_completeness(nodes)
    assert comp["articles_found"] == 2
    assert comp["recitals_found"] == 2
    assert comp["articles_complete"] is False
    assert comp["recitals_complete"] is False