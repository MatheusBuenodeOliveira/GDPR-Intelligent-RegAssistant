import pytest
from src.ingest import load_pdf
from src.graph_rag import extract_structured_nodes, build_graph, rank_graph, expand_with_neighbors


def test_graph_construction_and_expansion():
    raw = load_pdf()
    nodes = extract_structured_nodes(raw)
    assert len(nodes) > 10, "Expected to find multiple structural nodes (articles/recitals/chapters)."
    g = build_graph(nodes)
    # Graph may collapse duplicates; ensure non-empty proportion retained
    assert g.number_of_nodes() > 50, f"Graph unexpectedly small: {g.number_of_nodes()}"
    pr = rank_graph(g)
    # pick first few article ids
    anchor = [n["id"] for n in nodes if n["type"] == "article"][:2]
    if not anchor:
        pytest.skip("No articles parsed; skip expansion test.")
    expanded = expand_with_neighbors(anchor, g, depth=1)
    assert set(anchor).issubset(set(expanded))
    # PageRank returns floats
    assert all(isinstance(v, float) for v in pr.values()) or pr == {}