"""Graph-based retrieval utilities for GDPR regulation.

Builds a structural graph linking Chapters -> Articles -> Recitals.
Recitals referencing Articles (via text pattern "Article <number>") create edges.

Functions:
    extract_structured_nodes(raw_docs): parse raw PDF documents to identify sections.
    build_graph(nodes): build a NetworkX graph from parsed nodes.
    rank_graph(graph): return PageRank scores.
    expand_with_neighbors(anchor_ids, graph, depth=1): gather neighbor node IDs.
    retrieve_with_graph(store, query, graph, page_rank, k=5, neighbor_depth=1):
        standard vector retrieval for anchor docs + neighbor expansion + rerank by PageRank.

Notes:
    - This is a first iteration; more robust parsing (multi-line headers) can be added later.
    - Page numbers are taken from original Document metadata when present.
"""
from __future__ import annotations
import re
from typing import List, Dict, Any, Tuple, Set
import networkx as nx
from langchain.schema import Document

ARTICLE_RE = re.compile(r"(?i)\bArticle\s+(\d+)\b")
RECITAL_RE = re.compile(r"(?i)\bRecital\s+(\d+)\b")
CHAPTER_RE = re.compile(r"(?i)\bChapter\s+([IVXLC]+)\b")


def extract_structured_nodes(raw_docs: List[Document]) -> List[Dict[str, Any]]:
    nodes: List[Dict[str, Any]] = []
    for doc_id, d in enumerate(raw_docs):
        text = d.page_content
        page = d.metadata.get("page", d.metadata.get("page_number", doc_id))
        # Find all headers in page; produce nodes for each header region.
        # Simplistic: if page contains Article/Recital/Chapter mention, treat page as that node.
        articles = ARTICLE_RE.findall(text)
        recitals = RECITAL_RE.findall(text)
        chapters = CHAPTER_RE.findall(text)
        if articles:
            for a in articles:
                nodes.append({"type": "article", "id": f"article_{a}", "number": a, "page": page, "text": text})
        if recitals:
            for r in recitals:
                nodes.append({"type": "recital", "id": f"recital_{r}", "number": r, "page": page, "text": text})
        if chapters:
            for c in chapters:
                nodes.append({"type": "chapter", "id": f"chapter_{c}", "number": c, "page": page, "text": text})
    return nodes


def build_graph(nodes: List[Dict[str, Any]]) -> nx.Graph:
    g = nx.Graph()
    # Add nodes
    for n in nodes:
        g.add_node(n["id"], **n)
    # Link chapters to articles on same page (heuristic)
    articles = [n for n in nodes if n["type"] == "article"]
    chapters = [n for n in nodes if n["type"] == "chapter"]
    recitals = [n for n in nodes if n["type"] == "recital"]
    for ch in chapters:
        for art in articles:
            if art["page"] == ch["page"]:
                g.add_edge(ch["id"], art["id"], relation="chapter_contains")
    # Recital referencing article by explicit mention
    for rec in recitals:
        mentioned = ARTICLE_RE.findall(rec["text"])
        for a in mentioned:
            art_id = f"article_{a}"
            if g.has_node(art_id):
                g.add_edge(rec["id"], art_id, relation="recital_references")
    return g


def rank_graph(graph: nx.Graph) -> Dict[str, float]:
    if graph.number_of_nodes() == 0:
        return {}
    return nx.pagerank(graph, alpha=0.85)


def expand_with_neighbors(anchor_ids: List[str], graph: nx.Graph, depth: int = 1) -> List[str]:
    visited = set(anchor_ids)
    frontier = list(anchor_ids)
    for _ in range(depth):
        new_frontier = []
        for node in frontier:
            for nbr in graph.neighbors(node):
                if nbr not in visited:
                    visited.add(nbr)
                    new_frontier.append(nbr)
        frontier = new_frontier
    return list(visited)


def _doc_snippet(graph: nx.Graph, node_id: str, max_len: int = 400, full: bool = False) -> str:
    data = graph.nodes[node_id]
    text = data.get("text", "")
    content = text.replace("\n", " ") if full else text[:max_len].replace("\n", " ")
    tag = "full" if full else "snippet"
    return f"[{data.get('type')}:{data.get('number')}] p.{data.get('page')} ({tag}) {content}"


def retrieve_with_graph(store, query: str, graph: nx.Graph, page_rank: Dict[str, float], k: int = 5, neighbor_depth: int = 1, full_pages: bool = False) -> Tuple[str, List[str]]:
    # Vector retrieval for anchor docs
    retrieved = store.similarity_search(query, k=k)
    # Map to node IDs (heuristic: match by article/recital number if present)
    anchor_ids: List[str] = []
    for doc in retrieved:
        txt = doc.page_content
        arts = ARTICLE_RE.findall(txt)
        recs = RECITAL_RE.findall(txt)
        if arts:
            anchor_ids.extend([f"article_{a}" for a in arts if graph.has_node(f"article_{a}")])
        if recs:
            anchor_ids.extend([f"recital_{r}" for r in recs if graph.has_node(f"recital_{r}")])
    anchor_ids = list(dict.fromkeys(anchor_ids))  # deduplicate preserving order
    if not anchor_ids:
        return "No structural nodes matched anchor retrieval.", []
    expanded_ids = expand_with_neighbors(anchor_ids, graph, depth=neighbor_depth)
    # Rerank by PageRank score descending
    scored = sorted(expanded_ids, key=lambda nid: page_rank.get(nid, 0.0), reverse=True)
    context_blocks = [_doc_snippet(graph, nid, full=full_pages) for nid in scored[: k * 2]]
    context = "\n\n".join(context_blocks)
    return context, scored


def validate_completeness(nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Heuristic completeness check for Articles (1-99) and Recitals (1-173).

    Returns counts and truncated missing lists.
    """
    articles_present: Set[int] = set()
    recitals_present: Set[int] = set()
    for n in nodes:
        t = n.get("type")
        num = n.get("number")
        if not num:
            continue
        try:
            val = int(num)
        except Exception:
            continue
        if t == "article":
            articles_present.add(val)
        elif t == "recital":
            recitals_present.add(val)
    expected_articles = set(range(1, 100))
    expected_recitals = set(range(1, 174))
    missing_articles = sorted(expected_articles - articles_present)
    missing_recitals = sorted(expected_recitals - recitals_present)
    return {
        "articles_found": len(articles_present),
        "recitals_found": len(recitals_present),
        "missing_articles": missing_articles[:20],
        "missing_recitals": missing_recitals[:20],
        "articles_complete": len(missing_articles) == 0,
        "recitals_complete": len(missing_recitals) == 0,
    }


def rephrase_question(question: str) -> str:
    """Heuristic rephrase from natural phrasing to regulatory-targeted phrasing.

    This is intentionally lightweight: we map some common privacy inquiry forms to
    more explicit GDPR constructs to improve structural anchor matching.
    Future: integrate an LLM rephraser with constrained decoding, or rule-based
    extraction of Article references.
    """
    q = question.strip()
    # Simple keyword expansions
    replacements = {
        "personal data": "personal data (as defined by GDPR Article 4)",
        "data minimization": "data minimization principle (Article 5)",
        "lawful basis": "lawful basis for processing (Articles 6 and 9)",
        "data subject rights": "data subject rights (Articles 12-23)",
        "controller obligations": "controller obligations (Chapter IV, Articles 24-43)",
    }
    lower_q = q.lower()
    for k, v in replacements.items():
        if k in lower_q:
            # replace case-insensitively (crude):
            pattern = re.compile(re.escape(k), re.IGNORECASE)
            q = pattern.sub(v, q)
    return q
