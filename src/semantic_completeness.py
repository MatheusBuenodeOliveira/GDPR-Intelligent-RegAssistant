"""Semantic completeness heuristics.

Maps thematic queries to required GDPR Articles. Provides function to
check if retrieved context covers all required Articles and returns
missing list for auditing.
"""
from __future__ import annotations
import re
from typing import List, Dict, Set

TOPIC_ARTICLES: Dict[str, Set[int]] = {
    # Themes
    "lawful basis": {6, 9},
    "data minimization": {5},
    "data subject rights": set(range(12, 24)),  # 12-23 inclusive
    "controller obligations": set(range(24, 44)),  # 24-43
    "principles": {5},
}

ARTICLE_RE = re.compile(r"Article\s+(\d+)")


def infer_topics(question: str) -> List[str]:
    q = question.lower()
    hits = []
    for k in TOPIC_ARTICLES.keys():
        if k in q:
            hits.append(k)
    # fuzzy: rights -> data subject rights
    if "rights" in q and "data subject rights" not in hits:
        hits.append("data subject rights")
    return hits


def extract_articles_from_text(text: str) -> Set[int]:
    nums = set()
    for m in ARTICLE_RE.findall(text):
        try:
            nums.add(int(m))
        except Exception:
            pass
    return nums


def check_semantic_completeness(question: str, context_block: str) -> Dict[str, any]:
    topics = infer_topics(question)
    mentioned = extract_articles_from_text(context_block)
    missing_per_topic: Dict[str, List[int]] = {}
    for t in topics:
        required = TOPIC_ARTICLES[t]
        missing = sorted(required - mentioned)
        if missing:
            missing_per_topic[t] = missing[:10]
    return {
        "topics": topics,
        "mentioned_articles": sorted(mentioned),
        "missing": missing_per_topic,
        "complete": len(missing_per_topic) == 0,
    }
