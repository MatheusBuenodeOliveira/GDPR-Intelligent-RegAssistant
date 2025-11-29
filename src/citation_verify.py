"""Citation verification utilities.

Cross-checks that Articles cited in an answer appear in provided context
text. Returns missing citations for auditing / potential regeneration.
"""
from __future__ import annotations
import re
from typing import Dict, List

ARTICLE_RE = re.compile(r"Article\s+(\d+)")


def extract_citations(answer: str) -> List[int]:
    nums = []
    for m in ARTICLE_RE.findall(answer):
        try:
            nums.append(int(m))
        except Exception:
            pass
    return list(dict.fromkeys(nums))


def verify_answer_citations(answer: str, context: str) -> Dict[str, any]:
    cited = extract_citations(answer)
    context_nums = set(extract_citations(context))
    missing = [c for c in cited if c not in context_nums]
    return {"cited": cited, "missing": missing, "all_present": len(missing) == 0}
