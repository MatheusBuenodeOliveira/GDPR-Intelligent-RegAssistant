"""Responsible AI metrics aggregation from audit log events.

Reads in-memory audit events (session scope) and computes lightweight metrics:
 - citation_coverage_avg
 - agent_fallback_rate
 - regeneration_rate
 - low_support_avg
 - guardrail_trigger_counts by type
 - retrieval_avg_docs (approx from retrieval event length)

Future: persist historical runs; compute precision/recall with labeled evaluation set.
"""
from __future__ import annotations
from typing import Dict, Any
from .audit import get_events


def compute_metrics() -> Dict[str, Any]:
    events = get_events()
    if not events:
        return {"empty": True}
    citation_coverages = []
    fallback_used = 0
    regen_used = 0
    low_support_counts = []
    guardrail_counts: Dict[str, int] = {}
    retrieval_doc_counts = []
    for e in events:
        if e.get("event") == "agent_orchestrator_complete":
            citation_coverages.append(e.get("citation_coverage", 0.0))
            if e.get("fallback_used"):
                fallback_used += 1
            if e.get("regenerated"):
                regen_used += 1
            low_support_counts.append(e.get("low_support_count", 0))
        elif e.get("event") == "guardrail":
            trig = e.get("trigger", "unknown")
            guardrail_counts[trig] = guardrail_counts.get(trig, 0) + 1
        elif e.get("event") == "retrieval":
            retrieval_doc_counts.append(len(e.get("results", [])))
    total_agent = sum(1 for e in events if e.get("event") == "agent_orchestrator_complete")
    metrics = {
        "citation_coverage_avg": (sum(citation_coverages) / len(citation_coverages)) if citation_coverages else 0.0,
        "agent_fallback_rate": (fallback_used / total_agent) if total_agent else 0.0,
        "regeneration_rate": (regen_used / total_agent) if total_agent else 0.0,
        "low_support_avg": (sum(low_support_counts) / len(low_support_counts)) if low_support_counts else 0.0,
        "guardrail_trigger_counts": guardrail_counts,
        "retrieval_avg_docs": (sum(retrieval_doc_counts) / len(retrieval_doc_counts)) if retrieval_doc_counts else 0.0,
        "total_events": len(events),
    }
    return metrics


def format_metrics_report(metrics: Dict[str, Any]) -> str:
    if metrics.get("empty"):
        return "No audit events collected yet. Run some queries first."
    lines = ["Responsible AI Metrics Summary:"]
    lines.append(f"Citation Coverage (avg): {metrics['citation_coverage_avg']:.2f}")
    lines.append(f"Agent Fallback Rate: {metrics['agent_fallback_rate']:.2f}")
    lines.append(f"Regeneration Rate: {metrics['regeneration_rate']:.2f}")
    lines.append(f"Low-Support Sentences (avg): {metrics['low_support_avg']:.2f}")
    lines.append(f"Retrieval Docs (avg): {metrics['retrieval_avg_docs']:.2f}")
    lines.append("Guardrail Triggers:")
    for k, v in metrics['guardrail_trigger_counts'].items():
        lines.append(f"  - {k}: {v}")
    lines.append(f"Total Events: {metrics['total_events']}")
    return "\n".join(lines)
