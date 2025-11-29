"""Tests for metrics aggregation (synthetic events)."""
from src.audit import log_event, log_guardrail, log_retrieval
from langchain.schema import Document
from src.metrics import compute_metrics


def test_metrics_aggregation():
    # Synthetic retrieval
    docs = [Document(page_content="Article 1 ...", metadata={"page":1})]
    log_retrieval("q1", docs)
    # Agent completion events
    log_event("agent_orchestrator_complete", {"citation_coverage":0.5, "fallback_used":True, "regenerated":False, "low_support_count":2})
    log_event("agent_orchestrator_complete", {"citation_coverage":0.9, "fallback_used":False, "regenerated":True, "low_support_count":1})
    # Guardrail trigger
    log_guardrail("toxic_or_refusal", {"question":"bad"})
    metrics = compute_metrics()
    assert metrics["citation_coverage_avg"] > 0.6
    assert metrics["agent_fallback_rate"] > 0.0
    assert metrics["regeneration_rate"] > 0.0
    assert metrics["guardrail_trigger_counts"].get("toxic_or_refusal") == 1