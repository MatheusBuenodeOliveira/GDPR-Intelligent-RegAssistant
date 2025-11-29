"""Agent Orchestrator: structured multi-step pipeline for GDPR RAG.

Pipeline Steps (first iteration):
1. Retrieval: vector search for top-k chunks.
2. Baseline Answer: call rag.answer using retrieved context (question may already include conversation context).
3. Citation Validation: assess citation coverage; if low, trigger summarizer fallback.
4. Hallucination Check: naive semantic similarity scoring; flag low-support sentences.
5. Fallback Summarization (conditional): generate grounded summary and append.
6. Diagnostics & Auditing: log each step with metrics (counts, thresholds, decisions).

Design Goals:
- Deterministic offline behavior (works with missing API key: produces context blocks only).
- Clear audit trail enabling post-hoc evaluation of grounding & safety.
- Extensible: add future nodes (PII scrub, regeneration loop) without altering CLI surface.

Returns: dict with keys: answer, steps (list of step dicts), diagnostics (summary metrics).
"""
from __future__ import annotations
from typing import Dict, Any, List
from dataclasses import dataclass, field

from .rag import answer as rag_answer
from .agent_tools import retrieve, citation_checker, summarizer
from .hallucination import analyze_support, regenerate_if_needed
from .audit import log_event, log_retrieval
from .citation_verify import verify_answer_citations
from .tracing import span


@dataclass
class OrchestratorConfig:
    k_retrieval: int = 5
    citation_min_coverage: float = 0.3
    hallucination_threshold: float = 0.58  # similarity threshold
    regen_ratio: float = 0.4  # if >40% sentences low support => regeneration attempt
    enable_fallback: bool = True  # summarizer fallback
    enable_regeneration: bool = True
    append_diagnostics_footer: bool = True


class AgentOrchestrator:
    def __init__(self, store, config: OrchestratorConfig | None = None):
        self.store = store
        self.config = config or OrchestratorConfig()

    def run(self, question: str) -> Dict[str, Any]:
        with span("agent.run", {"question": question[:200], "k_retrieval": self.config.k_retrieval}):
            steps: List[Dict[str, Any]] = []
            # Step 1: Retrieval
            with span("agent.step.retrieval", {"k": self.config.k_retrieval}):
                docs = retrieve(self.store, question, k=self.config.k_retrieval)
                log_retrieval(question, docs)
                steps.append({"name": "retrieval", "k": self.config.k_retrieval, "retrieved": len(docs)})
            # Step 2: Baseline answer
            with span("agent.step.baseline_answer"):
                base_answer = rag_answer(self.store, question)
                steps.append({"name": "baseline_answer", "chars": len(base_answer)})
            # Step 3: Citation validation
            with span("agent.step.citation_check"):
                ccheck = citation_checker(base_answer)
                # Cross-verify citations actually appear in retrieval context
                context_block = "\n".join(d.page_content[:400] for d in docs) if docs else ""
                citation_ver = verify_answer_citations(base_answer, context_block)
                steps.append({
                    "name": "citation_check",
                        "citations": len(ccheck["citations"]),
                    "sentence_count": ccheck["sentence_count"],
                    "coverage": ccheck["citation_coverage"],
                    "low": ccheck["low"],
                        "missing_citations": citation_ver["missing"],
                })
                log_event("citation_check", {
                    "coverage": ccheck["citation_coverage"],
                    "citations": ccheck["citations"],
                    "low": ccheck["low"],
                    "missing_citations": citation_ver["missing"],
                })
            # Step 4: Hallucination support analysis
            with span("agent.step.support_analysis", {"threshold": self.config.hallucination_threshold}):
                support = analyze_support(base_answer, docs, threshold=self.config.hallucination_threshold)
                low_support_count = len(support.get("low_support", []))
                steps.append({
                    "name": "support_analysis",
                    "sentences": len(support.get("sentences", [])),
                    "low_support_count": low_support_count,
                    "threshold": support.get("threshold"),
                    "offline": support.get("offline", False),
                })
                log_event("support_analysis", {
                    "low_support_count": low_support_count,
                    "offline": support.get("offline", False),
                    "threshold": support.get("threshold"),
                })
            # Step 5: Optional regeneration
            regenerated_answer = None
            regen_meta = None
            with span("agent.step.regeneration", {"enable": self.config.enable_regeneration}):
                if self.config.enable_regeneration and not support.get("offline"):
                    regen_res = regenerate_if_needed(question, base_answer, docs, support, regen_ratio=self.config.regen_ratio)
                    if regen_res.get("regenerated"):
                        regenerated_answer = regen_res["answer"]
                        regen_meta = regen_res
                        steps.append({"name": "regeneration", "regenerated": True, "reason": regen_res.get("reason")})
                        log_event("regeneration", {"reason": regen_res.get("reason"), "low_support_before": regen_res.get("low_support_before")})
                        base_answer = regenerated_answer
                    else:
                        steps.append({"name": "regeneration", "regenerated": False, "reason": regen_res.get("reason")})
            # Step 6: Conditional fallback summarization
            fallback_used = False
            improved = ""
            sentence_total = len(support.get("sentences", [])) or 1
            low_ratio = low_support_count / sentence_total
            with span("agent.step.fallback", {"enable": self.config.enable_fallback, "low_ratio": low_ratio}):
                if self.config.enable_fallback and (ccheck["low"] or low_ratio > self.config.regen_ratio):
                    improved = summarizer(question, docs)
                    fallback_used = True
                    steps.append({"name": "summarizer_fallback", "chars": len(improved)})
                    log_event("summarizer_fallback", {"trigger": "citation_or_low_support", "fallback_used": True})
            # Compose final answer
            final_answer = base_answer
            if fallback_used:
                final_answer += "\n[orchestrator summarizer fallback]\n" + improved
            if self.config.append_diagnostics_footer:
                footer = ["[agent-orchestrator]",
                          f"citations={len(ccheck['citations'])}",
                          f"coverage={ccheck['citation_coverage']:.2f}",
                          f"low_support={low_support_count}",
                          f"regen={'1' if regenerated_answer else '0'}",
                          f"fallback={'1' if fallback_used else '0'}"]
                if fallback_used:
                    footer.append(f"support_threshold={support.get('threshold')}")
                final_answer += "\n" + " ".join(footer)
            diagnostics = {
                "citation_coverage": ccheck["citation_coverage"],
                "citations": len(ccheck["citations"]),  # numeric for tests
                "citation_list": ccheck["citations"],
                "missing_citations": citation_ver["missing"],
                "low_support_count": low_support_count,
                "fallback_used": fallback_used,
                "regenerated": bool(regenerated_answer),
            }
            log_event("agent_orchestrator_complete", diagnostics | {"answer_preview": final_answer[:400]})
            return {"answer": final_answer, "steps": steps, "diagnostics": diagnostics}

