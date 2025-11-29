"""LangGraph-based agent pipeline for GDPR RAG.

This provides an alternative to the custom AgentOrchestrator using LangGraph's
StateGraph. It mirrors the steps:
    retrieval -> baseline_answer -> citation_check -> support_analysis ->
    regeneration (optional) -> summarizer_fallback (conditional) -> finalize

Offline behavior: if no OPENAI_API_KEY the baseline answer becomes the offline
context block similar to rag.answer.
"""
from __future__ import annotations
from typing import List, Dict, Any, TypedDict, Optional
import os

try:  # pragma: no cover - optional import
    from langgraph.graph import StateGraph, END
except Exception:  # Fallback: define dummy interface so module import succeeds
    StateGraph = None
    END = "__END__"

from langchain.schema import Document
from .agent_tools import retrieve, citation_checker, summarizer
from .hallucination import analyze_support, regenerate_if_needed
from .rag import _format_docs  # reuse internal
from .audit import log_event, log_retrieval
from .tracing import span
from .rag import SYSTEM_PROMPT
from .memory import get_context
from .semantic_memory import build_semantic_context_block
from openai import OpenAI

class AgentState(TypedDict, total=False):
    question: str
    docs: List[Document]
    answer: str
    citations: List[str]
    citation_coverage: float
    support: Dict[str, Any]
    regenerated: bool
    fallback_used: bool


def _offline_answer(docs: List[Document]) -> str:
    return "[Offline Mode] Cannot access OpenAI API. Retrieved context:\n" + _format_docs(docs)


def node_context(state: AgentState) -> AgentState:
    """Inject conversation + semantic memory context before retrieval."""
    ctx = get_context()
    sem = build_semantic_context_block(state["question"], k=3) or ""
    combined = (ctx + ("\n" + sem if sem else "")) if ctx else sem
    q = state["question"]
    if combined:
        q = f"Conversation context:\n{combined}\n\nUser question: {state['question']}"
    return {**state, "question": q, "context_added": bool(combined)}


def node_retrieve(state: AgentState, store) -> AgentState:
    with span("lg.retrieve"):
        docs = retrieve(store, state["question"], k=5)
        log_retrieval(state["question"], docs)
        return {**state, "docs": docs}


def node_baseline(state: AgentState) -> AgentState:
    with span("lg.answer"):
        docs = state.get("docs", [])
        try:
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        except Exception:
            client = type("OfflineClient", (), {"api_key": None})()
        if not getattr(client, "api_key", None):
            ans = _offline_answer(docs)
        else:
            context_block = _format_docs(docs)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Question: {state['question']}\n\nContext:\n{context_block}"},
            ]
            resp = client.chat.completions.create(model="gpt-4o-mini", messages=messages, temperature=0)
            ans = resp.choices[0].message.content
        return {**state, "answer": ans}


def node_citation(state: AgentState) -> AgentState:
    with span("lg.citation"):
        cc = citation_checker(state.get("answer", ""))
        log_event("lg_citation", cc)
        return {**state, "citations": cc["citations"], "citation_coverage": cc["citation_coverage"]}


def node_support(state: AgentState) -> AgentState:
    with span("lg.support"):
        support = analyze_support(state.get("answer", ""), state.get("docs", []), threshold=0.58)
        log_event("lg_support", {"low_support": len(support.get("low_support", []))})
        return {**state, "support": support}


def node_regen(state: AgentState) -> AgentState:
    with span("lg.regen"):
        support = state.get("support", {})
        docs = state.get("docs", [])
        if support.get("offline"):
            return state
        regen_res = regenerate_if_needed(state["question"], state.get("answer", ""), docs, support, regen_ratio=0.4)
        if regen_res.get("regenerated"):
            log_event("lg_regen", {"reason": regen_res.get("reason")})
            return {**state, "answer": regen_res["answer"], "regenerated": True}
        return state


def node_fallback(state: AgentState) -> AgentState:
    with span("lg.fallback"):
        coverage = state.get("citation_coverage", 0.0)
        support = state.get("support", {})
        low_support = len(support.get("low_support", []))
        sentences = len(support.get("sentences", [])) or 1
        low_ratio = low_support / sentences
        if coverage < 0.3 or low_ratio > 0.4:
            fb = summarizer(state["question"], state.get("docs", []))
            log_event("lg_fallback", {"trigger": "citation_or_low_support"})
            return {**state, "answer": state.get("answer", "") + "\n[lg summarizer fallback]\n" + fb, "fallback_used": True}
        return state


def node_finalize(state: AgentState) -> AgentState:
    # Add footer diagnostics
    footer = ["[langgraph-agent]",
              f"coverage={state.get('citation_coverage', 0.0):.2f}",
              f"regen={'1' if state.get('regenerated') else '0'}",
              f"fallback={'1' if state.get('fallback_used') else '0'}"]
    answer = state.get("answer", "") + "\n" + " ".join(footer)
    log_event("lg_complete", {"answer_preview": answer[:400]})
    return {**state, "answer": answer}


def run_langgraph_agent(store, question: str) -> str:
    if StateGraph is None:
        return "LangGraph not available; falling back to offline context only."
    graph = StateGraph(AgentState)
    # Add nodes
    graph.add_node("retrieve", lambda s: node_retrieve(s, store))
    graph.add_node("baseline", node_baseline)
    graph.add_node("citation", node_citation)
    graph.add_node("support", node_support)
    graph.add_node("regen", node_regen)
    graph.add_node("fallback", node_fallback)
    graph.add_node("finalize", node_finalize)
    # Edges
    graph.set_entry_point("context")
    graph.add_node("context", node_context)
    graph.add_edge("context", "retrieve")
    graph.add_edge("retrieve", "baseline")
    graph.add_edge("baseline", "citation")
    graph.add_edge("citation", "support")
    graph.add_edge("support", "regen")
    graph.add_edge("regen", "fallback")
    graph.add_edge("fallback", "finalize")
    graph.add_edge("finalize", END)
    app = graph.compile()
    initial: AgentState = {"question": question}
    result = app.invoke(initial)
    return result.get("answer", "")
