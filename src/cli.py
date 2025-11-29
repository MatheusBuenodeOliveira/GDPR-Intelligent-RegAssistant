"""CLI entrypoint for GDPR Responsible RAG Assistant (baseline, agent, graph modes)."""
import argparse
from .guardrails import guard_input, guard_output, SAFE_REFUSAL
from .rag import answer as rag_answer
from .ingest import load_pdf, chunk_documents
from .graph_rag import extract_structured_nodes, build_graph, rank_graph, retrieve_with_graph, validate_completeness, rephrase_question
from .semantic_completeness import check_semantic_completeness
from .index_store import load_or_build
from .memory import append_message, get_context
from .semantic_memory import build_semantic_context_block
from .audit import log_event, log_retrieval, log_guardrail
from .agent_tools import retrieve, citation_checker, summarizer, hallucination_score
from .agent_orchestrator import AgentOrchestrator, OrchestratorConfig
from .langgraph_agent import run_langgraph_agent


def _ensure_store():
    raw_docs = load_pdf()
    chunks = chunk_documents(raw_docs)
    return load_or_build(chunks)


def baseline_answer(q: str, store):
    """Baseline RAG answer with prior conversation context prepended if available."""
    ctx = get_context()
    sem_block = build_semantic_context_block(q, k=3)
    full_ctx = ctx + ("\n" + sem_block if sem_block else "") if ctx else sem_block
    full_q = f"Conversation context:\n{full_ctx}\n\nUser question: {q}" if full_ctx else q
    return rag_answer(store, full_q)


def agent_answer(q: str, store):
    """Agent mode using structured orchestrator pipeline for multi-step reasoning and auditing."""
    orchestrator = AgentOrchestrator(store, OrchestratorConfig())
    result = orchestrator.run(q)
    return result["answer"]


def graph_answer(q: str, store, full_page: bool = False):
    """Graph mode: build structural graph, guided rephrase, expand neighbors, inject enriched context.

    full_page: if True include full page content for expanded nodes.
    """
    raw_docs = load_pdf()
    nodes = extract_structured_nodes(raw_docs)
    graph = build_graph(nodes)
    page_rank = rank_graph(graph)
    completeness = validate_completeness(nodes)
    # Guided regulatory rephrase step
    rephrased = rephrase_question(q)
    if rephrased != q:
        log_event("graph_rephrase", {"original": q, "rephrased": rephrased})
    store_local = _ensure_store()
    context, node_ids = retrieve_with_graph(store_local, rephrased, graph, page_rank, k=5, neighbor_depth=1, full_pages=full_page)
    sem_comp = check_semantic_completeness(rephrased, context)
    augmented_q = f"Use the following structural GDPR context blocks to answer.\n\n{context}\n\nOriginal Question: {q}\nRephrased: {rephrased}"
    ans = rag_answer(store, augmented_q)
    ans += f"\n[graph-mode] expanded_nodes={len(node_ids)} articles_found={completeness['articles_found']} recitals_found={completeness['recitals_found']} full_page={'1' if full_page else '0'}"
    if sem_comp["topics"]:
        ans += f" semantic_complete={'1' if sem_comp['complete'] else '0'} missing_topics={ {t: sem_comp['missing'].get(t, []) for t in sem_comp['topics']} }"
    log_event("graph_completeness", completeness | {"semantic": sem_comp})
    return ans


def run(mode: str, question: str, log: bool = True, full_page: bool = False) -> str:
    """Run a single question through selected mode with guardrails, memory, and audit logging."""
    original = question
    guarded = guard_input(original)
    # Always return refusal immediately regardless of logging flag
    if guarded == SAFE_REFUSAL:
        if log:
            log_guardrail("toxic_or_refusal", {"question": original})
        return SAFE_REFUSAL
    if guarded.startswith("[Safe Rewrite]") and log:
        log_guardrail("adversarial_rewrite", {"original": original, "rewritten": guarded})
    store = _ensure_store()
    # Log retrieval context independently (chain has its own retriever but we repeat lightweight store search for audit transparency)
    if mode != "agent":  # Orchestrator handles its own retrieval logging
        try:
            raw_docs = store.similarity_search(original, k=5)
            if log:
                log_retrieval(original, raw_docs)
        except Exception:
            pass
    if mode == "baseline":
        ans = baseline_answer(guarded, store)
    elif mode == "agent":
        ans = agent_answer(guarded, store)
    elif mode == "graph":
        ans = graph_answer(guarded, store, full_page=full_page)
    else:  # lgagent
        ans = run_langgraph_agent(store, guarded)
    ans = guard_output(ans)
    append_message("user", question)
    append_message("assistant", ans[:2000])
    if log:
        log_event("answer", {"mode": mode, "question": question, "answer_preview": ans[:400]})
    return ans


def main():  # pragma: no cover
    parser = argparse.ArgumentParser(description="GDPR Responsible RAG Assistant")
    parser.add_argument("question", help="Question to ask the assistant")
    parser.add_argument("--mode", default="baseline", choices=["baseline", "agent", "graph", "lgagent"], help="Execution mode")
    parser.add_argument("--full-page", action="store_true", help="Return full page content in graph mode")
    parser.add_argument("--no-log", action="store_true", help="Disable audit logging")
    args = parser.parse_args()
    print(run(args.mode, args.question, log=not args.no_log, full_page=args.full_page))


if __name__ == "__main__":  # pragma: no cover
    main()
