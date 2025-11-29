"""Architecture Diagram"""

```mermaid
flowchart TD
    A[GDPR PDF] --> B[ingest.load_pdf]
    B --> C[chunk_documents]
    C --> D[index_store.build_index]
    D --> E[rag.answer]
    C --> G[graph_rag.extract_structured_nodes]
    G --> H[build_graph + PageRank]
    H --> I[retrieve_with_graph]
    I --> E
    D --> J[agent_orchestrator.run]
    J --> K[hallucination.analyze_support]
    J --> L[summarizer fallback]
    E --> M[guardrails.guard_output]
    J --> M
    M --> N[audit.log_event]
    D --> O[semantic_memory.retrieve_semantic_context]
    O --> E
```

Baseline: ingest -> index -> retrieve -> answer -> guardrails -> audit.
Graph: adds structural extraction + neighbor expansion.
Agent: orchestrator adds support analysis, regeneration, summarization.
Memory: semantic recall augments question context.
Safety: guardrails + hallucination mitigation.