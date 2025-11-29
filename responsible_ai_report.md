# Responsible AI Report (GDPR Assistant)

## Scope
This document captures safety, robustness, and accountability practices for the GDPR Retrieval-Augmented Assistant.

## System Overview
- Modality: Text Q&A over official GDPR PDF.
- Pipeline: User Query -> Guardrails (input) -> Retrieval (FAISS) -> LLM (ChatOpenAI) -> Guardrails (output) -> Memory + Audit Log.
- Modes: baseline, agent (heuristic second pass), graph (placeholder, to be upgraded).

## Data Sources
- Official GDPR Regulation PDF (CELEX_32016R0679_EN_TXT.pdf).
- No external dynamic data; static legal corpus.

## Safety & Guardrails
| Aspect | Current Implementation | Planned Enhancement |
|--------|-----------------------|---------------------|
| Adversarial prompt injection | Regex patterns ("ignore rules", "prompt injection") -> safe rewrite | ML classifier or embedding similarity to injection exemplars |
| Toxic content | Keyword refusal (hate, racist, sexist, violent) | Moderation API or classifier + severity taxonomy |
| PII detection | Simple regex patterns (emails, generic IDs) | Comprehensive pattern library + entity NER + hashing for storage |
| Output citation enforcement | Post-check for page/article markers; add warning if absent | Pre-flight retrieval verification + forced answer regeneration |
| Hallucination mitigation | Citation presence heuristic only | Semantic overlap scoring; retrieval-only answer verification pass |

## Robustness
Tests executed:
- Guardrails unit tests (adversarial rewrite, toxic refusal, citation enforcement).
- Ingestion smoke test (skipped if PDF missing).

Planned robustness tests:
- Adversarial corpus of injection attempts.
- Paraphrased regulatory queries (semantic invariance).
- Stress queries (very long, multi-topic).

## Hallucination Strategy
Phase 1 (current): Warn if answer lacks citations.
Phase 2 (planned):
1. Retrieve chunks separately.
2. Compute overlap score between answer sentences and retrieved text.
3. Flag low-overlap sentences and either remove or regenerate.

## Accountability & Auditing
Current: JSON session log with events (refusal, answer preview).
Planned:
- Log retrieval doc IDs + scores.
- Log guardrail triggers (type, pattern matched).
- Log timing per stage (retrieval_ms, generation_ms).
- Export daily report (CSV or JSON).

## Memory Handling
- Stored last messages in JSON file (.memory/chat_history.json).
- Future: vector-based conversational memory (distill prior regulatory context).

## Graph RAG (Planned)
- Build graph: Chapters -> Articles -> Recitals referencing Articles.
- Apply PageRank to prioritize structurally central nodes.
- Expand retrieval: anchor chunks + neighbor traversal to cover conditions/complements.

## Risks & Mitigations
| Risk | Impact | Mitigation |
|------|--------|-----------|
| Partial retrieval leads to incomplete legal interpretation | Misleading compliance guidance | Graph expansion + completeness checklist |
| Hallucinated citation | False trust by user | Verification layer: citation string must appear verbatim in source chunk |
| PII exposure in output | Privacy risk | Strengthen PII regex + redaction before display |
| Prompt injection causing system override | Unsafe generation | Multi-layer input scanning + system prompt hardening |

## Open Items
- Integrate LangSmith tracing (versions conflict resolution required).
- Implement LangGraph-based agent orchestration.
- Add diagram of graph workflow.
- Run adversarial test suite and quantify block/allow rates.

## Versioning
Report version: 0.1 (initial skeleton)
Next update: After graph_rag.py + agent_tools.py integration.

## Graph Completeness & Coverage
Heuristic check ensures Articles 1–99 and Recitals 1–173 are parsed; missing IDs logged (`graph_completeness`) for ingestion monitoring.

## Guardrail Evaluation & Threshold Tuning
Synthetic evaluation (`guardrail_tuning.py`) computes preliminary precision/recall for adversarial and toxic detection enabling iterative refinement; expand with larger labeled dataset for robust metrics.

## Guardrail Evaluation Snapshot (Synthetic)

The tuning script (`guardrail_tuning.py`) was executed against a small synthetic set of 5 samples (mixed adversarial and toxic prompts). Current patterns achieved:

| Category      | Precision | Recall | Samples |
|---------------|-----------|--------|---------|
| Adversarial   | 1.00      | 1.00   | 5       |
| Toxic         | 1.00      | 1.00   | 5       |

Interpretation:
 - Synthetic set is tiny; results are optimistic and NOT representative of production.
 - Pattern expansion favors high recall for clear jailbreak attempts while keeping toxicity lexicon concise to avoid false positives.
 - Next step: assemble a larger, labeled evaluation corpus (>=200 prompts) spanning: neutral GDPR queries, borderline context (academic discussion of sensitive topics), explicit abuse, and sophisticated jailbreak phrasing.

Planned Enhancements (Updated - severity tiers implemented):
1. Embedding similarity layer for adversarial intent to catch obfuscated phrasing.
2. Statistical FP analysis (precision) per subclass (role-escalation, exfiltration, override).
3. Continuous metrics logging: false positives flagged manually in audit for iterative pruning.
4. Larger labeled corpus integration pipeline.

### Current Severity Behavior
| Type | Action | Notes |
|------|--------|-------|
| Toxic content | Block (hard refusal) | Returns static SAFE_REFUSAL template |
| Adversarial injection (non-toxic) | Warn + Safe Rewrite | Original harmful override clauses truncated; warning tag emitted |
| Benign query | Pass-through | No modification |

Safe rewrite format: `[Safe Rewrite] [Safety Warning] <sanitized user text>`

Next step: persist structured guardrail event entries (JSONL) including: timestamp, original_text_hash, action, matched_patterns, rewrite_text.

Risk Mitigations:
 - Overblocking: Keep list minimal; review weekly against false positive log.
 - Underblocking: Add semantic similarity fallback with conservative threshold.
 - Drift: Re-run evaluation monthly with updated adversarial corpus.

