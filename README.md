# GDPR-Intelligent-RegAssistant

Intelligent Regulatory Assistant for GDPR leveraging Responsible Retrieval-Augmented Generation (RAG).

## Implemented Features
* PDF ingestion and intelligent chunking (Articles / Recitals / Chapters heuristic)
* FAISS vector index with OpenAI embeddings (offline graceful fallback)
* Baseline RAG pipeline (query → retrieval → grounded answer + citations)
* Conversation memory (JSON history + semantic vector memory for recall)
* Guardrails: adversarial pattern detection, toxicity terms, advanced PII heuristics (regex + optional spaCy), citation enforcement
* Agentic reasoning orchestrator (multi-step: retrieval → answer → citation coverage → support analysis → regeneration → summarizer fallback)
* Hallucination mitigation: per-sentence similarity scoring + conditional regeneration + fallback summarization
* Graph-RAG: structural graph (Chapters / Articles / Recitals), PageRank neighbor expansion, guided regulatory rephrase, completeness check
* Audit logging: structured JSON events (retrieval metadata, guardrails, support analysis, regeneration, fallback use, graph completeness)
* Metrics aggregation: citation coverage, regeneration & fallback rates, low-support counts, guardrail triggers
* Guardrail tuning script (synthetic precision/recall) and expanded pattern lists
* Architecture diagram (Mermaid) and Responsible AI report
* Optional LangSmith tracing spans (`span()` wrappers) for RAG and agent steps
* Test suite: ingestion/index, graph, agent tools, orchestrator, hallucination, semantic memory, PII, guardrails, metrics, completeness, integration

## Estrutura
```
src/
	config.py
	ingest.py
	index_store.py
	rag.py
	guardrails.py
	cli.py
	tracing.py
notebooks/
	gdpr_rag.ipynb
tests/
	test_guardrails.py
	test_ingest_index.py
```

 (… additional test modules)
## Prerequisites
* Python 3.12+
* OpenAI API key exported in `OPENAI_API_KEY` (offline mode works if absent)
* Optional LangSmith tracing: set `LANGCHAIN_TRACING_V2=true` and `LANGCHAIN_API_KEY`

## Installation (PowerShell)
```powershell
python -m venv .venv
./.venv/Scripts/Activate.ps1
pip install -r requirements.txt
```

Set environment variables (PowerShell example):
```powershell
$env:OPENAI_API_KEY="SUA_CHAVE"
# Opcional LangSmith
$env:LANGCHAIN_TRACING_V2="true"
$env:LANGCHAIN_API_KEY="SUA_CHAVE_LANGSMITH"
```

## Run CLI
```powershell
python -m src.cli "What is personal data under GDPR?" --mode baseline
python -m src.cli "List obligations of controllers" --mode agent
python -m src.cli "Explain data minimization" --mode graph
```

## Notebook
Open `notebooks/gdpr_rag.ipynb` and run cells sequentially. The notebook demonstrates ingestion, indexing, baseline RAG, agent orchestration, graph expansion, hallucination analysis, guardrails, and metrics snapshot. If any helper is missing, re-run the earlier definition cells.

## Tests
```powershell
pip install pytest
pytest -q
```

## Export / Logs
Audit events and answer previews can be exported or aggregated. Session logs reside in `exports/` (extendable). LangSmith runs appear in your workspace if tracing enabled.

## Limitations
* LangGraph stateful agent pipeline not fully integrated (custom orchestrator used instead).
* PII detection remains heuristic; not production-grade.
* Offline embeddings fall back to zero vectors (reduces retrieval quality).
* Not a substitute for formal legal advice.

## Next Steps
1. Integrate full LangGraph pipeline for agent state management & branching.
2. Expand guardrail evaluation with larger labeled adversarial/toxic dataset.
3. Enhance citation verification (exact source span alignment scoring).
4. Add latency & token usage tracking into metrics module.
5. Improve semantic memory summarization for long sessions.
6. Deploy containerized service version with API endpoints & health checks.

## License
See `LICENSE`.
