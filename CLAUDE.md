# Deep Research Swarm

## Current State

- **Version**: V7 (niche retrieval: scholarly backends, archive fallback, citation chaining, provenance, grounded synthesis)
- **Tests**: 472 passing (run with `.venv/Scripts/python.exe -m pytest tests/ -v`)
- **Repo**: https://github.com/Darv0n/deep-research-swarm.git

## Architecture

LangGraph StateGraph orchestrating multi-agent research pipeline:
```
health_check -> plan -> [plan_gate?] -> search -> extract -> chunk_passages -> score -> citation_chain -> contradiction -> synthesize -> critique -> rollup_budget -> [converge?] -> report -> [report_gate?]
```
Gate nodes (`plan_gate`, `report_gate`) only present when `--mode hitl` and checkpointer is active.

## Key Files

- `contracts.py` — Single source of truth for all types, enums, protocols
- `config.py` — All settings from env vars (frozen dataclass)
- `graph/state.py` — ResearchState TypedDict with annotated reducers
- `graph/builder.py` — Graph construction, node closures, edge wiring

### V7 Additions
- `backends/openalex.py` — OpenAlex scholarly search backend (abstract reconstruction from inverted index)
- `backends/semantic_scholar.py` — Semantic Scholar backend + `get_paper_details()` for citation chaining
- `backends/crossref.py` — Crossref/Unpaywall DOI resolution utilities (not a SearchBackend)
- `backends/wayback.py` — Wayback Machine archive backend (CDX API, polite spacing, `id_` replay URLs)
- `extractors/chunker.py` — 4-tier passage chunking (headings -> paragraphs -> sentences -> hard split)
- `scoring/grounding.py` — Mechanical grounding verification (LLM-free, Jaccard-based, method="jaccard_v1")
- `scoring/authority.py` — Enhanced authority scoring with scholarly metadata signals
- `scoring/routing.py` — Deterministic query classification + backend routing (keyword heuristic)
- `scoring/provenance.py` — Content hash provenance tracking
- `reporting/provenance.py` — Provenance section rendering
- `agents/citation_chain.py` — BFS citation graph traversal via Semantic Scholar (50-paper budget)
- `agents/synthesizer.py` — 5-stage outline-first synthesis pipeline (validate -> outline -> draft -> verify -> refine -> compose)

### Previous Versions
- V6: HITL mode, run event log, evidence map
- V5: Cross-session memory, MCP export, PostgresSaver
- V4: Checkpoint persistence, PDF extraction
- V3: Search cache, streaming, contradiction detection, diversity scoring
- V2: Multi-iteration quality loop, critique chain
- V1: Core pipeline

### TypedDicts in contracts.py
SubQuery, SearchResult, ExtractedContent, ScoredDocument, GraderScores,
SectionDraft, Citation, ResearchGap, TokenUsage, Contradiction,
DiversityMetrics, SectionConfidenceSnapshot, RunEvent, IterationRecord, ResearchMemory,
ProvenanceRecord, ScholarlyMetadata, ArchiveCapture, SourcePassage, SectionOutline, IndexedContent

### State Fields (graph/state.py)
- Accumulating (operator.add): search_backends, perspectives, sub_queries, search_results, extracted_contents, token_usage, iteration_history, source_passages, citation_chain_results
- Replace-last-write: scored_documents, diversity_metrics, section_drafts, citations, contradictions, research_gaps, current_iteration, converged, convergence_reason, total_tokens_used, total_cost_usd, final_report, memory_context, citation_to_passage_map

## Conventions

- All agent nodes: `async def node_name(state: ResearchState) -> dict`
- List fields use `Annotated[list[T], operator.add]` for concurrent merge
- Replace fields use custom `_replace_*` reducers
- Backends implement `SearchBackend` Protocol (structural subtyping)
- Model tiering: Opus for plan/synthesize, Sonnet for critique/contradiction
- Passage IDs are content-hash-based, never random UUIDs
- `verify_grounding()` is LLM-free and deterministic (method="jaccard_v1")
- Use `.venv/Scripts/python.exe` — NOT the global Python (missing langgraph)

## CLI Flags

`python -m deep_research_swarm "question"` with:
- `--max-iterations`, `--token-budget`, `--output`, `--backends`
- `--no-cache` — Disable search result caching
- `--no-stream` — Use blocking ainvoke instead of astream
- `--verbose` — Detailed streaming progress
- `--resume THREAD_ID` — Resume a previous research run from checkpoint
- `--list-threads` — List recent research threads from checkpoint DB
- `--dump-state THREAD_ID` — Export checkpoint state as JSON
- `--no-memory` — Disable memory store/retrieval for this run
- `--list-memories` — Print stored memory records
- `--export-mcp` — Export memory in MCP entity format
- `--no-log` — Disable run event logging
- `--mode auto|hitl` — Execution mode (auto = default, hitl = human-in-the-loop gates)
- `--academic` — Add openalex and semantic_scholar to backends for this run
- `--no-archive` — Disable Wayback Machine archive fallback for this run

## Config Vars

- `ANTHROPIC_API_KEY` — Anthropic API key (required)
- `SEARXNG_URL` — SearXNG instance URL (default: `http://localhost:8080`)
- `EXA_API_KEY` — Exa semantic search API key (optional)
- `TAVILY_API_KEY` — Tavily factual search API key (optional)
- `OPUS_MODEL` — Model for planner, synthesizer (default: `claude-opus-4-6`)
- `SONNET_MODEL` — Model for critic graders (default: `claude-sonnet-4-6`)
- `MAX_ITERATIONS` — Maximum research iterations (default: `3`)
- `TOKEN_BUDGET` — Maximum total tokens (default: `200000`)
- `MAX_CONCURRENT_REQUESTS` — Concurrent API calls (default: `5`)
- `AUTHORITY_WEIGHT` — Weight of authority score in ranking (default: `0.2`)
- `RRF_K` — RRF smoothing parameter (default: `60`)
- `CONVERGENCE_THRESHOLD` — Minimum confidence delta for continued iteration (default: `0.05`)
- `SEARCH_CACHE_TTL` — Search cache TTL in seconds (default: `3600`)
- `SEARCH_CACHE_DIR` — Search cache directory (default: `.cache/search`)
- `CHECKPOINT_DB` — SQLite checkpoint database path (default: `checkpoints/research.db`)
- `CHECKPOINT_BACKEND` — `sqlite` (default), `postgres`, or `none`
- `POSTGRES_DSN` — PostgreSQL connection string (required when `CHECKPOINT_BACKEND=postgres`)
- `MEMORY_DIR` — Path to memory storage directory (default: `memory/`)
- `RUN_LOG_DIR` — Path to run event log directory (default: `runs/`)
- `MODE` — Execution mode: `auto` (default) or `hitl` (human-in-the-loop gates)
- `OPENALEX_EMAIL` — Email for OpenAlex polite pool access (enables openalex backend)
- `OPENALEX_API_KEY` — OpenAlex API key (optional; email is sufficient)
- `S2_API_KEY` — Semantic Scholar API key (optional; works unauthenticated)
- `WAYBACK_ENABLED` — Enable Wayback Machine backend (default: `true`)
- `WAYBACK_TIMEOUT` — Wayback request timeout in seconds (default: `15`)

## Testing

```bash
.venv/Scripts/python.exe -m pytest tests/ -v                    # All 472 tests
.venv/Scripts/python.exe -m pytest tests/ -k "not integration"  # Unit only
.venv/Scripts/python.exe -m ruff check . && .venv/Scripts/python.exe -m ruff format --check .
```

## Deferred to V8

- **Internal hybrid index** — OpenSearch k-NN (IndexedContent stub is the handshake)
- **Focused crawling** — priority frontier for deep site exploration
- **OCR / GROBID** — scanned PDF support
- **Embedding-based routing** — upgrade keyword heuristic to vector classification
- **Embedding-based passage retrieval** — upgrade Jaccard grounding (method field enables coexistence)
- **PROV-O JSON-LD export** — ProvenanceRecord is the foundation
- **MCP Memory server** — live server, not just export
- **Claim graph** — via SourcePassage.claim_ids
- **Incremental research** — content_hash-based passage IDs for diff-only updates
- **Per-claim confidence visualization** — via claim_details
- **Memory pruning** — automatic cleanup of old/low-value records
