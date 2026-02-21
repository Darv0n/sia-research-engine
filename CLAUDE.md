# Deep Research Swarm

## Current State

- **Version**: V5 complete (memory, state dump, PostgresSaver)
- **Tests**: 196 passing (run with `.venv/Scripts/python.exe -m pytest tests/ -v`)
- **Repo**: https://github.com/Darv0n/deep-research-swarm.git

## Architecture

LangGraph StateGraph orchestrating multi-agent research pipeline:
```
health_check -> plan -> search -> extract -> score -> contradiction -> synthesize -> critique -> rollup_budget -> [converge?] -> report
```

## Key Files

- `contracts.py` — Single source of truth for all types, enums, protocols
- `config.py` — All settings from env vars (frozen dataclass)
- `graph/state.py` — ResearchState TypedDict with annotated reducers
- `graph/builder.py` — Graph construction, node closures, edge wiring

### V3 Additions
- `backends/cache.py` — File-based search cache (SHA-256 keys, TTL)
- `scoring/diversity.py` — HHI-based source diversity scoring
- `reporting/trends.py` — Confidence sparklines across iterations
- `streaming.py` — StreamDisplay for astream progress to stderr
- `agents/contradiction.py` — Sonnet-powered contradiction detection between sources

### V4 Additions
- `config.py` — `checkpoint_db`, `checkpoint_backend` settings
- `graph/builder.py` — `checkpointer` param, `CompiledStateGraph` return type
- `__main__.py` — `--resume THREAD_ID`, `--list-threads`, thread_id generation, AsyncSqliteSaver lifecycle
- `extractors/pdf_extractor.py` — HTTP download via httpx, asyncio.to_thread for CPU-bound pymupdf4llm
- `extractors/__init__.py` — PDF-first cascade: PDF -> Crawl4AI -> Trafilatura

### V5 Additions
- `contracts.py` — `ResearchMemory` TypedDict
- `config.py` — `memory_dir`, `postgres_dsn` settings
- `graph/state.py` — `memory_context` field (pre-computed, injected before graph execution)
- `utils/text.py` — `jaccard_score()`, `is_duplicate()` (extracted from planner)
- `memory/store.py` — `MemoryStore` class (JSON-backed, Jaccard search)
- `memory/extract.py` — `extract_memory_record()` (deterministic, no LLM call)
- `memory/mcp_export.py` — `export_to_mcp_format()` (MCP entity format)
- `__main__.py` — `_make_checkpointer()` async context manager, memory lifecycle, 4 new CLI flags

### TypedDicts in contracts.py
SubQuery, SearchResult, ExtractedContent, ScoredDocument, GraderScores,
SectionDraft, Citation, ResearchGap, TokenUsage, Contradiction,
DiversityMetrics, SectionConfidenceSnapshot, IterationRecord, ResearchMemory

### State Fields (graph/state.py)
- Accumulating (operator.add): search_backends, perspectives, sub_queries, search_results, extracted_contents, token_usage, iteration_history
- Replace-last-write: scored_documents, diversity_metrics, section_drafts, citations, contradictions, research_gaps, current_iteration, converged, convergence_reason, total_tokens_used, total_cost_usd, final_report, memory_context

## Conventions

- All agent nodes: `async def node_name(state: ResearchState) -> dict`
- List fields use `Annotated[list[T], operator.add]` for concurrent merge
- Replace fields use custom `_replace_*` reducers
- Backends implement `SearchBackend` Protocol (structural subtyping)
- Model tiering: Opus for plan/synthesize, Sonnet for critique/contradiction
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

## Config Vars

- `CHECKPOINT_DB` — Path to SQLite checkpoint database (default: `checkpoints/research.db`)
- `CHECKPOINT_BACKEND` — `sqlite` (default), `postgres`, or `none`
- `MEMORY_DIR` — Path to memory storage directory (default: `memory/`)
- `POSTGRES_DSN` — PostgreSQL connection string (required when `CHECKPOINT_BACKEND=postgres`)
- `CONVERGENCE_THRESHOLD` — Minimum confidence delta for continued iteration (default: `0.05`)

## Testing

```bash
.venv/Scripts/python.exe -m pytest tests/ -v                    # All 196 tests
.venv/Scripts/python.exe -m pytest tests/ -k "not integration"  # Unit only
.venv/Scripts/python.exe -m ruff check . && .venv/Scripts/python.exe -m ruff format --check .
```

## Deferred to V6

- **MCP Memory server** — live MCP server for cross-session memory (V5 added export-only)
- **Memory pruning** — automatic cleanup of old/low-value memory records
- **Embedding-based retrieval** — replace Jaccard with vector similarity for memory search
