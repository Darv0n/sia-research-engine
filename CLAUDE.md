# Deep Research Swarm

## Current State

- **Version**: V4 complete, on branch `feat/v3-reports-observability`
- **Tests**: 152 passing (run with `.venv/Scripts/python.exe -m pytest tests/ -v`)
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

### TypedDicts in contracts.py
SubQuery, SearchResult, ExtractedContent, ScoredDocument, GraderScores,
SectionDraft, Citation, ResearchGap, TokenUsage, Contradiction,
DiversityMetrics, SectionConfidenceSnapshot, IterationRecord

### State Fields (graph/state.py)
- Accumulating (operator.add): search_backends, perspectives, sub_queries, search_results, extracted_contents, token_usage, iteration_history
- Replace-last-write: scored_documents, diversity_metrics, section_drafts, citations, contradictions, research_gaps, current_iteration, converged, convergence_reason, total_tokens_used, total_cost_usd, final_report

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

## Config Vars (V4)

- `CHECKPOINT_DB` — Path to SQLite checkpoint database (default: `checkpoints/research.db`)
- `CHECKPOINT_BACKEND` — `sqlite` (default) or `none` to disable checkpointing

## Testing

```bash
.venv/Scripts/python.exe -m pytest tests/ -v                    # All 152 tests
.venv/Scripts/python.exe -m pytest tests/ -k "not integration"  # Unit only
.venv/Scripts/python.exe -m ruff check . && .venv/Scripts/python.exe -m ruff format --check .
```

## Deferred to V5

- **MCP Memory server** — cross-session knowledge persistence
- **JSON state dump** — `--dump-state` for text-editor-inspectable checkpoint export
- **PostgresSaver** — config field exists (`checkpoint_backend`), implementation deferred
