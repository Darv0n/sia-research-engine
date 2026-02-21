# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.6.1] - 2026-02-21

### Fixed
- Robust JSON extraction in `call_json` — new `_extract_json()` helper finds JSON objects embedded in prose responses using brace-depth tracking
- Contradiction detector prompt hardened with strict JSON-only directive to prevent prose output
- Contradiction detector `max_tokens` increased 4096 -> 8192 for safety margin with large source sets

## [0.6.0] - 2026-02-21

### Added
- **HITL mode**: `--mode hitl` adds LangGraph `interrupt()` gates after plan and report nodes
  - `plan_gate`: review research plan before search begins
  - `report_gate`: review final report before accepting
  - Resume with `--resume THREAD_ID` to continue through gates
  - Graceful fallback to auto mode when no checkpointer is present
- **Run event log**: append-only JSONL per run at `runs/<thread_id>/events.jsonl`
  - `RunEvent` TypedDict: node name, iteration, timestamp, elapsed time, token/cost summary
  - `EventLog` class with emit/read/make_event factory
  - `_wrap_with_logging()` wraps node closures for transparent event capture
  - `--no-log` CLI flag to disable
- **Evidence map appendix**: post-synthesis claim-to-source mapping in reports
  - Extracts cited sentences from section drafts, maps to citation metadata
  - Markdown table: Claim | Source | Authority | Confidence
  - Rendered after Bibliography section
- Config: `RUN_LOG_DIR`, `MODE` settings with validation
- CLI: `--mode` flag (auto/hitl), `--no-log` flag

### Changed
- `build_graph()` accepts `event_log` and `mode` parameters
- Node map pattern for clean wrapping/extension in builder.py

## [0.5.1] - 2026-02-21

### Fixed
- Wire `convergence_threshold` from Settings through critique node to `should_replan()`
- Remove orphaned `budget/` package (dead code since `rollup_budget_node`)

### Added
- Tests for `export_to_mcp_format()`

## [0.5.0] - 2026-02-20

### Added
- Cross-session memory with JSON-backed `MemoryStore` (Jaccard search)
- `extract_memory_record()` — deterministic extraction, no LLM call
- `export_to_mcp_format()` — MCP entity format export
- `memory_context` field in `ResearchState` (pre-computed, injected before graph)
- CLI flags: `--no-memory`, `--list-memories`, `--export-mcp`
- `jaccard_score()` and `is_duplicate()` text utilities
- PostgresSaver support (`CHECKPOINT_BACKEND=postgres`, `POSTGRES_DSN`)

### Changed
- `_make_checkpointer()` async context manager handles sqlite/postgres/none

## [0.4.0] - 2026-02-19

### Added
- Checkpoint persistence with `AsyncSqliteSaver` / `AsyncPostgresSaver`
- CLI flags: `--resume THREAD_ID`, `--list-threads`, `--dump-state THREAD_ID`
- Thread ID generation (`research-YYYYMMDD-HHMMSS-XXXX`)
- PDF extraction via `pymupdf4llm` with HTTP download
- PDF-first extraction cascade: PDF -> Crawl4AI -> Trafilatura

## [0.3.0] - 2026-02-19

### Added
- File-based search cache with SHA-256 keys and TTL
- HHI-based source diversity scoring
- Confidence sparklines across iterations
- `StreamDisplay` for `astream` progress to stderr
- Contradiction detection between sources (Sonnet-powered)
- Report appendices: confidence heatmap, trends, diversity, contradictions, bibliography
- YAML frontmatter in rendered reports
- Citation deduplication and renumbering

## [0.2.0] - 2026-02-18

### Added
- Multi-iteration quality loop with critique and re-planning
- Convergence detection based on confidence delta
- RRF + authority scoring with configurable weights
- Grader scores (relevance, hallucination, quality) per section
- Section confidence levels (HIGH/MEDIUM/LOW)
- Iteration history tracking with `IterationRecord`

## [0.1.0] - 2026-02-17

### Added
- Core multi-agent research pipeline: plan -> search -> extract -> score -> synthesize
- LangGraph StateGraph with typed state and annotated reducers
- STORM-style question decomposition with perspectives
- Fan-out parallel search across SearXNG, Exa, Tavily
- Content extraction via Crawl4AI and Trafilatura
- Reciprocal Rank Fusion document scoring
- Source authority classification (institutional/professional/community/promotional)
- Opus/Sonnet model tiering (Opus for plan+synthesize, Sonnet for critique)
- Markdown report generation
- CLI entry point with configurable backends, iterations, token budget
