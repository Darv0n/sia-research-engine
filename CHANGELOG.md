# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.7.0] - 2026-02-23

### Added
- **OpenAlex backend**: Scholarly search via OpenAlex API with abstract reconstruction from inverted index, 429 retry with exponential backoff, polite pool support via `OPENALEX_EMAIL`
- **Semantic Scholar backend**: Scholarly search + `get_paper_details()` for citation chaining, Retry-After header handling, works unauthenticated
- **Crossref/Unpaywall utilities**: DOI resolution, open-access URL lookup, content negotiation, scholarly result enrichment
- **Wayback Machine backend**: Archive fallback via CDX API, polite 1s spacing, `id_` replay URLs, integrated into extractor cascade after live extractors fail
- **Passage chunking**: 4-tier chunking (headings -> paragraphs -> sentences -> hard split), deterministic SHA-256-based passage IDs, `chunk_all_documents()` wired between extract and score nodes
- **Mechanical grounding verification**: LLM-free `verify_grounding()` using Jaccard similarity (method="jaccard_v1"), `compute_section_grounding_score()` with 0.8 pass threshold, `assign_passages_to_sections()` for passage narrowing
- **Enhanced authority scoring**: `score_authority()` with scholarly metadata signals, SCHOLARLY_DOMAINS set, extended INSTITUTIONAL_SUFFIXES
- **Backend routing**: Deterministic query classification (`classify_query()`) using keyword signal density, `route_backends()` for type-specific backend ordering, QueryType enum (ACADEMIC/GENERAL/ARCHIVAL/TECHNICAL)
- **Provenance tracking**: Content hash provenance with `build_provenance_record()`, provenance section in rendered reports
- **Citation chaining**: BFS traversal of Semantic Scholar citation graphs, 50-paper budget, 2-hop depth, cross-iteration deduplication, relevance filtering via Jaccard similarity
- **Outline-first synthesis**: 5-stage pipeline replacing single-call synthesis — validate outline (deterministic), generate outline (1 LLM call), draft sections in parallel (N LLM calls), verify grounding (mechanical, LLM-free), refine failed sections (max 2 attempts), compose report (intro/transitions/conclusion)
- **Composition structure**: Reports now have LLM-generated introduction, section transitions, and conclusion (section content immutable during composition)
- **Grounding data survival**: `grounding_score` and `claim_details` persist through synthesis to final output, `citation_to_passage_map` as first-class state output
- **CLI flags**: `--academic` (add scholarly backends), `--no-archive` (disable Wayback fallback)
- **Config warnings**: Non-fatal warnings for aggressive Wayback timeout
- New TypedDicts: ProvenanceRecord, ScholarlyMetadata, ArchiveCapture, SourcePassage, SectionOutline, IndexedContent
- New state fields: `source_passages`, `citation_chain_results`, `citation_to_passage_map`
- Config: `OPENALEX_EMAIL`, `OPENALEX_API_KEY`, `S2_API_KEY`, `WAYBACK_ENABLED`, `WAYBACK_TIMEOUT`
- 233 new tests (472 total)

### Changed
- Pipeline now includes `chunk_passages` node between extract and score, and `citation_chain` node between score and contradiction
- Synthesizer completely rewritten: outline-first with parallel section drafting, passage-narrowed context, mechanical grounding verification
- Report renderer handles composition structure (introduction, transitions, conclusion)
- Planner integrates backend routing: `classify_query()` + `route_backends()` applied per sub-query when user doesn't specify backends
- Authority scoring uses scholarly metadata when available (citation count, journal prestige)

## [0.6.2] - 2026-02-21

### Fixed
- Opus-to-Sonnet fallback on 529 (OverloadedError) — `AgentCaller` catches overloaded errors with exponential backoff, falls back to Sonnet after retries exhausted
- Mode persistence in checkpoint state — `ResearchState` now stores `mode` field, resume path reads stored mode so `--mode hitl` no longer required on every `--resume`
- Suppress LangGraph `RunnableConfig` typing advisory warning during graph compilation

### Added
- `fallback_model` parameter on `AgentCaller` (wired as `sonnet_model` for opus caller)
- `model_override` support in `_track_usage()` for correct fallback pricing
- 9 new tests: fallback trigger, no-fallback on generic errors, no-fallback when unconfigured, correct sonnet pricing, stderr warnings, mode in state, warning suppression

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
