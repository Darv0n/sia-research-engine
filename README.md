# Deep Research Swarm

Multi-agent deep research system built on [LangGraph](https://github.com/langchain-ai/langgraph). Takes a research question, decomposes it via STORM-style perspective-guided questioning, dispatches parallel search agents across multiple backends (including scholarly APIs and web archives), synthesizes findings via outline-first grounded synthesis with passage-level verification, critiques through a three-grader chain, and iterates until convergence. Produces structured Markdown reports with passage-level citations, confidence heat maps, provenance tracking, and gap analysis.

## Table of Contents

- [Quick Start](#quick-start)
- [How It Works](#how-it-works)
- [Architecture](#architecture)
- [Pipeline Stages](#pipeline-stages)
- [Configuration](#configuration)
- [Search Backends](#search-backends)
- [Convergence & Scoring](#convergence--scoring)
- [Output Format](#output-format)
- [CLI Reference](#cli-reference)
- [Project Structure](#project-structure)
- [Testing](#testing)
- [Development](#development)
- [Roadmap](#roadmap)
- [License](#license)

## Quick Start

### Prerequisites

- Python 3.11+
- Docker Desktop (for SearXNG)
- [Anthropic API key](https://console.anthropic.com/)

### Setup

```bash
# 1. Clone
git clone https://github.com/Darv0n/deep-research-swarm.git
cd deep-research-swarm

# 2. Start SearXNG (local search engine)
docker compose -f docker/docker-compose.yml up -d

# 3. Create virtual environment and install
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -e ".[dev]"

# 4. Configure
cp .env.example .env
# Edit .env — set ANTHROPIC_API_KEY (required)

# 5. Run
python -m deep_research_swarm "What is quantum entanglement?"

# With scholarly backends (OpenAlex + Semantic Scholar)
python -m deep_research_swarm --academic "CRISPR gene editing for metabolic disorders"
```

The report prints to stdout and saves to `output/<timestamp>-<question>.md`.

## How It Works

Given a research question, the system:

1. **Decomposes** the question into 3+ sub-queries from diverse perspectives (STORM method)
2. **Routes** each sub-query to appropriate backends based on query type (academic, technical, archival, general)
3. **Searches** each sub-query in parallel across configured backends (including scholarly APIs)
4. **Extracts** clean content from result URLs via a cascade of extractors (with Wayback Machine fallback for 404s)
5. **Chunks** extracted documents into source passages with deterministic IDs
6. **Scores** documents using Reciprocal Rank Fusion + enhanced authority classification (scholarly metadata signals)
7. **Chains** citations via BFS traversal of the Semantic Scholar citation graph
8. **Detects** contradictions between sources
9. **Synthesizes** via outline-first pipeline: generate outline, draft sections in parallel with passage-narrowed context, mechanically verify grounding, refine failed sections, compose report
10. **Critiques** each section across three dimensions: relevance, hallucination, quality
11. **Iterates** if quality is insufficient — re-plans to address gaps and weak sections
12. **Renders** a final Markdown report with provenance, bibliography, confidence heat map, and gap analysis

On subsequent iterations, the synthesizer keeps HIGH-grounding sections intact while re-drafting MEDIUM/LOW sections with new sources.

## Architecture

```
                    +------------------+
                    |   health_check   |  Verify backends
                    +--------+---------+
                             |
                    +--------v---------+
             +----->|      plan        |  STORM decomposition + routing
             |      +--------+---------+
             |               |
             |      +--------v---------+
             |      |  [plan_gate]     |  HITL interrupt (--mode hitl only)
             |      +--------+---------+
             |               |
             |      +--------v---------+
             |      |  search (fan-out)|  Parallel per sub-query
             |      +--------+---------+
             |               |
             |      +--------v---------+
             |      | extract (fan-out)|  Parallel per URL (max 30)
             |      +--------+---------+
             |               |
             |      +--------v---------+
             |      | chunk_passages   |  4-tier passage chunking
             |      +--------+---------+
             |               |
             |      +--------v---------+
             |      |      score       |  RRF + authority ranking
             |      +--------+---------+
             |               |
             |      +--------v---------+
             |      | citation_chain   |  BFS via Semantic Scholar
             |      +--------+---------+
             |               |
             |      +--------v---------+
             |      |  contradiction   |  Cross-source conflict detection
             |      +--------+---------+
             |               |
             |      +--------v---------+
             |      |   synthesize     |  Outline-first grounded synthesis
             |      +--------+---------+
             |               |
             |      +--------v---------+
             |      |    critique      |  3 parallel graders
             |      +--------+---------+
             |               |
             |      +--------v---------+
             |      | rollup_budget    |  Aggregate token usage
             |      +--------+---------+
             |               |
             |         converged?
             |          /        \
             |        NO          YES
             |        /              \
             +-------+      +---------v--------+
                             |     report       |  Markdown render
                             +---------+--------+
                                       |
                             +---------v--------+
                             | [report_gate]    |  HITL interrupt (--mode hitl only)
                             +---------+--------+
                                       |
                                      END
```

**Execution backbone**: LangGraph StateGraph with annotated reducers for concurrent state merging. Accumulating fields (sub-queries, search results) use `operator.add`. Overwrite fields (section drafts, citations) use replace-last-write semantics.

**Model tiering**: Opus handles planning, synthesis, and report generation. Sonnet handles the three critic graders (5x cost reduction for structured evaluation tasks).

## Pipeline Stages

### Planner

STORM-style decomposition: identifies 3 diverse perspectives on the question, then generates 1-2 specific search queries per perspective. On re-iterations, receives previous queries and identified gaps — generates new queries that address gaps without repeating prior searches. Programmatic deduplication via Jaccard similarity (threshold 0.7) + substring containment prevents redundant queries.

### Searcher

Dispatches sub-queries in parallel across all configured backends using LangGraph's `Send` API. Each backend returns ranked results with metadata (title, URL, snippet, authority classification).

### Extractor

Extracts clean content from result URLs using a three-tier cascade:

| Priority | Extractor | Use Case |
|----------|-----------|----------|
| 1 | Crawl4AI | JavaScript-rendered pages, SPAs |
| 2 | Trafilatura | Static HTML (fallback) |
| 3 | PyMuPDF4LLM | PDF documents |

Capped at 30 URLs per iteration to control costs.

### Scorer

Combines two scoring signals:

- **Reciprocal Rank Fusion (RRF)**: Merges rankings from multiple backends using `1 / (k + rank)` with k=60. Documents appearing in multiple result lists receive fused scores.
- **Source Authority**: Classifies URLs into tiers with score multipliers:

| Tier | Score | Examples |
|------|-------|----------|
| Institutional | 1.0 | `.edu`, `.gov`, Nature, arXiv, PubMed |
| Professional | 0.7 | BBC, Reuters, Stack Overflow, tech blogs |
| Community | 0.4 | Wikipedia, Reddit, forums |
| Promotional | 0.2 | Marketing sites, product pages |
| Unknown | 0.3 | Unclassified domains |

Combined score: `rrf * (1 - w) + authority * w` where `w` defaults to 0.2.

### Synthesizer

Outline-first grounded synthesis with inline `[N]` citations. Five-stage pipeline:

1. **Validate** — Deterministic outline validation (no LLM): checks source_ids, passage coverage, section count, key_claims
2. **Outline** — 1 LLM call to generate section outline with source assignments and key claims
3. **Draft** — N parallel LLM calls (asyncio.gather), each section sees ONLY its assigned passages
4. **Verify** — Mechanical grounding verification (LLM-free, Jaccard similarity, 0.8 pass threshold)
5. **Refine** — Failed sections get up to 2 LLM refinement attempts with ungrounded claims context
6. **Compose** — 1 LLM call for introduction, section transitions, and conclusion (section content immutable)

On iteration 2+, sections with HIGH grounding scores are preserved; MEDIUM/LOW sections are re-drafted with new sources.

### Critic

Three-grader chain evaluating distinct failure modes in parallel:

| Grader | Evaluates | 1.0 = | 0.0 = |
|--------|-----------|-------|-------|
| Relevance | Does content answer the question? | Directly answers with depth | Completely off-topic |
| Hallucination | Are claims grounded in cited sources? | Every claim cited and verified | Fabricated content |
| Quality | Is writing clear, deep, organized? | Excellent structure and depth | Incoherent or empty |

Per-section confidence = average of three grader scores, classified as HIGH/MEDIUM/LOW.

### Report Renderer

Generates structured Markdown with:
- YAML frontmatter (metadata, iteration count, token usage, cost)
- LLM-generated introduction and section transitions
- Themed sections with inline citations
- LLM-generated conclusion
- Confidence heat map table
- Confidence trends with sparklines across iterations
- Source diversity metrics (HHI, domain count, authority mix)
- Contradictions detected between sources
- Research gaps section
- Deduplicated, sequentially-numbered bibliography
- Evidence map (claim-to-source mapping table with authority and confidence)
- Source provenance table (content hashes, access timestamps, archive status)

## Configuration

All settings are controlled via environment variables. Copy `.env.example` to `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | *(required)* | Anthropic API key |
| `SEARXNG_URL` | `http://localhost:8080` | SearXNG instance URL |
| `EXA_API_KEY` | *(optional)* | Exa semantic search API key |
| `TAVILY_API_KEY` | *(optional)* | Tavily factual grounding API key |
| `OPUS_MODEL` | `claude-opus-4-6` | Model for planner, synthesizer |
| `SONNET_MODEL` | `claude-sonnet-4-6` | Model for critic graders |
| `MAX_ITERATIONS` | `3` | Maximum research iterations |
| `TOKEN_BUDGET` | `200000` | Maximum total tokens across all agents |
| `MAX_CONCURRENT_REQUESTS` | `5` | Concurrent API calls (semaphore) |
| `AUTHORITY_WEIGHT` | `0.2` | Weight of authority score in ranking |
| `RRF_K` | `60` | RRF smoothing parameter |
| `CONVERGENCE_THRESHOLD` | `0.05` | Minimum improvement to continue iterating |
| `SEARCH_CACHE_TTL` | `3600` | Search cache TTL in seconds |
| `SEARCH_CACHE_DIR` | `.cache/search` | Search cache directory |
| `CHECKPOINT_DB` | `checkpoints/research.db` | SQLite checkpoint database path |
| `CHECKPOINT_BACKEND` | `sqlite` | Checkpoint backend: `sqlite`, `postgres`, or `none` |
| `POSTGRES_DSN` | *(optional)* | PostgreSQL connection string (required when backend=postgres) |
| `MEMORY_DIR` | `memory/` | Path to memory storage directory |
| `RUN_LOG_DIR` | `runs/` | Path to run event log directory |
| `MODE` | `auto` | Execution mode: `auto` or `hitl` (human-in-the-loop gates) |
| `OPENALEX_EMAIL` | *(optional)* | Email for OpenAlex polite pool (enables openalex backend) |
| `OPENALEX_API_KEY` | *(optional)* | OpenAlex API key (email is sufficient for access) |
| `S2_API_KEY` | *(optional)* | Semantic Scholar API key (works unauthenticated) |
| `WAYBACK_ENABLED` | `true` | Enable Wayback Machine archive backend |
| `WAYBACK_TIMEOUT` | `15` | Wayback request timeout in seconds |

## Search Backends

The system supports multiple search backends via a Protocol-based registry. Backends are selected automatically based on which API keys are configured.

| Backend | Type | Requires | Notes |
|---------|------|----------|-------|
| **SearXNG** | Self-hosted meta-search | Docker + `SEARXNG_URL` | Aggregates Google, Bing, DuckDuckGo, Wikipedia, Google Scholar, arXiv |
| **Exa** | Semantic search API | `EXA_API_KEY` | Neural search with content retrieval |
| **Tavily** | Factual search API | `TAVILY_API_KEY` | Optimized for factual grounding |
| **OpenAlex** | Scholarly search API | `OPENALEX_EMAIL` | 250M+ works, abstract reconstruction from inverted index, polite pool with email |
| **Semantic Scholar** | Scholarly search API | *(optional)* `S2_API_KEY` | Works unauthenticated; also used for citation chaining (BFS graph traversal) |
| **Wayback Machine** | Web archive | `WAYBACK_ENABLED=true` | CDX API for historical snapshots; fallback when live extractors fail |

### SearXNG Setup

The included Docker configuration runs a SearXNG instance with JSON output enabled and rate limiting disabled:

```bash
docker compose -f docker/docker-compose.yml up -d

# Verify it's running
curl "http://localhost:8080/search?q=test&format=json" | head -c 200
```

Pre-configured search engines: Google, Bing, DuckDuckGo, Wikipedia, Google Scholar, arXiv.

## Convergence & Scoring

The system iterates until one of these conditions is met:

| Condition | Trigger | Effect |
|-----------|---------|--------|
| **All acceptable** | Every section HIGH confidence (>= 0.8) | Converge |
| **Diminishing returns** | Avg confidence improved < 0.05 from previous iteration | Converge |
| **Max iterations** | Reached `MAX_ITERATIONS` (default 3) | Force converge |
| **Budget exhausted** | Token usage > 90% of `TOKEN_BUDGET` | Force converge |

**Replan triggers** (if not force-converged):
- Any section with LOW confidence (< 0.6)
- More than 30% of sections at MEDIUM confidence (0.6-0.8)

On replan, the system generates new sub-queries targeting identified gaps and weak sections, then runs through the full pipeline again with context from previous iterations.

## Output Format

Reports are saved as Markdown with YAML frontmatter:

```markdown
---
title: "Research Report: What is quantum entanglement?"
generated: 2026-02-20T15:30:00+00:00
iterations: 2
total_sections: 5
total_citations: 8
total_tokens: 12500
total_cost_usd: 0.3200
convergence_reason: all_acceptable
---

# What is quantum entanglement?

## Section Title

Synthesized content with inline [1] citations referencing sources [2].

## Confidence Assessment

| Section | Relevance | Hallucination | Quality | Avg | Level |
|---------|-----------|---------------|---------|-----|-------|
| Section Title | 0.90 | 0.85 | 0.88 | 0.88 | HIGH |

## Research Gaps

- **No sources found**: Description of what's missing

## Bibliography

1. [Source Title](https://example.com) - *institutional*
2. [Another Source](https://example.org) - *professional*
```

## CLI Reference

```
usage: deep-research-swarm [-h] [--max-iterations N] [--token-budget N]
                           [--output PATH] [--backends BACKEND [BACKEND ...]]
                           [--no-cache] [--no-stream] [--verbose]
                           [--resume THREAD_ID] [--list-threads]
                           [--dump-state THREAD_ID] [--no-memory]
                           [--list-memories] [--export-mcp]
                           [--academic] [--no-archive]
                           [--no-log] [--mode {auto,hitl}]
                           [question]

positional arguments:
  question              The research question to investigate

options:
  -h, --help            Show help message
  --max-iterations N    Maximum research iterations (default: from config)
  --token-budget N      Maximum token budget (default: from config)
  --output PATH         Output file path (default: output/<timestamp>.md)
  --backends B [B ...]  Search backends to use (default: all available)
  --no-cache            Disable search result caching
  --no-stream           Use blocking ainvoke instead of astream
  --verbose             Detailed streaming progress
  --resume THREAD_ID    Resume a previous research run from checkpoint
  --list-threads        List recent research threads from checkpoint DB
  --dump-state THREAD_ID  Export checkpoint state as JSON
  --no-memory           Disable memory store/retrieval for this run
  --list-memories       Print stored memory records and exit
  --export-mcp          Export memory in MCP entity format
  --academic            Add openalex and semantic_scholar to backends
  --no-archive          Disable Wayback Machine archive fallback
  --no-log              Disable run event logging
  --mode {auto,hitl}    Execution mode (auto=default, hitl=human-in-the-loop gates)
```

### Examples

```bash
# Basic research
python -m deep_research_swarm "What are the latest advances in quantum computing?"

# Limit to 1 iteration (faster, cheaper)
python -m deep_research_swarm --max-iterations 1 "Compare React vs Vue in 2026"

# Use specific backends
python -m deep_research_swarm --backends searxng exa "Effects of sleep on memory"

# Custom output path
python -m deep_research_swarm --output reports/ai-safety.md "Current state of AI safety research"

# Verbose streaming with progress details
python -m deep_research_swarm --verbose "Impact of remote work on productivity"

# Resume a crashed or interrupted run
python -m deep_research_swarm --resume research-20260220-131354-3767

# List previous research threads
python -m deep_research_swarm --list-threads

# Export checkpoint state for debugging
python -m deep_research_swarm --dump-state research-20260220-131354-3767

# Run without cross-session memory
python -m deep_research_swarm --no-memory "Ephemeral research question"

# View stored research memories
python -m deep_research_swarm --list-memories

# Human-in-the-loop mode (pause at plan and report for review)
python -m deep_research_swarm --mode hitl "Risks of artificial general intelligence"

# Resume through an HITL gate after review
python -m deep_research_swarm --resume research-20260221-212856-cd2e

# Scholarly research with academic backends (OpenAlex + Semantic Scholar)
python -m deep_research_swarm --academic "CRISPR gene editing for metabolic disorders"

# Scholarly research without archive fallback
python -m deep_research_swarm --academic --no-archive "Latest advances in quantum computing"

# Run without event logging
python -m deep_research_swarm --no-log "Quick ephemeral question"
```

## Project Structure

```
deep-research-swarm/
├── deep_research_swarm/
│   ├── __main__.py              # CLI entry point + checkpoint/memory lifecycle
│   ├── contracts.py             # All types, enums, protocols (SSOT)
│   ├── config.py                # Settings from environment variables
│   │
│   ├── graph/
│   │   ├── state.py             # ResearchState + annotated reducers
│   │   └── builder.py           # StateGraph construction + edge wiring
│   │
│   ├── agents/
│   │   ├── base.py              # AgentCaller (Anthropic SDK wrapper + retry)
│   │   ├── planner.py           # STORM decomposition + query dedup + routing
│   │   ├── searcher.py          # Parallel search dispatch
│   │   ├── extractor.py         # Content extraction coordinator
│   │   ├── synthesizer.py       # Outline-first grounded synthesis (5-stage)
│   │   ├── citation_chain.py    # BFS citation graph traversal via S2
│   │   ├── contradiction.py     # Sonnet-powered contradiction detection
│   │   └── critic.py            # Three-grader chain + convergence
│   │
│   ├── backends/
│   │   ├── protocol.py          # SearchBackend Protocol (PEP 544)
│   │   ├── searxng.py           # SearXNG implementation
│   │   ├── exa.py               # Exa semantic search
│   │   ├── tavily.py            # Tavily factual search
│   │   ├── openalex.py          # OpenAlex scholarly search
│   │   ├── semantic_scholar.py  # Semantic Scholar search + citation API
│   │   ├── crossref.py          # Crossref/Unpaywall DOI utilities
│   │   ├── wayback.py           # Wayback Machine archive backend
│   │   └── cache.py             # File-based search cache (SHA-256 keys, TTL)
│   │
│   ├── extractors/
│   │   ├── chunker.py           # 4-tier passage chunking
│   │   ├── crawl4ai_extractor.py
│   │   ├── trafilatura_extractor.py
│   │   └── pdf_extractor.py     # HTTP download + pymupdf4llm
│   │
│   ├── scoring/
│   │   ├── rrf.py               # Reciprocal Rank Fusion
│   │   ├── authority.py         # Source authority classification + scholarly signals
│   │   ├── confidence.py        # Confidence + replan logic
│   │   ├── diversity.py         # HHI-based source diversity scoring
│   │   ├── grounding.py         # Mechanical grounding verification (LLM-free)
│   │   ├── routing.py           # Query classification + backend routing
│   │   └── provenance.py        # Content hash provenance tracking
│   │
│   ├── reporting/
│   │   ├── renderer.py          # Markdown report generation + composition
│   │   ├── citations.py         # Dedup, renumber, bibliography
│   │   ├── heatmap.py           # Confidence heat map table
│   │   ├── trends.py            # Confidence sparklines across iterations
│   │   ├── evidence_map.py      # Claim-to-source mapping table
│   │   └── provenance.py        # Provenance section rendering
│   │
│   ├── memory/
│   │   ├── store.py             # JSON-backed cross-session memory store
│   │   ├── extract.py           # Deterministic memory record extraction
│   │   └── mcp_export.py        # MCP entity format export
│   │
│   ├── event_log/
│   │   └── writer.py            # JSONL run event logging
│   │
│   ├── utils/
│   │   └── text.py              # Jaccard score, dedup utilities
│   │
│   └── streaming.py             # StreamDisplay for astream progress
│
├── tests/                       # 480 tests across 30+ modules
├── docker/                      # SearXNG Docker configuration
├── output/                      # Generated reports (gitignored)
├── checkpoints/                 # SQLite checkpoint DB (gitignored)
├── memory/                      # Research memory store (gitignored)
├── runs/                        # Run event logs (gitignored)
└── .github/workflows/ci.yml    # Lint + test matrix (3.11, 3.12)
```

**Dependency flow**: `contracts.py` is the leaf — everything imports from it. `config.py` is the other leaf — agents and graph import settings. No circular dependencies.

## Testing

```bash
# Unit tests (no network, no API keys, no Docker)
pytest tests/ -v

# With coverage
pytest tests/ --cov=deep_research_swarm --cov-report=term-missing

# Specific test module
pytest tests/test_rrf.py -v

# Full e2e (requires API keys + Docker)
python -m deep_research_swarm "What is quantum entanglement?"
```

**480 tests** covering:

| Module | Tests | Coverage |
|--------|-------|----------|
| Authority scoring | 16 | URL classification, score ranges, scholarly metadata |
| RRF algorithm | 8 | Fusion, k parameter, empty input, scored documents |
| Confidence | 12 | Classification, aggregation, replan triggers |
| Contracts | 18 | TypedDict construction, enum values, Protocol conformance, V7 types |
| Budget tracker | 7 | Recording, limits, agent breakdown |
| Backends | 4 | Registry, Protocol conformance |
| Graph | 4 | Compilation, routing, checkpointer wiring |
| Reporting | 14 | Frontmatter, citations, dedup/renumber, heat map |
| Query dedup | 12 | Jaccard similarity, substring, thresholds |
| Contradiction | 4 | LLM parse, index mapping, out-of-range handling |
| Diversity | 6 | HHI math, domain normalization, authority counts |
| Search cache | 7 | Round-trip, TTL expiry, key determinism, corruption |
| Streaming | 7 | Node labels, iteration detection, custom events |
| PDF extraction | 9 | URL detection, cascade routing, local/remote extraction |
| Trends | 5 | Sparkline rendering, new/dropped sections |
| Resume/checkpoint | 15 | Arg parsing, resume validation, V7 CLI flags |
| Checkpoint robustness | 8 | WAL mode, busy_timeout, large state writes, concurrent contention, no-checkpointer fallback |
| Dump state | 5 | CLI flags, memory context formatting |
| Memory store | 11 | Add/list, search, persistence, graceful degradation |
| Memory extract | 3 | Full state, empty state, missing fields |
| Text utils | 9 | Jaccard score, is_duplicate, custom thresholds |
| Event log | 11 | Write/read roundtrip, append, corruption, make_event |
| Evidence map | 13 | Claim extraction, citation mapping, rendering, escaping |
| HITL gates | 10 | Auto/hitl mode, gate wiring, config validation |
| MCP export | 7 | Entity structure, observations, metadata, empty input |
| Config | 17 | V7 defaults, available_backends, warnings, registry |
| OpenAlex | 11 | Protocol conformance, search, error handling, abstract reconstruction |
| Semantic Scholar | 13 | Protocol conformance, search, paper details, retry |
| Crossref | 10 | DOI resolution, open access, content negotiation |
| Wayback | 11 | Protocol conformance, CDX captures, archive fetch, health |
| Chunker | 12 | 4-tier chunking, passage IDs, edge cases |
| Grounding | 12 | Passage assignment, grounding verification, section scores |
| Provenance | 7 | Provenance records, content hashing, rendering |
| Routing | 28 | Query classification, backend routing, planner integration |
| Citation chain | 20 | Relevance scoring, BFS traversal, budget, dedup |
| Synthesizer | 25 | Outline parsing, validation, citation map, renumbering, full pipeline |
| Integration | 1 | Mocked LLM planner e2e |

All tests run without network access, API keys, or Docker — LLM calls are mocked.

## Development

### Prerequisites

```bash
pip install -e ".[dev]"
```

### Lint & Format

```bash
ruff check .          # Lint
ruff format --check . # Format check
ruff format .         # Auto-format
```

### CI Pipeline

GitHub Actions runs on every PR:
- **Lint**: `ruff check` + `ruff format --check`
- **Test**: `pytest` on Python 3.11 and 3.12

Branch protection requires all checks to pass before merge. Squash-only merge policy.

### Adding a Search Backend

1. Create `backends/mybackend.py` implementing the `SearchBackend` Protocol:

```python
from deep_research_swarm.contracts import SearchBackend, SearchResult

class MyBackend:
    name = "mybackend"

    async def search(self, query, *, num_results=10, category=None) -> list[SearchResult]:
        ...

    async def health_check(self) -> bool:
        ...
```

2. Register in `backends/__init__.py`
3. Add API key to `config.py` and `.env.example`

No inheritance required — the Protocol uses structural subtyping (PEP 544).

## Roadmap

- [x] **V1** — Core loop: single backend, single iteration, basic synthesis
- [x] **V2** — Multi-iteration quality: three-grader chain, context-aware synthesis, query dedup, citation renumbering, sonnet cost optimization
- [x] **V3** — Reports & observability: streaming output, confidence trends, search caching, contradiction detection, source diversity scoring
- [x] **V4** — Persistence & resume: AsyncSqliteSaver checkpointing, `--resume` CLI flag, PDF extraction cascade, PostgresSaver option
- [x] **V5** — Memory & state: cross-session research memory (JSON-backed Jaccard search), `--dump-state`, `--list-memories`, `--export-mcp`, extracted text utilities, memory lifecycle in CLI
- [x] **V6** — Forensic mode: human-in-the-loop gates (`--mode hitl` with LangGraph `interrupt()`), run event log (JSONL per run), evidence map appendix (claim-to-source mapping), robust JSON extraction for agent responses
- [x] **V7** — Niche retrieval: scholarly backends (OpenAlex, Semantic Scholar), archive fallback (Wayback Machine), citation chaining (BFS via S2), passage chunking, mechanical grounding verification, backend routing, provenance tracking, outline-first synthesis with parallel drafting
- [x] **V8** — Adaptive intelligence: deterministic overseer (TunableRegistry + complexity multiplier), second-pass grounding (semantic neighborhood), embedding grounding (fastembed), claim graph, OCR/GROBID extraction cascade, incremental research, PROV-O JSON-LD export, adaptive report section
- [ ] **V9** — Planned: internal hybrid index (OpenSearch k-NN), focused crawling, embedding-based routing, MCP Memory server (live), memory pruning

## License

[MIT](LICENSE)

**Dependency license note:** The PDF extractor (`extractors/pdf_extractor.py`) uses PyMuPDF4LLM, which is AGPL-3.0 licensed. CLI distribution under MIT is unaffected. If you deploy deep-research-swarm as a network service or API, AGPL terms apply to the PDF extraction component. Apache-2.0 alternatives (GROBID, Unstructured, PaddleOCR) are available and documented in `CONTRIBUTING.md`.
