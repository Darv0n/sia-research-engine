# Deep Research Swarm

Multi-agent deep research system built on [LangGraph](https://github.com/langchain-ai/langgraph). Takes a research question, decomposes it via STORM-style perspective-guided questioning, dispatches parallel search agents across multiple backends, synthesizes findings via RAG-Fusion with Reciprocal Rank Fusion, critiques through a three-grader chain, and iterates until convergence. Produces structured Markdown reports with passage-level citations, confidence heat maps, and gap analysis.

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
```

The report prints to stdout and saves to `output/<timestamp>-<question>.md`.

## How It Works

Given a research question, the system:

1. **Decomposes** the question into 3+ sub-queries from diverse perspectives (STORM method)
2. **Searches** each sub-query in parallel across configured backends
3. **Extracts** clean content from result URLs via a cascade of extractors
4. **Scores** documents using Reciprocal Rank Fusion + source authority classification
5. **Synthesizes** a structured report with inline `[N]` citations
6. **Critiques** each section across three dimensions: relevance, hallucination, quality
7. **Iterates** if quality is insufficient — re-plans to address gaps and weak sections
8. **Renders** a final Markdown report with bibliography, confidence heat map, and gap analysis

On subsequent iterations, the synthesizer refines rather than regenerates — preserving strong sections while revising weak ones using newly gathered sources.

## Architecture

```
                    +------------------+
                    |   health_check   |  Verify backends
                    +--------+---------+
                             |
                    +--------v---------+
             +----->|      plan        |  STORM decomposition
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
             |      |      score       |  RRF + authority ranking
             |      +--------+---------+
             |               |
             |      +--------v---------+
             |      |  contradiction   |  Cross-source conflict detection
             |      +--------+---------+
             |               |
             |      +--------v---------+
             |      |   synthesize     |  RAG-Fusion + citations
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

RAG-Fusion synthesis with inline `[N]` citations referencing source documents. Two operational modes:

- **Initial** (iteration 1): Fresh synthesis from scored sources — identifies themes, organizes into sections, assesses confidence.
- **Refine** (iteration 2+): Context-aware refinement — keeps HIGH confidence sections intact, revises MEDIUM/LOW sections with new sources, adds new sections for uncovered topics, addresses identified research gaps.

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
- Themed sections with inline citations
- Confidence heat map table
- Confidence trends with sparklines across iterations
- Source diversity metrics (HHI, domain count, authority mix)
- Contradictions detected between sources
- Research gaps section
- Deduplicated, sequentially-numbered bibliography
- Evidence map (claim-to-source mapping table with authority and confidence)

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

## Search Backends

The system supports multiple search backends via a Protocol-based registry. Backends are selected automatically based on which API keys are configured.

| Backend | Type | Requires | Notes |
|---------|------|----------|-------|
| **SearXNG** | Self-hosted meta-search | Docker + `SEARXNG_URL` | Aggregates Google, Bing, DuckDuckGo, Wikipedia, Google Scholar, arXiv |
| **Exa** | Semantic search API | `EXA_API_KEY` | Neural search with content retrieval |
| **Tavily** | Factual search API | `TAVILY_API_KEY` | Optimized for factual grounding |

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
│   │   ├── planner.py           # STORM decomposition + query dedup
│   │   ├── searcher.py          # Parallel search dispatch
│   │   ├── extractor.py         # Content extraction coordinator
│   │   ├── synthesizer.py       # RAG-Fusion synthesis
│   │   ├── contradiction.py     # Sonnet-powered contradiction detection
│   │   └── critic.py            # Three-grader chain + convergence
│   │
│   ├── backends/
│   │   ├── protocol.py          # SearchBackend Protocol (PEP 544)
│   │   ├── searxng.py           # SearXNG implementation
│   │   ├── exa.py               # Exa semantic search
│   │   ├── tavily.py            # Tavily factual search
│   │   └── cache.py             # File-based search cache (SHA-256 keys, TTL)
│   │
│   ├── extractors/
│   │   ├── crawl4ai_extractor.py
│   │   ├── trafilatura_extractor.py
│   │   └── pdf_extractor.py     # HTTP download + pymupdf4llm
│   │
│   ├── scoring/
│   │   ├── rrf.py               # Reciprocal Rank Fusion
│   │   ├── authority.py         # Source authority classification
│   │   ├── confidence.py        # Confidence + replan logic
│   │   └── diversity.py         # HHI-based source diversity scoring
│   │
│   ├── reporting/
│   │   ├── renderer.py          # Markdown report generation
│   │   ├── citations.py         # Dedup, renumber, bibliography
│   │   ├── heatmap.py           # Confidence heat map table
│   │   ├── trends.py            # Confidence sparklines across iterations
│   │   └── evidence_map.py      # Claim-to-source mapping table
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
├── tests/                       # 230 tests across 20 modules
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

**230 tests** covering:

| Module | Tests | Coverage |
|--------|-------|----------|
| Authority scoring | 16 | URL classification, score ranges, edge cases |
| RRF algorithm | 8 | Fusion, k parameter, empty input, scored documents |
| Confidence | 12 | Classification, aggregation, replan triggers |
| Contracts | 10 | TypedDict construction, enum values, Protocol conformance |
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
| Resume/checkpoint | 12 | Arg parsing, resume validation, thread ID format |
| Dump state | 5 | CLI flags, memory context formatting |
| Memory store | 11 | Add/list, search, persistence, graceful degradation |
| Memory extract | 3 | Full state, empty state, missing fields |
| Text utils | 9 | Jaccard score, is_duplicate, custom thresholds |
| Event log | 11 | Write/read roundtrip, append, corruption, make_event |
| Evidence map | 13 | Claim extraction, citation mapping, rendering, escaping |
| HITL gates | 10 | Auto/hitl mode, gate wiring, config validation |
| MCP export | 7 | Entity structure, observations, metadata, empty input |
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
- [ ] **V7** — Planned: MCP Memory server (live, not just export), memory pruning (auto-cleanup old/low-value records), embedding-based retrieval (replace Jaccard with vector similarity)

## License

[MIT](LICENSE)
