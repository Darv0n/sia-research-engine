<div align="center">

# SIA Research Engine

**Question in. Deliberated, cross-validated report out.**

*Multi-agent deep research with [SIA](docs/SIA.md) cognitive deliberation, thermodynamic entropy control, and swarm intelligence — powered by [LangGraph](https://github.com/langchain-ai/langgraph) + Claude*

<p>
  <img src="https://img.shields.io/badge/python-3.11%2B-blue?logo=python&logoColor=white" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License: MIT">
  <img src="https://img.shields.io/badge/tests-1166%20passing-brightgreen?logo=pytest&logoColor=white" alt="Tests: 1166 passing">
  <img src="https://img.shields.io/badge/version-0.10.0-blue" alt="Version: 0.10.0">
  <img src="https://img.shields.io/badge/agents-7%20cognitive%20lenses-purple" alt="Agents: 7">
  <img src="https://img.shields.io/badge/backends-6%20search%20engines-teal" alt="Backends: 6">
</p>

<p>
  <a href="#quick-start">Quick Start</a>
  <span>&nbsp;&nbsp;&bull;&nbsp;&nbsp;</span>
  <a href="#architecture">Architecture</a>
  <span>&nbsp;&nbsp;&bull;&nbsp;&nbsp;</span>
  <a href="#sia-deliberation">SIA Deliberation</a>
  <span>&nbsp;&nbsp;&bull;&nbsp;&nbsp;</span>
  <a href="#swarm-mode">Swarm Mode</a>
  <span>&nbsp;&nbsp;&bull;&nbsp;&nbsp;</span>
  <a href="CHANGELOG.md">Changelog</a>
</p>

</div>

---

Evolved from [deep-research-swarm](https://github.com/Darv0n/deep-research-swarm) (V1-V9, archived).

Takes a research question, decomposes it into targeted queries, dispatches parallel searches across 6 backends, builds a structured **KnowledgeArtifact** through deliberation panels with cross-referenced claim verification, runs **multi-turn agent deliberation** where cognitive lenses challenge and constrain each other, synthesizes through reactor-informed 2-stage composition, evaluates through **adversarial critique** with singularity prevention, and iterates under **thermodynamic entropy control** until five-way convergence. Optionally runs **N parallel reactors** with cross-validation for high-stakes queries.

## Quick Start

```bash
# Clone and install
git clone https://github.com/Darv0n/sia-research-engine.git
cd sia-research-engine
python -m venv .venv && source .venv/bin/activate  # .venv\Scripts\activate on Windows
pip install -e ".[dev]"

# Configure
cp .env.example .env       # Edit .env — set ANTHROPIC_API_KEY (required)

# Start SearXNG (local meta-search engine)
docker compose -f docker/docker-compose.yml up -d

# Run (single reactor, SIA deliberation)
python -m deep_research_swarm "What is quantum entanglement?"

# Run with swarm mode (3 parallel reactors with cross-validation)
python -m deep_research_swarm --swarm 3 "CRISPR gene editing for metabolic disorders"

# Run with V9 behavior (no SIA deliberation)
python -m deep_research_swarm --no-sia "Simple factual question"
```

> [!TIP]
> Add `--academic` for scholarly research with OpenAlex + Semantic Scholar backends and BFS citation chaining:
> ```bash
> python -m deep_research_swarm --academic --swarm 3 "Effects of gut microbiome on neurological disorders"
> ```

## Architecture

```
health_check -> clarify -> plan -> [plan_gate?] ->
  WAVE LOOP:
    search_wave -> extract_wave -> score_wave ->
    DELIBERATION PANEL (parallel):
      [authority | grounding | contradiction | coverage]
    -> merge_judgments -> [coverage check] -> loop or continue
  compress -> REACTOR (multi-turn SIA deliberation) ->
  synthesize (2-stage) -> critique (adversarial) -> compute_entropy ->
  rollup_budget -> [five-way converge?] -> report
```

### Model Tiering

| Tier | Model | Used For | Cost Impact |
|------|-------|----------|-------------|
| **Opus** | claude-opus-4-6 | Plan, synthesis outline | High quality where it matters |
| **Sonnet** | claude-sonnet-4-6 | Reactor turns, adversarial critique, section drafts | Bulk intelligence work |
| **Haiku** | claude-haiku-4-5-20251001 | Compression summaries, composition | Cheap parallel calls |

### Five-Way Convergence

The system converges when ALL conditions are met:

```
converged = confidence_ok
            AND entropy_gate_ok
            AND NOT false_convergence
            AND NOT dominance_detected
            AND singularity_safe
```

| Check | What It Catches |
|-------|----------------|
| **Confidence** | Sections below quality threshold |
| **Entropy gate** | Synthesis attempted while entropy is in runaway band |
| **False convergence** | Low entropy with unresolved contradictions |
| **Dominance** | All sections scoring identically (grader collapse) |
| **Singularity** | Structural collapse: all constraints from one agent, no reframes accepted, hidden coalitions |

## SIA Deliberation

SIA (Semantic Intelligence Architecture) replaces flat parallel evaluation with true multi-agent deliberation. Agents have persistent cognitive stances, see each other's output, and build on or challenge prior reasoning.

### Cognitive Agents

| Agent | Archetype | Processing Bias |
|-------|-----------|----------------|
| **Lawliet** | Constraint Architect | Extract what MUST be true vs merely asserted. Check grounding. |
| **Light** | Strategic Director | Directional commitment. If replanning: which queries fill gaps? |
| **Rick** | Frame Disruptor | What hidden assumptions drive the synthesis? What if the opposite were true? |
| **Makishima** | Legitimacy Auditor | Test value assumptions treated as facts. Find missing perspectives. |
| **Shikamaru** | Efficiency Stabilizer | Can we ship this? Identify redundant arguments, prune branches. |
| **Shiro** | Rule Builder | *(Reserve)* Activated when structural formalism is low. |
| **Johan** | Alignment Auditor | *(Reserve)* Activated when coalition cohesion is unexamined. |

### Entropy Bands

Thermodynamic entropy governs pipeline behavior. Zero LLM calls — purely deterministic computation from research state observables.

| Band | Range | Pipeline Behavior |
|------|-------|-------------------|
| **Crystalline** | e <= 0.20 | Harvest. Synthesis allowed. |
| **Convergence** | 0.20-0.45 | Build phase. Deeper drafts. |
| **Turbulence** | 0.45-0.70 | Anchor after rupture. Synthesis conditional. |
| **Runaway** | e > 0.70 | Compression. Synthesis blocked. Reduce queries. |

Entropy is computed from four signals: **ambiguity** (unresolved gaps), **conflict** (contradiction density), **novelty** (new query ratio), **trust** (grounding variance).

### Reactor

The SIA Kernel orchestrates multi-turn deliberation over the KnowledgeArtifact:

1. **Ignition** — Kernel selects first speaker based on entropy band (never starts with a cooling agent)
2. **Turns** — Each agent sees the full conversation thread + Kernel framing. 6 turns default (tunable 3-10).
3. **Anti-dominance** — Max 2 consecutive turns per agent. Fatigue tracking prevents monopolization.
4. **Harvest** — Deduplicated constraints, rejected branches, active frames, and coalition map feed synthesis.

### Adversarial Critique

Replaces V9's 3 parallel graders with a multi-turn conversation:

```
Makishima (legitimacy stress-test)
  -> Lawliet (constraint extraction, grounding validation)
  -> Rick (frame audit — conditional on entropy > 0.40)
  -> Shikamaru (synthesis readiness, pruning)
  -> Light (replan direction — conditional on replan recommendation)
```

Score adjustments per finding severity: critical -0.15, significant -0.08, minor -0.03.

## Swarm Mode

Run N parallel full-pipeline invocations with perturbed initialization for high-stakes queries.

```bash
python -m deep_research_swarm --swarm 3 "Your important question"
```

### How It Works

Each reactor gets a different perturbation strategy:

| Strategy | Effect |
|----------|--------|
| **Baseline** | Control reactor — no perturbation |
| **Entropy High** | Starts in turbulence band — broader exploration |
| **Entropy Low** | Starts in crystalline band — faster convergence |
| **Perspective Shuffle** | Reversed perspective ordering — different exploration priority |
| **Depth Focus** | Fewer queries, deeper extraction per source |

### Winner Selection

Structural scoring across 6 dimensions with weighted aggregation:

| Dimension | Weight | Measures |
|-----------|:------:|----------|
| Entropy stability | 0.20 | Lower final entropy = more convergent |
| Constraint density | 0.15 | More constraints = better-explored |
| Grounding quality | 0.30 | Average section confidence |
| Comprehensiveness | 0.10 | Section coverage |
| Convergence | 0.10 | Natural vs forced convergence |
| Cross-validation | 0.15 | Heading overlap between reactors |

### Cross-Reactor Coupling

Reactors communicate via three channels:

- **Artifact injection** — Share KnowledgeArtifact coverage maps between reactors
- **Entropy broadcast** — Share entropy state for cross-calibration
- **Validation shock** — Inject critical/significant findings from one reactor into others

## Features

- **SIA multi-agent deliberation** — 7 cognitive lenses with persistent stances, thermodynamic entropy control, multi-turn reactor
- **Multi-reactor swarm** — N parallel pipelines with perturbed initialization, structural winner selection, cross-validation
- **Adversarial critique** — Multi-turn evaluation with singularity prevention (4 collapse checks + 7-axis stability)
- **Knowledge compression** — Passage clustering, KnowledgeArtifact builder, 3-tier claim verification (NLI > embedding > Jaccard)
- **Deliberation panels** — 4 parallel judges (authority, grounding, contradiction, coverage) with judgment reconciliation
- **Wave-based search** — Coverage-gated convergence with diminishing returns detection
- **Pre-research clarification** — Heuristic scope analysis (auto) or LLM-powered Q&A (HITL)
- **High query volume** — 10-15 queries per iteration via 5 perspectives x 2-3 queries each
- **Reactive search** — Within-iteration gap analysis with automatic follow-up queries
- **Multi-backend search** — SearXNG, Exa, Tavily, OpenAlex, Semantic Scholar, Wayback Machine
- **Scholarly research** — OpenAlex (250M+ works), Semantic Scholar citation chaining (BFS)
- **Adaptive overseer** — 20 tunables scaled mid-run based on pipeline metrics and entropy band
- **Grounded synthesis** — Reactor-informed 2-stage synthesis: Opus outline from KnowledgeArtifact, Sonnet section drafts from clusters
- **Deep reports** — 400-800 words/section. Export as Markdown, DOCX, or PDF
- **Five-way convergence** — Confidence + entropy gate + false convergence + dominance + singularity
- **Human-in-the-loop** — Plan and report gates via LangGraph `interrupt()` (`--mode hitl`)
- **Checkpoint resume** — SQLite or PostgreSQL, `--resume` for interrupted runs
- **Cross-session memory** — JSON-backed search, MCP export
- **Provenance tracking** — Content hashes, PROV-O JSON-LD export (W3C)
- **OCR/GROBID** — Scanned PDF support via GROBID TEI + PaddleOCR fallback
- **Graceful degradation** — SIA failure -> compression + 2-stage -> V9 behavior. Always produces output.

<details>
<summary><b>Search Backends</b></summary>

| Backend | Type | Requires | Notes |
|---------|------|----------|-------|
| **SearXNG** | Self-hosted meta-search | Docker + `SEARXNG_URL` | Aggregates Google, Bing, DuckDuckGo, Wikipedia, Scholar, arXiv |
| **Exa** | Semantic search API | `EXA_API_KEY` | Neural search with content retrieval |
| **Tavily** | Factual search API | `TAVILY_API_KEY` | Optimized for factual grounding |
| **OpenAlex** | Scholarly API | `OPENALEX_EMAIL` | 250M+ works, abstract reconstruction, polite pool |
| **Semantic Scholar** | Scholarly API | *(optional)* `S2_API_KEY` | Also used for BFS citation chaining |
| **Wayback Machine** | Web archive | `WAYBACK_ENABLED=true` | CDX API for historical snapshots, 404 fallback |

</details>

<details>
<summary><b>Configuration</b></summary>

All settings via environment variables. Copy `.env.example` to `.env`:

| Variable | Default | Required | Description |
|:---------|:--------|:--------:|:------------|
| **`ANTHROPIC_API_KEY`** | — | Yes | Anthropic API key |
| `SEARXNG_URL` | `http://localhost:8080` | | SearXNG instance URL |
| `EXA_API_KEY` | — | | Exa semantic search API key |
| `TAVILY_API_KEY` | — | | Tavily factual grounding API key |
| `OPUS_MODEL` | `claude-opus-4-6` | | Model for plan, synthesis outline |
| `SONNET_MODEL` | `claude-sonnet-4-6` | | Model for reactor, critique, drafts |
| `HAIKU_MODEL` | `claude-haiku-4-5-20251001` | | Model for compression, composition |
| `MAX_ITERATIONS` | `3` | | Maximum research iterations |
| `TOKEN_BUDGET` | `200000` | | Maximum total tokens |
| `MAX_CONCURRENT_REQUESTS` | `5` | | Concurrent API calls |
| `CONVERGENCE_THRESHOLD` | `0.05` | | Minimum improvement to continue |
| `SIA_ENABLED` | `true` | | Enable SIA multi-agent deliberation |
| `SWARM_ENABLED` | `true` | | Enable swarm mode capability |
| `SWARM_MAX_REACTORS` | `5` | | Maximum reactors in swarm mode (2-10) |
| `CHECKPOINT_BACKEND` | `sqlite` | | `sqlite`, `postgres`, or `none` |
| `MODE` | `auto` | | `auto` or `hitl` (human-in-the-loop) |
| `ADAPTIVE_MODE` | `true` | | Enable adaptive overseer |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | | Embedding model for grounding |

See [CLAUDE.md](CLAUDE.md) for the full configuration reference.

</details>

<details>
<summary><b>CLI Reference</b></summary>

```
usage: deep-research-swarm [-h] [--max-iterations N] [--token-budget N]
                           [--output PATH] [--backends BACKEND [BACKEND ...]]
                           [--no-cache] [--no-stream] [--verbose]
                           [--resume THREAD_ID] [--list-threads]
                           [--dump-state THREAD_ID] [--no-memory]
                           [--list-memories] [--export-mcp]
                           [--academic] [--no-archive] [--no-log]
                           [--mode {auto,hitl}]
                           [--no-adaptive] [--complexity]
                           [--export-prov-o PATH] [--embedding-model MODEL]
                           [--grobid-url URL]
                           [--format {md,docx,pdf}]
                           [--follow-up THREAD_ID QUESTION]
                           [--swarm N] [--no-sia]
                           [question]
```

```bash
# Single reactor with SIA deliberation (default)
python -m deep_research_swarm "Effects of gut microbiome on cognition"

# Swarm mode — 3 parallel reactors with cross-validation
python -m deep_research_swarm --swarm 3 "CRISPR delivery mechanisms for CNS disorders"

# Academic backends + swarm
python -m deep_research_swarm --academic --swarm 3 "Topological quantum computing approaches"

# Disable SIA — use V9 three-grader critique
python -m deep_research_swarm --no-sia "Simple factual question"

# Export as PDF
python -m deep_research_swarm --format pdf --swarm 2 "Fusion energy timeline"

# Human-in-the-loop with verbose streaming
python -m deep_research_swarm --mode hitl --verbose "AI alignment approaches"

# Resume interrupted run
python -m deep_research_swarm --resume research-20260224-120000-1234
```

</details>

<details>
<summary><b>Project Structure</b></summary>

```
sia-research-engine/
├── deep_research_swarm/
│   ├── __main__.py              # CLI entry point + lifecycle
│   ├── contracts.py             # All types (40+ TypedDicts, SSOT)
│   ├── config.py                # Settings from environment variables
│   │
│   ├── graph/
│   │   ├── state.py             # ResearchState + annotated reducers
│   │   └── builder.py           # StateGraph construction + edge wiring
│   │
│   ├── sia/                     # V10: Semantic Intelligence Architecture
│   │   ├── agents.py            # 7 cognitive lenses (frozen dataclasses)
│   │   ├── covenants.py         # Pairwise field physics
│   │   ├── entropy.py           # Thermodynamic convergence (4-signal)
│   │   ├── entropy_steering.py  # Tunable adjustment by entropy band
│   │   ├── kernel.py            # SIAKernel (speaker selection, framing)
│   │   ├── adversarial_critique.py  # Multi-turn evaluation
│   │   ├── singularity_prevention.py  # Pre-synthesis safety gate
│   │   ├── swarm.py             # SwarmOrchestrator (N parallel reactors)
│   │   └── reactor_coupling.py  # Cross-reactor channels
│   │
│   ├── deliberate/              # V10: Structured evidence pipeline
│   │   ├── panel.py             # 4 parallel judges
│   │   ├── merge.py             # Judgment reconciliation
│   │   ├── wave.py              # Wave orchestration
│   │   └── convergence.py       # Coverage-gated convergence
│   │
│   ├── compress/                # V10: Knowledge compression
│   │   ├── cluster.py           # Passage clustering (embedding + heading)
│   │   ├── artifact.py          # KnowledgeArtifact builder
│   │   └── grounding.py         # 3-tier claim verification cascade
│   │
│   ├── adaptive/                # Adaptive overseer (20 tunables)
│   ├── agents/                  # Core agents (plan, search, synthesize, critique)
│   ├── backends/                # 6 search backends
│   ├── extractors/              # 6-tier extraction cascade
│   ├── scoring/                 # RRF, authority, grounding, claims
│   ├── reporting/               # Markdown/DOCX/PDF, provenance, heatmaps
│   ├── memory/                  # Cross-session memory + incremental
│   ├── event_log/               # JSONL run event logging
│   └── streaming.py             # StreamDisplay for astream progress
│
├── tests/                       # 1166 tests across 60+ modules
├── docker/                      # SearXNG Docker configuration
└── docs/                        # Architecture diagrams
```

</details>

<details>
<summary><b>Testing</b></summary>

```bash
# All tests (no network, no API keys, no Docker)
pytest tests/ -v

# With coverage
pytest tests/ --cov=deep_research_swarm --cov-report=term-missing
```

**1166 tests** covering all modules. All tests run without network access, API keys, or Docker — LLM calls are mocked.

</details>

## Cost Model

### Single Reactor (typical query)

| Stage | Calls | Model | Cost/iter |
|-------|-------|-------|-----------|
| Planner | 1 | Opus | $0.13 |
| Wave loop (4 waves) | 16 | 2 Haiku + 2 Sonnet/panel | $0.10 |
| Compression | 10 | Haiku | $0.004 |
| **Reactor (6 turns)** | 6 | Sonnet | **$0.16** |
| Outline | 1 | Opus | $0.14 |
| Section drafts | 8 | Sonnet | $0.06 |
| Composition | 1 | Haiku | $0.001 |
| **Adversarial critique** | 5 | Sonnet | **$0.08** |
| **Total/iteration** | ~60 | — | **$0.68** |

| Scenario | V9 | V10 Single | V10 Swarm (n=3) |
|----------|:---:|:----------:|:---------------:|
| Simple (1 iter) | $0.61 | $0.52 | $1.56 |
| Moderate (2 iter) | $1.22 | $1.04 | $3.12 |
| Complex (3 iter) | $1.82 | $1.56 | $4.68 |

V10 single-reactor: ~18% cost reduction with dramatically richer output. Intelligence pays for itself.

## Roadmap

- [x] **V1-V6** — Core pipeline, quality loops, streaming, persistence, HITL, memory
- [x] **V7** — Scholarly backends, citation chaining, passage grounding, provenance
- [x] **V8** — Adaptive overseer, embedding grounding, claim graph, OCR/GROBID
- [x] **V9** — Reactive search, 3x query volume, clarifier, multi-format export
- [x] **V10** — SIA Tensegrity: multi-agent deliberation, entropy control, adversarial critique, knowledge compression, multi-reactor swarm
- [ ] **V11** — Internal hybrid index, focused crawling, embedding routing, swarm streaming, reactor coupling integration

## License

[MIT](LICENSE)

> [!NOTE]
> The PDF extractor uses PyMuPDF4LLM (AGPL-3.0). CLI distribution under MIT is unaffected. If deploying as a network service, AGPL terms apply to the PDF extraction component. Apache-2.0 alternatives (GROBID, PaddleOCR) are available.
</div>
