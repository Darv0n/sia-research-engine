# V10: Deliberative Research Intelligence — Tensegrity Architecture

## The Thesis

V9 treats research as a pipeline: collect, then synthesize. 12 specialized modules run in
sequence — authority scoring, grounding, claim extraction, contradiction detection, gap
analysis, citation chaining — each producing rich output that the next module ignores.
The synthesizer receives a flat blob of 20 documents and asks Opus to rediscover all the
structure those 12 modules already computed. Five times.

**The modules aren't broken. The wiring is.**

V10 doesn't replace the modules. It rewires them into a **tensegrity structure** — a
deliberation system where each module's output is structurally incomplete without the
others, and a merge layer forces reconciliation before synthesis can proceed.

## The Structural Problem: 17 Broken Wires

Forensic analysis of the V9 codebase reveals 17 information flows that should exist but
don't. Each is a place where one module produces intelligence that another needs but
never receives:

### Tier 1 — Critical (these cause measurable quality loss)

| # | From | To | What's Missing |
|---|------|----|----------------|
| 1 | Contradiction | Authority | Contradicted sources should be downweighted. Currently contradiction between Nature and a blog ranks equally. |
| 2 | Contradiction | Grounding | Contradictory claims should require stricter grounding thresholds. Currently fixed at Jaccard 0.3. |
| 3 | Grounding | Claim Graph | Claim confidence should merge grounding + authority + contradiction. Currently claim IDs have no confidence. |
| 4 | Grounding | Gap Analyzer | Gaps should be inferred from ungrounded claims. Currently gap analyzer only sees document summaries. |
| 5 | Contradiction | Planner | Re-planning should seek contradiction resolution. Currently planner never sees contradictions. |
| 6 | Contradiction | RRF | Contradicted documents should be flagged in ranking. Currently all documents merge equally. |

### Tier 2 — Structural (these prevent emergent intelligence)

| # | From | To | What's Missing |
|---|------|----|----------------|
| 7 | Embedding | Diversity | Diversity should measure semantic redundancy, not just domain count. Two tech blogs from different domains saying the same thing isn't diverse. |
| 8 | Gap Analysis | Citation Chain | Citation traversal should prioritize papers addressing identified gaps. Currently purely relevance-filtered. |
| 9 | Citation Patterns | Authority | Within-run citation frequency should boost authority. Source appearing in 30% of citations gets same score as 0%. |
| 10 | Diversity | Gap Analyzer | Gaps should identify under-represented perspectives/domains. |
| 11 | Diversity | Grounding | Over-represented sources should face stricter grounding. |
| 12 | Authority | Planner | Perspectives should weight toward under-represented authority levels. |

### Tier 3 — Optimization (valuable when Tier 1-2 are wired)

| # | From | To | What's Missing |
|---|------|----|----------------|
| 13 | Grounding | Citation Chain | Expand more from well-grounded papers. |
| 14 | Query Diversity | RRF | Results from diverse query *types* should rank higher. |
| 15 | Citation Chain | Claim Graph | Distinguish claims found via search vs. citation chain. |
| 16 | Redundancy | Complexity | 100 copies of same result ≠ 100 unique results for complexity. |
| 17 | Tunable Changes | Planner | Planner should know "we just increased extraction_cap." |

## The Architecture: Tensegrity Deliberation

### Core Concept: The Deliberation Panel

Instead of running modules in sequence, V10 runs them as a **parallel deliberation panel**
using LangGraph's `Send()` API for dynamic fan-out. Four specialized agents analyze the
collected evidence simultaneously, then a merge node cross-references their judgments before
synthesis can proceed.

```
V9:  score -> gap_analysis -> [followup?] -> adapt_synthesis -> citation_chain ->
     contradiction -> synthesize -> critique

V10: WAVE LOOP:
       search_wave -> extract_wave -> score_wave ->
       DELIBERATION PANEL (parallel via Send()):
         [authority_judge | grounding_judge | contradiction_judge | coverage_judge]
       -> merge_judgments -> [coverage < threshold?] -> search_wave (loop)
                          -> [coverage >= threshold?] -> compress ->
     END WAVE LOOP
     citation_chain (targeted) -> synthesize (2-stage) -> critique (consolidated)
```

### Why Tensegrity, Not Just Parallelism

Parallelism means "run things at the same time." Tensegrity means "the structure collapses
if you remove any member." In V10:

- **Without authority judgments**: Synthesis can't weight source credibility. Claims from
  Nature and random blogs look identical.
- **Without grounding judgments**: Synthesis can't distinguish supported from fabricated
  claims. The refine stage returns.
- **Without contradiction judgments**: Synthesis presents false consensus. Tensions invisible.
- **Without coverage judgments**: Synthesis can't know what's missing. Gaps persist.

The merge node **requires all four judgment vectors**. A report produced without any of them
is structurally invalid — not just lower quality, but architecturally incomplete. This is
the tensegrity property.

### The Four Agents

Each maps directly to existing V9 modules — we're not building new intelligence, we're
wiring existing intelligence together:

| Agent | Existing Code | Lens | Produces | Model |
|-------|--------------|------|----------|-------|
| **Authority Judge** | `scoring/authority.py` + `scoring/rrf.py` | "Is this source trustworthy?" | Authority vectors per document, credibility ranking | Deterministic + Haiku summary |
| **Grounding Judge** | `scoring/grounding.py` + `scoring/claim_graph.py` + `scoring/embedding_grounding.py` | "Does evidence support each claim?" | Per-claim grounding verdicts, ungrounded claim list | Deterministic (Jaccard/embedding) |
| **Contradiction Judge** | `agents/contradiction.py` | "Where do sources disagree?" | Contradiction list with severity, claim-level linking | Sonnet (nuanced reasoning) |
| **Coverage Judge** | `agents/gap_analyzer.py` + `scoring/diversity.py` | "What's missing?" | Coverage map, under-represented perspectives, follow-up queries | Sonnet (creative gap identification) |

**Cost**: 2 deterministic + 1 Haiku + 2 Sonnet = ~$0.032/panel
Comparable to V9's sequential contradiction($0.01) + gap_analysis($0.015) = $0.025.
Nearly cost-neutral, but with cross-referencing.

### The Merge Node: Where Tensions Become Intelligence

The merge node is deterministic (no LLM) but is the architectural keystone. It:

1. **Cross-references authority × grounding**: High-authority + ungrounded = investigate.
   Low-authority + well-grounded = still valuable. High-authority + contradicted = flag.
2. **Cross-references contradiction × coverage**: A contradiction IS a coverage gap that
   needs resolution. Feeds contradiction-targeted queries into next wave.
3. **Cross-references diversity × grounding**: Over-represented sources with weak grounding
   = concentration risk. Under-represented authorities with strong grounding = signal boost.
4. **Produces the unified judgment context**: A single structured artifact that the
   synthesizer sees instead of raw documents.

```python
class JudgmentContext(TypedDict):
    """Merged output from all four deliberation agents."""
    claim_verdicts: list[ClaimVerdict]       # Per-claim: grounding + authority + contradiction status
    source_credibility: dict[str, float]     # URL -> merged credibility (authority × grounding × contradiction)
    active_tensions: list[ActiveTension]     # Contradictions annotated with authority + grounding context
    coverage_map: CoverageMap                # Facets, gaps, under-represented perspectives
    next_wave_queries: list[SubQuery]        # Queries targeting specific gaps/tensions
    overall_coverage: float                  # 0.0-1.0 — drives wave convergence
    structural_risks: list[str]              # "3 claims from single domain", "no institutional sources", etc.
```

### The Wave Loop: IRCoT Interleaving

Instead of one massive search-then-synthesize, V10 runs 3-5 micro-waves. Each wave:

1. **Search**: 3-5 queries (wave 1 from planner, subsequent waves from merge_judgments.next_wave_queries)
2. **Extract**: Standard extraction (reuses V9 extractor + chunker)
3. **Score**: Standard RRF scoring (reuses V9 scorer)
4. **Deliberation Panel**: Fan-out to 4 agents, fan-in at merge
5. **Converge Check**: If `overall_coverage >= coverage_threshold` → exit loop

The key difference from V9's gap_analysis reactive loop:
- V9: One follow-up round, gap analysis sees only document summaries
- V10: N waves, each wave's agents see ALL previous wave results + each other's judgments

Cost per wave: ~$0.04 (search/extract/score: free locally, panel: $0.032, scoring: deterministic)
5 waves: $0.20 — vs V9's single-pass search+extract+score: ~$0.15 for the search phase.
Marginal cost increase for dramatically better evidence quality.

## Knowledge Compression: The Bridge to Synthesis

After deliberation converges, the accumulated findings (hundreds of passages across waves,
annotated with judgment context) must be compressed into a KnowledgeArtifact for synthesis.

This is the cost-optimization layer from the compression plan, now operating on
**pre-deliberated material** rather than raw dumps.

### The Pipeline

```
Judgment context + passages → Cluster (fastembed) → Rank (merged credibility) →
  Extract claims (claim_graph) → Verify (grounding cascade) → Authority profiles →
  Tension mapping → KnowledgeArtifact
```

### What's Different from the Compression-Only Plan

The compression plan tried to do intelligence AND compression in the same layer. That was
the design error. Now:

- **Deliberation** handles intelligence (what matters, what's missing, what contradicts)
- **Compression** handles efficiency (cluster, rank, prune, verify, summarize)
- **Synthesis** handles writing (structured input → structured output)

Each layer has a single responsibility. The KnowledgeArtifact contains pre-deliberated,
pre-verified, pre-ranked material. The synthesizer's job is purely narrative.

### The KnowledgeArtifact

```python
class KnowledgeArtifact(TypedDict):
    question: str
    facets: list[Facet]                     # From coverage judge's question decomposition
    clusters: list[PassageCluster]          # Embedding-clustered, ranked by merged credibility
    claim_verdicts: list[ClaimVerdict]      # Per-claim: grounded? authority? contradicted?
    active_tensions: list[ActiveTension]    # Contradictions with resolution context
    coverage: CoverageMap                   # What's covered, what's not
    insights: list[Insight]                 # Novel connections from cross-wave analysis
    authority_profiles: list[AuthorityProfile]
    structural_risks: list[str]
    compression_ratio: float                # passages_in / claims_out
    wave_count: int
```

Estimated size: ~2,500-3,500 tokens (vs ~20,000 flat passages in V9).

## 2-Stage Synthesis

### Stage 1: Outline from Artifact (1 Opus call)

Input: KnowledgeArtifact (~3,000 tokens)
- Opus sees pre-structured knowledge: themes, verified claims with authority+grounding
  status, active tensions, coverage gaps, novel insights, structural risks
- Prompt: "Design a report that addresses each facet, acknowledges tensions where they
  exist, highlights insights, and notes where evidence is thin"

Output: Section headings + assigned cluster indices + key claims per section + tension handling

### Stage 2: Section Drafts (N parallel Sonnet calls)

Input per section: ~500-700 tokens
- Cluster summary + top 5 pre-ranked passages + pre-verified claims
- Claim verdicts attached (authority, grounding, contradiction status)
- Active tensions relevant to this section
- Insight candidates

**Sonnet, not Opus** — the intellectual work (what to include, how to handle tensions,
what to emphasize) was decided by Opus in Stage 1. Sonnet executes.

### Composition: 1 Haiku call

Intro/transitions/conclusion. Haiku at $0.25/$1.25 per 1M tokens — effectively free.

### Dropped V9 Stages
- **Stage 3 (grounding verification loop)** → Done in compression layer
- **Stage 4 (refine failures)** → Eliminated by pre-verified input
- **Stage 5 (compose on Opus)** → Haiku (structured input = cheap model sufficient)

## Consolidated Critic

V9 sends identical `sections_text` to 3 separate Sonnet calls (6,200 tokens duplicated).
V10: single Sonnet call scoring relevance + hallucination + quality simultaneously.

Same output shape (`GraderScores` per section). Same convergence logic. 1 call not 3.

Additionally: coverage-gated convergence using the KnowledgeArtifact's coverage_map.
If coverage >= 0.8 AND quality >= 0.75 → converge early. Deterministic, zero tokens.

## Cost Math

### V10 Per-Iteration (with 4-wave deliberation)

| Stage | Calls | Model | Tokens In | Tokens Out | Cost |
|-------|-------|-------|-----------|------------|------|
| Planner | 1 | Opus | 1,500 | 1,500 | $0.13 |
| **Wave searches (4 waves x 3 queries)** | 12 | N/A | 0 | 0 | $0.00 |
| **Wave extractions** | 4 | N/A (local) | 0 | 0 | $0.00 |
| **Wave scoring** | 4 | N/A (deterministic) | 0 | 0 | $0.00 |
| **Deliberation panels (4 waves)** | | | | | |
|   Authority judge x4 | 4 | Deterministic+Haiku | 2,000 | 800 | $0.003 |
|   Grounding judge x4 | 4 | Deterministic | 0 | 0 | $0.00 |
|   Contradiction judge x4 | 4 | Sonnet | 8,000 | 1,600 | $0.05 |
|   Coverage judge x4 | 4 | Sonnet | 6,000 | 2,000 | $0.05 |
|   Merge x4 | 4 | Deterministic | 0 | 0 | $0.00 |
| **Compression** | | | | | |
|   Clustering | 0 | Deterministic | 0 | 0 | $0.00 |
|   Cluster summaries | 10 | Haiku | 7,500 | 2,000 | $0.004 |
|   Claim verification | 0 | Deterministic | 0 | 0 | $0.00 |
| **Synthesis** | | | | | |
|   Outline from artifact | 1 | Opus | 3,000 | 1,200 | $0.14 |
|   Section drafts | 8 | Sonnet | 5,600 | 3,200 | $0.06 |
|   Composition | 1 | Haiku | 500 | 550 | $0.001 |
| **Critic** | 1 | Sonnet | 3,100 | 450 | $0.01 |
| **Subtotal** | ~57 | — | 37,200 | 13,300 | **$0.45** |

### 3-Iteration Projection: $1.35

But with coverage-gated convergence:
- Simple queries: 1 iteration, 2-3 waves → **$0.35**
- Moderate queries: 2 iterations, 3-4 waves → **$0.75**
- Complex queries: 3 iterations, 4-5 waves → **$1.35**

**Weighted average (40/40/20 distribution): $0.61 — 67% reduction from V9's $1.82**

### Intelligence Per Dollar

| Metric | V9 | V10 | Change |
|--------|-----|------|--------|
| Cost per run (weighted avg) | $1.82 | $0.61 | -67% |
| Passage waste ratio | 99.2% | ~20% | -80pp |
| Cross-module information flow | 0 wires | 17 wires | +17 |
| Deliberation perspectives per wave | 0 | 4 parallel | +4 |
| Contradiction × authority cross-reference | None | Automatic | New capability |
| Coverage-gated convergence | No | Yes | New capability |
| Wave-by-wave evidence refinement | 1 reactive follow-up | 3-5 deliberated waves | 3-5x |

## New Contracts (TypedDicts)

### Deliberation Types

```python
class Facet(TypedDict):
    id: str                        # "facet-001"
    question: str                  # Sub-question from research_question
    weight: float                  # Importance (0.0-1.0)

class ClaimVerdict(TypedDict):
    claim_id: str                  # Links to claim_graph claim
    claim_text: str
    grounding_score: float         # From grounding judge
    grounding_method: str          # "jaccard_v1" | "embedding_v1" | "nli_v1"
    authority_score: float         # From authority judge
    authority_level: str           # SourceAuthority value
    contradicted: bool             # From contradiction judge
    contradiction_id: str | None   # If contradicted, which contradiction

class ActiveTension(TypedDict):
    id: str
    claim_a: ClaimVerdict
    claim_b: ClaimVerdict
    severity: str                  # "direct" | "nuanced" | "contextual"
    authority_differential: float  # How much credibility differs between sides
    resolution_hint: str           # "seek newer source" | "context-dependent" | etc.

class Insight(TypedDict):
    id: str
    description: str               # Novel connection
    source_finding_ids: list[str]  # What combines to produce this
    confidence: float

class JudgmentContext(TypedDict):
    claim_verdicts: list[ClaimVerdict]
    source_credibility: dict[str, float]  # URL -> merged score
    active_tensions: list[ActiveTension]
    coverage_map: CoverageMap
    next_wave_queries: list[SubQuery]
    overall_coverage: float
    structural_risks: list[str]
    wave_number: int
```

### Compression Types

```python
class AuthorityProfile(TypedDict):
    dominant_authority: str
    source_count: int
    avg_authority_score: float
    institutional_ratio: float

class PassageCluster(TypedDict):
    cluster_id: str
    theme: str
    passage_ids: list[str]
    claims: list[ClaimVerdict]
    authority: AuthorityProfile
    summary: str                   # Haiku-generated

class CrossClusterTension(TypedDict):
    cluster_a_id: str
    cluster_b_id: str
    tension: ActiveTension

class CoverageMap(TypedDict):
    facet_coverage: dict[str, float]
    overall_coverage: float
    uncovered_facets: list[str]
    under_represented_perspectives: list[str]

class KnowledgeArtifact(TypedDict):
    question: str
    facets: list[Facet]
    clusters: list[PassageCluster]
    claim_verdicts: list[ClaimVerdict]
    active_tensions: list[ActiveTension]
    coverage: CoverageMap
    insights: list[Insight]
    authority_profiles: list[AuthorityProfile]
    structural_risks: list[str]
    compression_ratio: float
    wave_count: int
```

## New State Fields

```python
# Deliberation (V10) — panel judgments accumulate across waves
panel_judgments: Annotated[list[dict], operator.add]
judgment_context: Annotated[dict, _replace_dict]       # JudgmentContext, updated per wave
knowledge_artifact: Annotated[dict, _replace_dict]     # KnowledgeArtifact, built by compress
deliberation_waves: Annotated[list[dict], operator.add] # Wave history for streaming/debug
wave_count: Annotated[int, _replace_int]
```

## New Tunables

| Name | Default | Floor | Ceiling | Purpose |
|------|---------|-------|---------|---------|
| max_waves | 5 | 2 | 8 | Deliberation waves per iteration |
| wave_batch_size | 3 | 1 | 5 | Queries per wave |
| wave_extract_cap | 15 | 5 | 30 | Docs extracted per wave |
| coverage_threshold | 0.75 | 0.5 | 0.95 | Convergence threshold for deliberation |
| max_clusters | 12 | 3 | 20 | Maximum passage clusters |
| claims_per_cluster | 8 | 3 | 15 | Max verified claims per cluster |

## Implementation Phases

### Phase 1: Deliberation Panel + Merge (the intelligence upgrade)

**New module: `deep_research_swarm/deliberate/`**

**`deliberate/__init__.py`** — module marker

**`deliberate/panel.py`** — the 4-agent deliberation panel
- `authority_judge(scored_docs, passages, contradictions) -> AuthorityJudgment`
  - Reuses `scoring/authority.py` for classification
  - Adds: cross-reference with contradiction records
  - Adds: within-run citation frequency boost
  - Haiku summary of authority landscape
- `grounding_judge(passages, section_drafts, claim_graph) -> GroundingJudgment`
  - Reuses `scoring/grounding.py` for Jaccard verification
  - Reuses `scoring/embedding_grounding.py` for semantic verification
  - Reuses `scoring/claim_graph.py` for claim extraction
  - Adds: contradiction-aware threshold adjustment
  - Deterministic (no LLM)
- `contradiction_judge(scored_docs, caller) -> ContradictionJudgment`
  - Reuses `agents/contradiction.py` core logic
  - Adds: claim-level linking (not just document-level)
  - Adds: authority differential annotation
  - Sonnet call (existing pattern)
- `coverage_judge(scored_docs, research_question, diversity_metrics, caller) -> CoverageJudgment`
  - Reuses `agents/gap_analyzer.py` core logic
  - Reuses `scoring/diversity.py` for diversity input
  - Adds: facet decomposition + coverage scoring
  - Adds: semantic redundancy check via embeddings
  - Sonnet call (existing pattern)

**`deliberate/merge.py`** — judgment reconciliation (deterministic)
- `merge_judgments(authority, grounding, contradiction, coverage) -> JudgmentContext`
  - Cross-references all 4 judgment vectors
  - Produces claim_verdicts with merged confidence
  - Generates next-wave queries targeting specific gaps
  - Computes overall_coverage
  - Identifies structural risks

**`deliberate/wave.py`** — wave orchestration
- `async def execute_wave(wave_num, state, callers, config) -> WaveResult`
  - Executes search queries (reuses existing searcher)
  - Extracts content (reuses existing extractor)
  - Chunks passages (reuses existing chunker)
  - Scores results (reuses existing RRF scorer)
  - Dispatches deliberation panel (via Send() in builder.py)
  - Returns accumulated results

**`deliberate/convergence.py`** — wave convergence logic
- `should_continue_waves(judgment_context, wave_count, max_waves) -> bool`
  - Coverage-gated: stop when overall_coverage >= threshold
  - Budget-gated: stop when max_waves reached
  - Diminishing returns: stop when coverage_delta < 0.05 between waves
  - Deterministic, zero tokens

**`contracts.py` additions:**
- All TypedDicts listed above (Facet, ClaimVerdict, ActiveTension, etc.)

**`graph/state.py` additions:**
- 5 new state fields (panel_judgments, judgment_context, etc.)

**Key principle: No existing module code is modified in Phase 1.** The panel agents
CALL the existing functions. The intelligence comes from cross-wiring, not rewriting.

### Phase 2: Compression Layer

**New module: `deep_research_swarm/compress/`**

**`compress/__init__.py`** — module marker

**`compress/cluster.py`** — passage clustering
- `cluster_by_embedding()` — k-means via fastembed (zero new deps)
- `cluster_by_heading()` — fallback when embeddings unavailable
- `rank_passages_in_cluster()` — uses merged credibility from judgment_context

**`compress/artifact.py`** — KnowledgeArtifact builder
- `async def build_knowledge_artifact(judgment_context, passages, caller) -> KnowledgeArtifact`
  - Clusters passages by embedding
  - Ranks using judgment_context.source_credibility (not raw scored_documents)
  - Attaches claim_verdicts (already computed by deliberation panel)
  - Generates cluster summaries (Haiku, parallel)
  - Maps tensions from judgment_context
  - Returns complete artifact

**`compress/grounding.py`** — claim verification cascade
- `verify_claim_grounding()` — Tier 1: NLI, Tier 2: embedding, Tier 3: Jaccard
- `NLIVerifier` class — lazy-loads cross-encoder (optional `[nli]` dep)

### Phase 3: 2-Stage Synthesis + Consolidated Critic

**Rewrite `agents/synthesizer.py`:**
- Stage 1: Outline from KnowledgeArtifact (1 Opus call, structured input)
- Stage 2: Section drafts from clusters (N Sonnet calls)
- Composition: 1 Haiku call
- Drop stages 3, 4, 5
- Output shape unchanged (backward compatible)

**Rewrite `agents/critic.py`:**
- Single Sonnet call for all 3 dimensions
- Coverage-gated convergence from knowledge_artifact

**Add NLI path to `agents/contradiction.py`:**
- Optional cross-encoder for local contradiction detection
- Falls back to existing Sonnet call

### Phase 4: Graph Rewiring + Config

**`graph/builder.py` changes:**
- Add haiku_caller alongside opus_caller and sonnet_caller
- New nodes: deliberation panel (via Send()), merge_judgments, compress
- Wave loop: search_wave → extract_wave → score_wave → panel → merge → [converge?]
- Wire: `plan → [wave loop] → compress → adapt_synthesis → citation_chain →
  contradiction → synthesize → critique → rollup_budget → [converge?]`
- Remove direct gap_analysis → search_followup → extract_followup → score_merge chain
  (functionality redistributed into wave loop)

**`config.py` addition:**
- HAIKU_MODEL env var

**`adaptive/registry.py` additions:**
- 6 new tunables

### Phase 5: Tests + Release

- Tests for all new modules
- Target: 850+ tests (777 existing + ~75 new)
- Update CLAUDE.md, pyproject.toml, README
- Live comparison test: same query, V9 vs V10

## Streaming Events

| Event Kind | Payload | When |
|-----------|---------|------|
| `wave_start` | wave_num, queries, target_facets | Each wave begins |
| `panel_authority` | credibility_summary, top_sources | Authority judge completes |
| `panel_grounding` | grounded_claims, ungrounded_claims | Grounding judge completes |
| `panel_contradiction` | tension_count, severities | Contradiction judge completes |
| `panel_coverage` | coverage_score, gaps_identified | Coverage judge completes |
| `merge_complete` | overall_coverage, structural_risks | Merge node completes |
| `wave_converged` | wave_count, reason | Deliberation loop exits |
| `compression_complete` | clusters, claims, ratio | Artifact built |

## What Gets Preserved vs Redistributed vs Rewritten

### Preserved (untouched)
- `agents/planner.py` — generates initial plan + perspectives
- `agents/clarifier.py` — produces scope_hints
- `agents/searcher.py` — reused by wave execution
- `agents/extractor.py` — reused by wave execution
- `agents/citation_chain.py` — runs after deliberation (targeted by judgment gaps)
- `extractors/chunker.py` — reused by wave execution
- `scoring/rrf.py` — reused by wave scoring
- `scoring/authority.py` — called by authority_judge
- `scoring/grounding.py` — called by grounding_judge
- `scoring/embedding_grounding.py` — called by grounding_judge
- `scoring/claim_graph.py` — called by grounding_judge
- `scoring/diversity.py` — called by coverage_judge
- `adaptive/adapt_extraction.py` — still scales per-wave tunables
- `adaptive/adapt_synthesis.py` — still scales synthesis tunables
- `reporting/` — all renderers/exporters unchanged

### Redistributed (moved to panel, original files preserved)
- `agents/gap_analyzer.py` — core logic reused by coverage_judge
- `agents/contradiction.py` — core logic reused by contradiction_judge
- V9 reactive loop nodes — functionality absorbed into wave loop

### Rewritten
- `agents/synthesizer.py` — 5-stage → 2-stage
- `agents/critic.py` — 3-call → 1-call

## Dependency Strategy

### Tier 0: Zero new dependencies (Phases 1-5)
- Haiku: Anthropic SDK already installed
- Fastembed: already in `[embeddings]`
- All deliberation logic: pure Python + existing callers
- Send() API: LangGraph already installed

### Tier 1: Optional (`[nli]` dep group)
- `sentence-transformers[onnx]` + `cross-encoder/nli-deberta-v3-xsmall`
- Upgrades grounding + contradiction from Jaccard/LLM to NLI
- Not required — graceful degradation

## Verification

After each phase:
```bash
.venv/Scripts/python.exe -m pytest tests/ -v
.venv/Scripts/python.exe -m ruff check . && .venv/Scripts/python.exe -m ruff format --check .
```

After Phase 4 (integration):
```bash
.venv/Scripts/python.exe -m deep_research_swarm \
  "astrology sources for professional astrologers" --verbose
# Targets:
#   Cost: <$0.80 (vs V9 $1.82)
#   Quality: richer (tensions acknowledged, coverage gaps noted, multi-perspective)
#   Streaming: wave-by-wave progress, panel results visible
```

## Critical Files Summary

| File | Change | Phase |
|------|--------|-------|
| `contracts.py` | Add ~12 TypedDicts | 1 |
| `graph/state.py` | Add 5 state fields | 1 |
| NEW: `deliberate/__init__.py` | Module marker | 1 |
| NEW: `deliberate/panel.py` | 4-agent deliberation panel | 1 |
| NEW: `deliberate/merge.py` | Judgment reconciliation | 1 |
| NEW: `deliberate/wave.py` | Wave orchestration | 1 |
| NEW: `deliberate/convergence.py` | Coverage-gated convergence | 1 |
| NEW: `compress/__init__.py` | Module marker | 2 |
| NEW: `compress/cluster.py` | Passage clustering | 2 |
| NEW: `compress/artifact.py` | KnowledgeArtifact builder | 2 |
| NEW: `compress/grounding.py` | NLI/embedding/Jaccard cascade | 2 |
| `agents/synthesizer.py` | Rewrite: 5-stage → 2-stage | 3 |
| `agents/critic.py` | Single-call 3-dimension + coverage convergence | 3 |
| `agents/contradiction.py` | Add NLI path | 3 |
| `graph/builder.py` | Wave loop, Send() panel, compress node, rewire | 4 |
| `config.py` | HAIKU_MODEL env var | 4 |
| `adaptive/registry.py` | 6 new tunables | 4 |
| `pyproject.toml` | Version 0.10.0, `[nli]` dep group | 5 |

## Research Sources

### Intelligence Architecture
- IRCoT: arXiv:2212.10509 — Interleaved retrieval + chain-of-thought
- STORM: arXiv:2402.14207 — Multi-perspective questioning
- Co-STORM: arXiv:2408.15232 — Collaborative discourse
- WebThinker: arXiv:2503.01428 — Think-while-searching (NeurIPS 2025)
- Multi-Agent Debate: arXiv:2305.14325 — 3 agents, 2 rounds optimal (Du et al.)
- Self-RAG: arXiv:2310.11511 — Reflection tokens
- Google Mass: arXiv:2502.02533 — Optimized prompts + topology
- MoA: arXiv:2406.04692 — Mixture-of-Agents

### Compression & Grounding
- CompactRAG: arXiv:2602.05728 — 81% token reduction
- RAPTOR: arXiv:2401.18059 — Recursive summarization trees
- MiniCheck: arXiv:2404.10774 — 770M fact-checker
- StructRAG: arXiv:2410.08815 — Adaptive structure selection

### Implementation Patterns
- LangGraph Send() API — dynamic parallel fan-out
- LangGraph Map-Reduce — parallel execution + merge
- Blackboard architecture — shared state communication
- HuggingFace models verified via MCP: bge-small, nli-deberta (CPU-viable, ONNX)
