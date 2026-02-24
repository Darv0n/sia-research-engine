# V10 Unified Plan: SIA Tensegrity Architecture

## What This Is

The definitive V10 plan. Merges two complementary architectures:

- **SIA** (Semantic Intelligence Architecture) — true multi-agent deliberation with
  persistent cognitive stances, thermodynamic entropy control, adversarial critique,
  and multi-reactor swarm evolution
- **Tensegrity Wiring** — 17 broken cross-module information flows repaired, knowledge
  compression layer, 2-stage synthesis, model tiering

Neither alone is complete. SIA adds intelligence to synthesis/critique but operates on
the same flat document dumps. Tensegrity adds structured evidence but has no deliberation
engine. Together: agents deliberate over pre-structured, cross-referenced knowledge
artifacts, governed by entropy thermodynamics.

**Branch**: `v10-sia-tensegrity` (forked from main at v0.9.0, 777 tests)

---

## The Two Blind Spots, Filled

| SIA Alone | Tensegrity Alone | Unified |
|-----------|-----------------|---------|
| Reactor deliberates over flat `_build_source_context()` blob | No deliberation — parallel judges with no conversation memory | Reactor deliberates over KnowledgeArtifact |
| Doesn't address 19K→160 passage waste (99.2%) | Fixes waste but no multi-turn reasoning | Compression feeds artifact to reactor |
| All 5-stage synthesis preserved (costly) | 2-stage synthesis (cheap) but no pre-synthesis deliberation | Reactor → compress → 2-stage synthesis |
| Critic upgraded to adversarial multi-turn | Critic downgraded to single call | Adversarial critique on pre-compressed sections |
| No cross-module wiring repair | 17 wires repaired but no persistent agent stances | Judges wire existing modules; reactor adds stances |
| Entropy governs convergence | Coverage governs convergence | Entropy + coverage + singularity prevention |

---

## Agent Deployment: 5 from 7

The SIAFRESH framework (Section 2.2.A, Dynamic Mode) defines 7 agents as a **talent pool**
but selects only the top-relevant subset per query: `a = [Top relevant > threshold]`.
The query analysis example shows 3 selected from 7 (`Thresh:0.65 -> a:[L, Light, Shika]`).

For the research pipeline context, the typical active set is **5 agents**:

| Agent | Role in Pipeline | When Activated | Model |
|-------|-----------------|----------------|-------|
| **Lawliet** | Constraint extraction, compression, grounding validation | Always (core) | Sonnet |
| **Light** | Directional commitment, strategic framing, plan execution | Always (core) | Sonnet |
| **Rick** | Frame rupture, hidden assumption exposure, reframe catalyst | When entropy > 0.40 (turbulence+) | Sonnet |
| **Makishima** | Legitimacy stress-test, ethical faultline exposure, value audit | Always in critique; conditional in reactor | Sonnet |
| **Shikamaru** | Energy minimization, branch pruning, efficiency stabilizer | Mid-run cooling, synthesis readiness | Sonnet |

**Reserve pool** (activated by Kernel when specific conditions trigger):

| Agent | When Deployed |
|-------|--------------|
| **Shiro** | When A₄ (structural formalism) is low — rule-building needed |
| **Johan** | When A₆ (coalition cohesion) is dangerously unexamined — alignment audit |

This maps to the SIA spec's Dynamic Mode threshold selection. The Kernel's
`select_speaker()` implements this via relevance scoring + fatigue + entropy band.

---

## Execution Order: 6 Phases

```
Phase 0  SIA Genesis           ← Agent definitions, covenants, data structures
Phase 1  Entropy Layer         ← Thermodynamic foundation (PR-SIA-01)
Phase 2  Tensegrity Wiring     ← 17 broken wires + compression + 2-stage synthesis
Phase 3  True Agent Reactor    ← Multi-turn deliberation core (PR-SIA-02 adapted)
Phase 4  Adversarial Critique  ← Multi-turn evaluation (PR-SIA-03 adapted)
Phase 5  Integration + Swarm   ← Full wiring + optional swarm (PR-SIA-04)
```

Each phase is independently shippable and backward compatible.

---

## Phase 0: SIA Genesis (Agent Definitions + Covenants)

**Purpose**: Define all agent cognitive lenses, covenant physics, IntType grammar,
and data structures. No runtime code — pure type definitions and frozen data.

### Deliverables

**`contracts.py` additions:**
```python
# --- SIA types ---

class EntropyState(TypedDict):
    e: float                    # scalar entropy in [0, 1]
    e_amb: float                # ambiguity component
    e_conf: float               # conflict component
    e_nov: float                # novelty component
    e_trust: float              # trust/coherence component
    band: str                   # "crystalline" | "convergence" | "turbulence" | "runaway"
    turn: int
    stagnation_count: int       # consecutive turns with |delta_e| < 0.03

class TurnRecord(TypedDict):
    turn: int
    agent: str
    int_type: str               # B, C, RF, CL, CO, A, S, I
    constraints: list[str]
    challenges: list[str]
    reframes: list[str]
    response_to_prior: list[str]
    raw_output: str

class ReactorState(TypedDict):
    constraints: list[str]
    rejected_branches: list[str]
    active_frames: list[str]
    key_claims: list[str]
    coalition_map: dict[str, list[str]]
    unresolved: list[str]
    turn_log: list[TurnRecord]

class ReactorTrace(TypedDict):
    turns_executed: int
    agents_used: list[str]
    constraints_produced: int
    branches_killed: int
    challenges_issued: int
    final_entropy: float
    termination_reason: str
    ignition_pattern: str

class AdversarialFinding(TypedDict):
    agent: str
    int_type: str
    target_section: str         # section_id or "global"
    finding: str
    severity: str               # "critical" | "significant" | "minor"
    actionable: bool
    response_to: str

class CritiqueTrace(TypedDict):
    turns: int
    findings_count: int
    critical_findings: int
    constraints_extracted: int
    missing_variables: list[str]
    alternative_frames: list[str]
    recommendation: str         # "converge" | "replan" | "refine_targeted"

# --- Tensegrity types ---

class Facet(TypedDict):
    id: str
    question: str
    weight: float

class ClaimVerdict(TypedDict):
    claim_id: str
    claim_text: str
    grounding_score: float
    grounding_method: str
    authority_score: float
    authority_level: str
    contradicted: bool
    contradiction_id: NotRequired[str]

class ActiveTension(TypedDict):
    id: str
    claim_a: ClaimVerdict
    claim_b: ClaimVerdict
    severity: str
    authority_differential: float
    resolution_hint: str

class JudgmentContext(TypedDict):
    claim_verdicts: list[ClaimVerdict]
    source_credibility: dict[str, float]
    active_tensions: list[ActiveTension]
    coverage_map: CoverageMap
    next_wave_queries: list[SubQuery]
    overall_coverage: float
    structural_risks: list[str]
    wave_number: int

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
    summary: str

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
    insights: list[dict]
    authority_profiles: list[AuthorityProfile]
    structural_risks: list[str]
    compression_ratio: float
    wave_count: int

class SwarmMetadata(TypedDict):
    n_reactors: int
    reactor_configs: list[dict]
    reactor_entropies: list[float]
    reactor_tokens: list[int]
    reactor_costs: list[float]
    winner_id: str
    selection_reason: str
    selection_scores: dict[str, float]
    cross_validation_scores: dict[str, float]
    total_tokens_all: int
    total_cost_all: float
    failed_reactors: list[str]
```

**New module: `deep_research_swarm/sia/__init__.py`** — package marker

**New module: `deep_research_swarm/sia/agents.py`** — 7 SIAAgent frozen dataclasses
- Cognitive lens system prompts (processing bias, NOT role-play)
- Basin profiles with 7-axis deformation signatures
- Preferred IntTypes, repulsion patterns, instability signatures
- Template variables: {research_question}, {source_summary}, {entropy_band}, etc.

**New module: `deep_research_swarm/sia/covenants.py`** — pairwise field physics
- All covenant pairs from Inter-Agent-Covenants spec
- Covenant schema: coupling, yield_pattern, failure_mode, escalation_trigger
- High-risk triads as separate data structures
- Lookup function: `get_covenant(a, b)` (order-independent)

### Tests: ~15

- All 7 agents instantiate with correct fields
- Basin profiles have valid axes and bounded weights
- System prompts contain required template variables
- All covenant pairs defined, lookup works both orderings
- High-risk triads defined
- IntTypes are valid across all covenants

---

## Phase 1: Entropy Layer (PR-SIA-01)

**Purpose**: Thermodynamic convergence control. Replaces confidence-delta scalar
with multi-signal entropy computation. Zero LLM calls.

### Deliverables

**`sia/entropy.py`** — pure deterministic computation from ResearchState observables
- `classify_band(e)` — 4 bands: crystalline (<=0.20), convergence (0.20-0.45),
  turbulence (0.45-0.70), runaway (>0.70)
- `compute_entropy(state, prev)` — signal extraction:
  - e_amb: MEDIUM confidence sections + unresolved gaps
  - e_conf: contradiction density + grader score divergence
  - e_nov: new query ratio + outline heading delta
  - e_trust: grounding score variance + unmapped citations
  - Scalar: `e = 0.30*amb + 0.30*conf + 0.20*nov + 0.20*trust`
- `entropy_gate(entropy, sections)` — synthesis allowed/blocked
- `detect_false_convergence(entropy, sections, contradictions)`
- `detect_dominance(history, sections)`

**`graph/state.py` additions:**
```python
entropy_state: Annotated[dict, _replace_dict]
entropy_history: Annotated[list[dict], operator.add]
```

**`graph/builder.py` changes:**
- New node: `compute_entropy` wired after critique, before rollup_budget
- Convergence check augmented: confidence AND entropy_gate AND NOT false_convergence
  AND NOT dominance

**Optional `sia/entropy_steering.py`:**
- Adjusts existing tunables based on entropy band
- Runaway: reduce perspectives_count, zero follow_up_budget
- Turbulence: reduce target_queries
- Convergence: deeper drafts
- Crystalline: no-op

### Tests: ~35

- classify_band: all 4 bands + boundary values
- compute_entropy: synthetic states with known signals
- entropy_gate: all convergence/block conditions
- False convergence: low-e-with-contradictions, rapid-drop-without-constraint-gain
- Dominance: uniform scores, grader dimension convergence
- Graph integration: node position, convergence routing, backward compat

### Test target: 777 + 35 = ~812

---

## Phase 2: Tensegrity Wiring + Compression + 2-Stage Synthesis

**Purpose**: Fix the evidence pipeline. Repair 17 broken cross-module information
flows. Build KnowledgeArtifact from cross-referenced judgments. Rewrite synthesis
from 5-stage Opus to 2-stage Sonnet.

This is the largest phase. It delivers the cost reduction AND the structured
evidence that makes the reactor (Phase 3) dramatically more effective.

### Deliverables

**New module: `deep_research_swarm/deliberate/`**

**`deliberate/__init__.py`** — module marker

**`deliberate/panel.py`** — 4-agent deliberation panel (parallel via Send())
- `authority_judge(scored_docs, passages, contradictions)` — reuses `scoring/authority.py`,
  cross-references with contradictions, computes within-run citation frequency.
  Deterministic + 1 Haiku summary.
- `grounding_judge(passages, claim_graph)` — reuses `scoring/grounding.py` +
  `scoring/embedding_grounding.py` + `scoring/claim_graph.py`. Contradiction-aware
  threshold adjustment. Deterministic (no LLM).
- `contradiction_judge(scored_docs, caller)` — reuses `agents/contradiction.py` core.
  Adds claim-level linking + authority differential annotation. 1 Sonnet call.
- `coverage_judge(scored_docs, question, diversity, caller)` — reuses
  `agents/gap_analyzer.py` + `scoring/diversity.py`. Facet decomposition + coverage
  scoring + semantic redundancy check. 1 Sonnet call.

**`deliberate/merge.py`** — judgment reconciliation (deterministic, no LLM)
- `merge_judgments(authority, grounding, contradiction, coverage) -> JudgmentContext`
- Cross-references: authority x grounding, contradiction x coverage, diversity x grounding
- Produces: claim_verdicts, source_credibility, active_tensions, coverage_map
- Generates next-wave queries targeting specific gaps/tensions
- Computes overall_coverage

**`deliberate/wave.py`** — wave orchestration
- `execute_wave(wave_num, state, callers, config) -> WaveResult`
- Reuses existing searcher, extractor, chunker, scorer
- 3-5 micro-waves per iteration

**`deliberate/convergence.py`** — wave convergence
- Coverage-gated: stop when overall_coverage >= threshold
- Diminishing returns: stop when coverage_delta < 0.05

**New module: `deep_research_swarm/compress/`**

**`compress/__init__.py`** — module marker

**`compress/cluster.py`** — passage clustering
- `cluster_by_embedding()` via fastembed (existing dep)
- `cluster_by_heading()` fallback
- `rank_passages_in_cluster()` using merged credibility from JudgmentContext

**`compress/artifact.py`** — KnowledgeArtifact builder
- `build_knowledge_artifact(judgment_context, passages, caller) -> KnowledgeArtifact`
- Clusters by embedding, ranks by merged credibility
- Attaches claim_verdicts from deliberation panel
- Generates cluster summaries (Haiku, parallel)
- Maps tensions and insights

**`compress/grounding.py`** — claim verification cascade
- 3-tier: NLI (optional [nli] dep) > embedding (fastembed) > Jaccard (pure Python)

**Rewrite: `agents/synthesizer.py`** — 5-stage to 2-stage
- Stage 1: Outline from KnowledgeArtifact (1 Opus call, ~3K tokens input)
  - If reactor_products available (Phase 3): constraints, rejected_branches,
    active_frames, unresolved variables injected as additional context
  - If no reactor: artifact alone provides structured input
- Stage 2: Section drafts from clusters (N parallel Sonnet calls)
  - Each section gets: cluster summary + top passages + pre-verified claims
- Composition: 1 Haiku call
- **Drops**: stages 3 (grounding loop), 4 (refine), 5 (compose on Opus)
- **Interface preserved**: same return dict shape

**`config.py` addition:** HAIKU_MODEL env var

**`graph/state.py` additions:**
```python
panel_judgments: Annotated[list[dict], operator.add]
judgment_context: Annotated[dict, _replace_dict]
knowledge_artifact: Annotated[dict, _replace_dict]
deliberation_waves: Annotated[list[dict], operator.add]
wave_count: Annotated[int, _replace_int]
```

**`graph/builder.py` changes:**
- Add haiku_caller
- New nodes: deliberation panel (via Send()), merge_judgments, compress
- Wave loop: plan -> [search_wave -> extract_wave -> score_wave -> panel -> merge
  -> converge?] -> compress -> adapt_synthesis -> ...
- Replaces: gap_analysis -> search_followup -> extract_followup -> score_merge chain

**`adaptive/registry.py` additions:** 6 new tunables
- max_waves (5, 2-8), wave_batch_size (3, 1-5), wave_extract_cap (15, 5-30)
- coverage_threshold (0.75, 0.5-0.95), max_clusters (12, 3-20), claims_per_cluster (8, 3-15)

### Tests: ~75

- Deliberation panel: all 4 judges produce correct output shapes
- Merge: cross-referencing logic (authority x grounding, contradiction x coverage)
- Wave: convergence, diminishing returns, budget respect
- Clustering: embedding and heading fallback
- Artifact: compression ratio, claim count, coverage map
- Grounding cascade: 3-tier fallback
- Synthesizer: 2-stage output matches V9 shape, backward compat
- Graph: new topology compiles, nodes in correct position

### Test target: ~812 + 75 = ~887

---

## Phase 3: True Agent Reactor (PR-SIA-02 adapted)

**Purpose**: Multi-turn deliberation with persistent cognitive stances. Agents see
and respond to each other in a shared conversation thread. Reactor operates on
KnowledgeArtifact (from Phase 2), not flat document dumps.

### Key Adaptation from Original SIA Plan

The original PR-SIA-02 has the reactor wrap the existing 5-stage synthesis pipeline
and feed constraints into the outline prompt. In the unified architecture:

- The reactor operates **between compression and synthesis**
- It deliberates over the **KnowledgeArtifact** (structured, pre-verified knowledge)
- Its output (constraints, rejected branches, active frames) feeds directly into
  the 2-stage synthesis from Phase 2
- The synthesis receives both: KnowledgeArtifact + ReactorState

This is dramatically more effective. The reactor no longer needs to rediscover
structure from raw documents — it reasons about pre-structured claims, tensions,
coverage gaps, and authority profiles. Each agent's cognitive lens operates on
rich input instead of flat text.

### Deliverables

**`sia/kernel.py`** — thermodynamic steering engine
- `SIAKernel` class with select_speaker(), frame_turn(), parse_turn_output(),
  update_state(), should_terminate(), harvest()
- Ignition doctrine: 4 valid patterns (never start with cooling sink)
- Entropy-band steering: runaway -> compression, turbulence -> anchor after rupture,
  convergence -> build, crystalline -> harvest
- Anti-dominance: max 2 consecutive turns, fatigue tracking
- Covenant enforcement: anchor requirements, critique tax
- Agent selection by relevance threshold from available pool of 7

**Refactored `agents/synthesizer.py`:**
- Phase A: Reactor deliberation (when SIA enabled + entropy_state present)
  - Shared conversation thread grows with each turn
  - Kernel selects speaker from 5-agent active set (threshold from 7)
  - Each agent sees full prior conversation + Kernel framing
  - 6 turns default (tunable: sia_reactor_turns, 3-10)
  - Sonnet tier for all reactor calls
  - Token budget: sia_reactor_budget (20K default)
- Phase B: 2-stage synthesis (from Phase 2)
  - Outline receives: KnowledgeArtifact + reactor constraints + rejected branches
  - Section drafts: Sonnet with cluster context + reactor active_frames
- Graceful degradation: SIA disabled -> Phase 2 behavior exactly

**`adaptive/registry.py` additions:**
```python
Tunable(name="sia_reactor_turns", default=6, floor=3, ceiling=10, category="synthesis")
Tunable(name="sia_reactor_budget", default=20000, floor=8000, ceiling=40000, category="synthesis")
```

**`graph/state.py` addition:**
```python
reactor_trace: Annotated[dict, _replace_dict]
```

### Graph Topology (after Phase 3)

```
health_check -> clarify -> plan -> [plan_gate?] ->
  WAVE LOOP:
    search_wave -> extract_wave -> score_wave ->
    DELIBERATION PANEL (parallel via Send()):
      [authority_judge | grounding_judge | contradiction_judge | coverage_judge]
    -> merge_judgments -> [coverage < threshold?] -> search_wave (loop)
                       -> [coverage >= threshold?] ->
  END WAVE LOOP
  compress -> REACTOR (multi-turn SIA deliberation) ->
  synthesize (2-stage) -> critique -> compute_entropy ->
  rollup_budget -> [converge?] -> report -> [report_gate?]
```

### Tests: ~80

- All agent definitions: instantiation, basin profiles, template vars
- Kernel: ignition patterns, entropy-band selection, anti-dominance, fatigue
- Kernel: anchor requirements, critique tax, termination conditions
- Kernel: harvest produces deduplicated constraints, coalition map
- Covenants: all pairs, lookup, valid IntTypes
- Synthesizer: reactor-augmented output matches V9 shape
- Synthesizer: graceful degradation (SIA disabled -> V9)
- Synthesizer: conversation thread grows (each turn adds 2 messages)
- Synthesizer: multiple agents participate (>= 3 distinct in 6-turn reactor)
- Synthesizer: constraints flow to outline, rejected branches as exclusions

### Test target: ~887 + 80 = ~967

---

## Phase 4: Adversarial Critique (PR-SIA-03 adapted)

**Purpose**: Multi-turn adversarial evaluation using same true-agent architecture.
Replaces V9's 3 parallel graders with a conversation where agents challenge the
synthesis from different cognitive stances.

### Deliverables

**`sia/adversarial_critique.py`** — multi-turn adversarial evaluation
- Sequence: Makishima (legitimacy stress-test) -> Lawliet (constraint extraction)
  -> Rick (frame audit, conditional on entropy > 0.40) -> Shikamaru (synthesis
  readiness) -> Light (replan direction, conditional on replan recommendation)
- Shared conversation thread: each agent sees prior agents' critiques
- Score adjustment: critical finding -0.15, significant -0.08, minor -0.03
- Output compatible with existing critique consumers

**`sia/singularity_prevention.py`** — pre-synthesis safety gate
- 4 singularity checks: constraint, directional, reframe, coalition shadow
- 7-axis stability check (A₁-A₇ from LatentBasinMap)
- Blocks synthesis when structural collapse detected

**Modified `agents/critic.py`:**
- Mode switch: `sia_enabled -> adversarial_critique()` else `_classic_critique()`
- Backward compatible

**Convergence now uses five-way check:**
```python
converged = (
    confidence_ok
    AND entropy_gate_ok
    AND NOT false_convergence
    AND NOT dominance
    AND singularity_safe
)
```

**`graph/state.py` additions:**
```python
adversarial_findings: Annotated[list[dict], _replace_list]
critique_trace: Annotated[dict, _replace_dict]
```

### Tests: ~47

- Adversarial sequence: correct agent order
- Rick conditional on entropy > 0.40
- Conversation thread: each turn sees prior
- Score adjustments: severity -> confidence reduction
- Singularity prevention: all 4 types, 7-axis stability
- Five-way convergence: any veto blocks
- Backward compat: SIA disabled -> classic critique

### Test target: ~967 + 47 = ~1014

---

## Phase 5: Integration + Optional Swarm

**Purpose**: Full integration testing, CLI flags, optional multi-reactor swarm
for high-stakes queries.

### Deliverables

**`sia/swarm.py`** — SwarmOrchestrator (opt-in only, `--swarm N`)
- N parallel full pipeline invocations with perturbed initialization
- Each reactor runs genuine SIA deliberation
- Structural winner selection (entropy stability, constraint density, grounding, cross-validation)
- Graceful degradation: failures caught, minimum 2 for cross-validation

**`sia/reactor_coupling.py`** — cross-reactor channels
- artifact_injection, entropy_broadcast, validation_shock

**CLI: `__main__.py` additions:**
- `--swarm N` (0=disabled, 2-5 reactors)
- `--no-sia` — disable SIA, use V9 behavior

**`config.py` additions:**
```python
HAIKU_MODEL: str = "claude-haiku-4-5-20251001"
SWARM_ENABLED: bool = True
SWARM_MAX_REACTORS: int = 5
SIA_ENABLED: bool = True
```

**CLAUDE.md update:** Full V10 architecture, new graph topology, new state fields,
new tunables, new CLI flags

**pyproject.toml:** Version 0.10.0, optional `[nli]` dep group

### Tests: ~40

- Swarm: config generation, budget splitting, winner selection, cross-validation
- Coupling: artifact injection, entropy broadcast, validation shock
- CLI: --swarm parsing, --no-sia flag, cost estimate display
- Full integration: V10 pipeline end-to-end with mock callers

### Test target: ~1014 + 40 = ~1054

---

## Cost Model

### Single-Reactor V10 (typical query)

| Stage | Calls | Model | Cost/iter |
|-------|-------|-------|-----------|
| Planner | 1 | Opus | $0.13 |
| Wave loop (4 waves) | | | |
|   Wave searches + extractions | 12 | N/A | $0.00 |
|   Deliberation panels x4 | 16 | 2 Haiku + 2 Sonnet/panel | $0.10 |
|   Merge x4 | 0 | Deterministic | $0.00 |
| Compression | 10 | Haiku (summaries) | $0.004 |
| **Reactor (6 turns)** | 6 | **Sonnet** | **$0.16** |
| Outline from artifact | 1 | Opus | $0.14 |
| Section drafts | 8 | Sonnet | $0.06 |
| Composition | 1 | Haiku | $0.001 |
| **Adversarial critique (4-5 turns)** | 5 | **Sonnet** | **$0.08** |
| Entropy + convergence | 0 | Deterministic | $0.00 |
| **Total/iteration** | ~60 | — | **$0.68** |

### Projections

| Scenario | V9 | V10 Single | V10 Swarm (n=3) |
|----------|-----|------------|-----------------|
| Simple (1 iter, 2-3 waves) | $0.61 | $0.52 | $1.56 |
| Moderate (2 iter, 3-4 waves) | $1.22 | $1.04 | $3.12 |
| Complex (3 iter, 4-5 waves) | $1.82 | $1.56 | $4.68 |
| **Weighted avg (40/40/20)** | **$1.06** | **$0.87** | **$2.61** |

V10 single-reactor: ~18% cost reduction with dramatically richer output.
V10 swarm: ~2.5x cost for parallel exploration + cross-validation. Use for high-stakes.

**Where the savings come from:**
- Sonnet for section drafts instead of Opus: saves ~$0.19/iter
- Drop verify/refine stages: saves ~$0.15/iter
- Single-call composition on Haiku: saves ~$0.05/iter

**Where the cost goes:**
- Reactor deliberation: +$0.16/iter (6 Sonnet turns)
- Adversarial critique: +$0.03/iter (vs V9's 3 parallel graders)
- Deliberation panels: +$0.08/iter (4-wave coverage)

Net: intelligence layer costs +$0.27/iter, efficiency gains save -$0.39/iter.
**Intelligence pays for itself.**

---

## Invariants (never break these)

1. `synthesize()` returns the same dict shape
2. `critique()` returns the same dict shape
3. Old checkpoints without SIA/V10 fields still work
4. All 777+ existing tests pass after every phase
5. Model tiering: Opus for plan + outline, Sonnet for reactor + critique + drafts,
   Haiku for compression + composition
6. All new types in contracts.py
7. No new required pip dependencies (Haiku via existing Anthropic SDK, clustering
   via existing fastembed)
8. Any SIA failure -> graceful fallback to Phase 2 behavior (compression + 2-stage)
9. Any Phase 2 failure -> graceful fallback to V9 behavior
10. Entropy is purely deterministic — zero LLM calls

---

## Critical Files (cumulative)

| File | Change | Phase |
|------|--------|-------|
| `contracts.py` | Add ~20 TypedDicts (SIA + Tensegrity families) | 0 |
| `graph/state.py` | Add ~10 state fields | 0-4 |
| `graph/builder.py` | Wave loop, Send() panel, compress, reactor, entropy node | 2-3 |
| `agents/synthesizer.py` | Rewrite: 5-stage -> reactor + 2-stage | 2-3 |
| `agents/critic.py` | Mode switch: classic vs adversarial | 4 |
| `config.py` | HAIKU_MODEL, SIA_ENABLED, SWARM_* | 2-5 |
| `adaptive/registry.py` | 8 new tunables | 2-3 |
| `scoring/confidence.py` | entropy_gate augments convergence | 1 |
| `__main__.py` | --swarm, --no-sia flags | 5 |
| `pyproject.toml` | Version 0.10.0, [nli] dep group | 5 |
| NEW: `sia/__init__.py` | Package marker | 0 |
| NEW: `sia/agents.py` | 7 cognitive lenses | 0 |
| NEW: `sia/covenants.py` | Pairwise field physics | 0 |
| NEW: `sia/entropy.py` | Thermodynamic tracking | 1 |
| NEW: `sia/entropy_steering.py` | Tunable adjustment by band | 1 |
| NEW: `sia/kernel.py` | Speaker selection, framing, harvest | 3 |
| NEW: `sia/adversarial_critique.py` | Multi-turn evaluation | 4 |
| NEW: `sia/singularity_prevention.py` | Pre-synthesis safety | 4 |
| NEW: `sia/swarm.py` | Multi-reactor orchestration | 5 |
| NEW: `sia/reactor_coupling.py` | Cross-reactor channels | 5 |
| NEW: `deliberate/__init__.py` | Package marker | 2 |
| NEW: `deliberate/panel.py` | 4-agent deliberation panel | 2 |
| NEW: `deliberate/merge.py` | Judgment reconciliation | 2 |
| NEW: `deliberate/wave.py` | Wave orchestration | 2 |
| NEW: `deliberate/convergence.py` | Coverage-gated convergence | 2 |
| NEW: `compress/__init__.py` | Package marker | 2 |
| NEW: `compress/cluster.py` | Passage clustering | 2 |
| NEW: `compress/artifact.py` | KnowledgeArtifact builder | 2 |
| NEW: `compress/grounding.py` | NLI/embedding/Jaccard cascade | 2 |

---

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
- `backends/` — all search backends unchanged
- `memory/` — incremental research unchanged

### Redistributed (logic reused, graph edges change)
- `agents/gap_analyzer.py` — core logic reused by coverage_judge in panel
- `agents/contradiction.py` — core logic reused by contradiction_judge in panel
- V9 reactive loop (search_followup, extract_followup, score_merge) — absorbed
  into wave loop

### Rewritten
- `agents/synthesizer.py` — 5-stage -> reactor + 2-stage
- `agents/critic.py` — 3-grader -> adversarial multi-turn (with classic fallback)

---

## Streaming Events

| Event Kind | Payload | Phase |
|-----------|---------|-------|
| `entropy_computed` | e, band, components | 1 |
| `entropy_gate` | allowed, reason | 1 |
| `wave_start` | wave_num, queries | 2 |
| `panel_complete` | judge, summary | 2 |
| `merge_complete` | coverage, risks | 2 |
| `wave_converged` | wave_count, reason | 2 |
| `compression_complete` | clusters, claims, ratio | 2 |
| `reactor_turn` | agent, int_type, constraints | 3 |
| `reactor_complete` | turns, constraints, entropy | 3 |
| `adversarial_finding` | agent, severity, finding | 4 |
| `singularity_check` | safe, axes | 4 |
| `swarm_reactor_complete` | reactor_id, entropy, cost | 5 |
| `swarm_winner` | reactor_id, reason, scores | 5 |

---

## Dependency Strategy

### Tier 0: Zero new dependencies (Phases 0-5)
- Haiku: Anthropic SDK already installed, add model ID
- Fastembed: already in `[embeddings]` optional dep
- Send() API: LangGraph already installed
- All SIA logic: pure Python + existing callers

### Tier 1: Optional (`[nli]` dep group)
- `sentence-transformers[onnx]` + `cross-encoder/nli-deberta-v3-xsmall`
- Upgrades claim verification + contradiction detection
- Not required — graceful degradation

---

## Verification Protocol

After each phase:
```bash
.venv/Scripts/python.exe -m pytest tests/ -v
.venv/Scripts/python.exe -m ruff check . && .venv/Scripts/python.exe -m ruff format --check .
```

After Phase 3 (reactor + compression):
```bash
.venv/Scripts/python.exe -m deep_research_swarm \
  "astrology sources for professional astrologers" --verbose
# Compare: cost, time, quality, streaming output
```

After Phase 5 (full V10):
```bash
# Single reactor
.venv/Scripts/python.exe -m deep_research_swarm \
  "astrology sources for professional astrologers" --verbose
# Target: richer output (tensions acknowledged, coverage gaps noted,
#   multi-perspective deliberation, reactor constraints visible)

# Swarm mode
.venv/Scripts/python.exe -m deep_research_swarm \
  "astrology sources for professional astrologers" --swarm 3 --verbose
# Target: cross-validated synthesis, winner selection visible

# V9 fallback
.venv/Scripts/python.exe -m deep_research_swarm \
  "astrology sources for professional astrologers" --no-sia --verbose
# Target: identical to V9 behavior
```

---

## Research Sources

### SIA Framework
- SIAFRESH.txt — Comprehensive agent profiles, execution framework, entropy dynamics
- Inter-Agent-Covenants — Pairwise field physics, covenant schema
- Entropy-Steering-Spec — Thermodynamic bands, steering primitives, stagnation detection
- Latent Basin Map — 7-axis semantic deformation topology
- Reactor Initialization Doctrine — Ignition patterns, first speaker selection
- Multi-Reactor Swarm Architecture — Parallel reactor types, coupling dynamics

### Intelligence Architecture (arXiv)
- IRCoT: arXiv:2212.10509 — Interleaved retrieval + chain-of-thought
- STORM: arXiv:2402.14207 — Multi-perspective questioning (Stanford)
- Co-STORM: arXiv:2408.15232 — Collaborative discourse with convergence interrupts
- WebThinker: arXiv:2503.01428 — Think-while-searching (NeurIPS 2025)
- Multi-Agent Debate: arXiv:2305.14325 — 3 agents, 2 rounds optimal (Du et al.)
- Google Mass: arXiv:2502.02533 — Optimized prompts + topology beats naive multi-agent

### Compression & Grounding (arXiv)
- CompactRAG: arXiv:2602.05728 — 81% token reduction
- RAPTOR: arXiv:2401.18059 — Recursive summarization trees
- MiniCheck: arXiv:2404.10774 — 770M fact-checker
- StructRAG: arXiv:2410.08815 — Adaptive structure selection

### HuggingFace Models (verified via MCP)
- `BAAI/bge-small-en-v1.5` — 33.4M, MIT, ONNX (already in deps)
- `cross-encoder/nli-deberta-v3-xsmall` — 70.8M, Apache-2.0, ONNX, CPU-viable

### Implementation Patterns
- LangGraph Send() API — dynamic parallel fan-out
- LangGraph Map-Reduce — parallel execution + merge
- Du et al. multi-agent debate — 3-4 agents optimal, 2-3 rounds to convergence
