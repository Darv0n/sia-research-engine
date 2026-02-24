# PR-SIA-01: Entropy Layer — Thermodynamic Control Foundation

## Context

You are working on `deep-research-swarm` (V9, 777+ tests, LangGraph StateGraph). The current convergence system uses confidence-delta thresholds — a scalar check that cannot detect false convergence, dominance collapse, or structural stagnation. You are installing the entropy layer from the SIA (Semantic Intelligence Architecture) reactor specification. This is the foundation that all subsequent SIA phases build on.

Read these files before writing any code:
- `CLAUDE.md` (repo architecture, conventions, state fields)
- `deep_research_swarm/graph/state.py` (ResearchState, reducers)
- `deep_research_swarm/scoring/confidence.py` (current convergence logic)
- `deep_research_swarm/agents/critic.py` (where convergence decisions happen)
- `deep_research_swarm/adaptive/registry.py` (TunableRegistry pattern)
- `docs/sia/Entropy-Steering-Spec.md` (the thermodynamic spec you're implementing)
- `docs/sia/KernelControlLoops.md` (control loop architecture)

## Deliverables

### 1. New package: `deep_research_swarm/sia/__init__.py`

Empty init. This is the SIA package root.

### 2. New type in `contracts.py`

```python
class EntropyState(TypedDict):
    e: float                    # scalar entropy ∈ [0, 1]
    e_amb: float                # ambiguity component
    e_conf: float               # conflict component
    e_nov: float                # novelty component
    e_trust: float              # trust/coherence component
    band: str                   # "crystalline" | "convergence" | "turbulence" | "runaway"
    turn: int                   # which pipeline turn this was computed at
    stagnation_count: int       # consecutive turns with |Δe| < 0.03
```

### 3. New module: `deep_research_swarm/sia/entropy.py`

Pure computation — zero LLM calls. All functions operate on pipeline observables already present in `ResearchState`.

**Functions:**

`classify_band(e: float) -> str`
- Ω: e ≤ 0.20 → `"crystalline"` (synthesis zone)
- α: 0.20 < e ≤ 0.45 → `"convergence"` (productive compression)
- β: 0.45 < e ≤ 0.70 → `"turbulence"` (creative collision)
- δ: e > 0.70 → `"runaway"` (coherence loss)

`compute_entropy(state: ResearchState, prev_entropy: EntropyState | None = None) -> EntropyState`

Signal extraction from existing state fields:
- **e_amb** (ambiguity): proportion of sections with MEDIUM confidence + (unresolved `research_gaps` count / total sections). High when the pipeline knows it doesn't know things.
- **e_conf** (conflict): `len(contradictions)` / `len(scored_documents)` + proportion of sections where grader sub-scores diverge > 0.2 from each other. High when sources disagree.
- **e_nov** (novelty): (new `sub_queries` this iteration) / (total `sub_queries`) + heading delta between iterations (how much the outline changed). High when the research direction is shifting.
- **e_trust** (coherence): variance of `grounding_score` across sections + proportion of citations with no passage mapping in `citation_to_passage_map`. High when the grounding is uneven.

Scalar projection: `e = 0.30 * e_amb + 0.30 * e_conf + 0.20 * e_nov + 0.20 * e_trust`

Stagnation: if `|e - prev_e| < 0.03`, increment `stagnation_count`; else reset to 0.

`entropy_gate(entropy: EntropyState, section_drafts: list) -> tuple[bool, str]`

Synthesis ALLOWED when:
- band == `"crystalline"` (e ≤ 0.20), OR
- band == `"convergence"` AND `stagnation_count >= 3` (cycling with diminishing returns)

Synthesis BLOCKED when:
- band == `"runaway"` (always)
- band == `"turbulence"` AND `stagnation_count < 2`

Returns `(should_converge, reason)`.

`detect_false_convergence(entropy: EntropyState, section_drafts: list, contradictions: list) -> bool`

True when:
- e < 0.30 but `len(contradictions) > 0`
- e < 0.30 but any section's `grounding_score < 0.5`
- e dropping > 0.15/turn without constraint count increasing

`detect_dominance(iteration_history: list, section_drafts: list) -> tuple[bool, str]`

True when:
- All section confidence scores have variance < 0.01 (suspiciously uniform)
- All three grader dimensions within 0.05 of each other across all sections

### 4. New state fields in `graph/state.py`

```python
# SIA entropy layer (V10)
entropy_state: Annotated[dict, _replace_dict]
entropy_history: Annotated[list[dict], operator.add]
```

### 5. New graph node: `compute_entropy`

Wire after `critique`, before convergence routing:

```
... → critique → compute_entropy → rollup_budget → [converge?] → ...
```

The node calls `compute_entropy()` and writes to state. The convergence check now combines:
- Confidence logic (existing) AND entropy gate AND false convergence check AND dominance check
- Logic: `converged = confidence_ok AND entropy_allows AND NOT false_convergence AND NOT dominance`
- Any veto blocks convergence. Entropy gate can also FORCE convergence when band is crystalline regardless of confidence.

### 6. Optional node: `entropy_steering`

Runs after `compute_entropy`. Adjusts existing adaptive tunables based on entropy band:
- **Runaway**: reduce `perspectives_count`, increase `convergence_threshold`, zero `follow_up_budget`
- **Turbulence**: reduce `target_queries` slightly
- **Convergence**: allow deeper drafts
- **Crystalline**: no-op

Integrates with existing `TunableRegistry`.

### 7. Tests: `tests/test_entropy.py` — minimum 25 tests

- `classify_band()`: all four bands + exact boundary values (0.20, 0.45, 0.70)
- `compute_entropy()`: synthetic states with known signal levels producing expected component values
- `entropy_gate()`: all convergence/block conditions
- `detect_false_convergence()`: low-e-with-contradictions, low-e-with-bad-grounding, rapid-drop
- `detect_dominance()`: uniform scores, divergent scores
- Stagnation counter: increment, reset, boundary

### 8. Tests: `tests/test_entropy_integration.py` — minimum 10 tests

- Graph compiles with new node
- Node appears in correct position (after critique, before rollup_budget)
- Convergence routing respects entropy veto
- State fields present and correctly typed
- Backward compat: missing `entropy_state` → existing convergence logic takes over

## Constraints

- **Non-breaking**: All 777+ existing tests pass unchanged
- **No LLM calls**: Entropy is purely deterministic from state observables
- **Backward compatible**: Missing `entropy_state` (old checkpoints) → safe defaults, existing logic takes over
- **Contracts-first**: `EntropyState` goes in `contracts.py`
- **Reducer discipline**: `entropy_state` = replace-last-write, `entropy_history` = accumulating

## After completion

Update `CLAUDE.md`: V10-alpha (SIA Phase 1: Entropy Layer)
Update `CHANGELOG.md`: `[0.10.0-alpha]`
Test count: 777 + ~35 = ~812
