# PR-SIA-04: Multi-Reactor Swarm — Evolutionary Semantic Competition

## Context

You are working on `deep-research-swarm` with SIA Phases 1-3 installed: entropy layer, true multi-turn agent reactor, and adversarial critique with singularity prevention. The system now has genuine multi-agent deliberation — agents that see and respond to each other in shared conversation threads, producing richer synthesis and structural critique. But it still runs ONE path. One plan, one reactor, one collapse basin. Path dependence means the first framing that stabilizes wins.

This phase installs multi-reactor swarm mode: N parallel research pipelines with perturbed initialization, competing to produce the best synthesis artifact.

Read these files before writing any code:
- `CLAUDE.md` (current state after Phase 3)
- `docs/sia/MultiReactorSwarmArchitecture.md` (swarm spec)
- `docs/sia/CrossReactorEntropyCoupling.md` (coupling dynamics)
- `deep_research_swarm/graph/builder.py` (graph construction)
- `deep_research_swarm/__main__.py` (CLI entry)
- `deep_research_swarm/sia/entropy.py` (entropy system)
- `deep_research_swarm/sia/kernel.py` (reactor kernel)

## Core Architecture

The swarm does NOT modify the graph topology. Each reactor runs the SAME compiled graph with different initial state perturbations. The SwarmOrchestrator sits ABOVE the graph:

```
                    SwarmOrchestrator
                   /       |        \
                  R₁       R₂       R₃
                  │        │        │
           [full pipeline] [full pipeline] [full pipeline]
                  │        │        │
           synthesis₁   synthesis₂   synthesis₃
           (true agent   (true agent   (true agent
            reactor)      reactor)      reactor)
                  \       │        /
                 select_winner()
                       │
                  final_report
```

Each reactor has its own true-agent reactor deliberation with different ignition patterns, different agent weighting, and different perspective biases. The deliberation is REAL in every reactor — genuine multi-turn conversation, not shortened or faked.

## Deliverables

### 1. New module: `deep_research_swarm/sia/swarm.py`

```python
@dataclass
class ReactorConfig:
    """Configuration for one reactor instance."""
    reactor_id: str
    reactor_type: str              # "strategic" | "exploratory" | "formal" | "minimal"
    entropy_seed: float            # initial entropy (perturbed around 0.85)
    ignition_pattern: str          # "A" | "B" | "C" | "D" (from InitializationDoctrine)
    agent_weight_overrides: dict[str, float]  # bias specific agents' selection probability
    perspective_bias: str | None   # force a starting perspective for the planner
    token_budget: int              # per-reactor budget


class SwarmOrchestrator:
    """Manages N parallel reactor instances and selects best synthesis."""
    
    def __init__(self, n_reactors: int = 3, settings: Settings = None):
        self.n_reactors = min(n_reactors, 5)  # hard cap
        self.settings = settings
        self.configs = self._generate_configs()
    
    def _generate_configs(self) -> list[ReactorConfig]:
        """Generate perturbed configurations.
        
        Type distribution:
          n=2: 1 strategic + 1 exploratory
          n=3: 1 strategic + 1 exploratory + 1 formal
          n=4: + 1 minimal
          n=5: + 1 strategic (different ignition)
        
        Perturbation:
          entropy_seed: 0.85 ± uniform(0.05)
          ignition_pattern: rotated across reactors (A, B, C, D)
          agent_weight_overrides:
            strategic: Light +0.2, Lawliet +0.1
            exploratory: Rick +0.2, Makishima +0.1
            formal: Shiro +0.2, Lawliet +0.1
            minimal: Shikamaru +0.2
          perspective_bias: each reactor gets a different starting perspective
          token_budget: total_budget / n * 1.1 (10% overhead for selection)
        """
    
    async def run_swarm(
        self,
        base_state: ResearchState,
        build_graph_fn: callable,
        settings: Settings,
        *,
        event_log=None,
        mode: str = "auto",
    ) -> tuple[dict, SwarmMetadata]:
        """Run N reactors in parallel and select best synthesis.
        
        1. For each reactor config:
           a. Clone base_state
           b. Apply perturbations (entropy_seed, perspective_bias, agent weights)
              - Store ignition_pattern in state for Kernel to read
              - Store agent_weight_overrides in tunable_snapshot
           c. Build fresh graph instance
           d. Invoke graph with perturbed state
        
        2. Run all reactors concurrently: asyncio.gather(*reactor_tasks)
           - Each reactor is a full pipeline invocation
           - Each has its own true-agent reactor deliberation
           - Failed reactors are caught and logged, not fatal
        
        3. Collect results from successful reactors
           - Minimum 2 successful for cross-validation
           - If only 1 succeeds, use it directly (no cross-validation)
           - If 0 succeed, fall back to single non-swarm run
        
        4. Select winner via structural comparison
        
        5. Optionally: cross-validate winner against challenger findings
        
        Returns (winning_result, swarm_metadata)
        """
    
    def select_winner(self, results: list[dict]) -> tuple[dict, str]:
        """Select best synthesis artifact. STRUCTURAL, not aesthetic.
        
        Scoring (each 0-1, weighted):
          0.30 — Entropy stability: lower final entropy = better
          0.25 — Constraint density: more explicit constraints = better
          0.20 — Grounding coverage: higher avg grounding_score = better
          0.15 — Cross-validation: claims that appear in 2+ reactors = bonus
          0.10 — Efficiency: useful output / tokens spent
        
        Returns (winning_result, selection_reason)
        """
    
    def cross_validate(self, results: list[dict]) -> dict[str, float]:
        """Cross-validate claims across reactors.
        
        For each section heading appearing in 2+ reactors:
        - Compare claims using Jaccard similarity on claim text
        - Claims independently produced by multiple reactors get bonus
        - Claims unique to one reactor get flagged
        
        Returns section_id -> cross_validation_score
        """
```

### 2. New module: `deep_research_swarm/sia/reactor_coupling.py`

Cross-reactor communication. Used optionally between rounds if implementing iterative swarm.

```python
class CouplingChannel:
    @staticmethod
    def artifact_injection(source_result: dict, target_state: dict) -> dict:
        """Inject constraints from one reactor as context in another.
        Not as fixed constraints — as 'another analysis found these.'"""
    
    @staticmethod
    def entropy_broadcast(entropies: list[dict]) -> list[float]:
        """Compute adjustment signals from peer reactor entropy states.
        If peer collapsed → encourage convergence. If peer stagnant → encourage exploration."""
    
    @staticmethod
    def validation_shock(winner: dict, challenger: dict) -> list[str]:
        """Find claims in challenger that contradict winner.
        If winner survives → high confidence. If not → merge insights."""
```

### 3. New types in `contracts.py`

```python
class SwarmMetadata(TypedDict):
    n_reactors: int
    reactor_configs: list[dict]
    reactor_entropies: list[float]
    reactor_tokens: list[int]
    reactor_costs: list[float]
    winner_id: str
    selection_reason: str
    selection_scores: dict[str, float]    # per-reactor composite scores
    cross_validation_scores: dict[str, float]
    total_tokens_all: int
    total_cost_all: float
    failed_reactors: list[str]
```

### 4. New state fields in `graph/state.py`

```python
swarm_metadata: Annotated[dict, _replace_dict]
swarm_mode: Annotated[bool, _replace_bool]
```

### 5. CLI integration in `__main__.py`

```python
parser.add_argument(
    "--swarm", type=int, default=0, metavar="N",
    help="Run N parallel reactors and select best synthesis (2-5, 0=disabled)"
)
```

When `--swarm N` (N >= 2):
1. Print estimated cost: `Cost estimate: ~{N}x base cost ($X-Y)`
2. Build SwarmOrchestrator
3. Run `orchestrator.run_swarm()` instead of single `graph.ainvoke()`
4. Use winning result for report generation
5. Print swarm metadata: which reactor won, why, how many failed

When `--swarm 0` (default): existing behavior unchanged.

### 6. Config in `config.py`

```python
SWARM_ENABLED: bool = True           # allow swarm mode
SWARM_MAX_REACTORS: int = 5          # hard ceiling
SWARM_COUPLING_STRENGTH: float = 0.15  # cross-reactor influence weight
```

### 7. Budget management

Each reactor gets `token_budget / n_reactors * 1.1`.

Early termination: if a reactor's entropy stays in runaway band (δ) for 3+ turns in the critique loop, terminate it early. Don't waste budget on a failing reactor.

In swarm mode, default `max_iterations=2` per reactor (instead of 3). Two iterations with true-agent deliberation produce better synthesis than three iterations without.

### 8. Tests: `tests/test_swarm.py` — minimum 22 tests

- Orchestrator initializes with correct reactor count
- Configs are properly perturbed (different entropy seeds, ignition patterns, weights)
- Config generation respects n=2..5 type distributions
- Budget splitting is correct (total/n * 1.1 per reactor)
- `select_winner()` picks highest composite score
- `select_winner()` handles tie-breaking
- `cross_validate()` detects shared claims (Jaccard)
- `cross_validate()` flags unique claims
- Early termination for runaway reactors
- Graceful degradation: 1 reactor fails → use remaining
- Graceful degradation: all fail → single non-swarm fallback
- SwarmMetadata fully populated

### 9. Tests: `tests/test_reactor_coupling.py` — minimum 10 tests

- Artifact injection adds context without overwriting state
- Entropy broadcast produces bounded adjustments
- Validation shock finds contradicting claims
- Coupling strength is bounded (no cascade instability)

### 10. Tests: `tests/test_swarm_integration.py` — minimum 10 tests

- CLI `--swarm 3` parsed correctly
- `--swarm 0` → no swarm imports triggered
- `--swarm 6` → clamped to 5
- Cost estimate printed before execution
- Swarm metadata in final state
- Report generation works with swarm winner

## Constraints

- **Opt-in only**: `--swarm 0` is default. Swarm never runs unless explicitly requested.
- **No graph topology changes**: Swarm wraps the graph, doesn't modify it
- **Cost transparency**: Estimated cost printed before execution
- **Bounded**: Maximum 5 reactors (diminishing returns beyond that)
- **Deterministic selection**: Winner chosen by structural metrics, not LLM judge
- **True agents in every reactor**: Each reactor runs genuine multi-turn deliberation. No shortcuts.
- **All ~939 existing tests pass**

## After completion

Update `CLAUDE.md`: V10 (SIA Complete)
Update `CHANGELOG.md`: full `[0.10.0]` release notes
Update `README.md`:
- Architecture section: swarm diagram
- Features: SIA bullets (true multi-agent deliberation, entropy-gated convergence, adversarial critique, multi-reactor swarm)
- CLI: `--swarm N` documentation
- Config: new env vars
Test count: ~939 + ~42 = ~981
