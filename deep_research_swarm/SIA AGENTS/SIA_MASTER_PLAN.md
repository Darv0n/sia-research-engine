# SIA Integration Master Plan — deep-research-swarm V10

## What This Is

Four phased PRs that convert deep-research-swarm from a sequential research pipeline
into a governed semantic fusion reactor with true multi-agent deliberation
and evolutionary pressure.

**"True agent" means**: persistent cognitive stances with multi-turn conversation memory
that see, respond to, and disagree with each other. NOT system prompt swaps.
NOT single-shot role-labeled calls. Genuine structured friction between distinct
processing biases operating on the same research material.

Each phase is independently shippable, backward compatible, and testable.
Execute in strict order.

---

## Execution Order

```
PR-SIA-01  Entropy Layer              ← FOUNDATION
    │
    ▼
PR-SIA-02  True Agent Reactor         ← CORE (true multi-turn deliberation)
    │
    ▼
PR-SIA-03  Adversarial Critique       ← RISK (true multi-turn evaluation)
    │
    ▼
PR-SIA-04  Multi-Reactor Swarm        ← EVOLUTION (parallel true-agent reactors)
```

---

## The Architectural Truth

What makes SIA real vs theater:

```
┌─ THEATER (what we rejected) ────────────────────────────────────────────────┐
│                                                                              │
│  for turn in range(N):                                                      │
│      agent = pick_next()                                                    │
│      response = caller.call(system=agent.prompt, messages=[one_shot_input]) │
│      # Agent has no memory. No response to prior agents. Costume change.   │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘

┌─ REAL (what we're building) ────────────────────────────────────────────────┐
│                                                                              │
│  conversation: list[dict] = []                                              │
│                                                                              │
│  for turn in range(N):                                                      │
│      agent = kernel.select_speaker(entropy)                                 │
│      frame = kernel.frame_turn(agent, context, prior_summary)               │
│                                                                              │
│      messages = conversation + [{"role": "user", "content": frame}]         │
│      response = agent_caller.call(                                          │
│          system=agent.cognitive_lens,  # processing bias, not role-play     │
│          messages=messages,            # SEES EVERYTHING PRIOR              │
│      )                                                                       │
│                                                                              │
│      conversation.append(frame_msg)                                         │
│      conversation.append(response_msg)                                      │
│      # Next agent inherits full thread. Can agree, disagree, build on.     │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

The conversation thread IS the reactor. Each agent's system prompt is a cognitive lens
that makes the model ATTEND to different features of the same input. Same Sonnet model,
different processing bias, growing shared context. This produces genuine disagreement,
genuine constraint extraction, genuine adversarial pressure.

---

## New Files (cumulative)

```
deep_research_swarm/
└── sia/
    ├── __init__.py                    # Phase 1
    ├── entropy.py                     # Phase 1: thermodynamic tracking
    ├── agents.py                      # Phase 2: 7 cognitive lenses
    ├── covenants.py                   # Phase 2: pairwise field physics
    ├── kernel.py                      # Phase 2: thermodynamic steering
    ├── adversarial_critique.py        # Phase 3: multi-turn evaluation
    ├── singularity_prevention.py      # Phase 3: pre-synthesis safety
    ├── swarm.py                       # Phase 4: multi-reactor orchestration
    └── reactor_coupling.py            # Phase 4: cross-reactor channels

tests/
├── test_entropy.py                    # Phase 1: ~25 tests
├── test_entropy_integration.py        # Phase 1: ~10 tests
├── test_sia_agents.py                 # Phase 2: ~20 tests
├── test_sia_kernel.py                 # Phase 2: ~30 tests
├── test_sia_covenants.py              # Phase 2: ~12 tests
├── test_synthesizer_reactor.py        # Phase 2: ~18 tests
├── test_adversarial_critique.py       # Phase 3: ~22 tests
├── test_singularity_prevention.py     # Phase 3: ~15 tests
├── test_convergence_integration.py    # Phase 3: ~10 tests
├── test_swarm.py                      # Phase 4: ~22 tests
├── test_reactor_coupling.py           # Phase 4: ~10 tests
└── test_swarm_integration.py          # Phase 4: ~10 tests
```

Estimated new tests: ~204
Target total: 777 + 204 = ~981

---

## Modified Files (per phase)

### Phase 1 (Entropy Layer)
- `contracts.py` → add EntropyState
- `graph/state.py` → add entropy_state, entropy_history
- `graph/builder.py` → add compute_entropy node after critique
- `scoring/confidence.py` → entropy_gate augments convergence

### Phase 2 (True Agent Reactor)
- `contracts.py` → add TurnRecord, ReactorState, ReactorTrace
- `graph/state.py` → add reactor_trace
- `agents/synthesizer.py` → reactor loop wraps existing 5-stage pipeline
- `adaptive/registry.py` → add sia_reactor_turns, sia_reactor_budget

### Phase 3 (Adversarial Critique)
- `contracts.py` → add AdversarialFinding, CritiqueTrace
- `graph/state.py` → add adversarial_findings, critique_trace
- `agents/critic.py` → mode switch: classic vs adversarial
- `graph/builder.py` → convergence adds singularity check

### Phase 4 (Swarm)
- `contracts.py` → add SwarmMetadata
- `graph/state.py` → add swarm_metadata, swarm_mode
- `__main__.py` → add --swarm CLI flag
- `config.py` → add SWARM_* env vars

---

## Invariants (never break these)

1. `synthesize()` returns the same dict shape
2. `critique()` returns the same dict shape
3. Old checkpoints without SIA fields still work
4. All 777+ existing tests pass after every phase
5. Opus for synthesis drafting, Sonnet for reactor + critique
6. All new types in contracts.py
7. No new pip dependencies
8. Any SIA failure → graceful fallback to V9 behavior

---

## GCROE Mapping

```
Phase 1 (Entropy)     = GOVERNANCE    explicit system laws replace implicit heuristics
Phase 2 (Reactor)     = CONTROL       true agent friction replaces single-voice generation
Phase 3 (Critique)    = RISK          adversarial evaluation catches structural failure
Phase 4 (Swarm)       = EVOLUTION     parallel collapse basins prevent path dependence

OPTIMIZATION is distributed across all four: entropy reduces waste, reactor
improves depth, critique catches failures, swarm prevents local minima.
```

---

## Cost Model

```
                     V9 (current)    V10 single-reactor    V10 swarm (n=3)
Planner (Opus)       ~$0.05          ~$0.05                ~$0.15
Search/Extract       ~$0.00          ~$0.00                ~$0.00
Reactor (Sonnet)     —               ~$0.16                ~$0.48
Synthesis (Opus)     ~$0.30          ~$0.30                ~$0.90
Critique (Sonnet)    ~$0.05          ~$0.08                ~$0.24
Entropy/Steering     ~$0.00          ~$0.00                ~$0.00
Selection            —               —                     ~$0.00
─────────────────────────────────────────────────────────────────────
TOTAL               ~$0.40           ~$0.59                ~$1.77

Cost multiplier:     1.0x             1.5x                  4.4x
```

Single-reactor SIA: 50% more for substantially richer synthesis.
Swarm (n=3): 4.4x for best-of-three with cross-validation. Use for high-stakes.

---

## Pre-Execution Checklist

```
□ git status clean
□ All tests pass
□ Lint clean
□ Read CLAUDE.md
□ Read the specific PR-SIA prompt completely
□ Read all referenced SIA spec docs
□ Identify files to modify vs create
□ Plan commit sequence: types → modules → tests → wiring
```

## Post-Execution Checklist

```
□ All tests pass (existing + new)
□ Lint clean
□ CLAUDE.md updated
□ CHANGELOG.md updated
□ Commit: feat(sia): Phase N — description
□ Verify graceful degradation (SIA disabled → V9 behavior)
```

---

## The One-Sentence Summary

We are installing true multi-agent deliberation — persistent cognitive stances
that see, respond to, and disagree with each other in shared conversation threads —
on top of a high-performance research engine, governed by thermodynamic control,
validated by adversarial critique, and evolved through multi-reactor competition.
