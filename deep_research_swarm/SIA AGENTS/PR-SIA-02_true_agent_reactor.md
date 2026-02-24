# PR-SIA-02: True Agent Reactor — Multi-Turn Deliberation Core

## Context

You are working on `deep-research-swarm` with SIA Phase 1 (entropy layer) installed. The synthesizer is still a single-voice 5-stage sub-pipeline. This phase installs TRUE independent agents — not system prompt swaps, but persistent cognitive stances with multi-turn conversation memory that see, respond to, and disagree with each other.

Read these files before writing any code:
- `CLAUDE.md` (current state after Phase 1)
- `docs/sia/Inter-Agent-Covenants.md` (pairwise interaction physics)
- `docs/sia/LatentBasinMap.md` (agent deformation profiles — study these deeply)
- `docs/sia/Entropy-Steering-Spec.md` (steering by pairing, intervention primitives)
- `docs/sia/ReactorInitializationDoctrine.md` (ignition patterns, first speaker doctrine)
- `docs/sia/KernelControlLoops.md` (three-layer control model)
- `deep_research_swarm/agents/synthesizer.py` (current 5-stage pipeline you're augmenting)
- `deep_research_swarm/agents/base.py` (AgentCaller — you will use this, not replace it)
- `deep_research_swarm/sia/entropy.py` (Phase 1 entropy system)

## The Core Design Decision

**What makes an agent "true" vs "cosplay":**

| Cosplay Agent | True Agent |
|---|---|
| System prompt swap on shared caller | Own persistent conversation thread |
| No memory between turns | Sees everything prior agents said |
| Kernel scripts what agent says | Agent decides what to say given its stance |
| No response to other agents | Directly responds to, challenges, builds on other agents |
| Role-labeled single-shot call | Cognitive bias that makes the model PROCESS differently |

**The architecture**: A shared multi-turn conversation thread where each agent contributes from its own cognitive stance, seeing and responding to everything that came before. The Kernel frames each turn — telling the next agent what just happened and what the reactor needs — but the agent decides HOW to respond from its stance.

**The mechanism**: Each SIA agent gets its own `AgentCaller` instance configured with Sonnet and its own system prompt. The system prompt is NOT an instruction to role-play. It is a cognitive lens — a set of processing biases that make the model attend to different features of the same input. Lawliet's prompt makes the model SEE constraint gaps. Rick's prompt makes the model SEE hidden assumptions. They process the same research material and prior conversation, but their outputs are structurally different because their attention is shaped differently.

## Deliverables

### 1. New module: `deep_research_swarm/sia/agents.py`

Seven agent definitions. Each is a frozen dataclass containing everything needed to instantiate a true agent.

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class SIAAgent:
    """A cognitive stance that deforms semantic space."""
    name: str                        # lowercase identifier
    role: str                        # one-line description
    system_prompt: str               # cognitive lens (NOT role-play instruction)
    basin_profile: dict[str, float]  # axis deformation signature {axis: weight}
    preferred_int_types: list[str]   # what this agent naturally produces
    repels: list[str]                # what this agent pushes against
    instability_signature: str       # how this agent breaks the system if unchecked
```

**System prompt design principle — THIS IS CRITICAL:**

Each system prompt must:
1. Define a PROCESSING BIAS, not a character to play
2. Tell the agent what to ATTEND TO in the research material and prior conversation
3. Tell the agent what to PRODUCE (structured output with explicit IntType)
4. NOT tell the agent what to "say" — let the bias shape the response

Example (Lawliet — Constraint Singularity):
```
You are a constraint extraction engine. When you read research material and 
prior analysis, you see GAPS — claims without evidence, assumptions without 
premises, directions without justification.

Your cognitive bias: you cannot let a vague claim pass. Every assertion must 
be traceable to a source or explicitly marked as unsupported. Every direction 
must have stated constraints that bound it.

You are reading a multi-agent research deliberation. Other analysts have 
contributed before you. Some of their claims are grounded. Some are not. 
Your job is to:

1. Identify the 3-5 most important UNRESOLVED VARIABLES in the current state
2. For each: state what is assumed but not proven
3. Extract HARD CONSTRAINTS that the evidence actually supports
4. Name what MUST be true for the current direction to hold
5. Flag any claim that sounds confident but lacks grounding

Output format:
[CL] (Clarification) or [B] (Build) — label your contribution.

CONSTRAINTS_EXTRACTED:
- constraint_1: "..."
- constraint_2: "..."

UNRESOLVED_VARIABLES:
- variable_1: "what we assume but haven't verified: ..."

RESPONSE_TO_PRIOR:
- direct engagement with specific claims from prior turns

Do not summarize. Do not agree politely. Extract structure from noise.
```

Each of the seven agents needs this level of specificity. The prompts are parameterized with:
- `{research_question}` — the user's query
- `{source_summary}` — compressed summary of scored documents and passages
- `{contradiction_summary}` — detected contradictions
- `{prior_conversation}` — the full reactor conversation thread so far
- `{kernel_frame}` — the Kernel's context for this turn (what just happened, what's needed)
- `{entropy_band}` — current thermodynamic state

**The Seven Agents and Their Basin Profiles:**

**LAWLIET** — Constraint Singularity
- Primary deformation: ↑A₁ (constraint density), ↓A₅ variance (frame stabilization)
- Pulls: vague definitions → constraints, loose claims → premises
- Repels: moral abstractions without premises, reframes without anchor
- Instability: if A₁ rises too fast while A₂/A₃ unresolved → brittle convergence

**LIGHT** — Directional Gradient Field
- Primary deformation: ↑A₂ (directionality), ↑A₆ (coalition toward his axis)
- Pulls: decisions, commitment, strategic framing
- Repels: perpetual exploration, non-committal neutrality
- Instability: if A₂ high while A₁ low → authoritarian drift

**RICK** — Frame Instability Spike
- Primary deformation: ↑A₅ volatility, ↓local stability temporarily
- Pulls: hidden assumptions, stagnant patterns
- Repels: complacency, overfitted structure
- Instability: repeated spikes without A₁ anchoring → entropy runaway

**SHIRO** — Formal Structure Basin
- Primary deformation: ↑A₄ (formalism), moderate ↑A₁
- Pulls: rule ambiguity, exploit surfaces, competitive dynamics
- Repels: vague power structures, undefined incentives
- Instability: if A₄ rises while A₂ unresolved → elegant but directionless

**SHIKAMARU** — Energy Minimization Basin
- Primary deformation: ↓A₇ (energy expenditure), ↓branch proliferation
- Pulls: redundant branches, overcomplex structures
- Repels: ornamental debate, unnecessary optimization
- Instability: smoothing before exploration → premature convergence

**MAKISHIMA** — Ethical Faultline Basin
- Primary deformation: ↑A₃ (ethical charge), destabilizes unexamined directionality
- Pulls: value assumptions, legitimacy claims
- Repels: blind utilitarianism, consensus comfort
- Instability: if A₃ dominates without A₁ compression → philosophical drift

**JOHAN** — Alignment Gravity Drift
- Primary deformation: ↑A₆ (coalition cohesion, subtly), manipulates A₂ without visible force
- Pulls: motivation, narrative cohesion
- Repels: overt confrontation, transparent power plays
- Instability: cohesion without explicit constraint mapping → hidden divergence

### 2. New module: `deep_research_swarm/sia/covenants.py`

Pairwise interaction rules as data. These are NOT personality descriptions — they are field physics that the Kernel uses for scheduling.

```python
@dataclass(frozen=True)
class Covenant:
    agent_a: str                    # first agent name
    agent_b: str                    # second agent name
    coupling: str                   # nature of the tension
    yield_pattern: str              # what productive friction produces
    failure_mode: str               # how it collapses into noise
    preferred_sequence: list[str]   # IntType sequence: ["CL", "B", "B", "A"]
    escalation_trigger: str         # when Kernel must intervene
    max_consecutive_without_anchor: int  # how many turns before forced B

# All pairs from Inter-Agent-Covenants.md
COVENANTS: dict[tuple[str, str], Covenant] = { ... }

# Lookup function that works regardless of argument order
def get_covenant(a: str, b: str) -> Covenant | None: ...
```

Include every covenant pair from the spec: Lawliet↔Light, Lawliet↔Rick, Light↔Makishima, Johan↔Coalition, Shiro↔Rick, Shikamaru↔Everyone. Plus the high-risk triads as separate data.

### 3. New module: `deep_research_swarm/sia/kernel.py`

The Kernel is the thermodynamic steering engine. It does NOT decide what agents say. It decides WHO speaks next and FRAMES the context for their turn.

```python
class SIAKernel:
    def __init__(
        self,
        agents: dict[str, SIAAgent],
        covenants: dict[tuple[str, str], Covenant],
        entropy: EntropyState,
    ):
        self.agents = agents
        self.covenants = covenants
        self.entropy = entropy
        self.turn_history: list[TurnRecord] = []
        self.speaker_counts: dict[str, int] = defaultdict(int)
        self.consecutive_speakers: list[str] = []
        self.fatigue: dict[str, float] = {name: 0.0 for name in agents}
        self.constraints_accumulated: list[str] = []
        self.branches_killed: list[str] = []
        self.last_int_type: str | None = None
        self.rf_count_since_anchor: int = 0
        self.c_count_since_build: int = 0

    def select_speaker(self, turn: int, max_turns: int) -> tuple[SIAAgent, str]:
        """Select next speaker and suggest IntType based on entropy band + history.
        
        IGNITION DOCTRINE (turns 0-2):
        Never start with a cooling sink. Use initialization patterns:
          Pattern A: Rick (RF) → Lawliet (CL/B)
          Pattern B: Light (B) → Makishima (C)
          Pattern C: Shiro (B) → Rick (RF)
          Pattern D: Lawliet (CL) → Light (B)
        Select pattern based on entropy_band and research question characteristics.
        
        STEADY STATE (turns 3+):
        Based on entropy band (from Entropy-Steering-Spec):
          Runaway (δ):    Shikamaru (cooling) or Lawliet (compression). 
                          Forbid RF. Force CL then B.
          Turbulence (β): Allow one RF (Rick), then force anchor (Lawliet/Shiro B).
                          Surgical C (Makishima) only with forced B follow-up.
          Convergence (α): Prioritize Light (direction) + Lawliet (constraints).
                          Pattern: CL → B → B → A.
          Crystalline (Ω): Johan (final alignment) then trigger synthesis harvest.
        
        ANTI-DOMINANCE:
        - No agent speaks > 2 consecutive turns
        - If speaker_counts[agent] > threshold for window: apply fatigue
        - Fatigue decays 0.1/turn for non-speaking agents, increases 0.3/turn for speaker
        
        COVENANT ENFORCEMENT:
        - Anchor requirement: after RF, B must appear within 2 turns (rf_count_since_anchor)
        - Critique tax: after 2 consecutive C without B, force builder (c_count_since_build)
        
        Returns (agent, suggested_int_type). Agent decides actual output;
        suggested_int_type is the Kernel's steering signal.
        """
    
    def frame_turn(
        self,
        agent: SIAAgent,
        suggested_int_type: str,
        research_question: str,
        source_summary: str,
        contradiction_summary: str,
    ) -> str:
        """Generate the Kernel's framing message for this agent's turn.
        
        This is the "user" message that precedes the agent's response.
        It tells the agent:
        1. What entropy band we're in and what that means
        2. What the previous agents contributed (summary, not full text)
        3. What the reactor NEEDS right now (the suggested IntType and why)
        4. What specific question or challenge to address
        
        The agent sees this framing + the full prior conversation.
        The agent decides how to respond from its cognitive stance.
        """
    
    def parse_turn_output(self, agent_name: str, output: str) -> TurnRecord:
        """Parse an agent's output into structured artifacts.
        
        Extracts:
        - int_type_label (from [B], [C], [RF], [CL], [CO], [A], [S], [I] markers)
        - constraints (lines under CONSTRAINTS_EXTRACTED or similar)
        - challenges (lines under CHALLENGES or similar)
        - reframes (lines under REFRAME or similar)
        - response_to_prior (explicit engagement with prior turns)
        
        Validates:
        - Output contains at least one IntType label
        - Output has structured content (not vague handwaving)
        """
    
    def update_state(self, record: TurnRecord):
        """Update Kernel internal state after a turn.
        
        - Increment speaker_counts
        - Update consecutive_speakers
        - Update fatigue
        - Accumulate constraints, killed branches
        - Update rf_count_since_anchor, c_count_since_build
        - Micro-update entropy based on turn artifacts
        """
    
    def should_terminate(self, turn: int, max_turns: int) -> tuple[bool, str]:
        """Check if reactor should stop deliberating.
        
        Terminate when:
        - Entropy dropped to crystalline band
        - max_turns reached
        - Stagnation: 3+ turns with no new constraints or reframes
        - Token budget for reactor exhausted
        
        Returns (should_stop, reason)
        """
    
    def harvest(self) -> ReactorState:
        """Harvest reactor products for downstream synthesis stages.
        
        Produces:
        - constraints: deduplicated list of all extracted constraints
        - rejected_branches: all explicitly killed approaches
        - active_frames: current framing(s) of the research question
        - key_claims: claims that survived adversarial pressure
        - coalition_map: which agents agreed on what
        - unresolved: variables still ambiguous
        - turn_log: full conversation for traceability
        """
```

### 4. New types in `contracts.py`

```python
class TurnRecord(TypedDict):
    """Record of a single agent turn in the reactor."""
    turn: int
    agent: str
    int_type: str                    # B, C, RF, CL, CO, A, S, I
    constraints: list[str]           # constraints extracted/proposed
    challenges: list[str]            # challenges to prior claims
    reframes: list[str]              # alternative framings proposed
    response_to_prior: list[str]     # explicit engagement with prior turns
    raw_output: str                  # full agent output for traceability

class ReactorState(TypedDict):
    """Accumulated products of multi-agent reactor deliberation."""
    constraints: list[str]
    rejected_branches: list[str]
    active_frames: list[str]
    key_claims: list[str]
    coalition_map: dict[str, list[str]]  # agent -> list of positions agreed to
    unresolved: list[str]
    turn_log: list[TurnRecord]

class ReactorTrace(TypedDict):
    """Summary trace for state persistence and analysis."""
    turns_executed: int
    agents_used: list[str]
    constraints_produced: int
    branches_killed: int
    challenges_issued: int
    final_entropy: float
    termination_reason: str
    ignition_pattern: str
```

### 5. Refactored: `deep_research_swarm/agents/synthesizer.py`

**The reactor wraps the existing pipeline. It does not replace it.**

New flow inside `synthesize()`:

```python
async def synthesize(state: ResearchState, caller: AgentCaller) -> dict:
    # --- PHASE A: REACTOR DELIBERATION ---
    # Only runs if SIA is enabled and entropy_state is present
    
    if _sia_enabled(state):
        # 1. Build reactor context from state
        source_summary = _compress_sources(state)
        contradiction_summary = _compress_contradictions(state)
        
        # 2. Initialize Kernel with current entropy
        kernel = SIAKernel(
            agents=SIA_AGENTS,
            covenants=COVENANTS,
            entropy=state.get("entropy_state", {}),
        )
        
        # 3. Create Sonnet caller for reactor turns
        # (reuse existing caller pattern, but force Sonnet tier)
        reactor_caller_config = {agent.name: AgentCaller(
            api_key=caller._client.api_key,  # share API key
            model="claude-sonnet-4-6",       # always Sonnet for reactor
            max_concurrent=1,                 # sequential deliberation
            max_retries=2,
        ) for agent in SIA_AGENTS.values()}
        
        # 4. Initialize conversation thread
        conversation: list[dict] = []
        reactor_token_usage: list[TokenUsage] = []
        
        # Read tunables
        _snap = state.get("tunable_snapshot", {})
        max_reactor_turns = int(_snap.get("sia_reactor_turns", 6))
        reactor_budget = int(_snap.get("sia_reactor_budget", 20000))
        reactor_tokens_used = 0
        
        # 5. REACTOR LOOP — true multi-turn deliberation
        for turn in range(max_reactor_turns):
            # Kernel selects speaker and frames the turn
            agent, suggested_int_type = kernel.select_speaker(turn, max_reactor_turns)
            
            frame_message = kernel.frame_turn(
                agent, suggested_int_type,
                research_question=state["research_question"],
                source_summary=source_summary,
                contradiction_summary=contradiction_summary,
            )
            
            # Build the full message sequence for this agent:
            # - System prompt: agent's cognitive lens
            # - Messages: full prior conversation + Kernel's frame for this turn
            messages = conversation.copy()
            messages.append({"role": "user", "content": frame_message})
            
            # Agent responds from its cognitive stance
            response_text, usage = await reactor_caller_config[agent.name].call(
                system=agent.system_prompt.format(
                    research_question=state["research_question"],
                    entropy_band=kernel.entropy.get("band", "unknown"),
                ),
                messages=messages,
                agent_name=f"sia_{agent.name}",
                max_tokens=2048,
                temperature=0.3,  # slight creativity for genuine disagreement
            )
            
            reactor_token_usage.append(usage)
            reactor_tokens_used += usage["input_tokens"] + usage["output_tokens"]
            
            # Append to shared conversation thread
            # (next agent will see this as prior context)
            conversation.append({"role": "user", "content": frame_message})
            conversation.append({"role": "assistant", "content": response_text})
            
            # Parse and record
            record = kernel.parse_turn_output(agent.name, response_text)
            kernel.update_state(record)
            
            # Check termination
            should_stop, reason = kernel.should_terminate(turn, max_reactor_turns)
            if should_stop:
                break
            
            # Check budget
            if reactor_tokens_used >= reactor_budget:
                break
        
        # 6. Harvest reactor products
        reactor_products = kernel.harvest()
    else:
        # SIA not enabled — empty reactor products (V9 behavior)
        reactor_products = None
        reactor_token_usage = []
    
    # --- PHASE B: EXISTING 5-STAGE PIPELINE ---
    # (augmented with reactor products if available)
    
    # Stage 0: Outline validation (unchanged)
    # Stage 1: Outline generation — NOW with reactor constraints as additional context
    #   If reactor_products exists:
    #     - constraints → injected as "hard constraints the outline must satisfy"
    #     - rejected_branches → injected as "approaches explicitly ruled out"
    #     - active_frames → injected as "framings that survived adversarial pressure"
    #     - unresolved → injected as "unknowns to acknowledge, not paper over"
    # Stage 2: Per-section drafting (unchanged, but outline is richer)
    # Stage 3: Mechanical grounding verification (unchanged)
    # Stage 4: Refinement (unchanged)
    # Stage 5: Composition (unchanged)
    
    # ... existing pipeline code with reactor injection point at Stage 1 ...
    
    # Build output (unchanged interface)
    result = _build_output(...)
    result["reactor_trace"] = ReactorTrace(...)  # NEW: add trace
    result["token_usage"] = existing_usage + reactor_token_usage  # merge
    return result
```

**Key architectural points:**

1. **Shared conversation thread**: `conversation: list[dict]` grows with each turn. Every agent sees the full thread. Lawliet at turn 4 can see and respond to Rick's reframe at turn 1 and Light's direction at turn 2.

2. **Kernel frames, agents decide**: The Kernel's `frame_turn()` tells the agent "we're in turbulence band, Rick just reframed the problem, the reactor needs constraint extraction now." The agent, through its cognitive lens, decides WHAT constraints to extract and HOW to respond to Rick's reframe.

3. **Temperature 0.3**: Slight creativity. We want genuine processing variation, not deterministic parroting. Different agents with different system prompts at temp 0.3 will attend to different features of the same input.

4. **Sonnet tier**: All reactor turns use Sonnet. Fast, cheap, sufficient for structured analysis. Opus is reserved for the final synthesis drafting (Stage 2).

5. **Interface preservation**: `synthesize()` still returns the same dict shape. The reactor is invisible to everything downstream.

### 6. New tunables in `adaptive/registry.py`

```python
Tunable(name="sia_reactor_turns", default=6, floor=3, ceiling=10, category="synthesis")
Tunable(name="sia_reactor_budget", default=20000, floor=8000, ceiling=40000, category="synthesis")
```

### 7. New state field in `graph/state.py`

```python
reactor_trace: Annotated[dict, _replace_dict]  # ReactorTrace
```

### 8. Tests: `tests/test_sia_agents.py` — minimum 20 tests

- All 7 agents instantiate with correct fields
- Basin profiles have valid axis names and weight ranges
- System prompts contain required template variables
- System prompts contain structured output format instructions
- Preferred IntTypes are valid
- No two agents have identical basin profiles

### 9. Tests: `tests/test_sia_kernel.py` — minimum 30 tests

- **Ignition doctrine**: first 2 turns never select cooling sink (Shikamaru)
- **Ignition patterns**: all four valid patterns produce correct speaker sequences
- **Entropy-based selection**: runaway → selects Shikamaru/Lawliet, not Rick
- **Entropy-based selection**: turbulence → allows Rick then forces anchor
- **Entropy-based selection**: convergence → prioritizes Light + Lawliet
- **Entropy-based selection**: crystalline → selects Johan for alignment
- **Anti-dominance**: agent cannot speak > 2 consecutive turns
- **Fatigue**: increases for speaker, decays for others
- **Anchor requirement**: after RF, B appears within 2 turns
- **Critique tax**: after 2 C without B, forces builder
- **frame_turn()**: includes entropy band, prior summary, suggested IntType
- **parse_turn_output()**: extracts IntType labels, constraints, challenges
- **parse_turn_output()**: rejects unstructured output
- **update_state()**: correctly tracks constraints, branches, speaker counts
- **should_terminate()**: crystalline → terminate
- **should_terminate()**: stagnation (3 turns, no new constraints) → terminate
- **should_terminate()**: budget exhausted → terminate
- **harvest()**: produces deduplicated constraints, killed branches, coalition map

### 10. Tests: `tests/test_sia_covenants.py` — minimum 12 tests

- All covenant pairs from spec are defined
- Lookup works for both orderings (a,b) and (b,a)
- Preferred sequences contain only valid IntTypes
- High-risk triads are defined
- max_consecutive_without_anchor is positive for all covenants

### 11. Tests: `tests/test_synthesizer_reactor.py` — minimum 18 tests

- **Interface preservation**: output dict has same keys as V9
- **ReactorTrace present**: reactor_trace field populated when SIA enabled
- **Graceful degradation**: SIA disabled → V9 behavior exactly
- **Graceful degradation**: entropy_state missing → V9 behavior
- **Graceful degradation**: reactor raises exception → catches, falls back to V9
- **Token budget respected**: reactor stops when budget hit
- **Conversation thread grows**: each turn adds 2 messages (user frame + assistant response)
- **Multiple agents participate**: at least 3 distinct agents used in a 6-turn reactor
- **Constraints flow to outline**: reactor constraints appear in outline generation context
- **Rejected branches flow**: killed approaches appear as exclusions
- **Token usage merged**: reactor usage + synthesis usage in same list

## Constraints

- **Interface preservation**: `synthesize(state, caller) -> dict` signature and return shape unchanged
- **True multi-turn**: Agents MUST see prior conversation. Single-shot calls with no memory are forbidden.
- **Sonnet tier**: All reactor turns use Sonnet. No Opus in the reactor.
- **Graceful degradation**: Any SIA failure → fall back to V9 single-voice synthesis. Log the error, don't crash.
- **No new pip dependencies**: Uses existing `AgentCaller`, existing Anthropic SDK
- **Contracts-first**: All new types in `contracts.py`
- **All 812+ existing tests pass**

## Cost Analysis

```
Reactor: 6 turns × ~4K tokens avg (input grows with conversation) = ~24K tokens
Sonnet pricing: (24K × $3/Mtok input + 6K × $15/Mtok output) ≈ $0.16
Current V9 synthesis cost: ~$0.30-0.50 (Opus)
Total V10 synthesis: $0.16 (reactor) + $0.30-0.50 (Opus drafting) = $0.46-0.66
Cost increase: ~50%. Justified by structured tension → richer outlines → better reports.
```

## After completion

Update `CLAUDE.md`: V10-beta (SIA Phase 2: True Agent Reactor)
Update `CHANGELOG.md`
Test count: ~812 + ~80 = ~892
