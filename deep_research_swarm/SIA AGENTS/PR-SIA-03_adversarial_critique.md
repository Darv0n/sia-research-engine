# PR-SIA-03: Adversarial Critique — Covenant-Driven Quality Assurance

## Context

You are working on `deep-research-swarm` with SIA Phase 1 (entropy layer) and Phase 2 (true agent reactor) installed. The synthesizer now produces richer outlines through genuine multi-agent deliberation — agents that see, respond to, and disagree with each other in a shared conversation thread. But the CRITIC still operates as three parallel graders producing 0-1 scores. It cannot produce actionable constraints, stress-test legitimacy, or detect structural failures that hide behind "acceptable" numbers.

This phase upgrades critique to use the same true-agent architecture: a multi-turn adversarial sequence where agents challenge the synthesis from different cognitive stances, producing constraints and findings — not just scores.

Read these files before writing any code:
- `CLAUDE.md` (current state after Phase 2)
- `deep_research_swarm/agents/critic.py` (current three-grader chain)
- `deep_research_swarm/sia/kernel.py` (SIAKernel — reuse for critique)
- `deep_research_swarm/sia/agents.py` (agent definitions — reuse)
- `docs/sia/SingularityPreventionSpec.md` (what you're implementing)
- `docs/sia/Entropy-Steering-Spec.md` (false convergence, synthesis readiness)
- `docs/sia/Inter-Agent-Covenants.md` (covenant enforcement mechanisms)

## Deliverables

### 1. New module: `deep_research_swarm/sia/adversarial_critique.py`

A multi-turn adversarial evaluation using the same true-agent architecture as the reactor. NOT parallel single-shot graders — a CONVERSATION where each agent can see and respond to prior agents' findings.

```python
async def adversarial_critique(
    state: ResearchState,
    sonnet_caller: AgentCaller,
) -> dict:
    """Run covenant-driven adversarial critique as multi-turn deliberation.
    
    Architecture: shared conversation thread, same as reactor in Phase 2.
    Each agent sees the synthesis AND all prior agents' critiques.
    Agents can build on, challenge, or prioritize prior findings.
    
    SEQUENCE (4-5 turns, Sonnet tier):
    
    Turn 1 — MAKISHIMA (Legitimacy Stress-Test):
      Cognitive lens: sees value assumptions, untested premises, claims
      that LOOK grounded but rest on contested foundations.
      Produces: [C] challenges with specific claims cited.
      
    Turn 2 — LAWLIET (Constraint Extraction):
      Sees Makishima's challenges. Cognitive lens: what MUST be true
      for the report to hold? What variables are assumed but not verified?
      Produces: [CL] constraint list + missing variable list.
      Can AGREE or DISAGREE with Makishima's specific challenges.
    
    Turn 3 — RICK (Frame Audit — conditional):
      Only runs if entropy > 0.40 (turbulence or runaway).
      Sees Makishima + Lawliet. Cognitive lens: is the report's
      framing the ONLY valid framing? What assumption is everyone
      making that nobody questioned?
      Produces: [RF] alternative frame + what changes if adopted.
      
    Turn 4 — SHIKAMARU (Synthesis Readiness Assessment):
      Sees all prior critique turns. Cognitive lens: what is
      ACTUALLY fixable vs philosophical? What's the minimal path
      to acceptable quality?
      Produces: [B] prioritized fix list + converge/replan recommendation.
    
    Turn 5 — LIGHT (Decision, only if Shikamaru recommends replan):
      Sees everything. Cognitive lens: given all findings, what is
      the SPECIFIC direction for the next iteration? What changes?
      Produces: [B] concrete replan directive.
    
    OUTPUT: Compatible with existing critique consumers.
    Returns dict with:
    - section_drafts (scores updated based on adversarial findings)
    - converged / convergence_reason
    - iteration_history
    - adversarial_findings (NEW: structured challenges)
    - critique_trace (NEW: full deliberation trace)
    """
```

**Implementation pattern** — identical to reactor loop in Phase 2:

```python
    # Shared conversation thread
    conversation: list[dict] = []
    
    # Build context: the synthesis output being critiqued
    synthesis_context = _format_synthesis_for_critique(state)
    
    # Adversarial sequence
    critique_agents = [
        (MAKISHIMA, "C"),    # always
        (LAWLIET, "CL"),    # always
    ]
    
    entropy = state.get("entropy_state", {})
    if entropy.get("e", 0.5) > 0.40:
        critique_agents.append((RICK, "RF"))  # conditional
    
    critique_agents.append((SHIKAMARU, "B"))  # always
    
    findings: list[AdversarialFinding] = []
    all_usage: list[TokenUsage] = []
    
    for agent, suggested_int_type in critique_agents:
        # Frame this turn with full context
        frame = _frame_critique_turn(
            agent=agent,
            suggested_int_type=suggested_int_type,
            synthesis_context=synthesis_context,
            entropy=entropy,
            prior_findings=findings,
        )
        
        messages = conversation.copy()
        messages.append({"role": "user", "content": frame})
        
        # Agent critiques from its cognitive stance
        response, usage = await sonnet_caller.call(
            system=agent.system_prompt.format(
                research_question=state["research_question"],
                entropy_band=entropy.get("band", "unknown"),
            ),
            messages=messages,
            agent_name=f"sia_critique_{agent.name}",
            max_tokens=2048,
            temperature=0.2,
        )
        
        all_usage.append(usage)
        conversation.append({"role": "user", "content": frame})
        conversation.append({"role": "assistant", "content": response})
        
        # Parse findings from this turn
        turn_findings = _parse_critique_findings(agent.name, suggested_int_type, response)
        findings.extend(turn_findings)
    
    # Check if Shikamaru recommended replan → optional Light turn
    shikamaru_recommendation = _extract_recommendation(findings)
    if shikamaru_recommendation == "replan":
        # Light provides concrete direction
        # ... (same pattern, one more turn)
    
    # Convert adversarial findings → section score adjustments
    updated_sections = _apply_findings_to_scores(state["section_drafts"], findings)
    
    # Determine convergence (incorporates adversarial severity)
    converged, reason = _determine_convergence(updated_sections, findings, entropy)
    
    return {
        "section_drafts": updated_sections,
        "converged": converged,
        "convergence_reason": reason,
        "iteration_history": [_build_iteration_record(...)],
        "token_usage": all_usage,
        "adversarial_findings": [asdict(f) for f in findings],
        "critique_trace": {
            "turns": len(critique_agents),
            "findings_count": len(findings),
            "critical_findings": sum(1 for f in findings if f.severity == "critical"),
            "recommendation": shikamaru_recommendation,
        },
    }
```

**Score adjustment logic:**
- Critical finding targeting a section → reduce that section's confidence by 0.15
- Significant finding → reduce by 0.08
- Minor finding → reduce by 0.03
- Unresolved constraint from Lawliet → flag section for replan
- This produces scores that LOOK like existing grader output but incorporate adversarial signal

### 2. New types in `contracts.py`

```python
class AdversarialFinding(TypedDict):
    agent: str                   # which SIA agent produced this
    int_type: str                # C, CL, RF, B
    target_section: str          # section_id or "global"
    finding: str                 # the specific challenge/constraint/reframe
    severity: str                # "critical" | "significant" | "minor"
    actionable: bool             # can this be addressed in refinement?
    response_to: str             # which prior finding this responds to (or "")

class CritiqueTrace(TypedDict):
    turns: int
    findings_count: int
    critical_findings: int
    constraints_extracted: int
    missing_variables: list[str]
    alternative_frames: list[str]
    recommendation: str          # "converge" | "replan" | "refine_targeted"
```

### 3. New module: `deep_research_swarm/sia/singularity_prevention.py`

Pre-synthesis safety gate. Runs BEFORE convergence is allowed.

```python
def check_singularity_conditions(state: ResearchState) -> tuple[bool, str, list[str]]:
    """Check for pathological collapse states.
    
    Returns (safe_to_synthesize, reason, required_interventions)
    
    Checks:
    1. Constraint Singularity: constraints high but no rejected branches
    2. Directional Singularity: strong direction, no ethical validation
    3. Reframe Singularity: many reframes, no constraint accumulation
    4. Coalition Shadow: agreement increasing, no explicit premise mapping
    
    Uses reactor_trace (if available) and adversarial_findings.
    """

def multi_axis_stability_check(
    reactor_trace: dict | None,
    entropy: dict,
    adversarial_findings: list[dict],
    section_drafts: list,
) -> dict[str, float]:
    """Compute stability score for each semantic axis [0, 1].
    
    A₁ (constraints): reactor constraints / expected minimum
    A₂ (direction): whether outline has coherent narrative_arc
    A₃ (values): whether ethical challenges were addressed (from adversarial)
    A₄ (structure): section count + grounding coverage
    A₅ (frame stability): inverse of reframe count
    A₆ (coalition): agent agreement ratio from reactor
    A₇ (efficiency): useful output / total tokens ratio
    """
```

### 4. Modified: `deep_research_swarm/agents/critic.py`

Mode switch:

```python
async def critique(state, caller, *, sia_enabled: bool = True):
    if sia_enabled and state.get("entropy_state"):
        return await adversarial_critique(state, caller)
    else:
        return await _classic_critique(state, caller)  # existing logic, renamed
```

### 5. Integration: convergence now uses three-way check

In `graph/builder.py` convergence routing:
```python
converged = (
    confidence_ok
    AND entropy_gate_ok
    AND NOT false_convergence
    AND NOT dominance
    AND singularity_safe
)
```

### 6. Adversarial findings flow into next iteration

When replan triggers, critical+actionable findings become constraints for the next iteration:
- Missing variables → added to planner context
- Legitimacy challenges → added to section draft prompts
- Alternative frames → added to outline context

This creates the GCROE feedback loop: adversarial findings from iteration N constrain iteration N+1.

### 7. New state fields in `graph/state.py`

```python
adversarial_findings: Annotated[list[dict], _replace_list]
critique_trace: Annotated[dict, _replace_dict]
```

### 8. Tests: `tests/test_adversarial_critique.py` — minimum 22 tests

- Sequence produces correct agent order (Makishima → Lawliet → [Rick] → Shikamaru)
- Rick's turn is conditional on entropy > 0.40
- Each turn sees prior conversation (conversation thread grows)
- Findings are properly typed with all required fields
- Severity classification works (critical/significant/minor)
- Score adjustment: critical finding reduces section confidence
- Actionable/non-actionable separation works
- Output compatible with existing critique consumers (section_drafts shape)
- Graceful degradation to classic critique when SIA unavailable
- Token usage from all turns merged correctly
- Shikamaru's recommendation determines if Light gets a turn

### 9. Tests: `tests/test_singularity_prevention.py` — minimum 15 tests

- All four singularity types detected correctly
- Multi-axis stability check covers all seven axes
- Missing axis blocks synthesis
- All axes stable → safe to synthesize
- Works with reactor_trace present and absent
- Works with adversarial_findings present and absent

### 10. Tests: `tests/test_convergence_integration.py` — minimum 10 tests

- Five-way convergence check works (confidence + entropy + false_convergence + dominance + singularity)
- Any single veto blocks convergence
- All agreeing allows convergence
- Without SIA, original convergence logic works unchanged

## Constraints

- **Output compatibility**: critique() returns same dict shape. New fields are additive.
- **True multi-turn**: Critique agents see prior critique agents' outputs. NOT parallel single-shot.
- **Cost parity**: 4-5 Sonnet calls ≈ current 3 parallel Sonnet graders. Slight increase justified.
- **Fallback**: Classic critique remains available. SIA critique only when entropy layer present.
- **All ~892 existing tests pass**

## After completion

Update `CLAUDE.md`: V10-rc1 (SIA Phase 3: Adversarial Critique + Singularity Prevention)
Update `CHANGELOG.md`
Test count: ~892 + ~47 = ~939
