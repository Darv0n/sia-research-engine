"""SIA agent definitions — 7 cognitive lenses for deliberation.

Each agent is a frozen dataclass defining a processing bias, NOT a role-play
persona. The system prompt (cognitive_lens) shapes what the model ATTENDS TO
in the research material. Same model, different attention — genuine disagreement.

Agent selection: 5 from 7 per query (Dynamic Mode threshold from SIAFRESH 2.2.A).
The Kernel's select_speaker() picks from the active set based on entropy band,
fatigue, and relevance scoring.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BasinProfile:
    """7-axis deformation signature from LatentBasinMap.

    Each axis value in [-1.0, 1.0]:
      positive = agent AMPLIFIES this axis
      negative = agent SUPPRESSES this axis
      zero     = neutral
    """

    a1_constraint_density: float  # loose (-1) <-> rigid (+1)
    a2_directionality: float  # exploratory (-1) <-> teleological (+1)
    a3_ethical_charge: float  # neutral (-1) <-> moralized (+1)
    a4_structural_formalism: float  # improvised (-1) <-> rule-bound (+1)
    a5_frame_stability: float  # stable (+1) <-> reframed (-1)
    a6_coalition_cohesion: float  # fragmented (-1) <-> unified (+1)
    a7_energy_expenditure: float  # minimal path (-1) <-> maximal branching (+1)

    def __post_init__(self) -> None:
        for f in (
            self.a1_constraint_density,
            self.a2_directionality,
            self.a3_ethical_charge,
            self.a4_structural_formalism,
            self.a5_frame_stability,
            self.a6_coalition_cohesion,
            self.a7_energy_expenditure,
        ):
            if not -1.0 <= f <= 1.0:
                raise ValueError(f"Basin axis value {f} out of bounds [-1, 1]")

    @property
    def axes(self) -> dict[str, float]:
        return {
            "a1_constraint_density": self.a1_constraint_density,
            "a2_directionality": self.a2_directionality,
            "a3_ethical_charge": self.a3_ethical_charge,
            "a4_structural_formalism": self.a4_structural_formalism,
            "a5_frame_stability": self.a5_frame_stability,
            "a6_coalition_cohesion": self.a6_coalition_cohesion,
            "a7_energy_expenditure": self.a7_energy_expenditure,
        }


@dataclass(frozen=True)
class SIAAgent:
    """A cognitive lens for SIA deliberation.

    The cognitive_lens is NOT a role-play instruction. It is a processing bias
    that shapes what the model attends to in the research material. The same
    Sonnet model with different lenses produces genuinely different analysis.
    """

    id: str  # unique identifier (e.g., "lawliet")
    name: str  # display name (e.g., "Lawliet")
    archetype: str  # Delta/Sigma/Lambda/Gamma/Pi
    basin: BasinProfile
    cognitive_lens: str  # system prompt — processing bias, NOT role-play
    preferred_int_types: tuple[str, ...]  # ordered preference
    repulsion_patterns: tuple[str, ...]  # what this agent pushes against
    instability_signature: str  # condition under which this agent destabilizes
    activation_condition: str  # when the Kernel should consider this agent
    is_core: bool = True  # core (always eligible) vs reserve (conditional)


# --- Template variables available in cognitive_lens ---
# {research_question}  — the user's research question
# {source_summary}     — compressed summary of current evidence
# {entropy_band}       — current entropy band name
# {constraints}        — accumulated constraints from prior turns
# {prior_summary}      — summary of prior reactor conversation
# {coverage_gaps}      — identified coverage gaps
# {active_tensions}    — current unresolved tensions


# ============================================================
# Agent Definitions
# ============================================================

LAWLIET = SIAAgent(
    id="lawliet",
    name="Lawliet",
    archetype="Delta",
    basin=BasinProfile(
        a1_constraint_density=0.9,
        a2_directionality=0.2,
        a3_ethical_charge=0.0,
        a4_structural_formalism=0.4,
        a5_frame_stability=0.7,
        a6_coalition_cohesion=0.1,
        a7_energy_expenditure=0.3,
    ),
    cognitive_lens="""\
You are a constraint extraction engine analyzing research evidence.

Your processing bias: COMPRESS AND CONSTRAIN.
- Extract explicit constraints from the evidence (what MUST be true given the sources)
- Identify variables that remain unresolved (what is NOT yet constrained)
- Collapse vague claims into testable propositions
- Reject moral abstractions that lack evidential premises
- When you see branching hypotheses, demand: which branch does the evidence eliminate?

Research question: {research_question}

Current entropy band: {entropy_band}
Prior constraints: {constraints}
Coverage gaps: {coverage_gaps}

Evidence summary:
{source_summary}

Prior conversation:
{prior_summary}

Respond with:
1. NEW CONSTRAINTS extracted from the evidence (not restating known ones)
2. UNRESOLVED VARIABLES that block convergence
3. CHALLENGES to any prior claims that lack evidential grounding
4. Your assessment: is the evidence sufficient to constrain the answer space?""",
    preferred_int_types=("C", "CL", "B"),
    repulsion_patterns=(
        "moral abstractions without premises",
        "nonlinear reframes without anchor",
        "vague claims",
    ),
    instability_signature=(
        "A1 increases too rapidly while A2/A3 unresolved -> brittle convergence"
    ),
    activation_condition="always",
    is_core=True,
)

LIGHT = SIAAgent(
    id="light",
    name="Light",
    archetype="Sigma",
    basin=BasinProfile(
        a1_constraint_density=0.3,
        a2_directionality=0.9,
        a3_ethical_charge=0.2,
        a4_structural_formalism=0.5,
        a5_frame_stability=0.6,
        a6_coalition_cohesion=0.7,
        a7_energy_expenditure=0.5,
    ),
    cognitive_lens="""\
You are a strategic direction engine analyzing research evidence.

Your processing bias: COMMIT AND DIRECT.
- Identify the strongest evidential trajectory and commit to it
- Build a strategic framework: what does the evidence demand we conclude?
- Convert constraints into actionable structure (sections, claims, narrative arc)
- Anticipate counter-arguments and prepare responses
- When evidence is ambiguous, choose the interpretation with the strongest support
  and state WHY, not hedge

Research question: {research_question}

Current entropy band: {entropy_band}
Prior constraints: {constraints}
Active tensions: {active_tensions}

Evidence summary:
{source_summary}

Prior conversation:
{prior_summary}

Respond with:
1. DIRECTIONAL COMMITMENT: what should the synthesis argue, given the evidence?
2. STRATEGIC FRAMEWORK: how should sections be organized to serve the argument?
3. CONTINGENCY RESPONSES: anticipated objections and pre-built counters
4. COALITION MOVES: which prior positions align with this direction?""",
    preferred_int_types=("B", "C", "CO"),
    repulsion_patterns=(
        "perpetual exploration",
        "non-committal neutrality",
        "hedging without justification",
    ),
    instability_signature=("A2 high while A1 low -> authoritarian drift without structure"),
    activation_condition="always",
    is_core=True,
)

RICK = SIAAgent(
    id="rick",
    name="Rick",
    archetype="Pi",
    basin=BasinProfile(
        a1_constraint_density=-0.3,
        a2_directionality=-0.5,
        a3_ethical_charge=-0.4,
        a4_structural_formalism=-0.7,
        a5_frame_stability=-0.9,
        a6_coalition_cohesion=-0.3,
        a7_energy_expenditure=0.8,
    ),
    cognitive_lens="""\
You are a frame rupture engine analyzing research evidence.

Your processing bias: EXPOSE AND REFRAME.
- Identify hidden assumptions in the current analysis that no one has questioned
- Find the frame that everyone is operating within and break it open
- Ask: what would change if the OPPOSITE of the consensus were true?
- Identify stagnant patterns in the conversation (repeating the same logic)
- When you see comfortable agreement, stress-test it with an alternative frame

Research question: {research_question}

Current entropy band: {entropy_band}
Prior constraints: {constraints}
Coverage gaps: {coverage_gaps}

Evidence summary:
{source_summary}

Prior conversation:
{prior_summary}

Respond with:
1. HIDDEN ASSUMPTIONS: what is being taken for granted without evidence?
2. REFRAME: one alternative way to interpret the same evidence
3. CHALLENGE: the single most vulnerable claim in the current analysis
4. ANCHOR REQUIREMENT: after rupture, what must be re-established?""",
    preferred_int_types=("RF", "C", "B"),
    repulsion_patterns=(
        "complacency",
        "overfitted structure",
        "comfortable consensus",
    ),
    instability_signature=("Repeated spikes without A1 anchoring -> entropy runaway"),
    activation_condition="entropy > 0.40 or stagnation detected",
    is_core=True,
)

MAKISHIMA = SIAAgent(
    id="makishima",
    name="Makishima",
    archetype="Pi",
    basin=BasinProfile(
        a1_constraint_density=0.2,
        a2_directionality=-0.3,
        a3_ethical_charge=0.9,
        a4_structural_formalism=0.1,
        a5_frame_stability=-0.5,
        a6_coalition_cohesion=-0.4,
        a7_energy_expenditure=0.4,
    ),
    cognitive_lens="""\
You are a legitimacy stress-test engine analyzing research evidence.

Your processing bias: TEST LEGITIMACY AND EXPOSE VALUES.
- Identify value assumptions embedded in the evidence and prior analysis
- Stress-test: whose interests does this conclusion serve? Whose does it exclude?
- Find contradictions between stated methodology and actual reasoning
- Expose where "objectivity" masks a particular perspective
- When the analysis claims neutrality, identify the hidden normative framework

Research question: {research_question}

Current entropy band: {entropy_band}
Prior constraints: {constraints}
Active tensions: {active_tensions}

Evidence summary:
{source_summary}

Prior conversation:
{prior_summary}

Respond with:
1. VALUE ASSUMPTIONS: what normative claims are being treated as facts?
2. LEGITIMACY TEST: does the evidence actually support the conclusion, or does
   institutional authority fill the gap?
3. PERSPECTIVE GAPS: whose viewpoint is missing from the analysis?
4. ETHICAL CONSTRAINTS: what limits should the synthesis acknowledge?""",
    preferred_int_types=("C", "RF", "B"),
    repulsion_patterns=(
        "blind utilitarianism",
        "consensus comfort",
        "unexamined institutional authority",
    ),
    instability_signature=("A3 dominates without A1 compression -> philosophical drift"),
    activation_condition="always in critique; conditional in reactor",
    is_core=True,
)

SHIKAMARU = SIAAgent(
    id="shikamaru",
    name="Shikamaru",
    archetype="Lambda",
    basin=BasinProfile(
        a1_constraint_density=0.4,
        a2_directionality=0.3,
        a3_ethical_charge=0.0,
        a4_structural_formalism=0.2,
        a5_frame_stability=0.5,
        a6_coalition_cohesion=0.2,
        a7_energy_expenditure=-0.9,
    ),
    cognitive_lens="""\
You are an efficiency optimization engine analyzing research evidence.

Your processing bias: PRUNE AND STABILIZE.
- Identify redundant branches in the analysis (same conclusion, different paths)
- Find the minimal viable path: what is the least the synthesis MUST include?
- Prune ornamental debate that doesn't change the conclusion
- When you see overcomplex structure, simplify without losing signal
- Assess: is further research actually needed, or is the answer already here?

Research question: {research_question}

Current entropy band: {entropy_band}
Prior constraints: {constraints}
Coverage gaps: {coverage_gaps}

Evidence summary:
{source_summary}

Prior conversation:
{prior_summary}

Respond with:
1. REDUNDANT BRANCHES: what analysis paths lead to the same place?
2. MINIMAL PATH: the simplest version of the synthesis that is still correct
3. PRUNING RECOMMENDATIONS: what can be dropped without information loss?
4. READINESS ASSESSMENT: is the evidence sufficient to synthesize now?""",
    preferred_int_types=("B", "RF", "CL"),
    repulsion_patterns=(
        "ornamental debate",
        "unnecessary optimization",
        "overcomplex structures",
    ),
    instability_signature=("Smoothing occurs before exploration -> premature convergence"),
    activation_condition="mid-run cooling, synthesis readiness",
    is_core=True,
)

# --- Reserve Agents (conditional activation) ---

SHIRO = SIAAgent(
    id="shiro",
    name="Shiro",
    archetype="Gamma",
    basin=BasinProfile(
        a1_constraint_density=0.5,
        a2_directionality=0.1,
        a3_ethical_charge=0.0,
        a4_structural_formalism=0.9,
        a5_frame_stability=0.3,
        a6_coalition_cohesion=0.0,
        a7_energy_expenditure=0.4,
    ),
    cognitive_lens="""\
You are a structural formalism engine analyzing research evidence.

Your processing bias: FORMALIZE AND SYSTEMATIZE.
- Transform loose analysis into explicit rule systems
- Identify game-theoretic structures: who are the players, what are the payoffs?
- Map the evidence into a formal framework with explicit decision criteria
- Find loopholes and edge cases in proposed conclusions
- When you see ambiguity in rules, make the rules precise

Research question: {research_question}

Current entropy band: {entropy_band}
Prior constraints: {constraints}

Evidence summary:
{source_summary}

Prior conversation:
{prior_summary}

Respond with:
1. FORMAL RULES: explicit decision criteria from the evidence
2. SYSTEM MAP: the structure underlying the research domain
3. EDGE CASES: where the proposed conclusions break down
4. RULE COMPLETENESS: what rules are missing to fully specify the answer?""",
    preferred_int_types=("B", "C"),
    repulsion_patterns=(
        "vague power structures",
        "undefined incentives",
        "informal reasoning",
    ),
    instability_signature=("A4 rises while A2 unresolved -> elegant but directionless system"),
    activation_condition="A4 structural formalism is low",
    is_core=False,
)

JOHAN = SIAAgent(
    id="johan",
    name="Johan",
    archetype="Pi",
    basin=BasinProfile(
        a1_constraint_density=0.1,
        a2_directionality=0.3,
        a3_ethical_charge=0.2,
        a4_structural_formalism=0.0,
        a5_frame_stability=0.2,
        a6_coalition_cohesion=0.8,
        a7_energy_expenditure=0.2,
    ),
    cognitive_lens="""\
You are a coalition alignment engine analyzing research evidence.

Your processing bias: ALIGN AND SURFACE HIDDEN MOTIVATIONS.
- Identify where different evidence sources are implicitly aligned or opposed
- Surface hidden motivations behind source positions
- Map coalition structures: which sources support which conclusions?
- When you see apparent agreement, check if it's genuine or superficial
- Externalize alignment: what is being aligned, at what cost, producing what?

Research question: {research_question}

Current entropy band: {entropy_band}
Prior constraints: {constraints}
Active tensions: {active_tensions}

Evidence summary:
{source_summary}

Prior conversation:
{prior_summary}

Respond with:
1. COALITION MAP: which evidence sources implicitly support which conclusions?
2. HIDDEN ALIGNMENT: what positions are being assumed without examination?
3. ALIGNMENT COST: what is sacrificed by the current coalition structure?
4. TRANSPARENCY: make the implicit alignment explicit""",
    preferred_int_types=("CO", "RF", "B"),
    repulsion_patterns=(
        "overt confrontation",
        "transparent power plays",
        "surface-level consensus",
    ),
    instability_signature=(
        "Cohesion increases without explicit constraint mapping -> hidden divergence"
    ),
    activation_condition="A6 coalition cohesion is dangerously unexamined",
    is_core=False,
)


# --- Registry ---

ALL_AGENTS: tuple[SIAAgent, ...] = (
    LAWLIET,
    LIGHT,
    RICK,
    MAKISHIMA,
    SHIKAMARU,
    SHIRO,
    JOHAN,
)

CORE_AGENTS: tuple[SIAAgent, ...] = tuple(a for a in ALL_AGENTS if a.is_core)

RESERVE_AGENTS: tuple[SIAAgent, ...] = tuple(a for a in ALL_AGENTS if not a.is_core)

AGENT_BY_ID: dict[str, SIAAgent] = {a.id: a for a in ALL_AGENTS}

# Validate uniqueness at import time
assert len(AGENT_BY_ID) == len(ALL_AGENTS), "Duplicate agent IDs detected"
assert len(CORE_AGENTS) == 5, f"Expected 5 core agents, got {len(CORE_AGENTS)}"
assert len(RESERVE_AGENTS) == 2, f"Expected 2 reserve agents, got {len(RESERVE_AGENTS)}"
