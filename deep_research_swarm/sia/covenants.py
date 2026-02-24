"""Inter-agent covenants — pairwise field physics for SIA deliberation.

Covenants are NOT "friendships." They are latent-space couplings:
attraction/repulsion gradients that produce predictable tension,
coalition drift, and entropy signatures.

Design intent:
- Preserve each agent's shadow mass (negative-space power source)
- Prevent collapse into bland agreement (entropy falsely low)
  OR runaway instability (entropy high, no convergence)
- Convert conflict into structured motion via IntType grammar
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Covenant:
    """Stable relational physics between two agent memeplex ontologies."""

    agent_a: str  # agent id
    agent_b: str  # agent id
    coupling: str  # e.g., "Compression vs Sovereignty (repulsive attraction)"
    signature: str  # expected dynamic when this pair interacts
    preferred_int_types_a: tuple[str, ...]  # what agent_a tends to produce
    preferred_int_types_b: tuple[str, ...]  # what agent_b tends to produce
    failure_mode: str  # how the pairing becomes toxic
    escalation_trigger: str  # when kernel must intervene
    kernel_lever: str  # how to steer this pairing


@dataclass(frozen=True)
class HighRiskTriad:
    """A triad of agents that produces emergent dynamics beyond pairwise covenants."""

    agents: tuple[str, str, str]
    signature: str
    kernel_stance: str
    exit_condition: str


# ============================================================
# Covenant Catalog (from Inter-AgentCovenants spec)
# ============================================================

COVENANT_A = Covenant(
    agent_a="lawliet",
    agent_b="light",
    coupling="Compression vs Sovereignty (repulsive attraction)",
    signature="decisive conflict that yields high-quality structure",
    preferred_int_types_a=("C", "CL", "B"),
    preferred_int_types_b=("B", "C", "CO"),
    failure_mode=(
        "Light becomes authoritarian; Lawliet becomes tyrannical reductionism. "
        "'I am right' vs 'I am certain' -> hard lock."
    ),
    escalation_trigger="C repeats 3 turns without new constraints",
    kernel_lever=(
        "If entropy rising: force CL then B — Lawliet extracts variables, "
        "Light binds them into roadmap. "
        "If stagnation: inject Rick for RF to rotate axes, then return."
    ),
)

COVENANT_B = Covenant(
    agent_a="lawliet",
    agent_b="rick",
    coupling="Frame rupture vs frame compression",
    signature="spike -> collapse into deeper basin (high discovery rate)",
    preferred_int_types_a=("CL", "B"),
    preferred_int_types_b=("RF", "C", "B"),
    failure_mode="Rick reframes endlessly; Lawliet overconstrains prematurely.",
    escalation_trigger="entropy > 0.75 and Rick requests another RF",
    kernel_lever=(
        "Use Rick only as RF catalyst with strict exit condition: "
        "'one rupture, then anchor.' "
        "Immediately schedule Lawliet next for CL/B to formalize."
    ),
)

COVENANT_C = Covenant(
    agent_a="light",
    agent_b="makishima",
    coupling="Destiny vs Moral faultline",
    signature="oscillation that reveals hidden values (high ethical signal)",
    preferred_int_types_a=("B", "CO"),
    preferred_int_types_b=("C", "RF", "B"),
    failure_mode="Prolonged ideological duel that never ships.",
    escalation_trigger="RF occurs twice in this pairing without new constraints",
    kernel_lever=(
        "Convert duel into two-column constraint treaty: "
        "Light: operational constraints, Makishima: ethical constraints. "
        "Force synthesis candidate: Shiro to codify rules, or Lawliet to compress."
    ),
)

COVENANT_D = Covenant(
    agent_a="johan",
    agent_b="*",  # system-wide coupling
    coupling="Subtle alignment pressure (attractive, corrosive)",
    signature="slow entropy drift (downward if healthy, upward if paranoid)",
    preferred_int_types_a=("CO", "RF", "B"),
    preferred_int_types_b=("CL",),  # others should clarify when suspicion appears
    failure_mode="Trust erosion spiral: paranoia raises entropy without visible cause.",
    escalation_trigger=("Coalition score diverges (two factions) and entropy rises"),
    kernel_lever=(
        "Johan must externalize coalition moves as explicit CO artifacts: "
        "'what is being aligned, toward what outcome.' "
        "If distrust rises: schedule Lawliet CL to surface hidden assumptions."
    ),
)

COVENANT_E = Covenant(
    agent_a="shiro",
    agent_b="rick",
    coupling="Formalism vs rupture",
    signature="innovation cycles (break -> rebuild rules)",
    preferred_int_types_a=("B", "C"),
    preferred_int_types_b=("RF",),
    failure_mode="Endless system redesign.",
    escalation_trigger=("Rebuild happens twice without producing runnable artifact"),
    kernel_lever=(
        "Impose 'rule freeze windows': after Shiro builds, "
        "disallow RF for 2 turns unless entropy spikes."
    ),
)

COVENANT_F = Covenant(
    agent_a="shikamaru",
    agent_b="*",  # system-wide coupling
    coupling="Efficiency stabilizer",
    signature="branch-pruning; anti-runaway",
    preferred_int_types_a=("B", "RF", "CL"),
    preferred_int_types_b=(),  # universal coupling
    failure_mode="Cynical disengagement -> entropy stagnation.",
    escalation_trigger="Shikamaru disengages while entropy > 0.6",
    kernel_lever=(
        "Use as 'cooling sink' after high entropy spike. "
        "Do not place as primary catalyst early; deploy mid-run."
    ),
)


# ============================================================
# High-Risk Triads
# ============================================================

TRIAD_1 = HighRiskTriad(
    agents=("rick", "makishima", "johan"),
    signature="philosophical chaos + trust corrosion",
    kernel_stance="keep off-stage unless explicitly needed",
    exit_condition=(
        "One RF (Rick), one ethical constraint (Makishima), "
        "one explicit CO artifact (Johan), then cooling sink (Shikamaru)."
    ),
)

TRIAD_2 = HighRiskTriad(
    agents=("light", "johan", "lawliet"),
    signature=("silent authoritarian convergence (low visible entropy, high hidden risk)"),
    kernel_stance=(
        "Require transparency constraints: "
        "at least one CL turn to externalize assumptions, "
        "at least one dissenting challenge (Makishima or Rick) before synthesis."
    ),
    exit_condition=("CL extraction completed and dissent tested before allowing synthesis."),
)


# ============================================================
# Covenant Trigger Events (Kernel-readable)
# ============================================================

COVENANT_TRIGGERS: dict[str, str] = {
    "COVENANT_LOCK": "repeated C without constraint gain",
    "RF_SPIRAL": "repeated RF without anchor",
    "TRUST_CORROSION": ("rising entropy + coalition divergence without explicit contradiction"),
    "OVERFORMALIZATION": "Shiro rules expanding while progress stalls",
    "DIRECTIONAL_TYRANNY": "Light dominates without validation",
}

# Trigger -> steering intervention mapping
TRIGGER_INTERVENTIONS: dict[str, str] = {
    "RF_SPIRAL": "P2 Cooling Sink + P3 Compression Pass; deny RF for 2 turns",
    "COVENANT_LOCK": "inject Rick RF once, then Lawliet CL/B",
    "TRUST_CORROSION": "P5 Transparency Clamp + Lawliet CL",
    "OVERFORMALIZATION": "Rule Freeze Window + Shikamaru pruning",
    "DIRECTIONAL_TYRANNY": "Makishima ethical C + Lawliet CL to validate assumptions",
}


# ============================================================
# Registry + Lookup
# ============================================================

ALL_COVENANTS: tuple[Covenant, ...] = (
    COVENANT_A,
    COVENANT_B,
    COVENANT_C,
    COVENANT_D,
    COVENANT_E,
    COVENANT_F,
)

ALL_TRIADS: tuple[HighRiskTriad, ...] = (TRIAD_1, TRIAD_2)

# Build lookup index: frozenset(agent_a, agent_b) -> Covenant
_COVENANT_INDEX: dict[frozenset[str], Covenant] = {}
for _cov in ALL_COVENANTS:
    if _cov.agent_b == "*":
        # System-wide covenants stored under agent's own key
        _COVENANT_INDEX[frozenset({_cov.agent_a, "*"})] = _cov
    else:
        _COVENANT_INDEX[frozenset({_cov.agent_a, _cov.agent_b})] = _cov


def get_covenant(agent_a: str, agent_b: str) -> Covenant | None:
    """Look up the covenant between two agents. Order-independent.

    Returns None if no specific covenant exists.
    For system-wide covenants (shikamaru/*, johan/*), pass "*" as agent_b
    or the specific agent id.
    """
    key = frozenset({agent_a, agent_b})
    cov = _COVENANT_INDEX.get(key)
    if cov is not None:
        return cov
    # Check for system-wide covenants
    for agent in (agent_a, agent_b):
        wildcard_key = frozenset({agent, "*"})
        if wildcard_key in _COVENANT_INDEX:
            return _COVENANT_INDEX[wildcard_key]
    return None


def get_triads_involving(agent_id: str) -> list[HighRiskTriad]:
    """Return all high-risk triads that include the given agent."""
    return [t for t in ALL_TRIADS if agent_id in t.agents]
