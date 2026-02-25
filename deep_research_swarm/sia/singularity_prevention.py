"""Singularity prevention — pre-synthesis safety gate.

Detects structural collapse patterns that indicate the deliberation has
converged on a false attractor. If any check fails, synthesis is blocked
and the pipeline should re-iterate.

Four singularity checks:
  1. Constraint singularity — all constraints from one agent
  2. Directional singularity — no reframes accepted (mono-perspective)
  3. Reframe singularity — all reframes, no constraints (entropy without signal)
  4. Coalition shadow — hidden alignment (agents appear to agree but
     coalition map shows no explicit coalition moves)

Plus 7-axis stability check from LatentBasinMap (A1-A7).
"""

from __future__ import annotations

from typing import Any

from deep_research_swarm.sia.agents import AGENT_BY_ID


def check_constraint_singularity(
    reactor_state: dict[str, Any],
) -> tuple[bool, str]:
    """Check if all constraints come from a single agent.

    Returns (safe, reason). safe=True means no singularity detected.
    """
    turn_log = reactor_state.get("turn_log", [])
    if not turn_log:
        return True, "no_turn_log"

    constraints = reactor_state.get("constraints", [])
    if len(constraints) < 2:
        return True, "insufficient_constraints"

    # Count constraints by agent
    agent_constraints: dict[str, int] = {}
    for record in turn_log:
        agent = record.get("agent", "")
        count = len(record.get("constraints", []))
        if count > 0:
            agent_constraints[agent] = agent_constraints.get(agent, 0) + count

    if not agent_constraints:
        return True, "no_agent_constraints"

    # Singularity: one agent produced all constraints
    total = sum(agent_constraints.values())
    if total == 0:
        return True, "zero_total_constraints"

    for agent_id, count in agent_constraints.items():
        if count == total:
            return False, (f"constraint_singularity: all {total} constraints from {agent_id}")

    # Also flag if one agent has > 80% of constraints
    max_agent = max(agent_constraints, key=agent_constraints.get)
    max_ratio = agent_constraints[max_agent] / total
    if max_ratio > 0.8 and total >= 3:
        return False, (
            f"constraint_near_singularity: {max_agent} has "
            f"{agent_constraints[max_agent]}/{total} ({max_ratio:.0%})"
        )

    return True, "constraint_diversity_ok"


def check_directional_singularity(
    reactor_state: dict[str, Any],
) -> tuple[bool, str]:
    """Check if no reframes were accepted (mono-perspective).

    Returns (safe, reason). safe=True means no singularity detected.
    """
    turn_log = reactor_state.get("turn_log", [])
    active_frames = reactor_state.get("active_frames", [])

    if len(turn_log) < 3:
        return True, "insufficient_turns"

    # Check if any RF-type turns occurred
    rf_turns = [r for r in turn_log if r.get("int_type") == "RF"]

    if not rf_turns and len(turn_log) >= 4:
        return False, (f"directional_singularity: no reframes in {len(turn_log)} turns")

    # RF occurred but no frames were accepted
    if rf_turns and not active_frames and len(turn_log) >= 4:
        return False, (
            f"directional_singularity: {len(rf_turns)} reframes attempted but none accepted"
        )

    return True, "directional_diversity_ok"


def check_reframe_singularity(
    reactor_state: dict[str, Any],
) -> tuple[bool, str]:
    """Check for all reframes with no constraints (entropy without signal).

    Returns (safe, reason). safe=True means no singularity detected.
    """
    constraints = reactor_state.get("constraints", [])
    active_frames = reactor_state.get("active_frames", [])
    turn_log = reactor_state.get("turn_log", [])

    if len(turn_log) < 3:
        return True, "insufficient_turns"

    # Singularity: many reframes, zero constraints
    if len(active_frames) >= 3 and len(constraints) == 0:
        return False, (f"reframe_singularity: {len(active_frames)} frames with 0 constraints")

    # Near-singularity: reframes vastly outnumber constraints
    if (
        len(active_frames) > 0
        and len(constraints) > 0
        and len(active_frames) / max(len(constraints), 1) > 3.0
        and len(turn_log) >= 4
    ):
        return False, (
            f"reframe_near_singularity: {len(active_frames)} frames "
            f"vs {len(constraints)} constraints (ratio > 3:1)"
        )

    return True, "reframe_balance_ok"


def check_coalition_shadow(
    reactor_state: dict[str, Any],
) -> tuple[bool, str]:
    """Check for hidden alignment (apparent agreement without explicit coalitions).

    Returns (safe, reason). safe=True means no singularity detected.
    """
    turn_log = reactor_state.get("turn_log", [])
    coalition_map = reactor_state.get("coalition_map", {})
    constraints = reactor_state.get("constraints", [])

    if len(turn_log) < 4:
        return True, "insufficient_turns"

    # Count agents who participated
    agents_participated = set(r.get("agent", "") for r in turn_log)

    # Check: many agents agree (few challenges) but no explicit coalitions
    challenges = []
    for r in turn_log:
        challenges.extend(r.get("challenges", []))

    if (
        len(agents_participated) >= 3
        and len(challenges) == 0
        and len(coalition_map) == 0
        and len(constraints) >= 2
    ):
        return False, (
            f"coalition_shadow: {len(agents_participated)} agents, "
            f"0 challenges, 0 explicit coalitions — hidden alignment"
        )

    return True, "coalition_transparency_ok"


def check_axis_stability(
    reactor_state: dict[str, Any],
) -> tuple[bool, str, dict[str, float]]:
    """Check 7-axis stability from participating agents' basin profiles.

    Returns (safe, reason, axis_summary).
    An axis is unstable if all participating agents push it the same direction
    (no opposing force on that axis).
    """
    turn_log = reactor_state.get("turn_log", [])
    if not turn_log:
        return True, "no_turn_log", {}

    # Get unique agents who participated
    agent_ids = list(dict.fromkeys(r.get("agent", "") for r in turn_log))
    agents = [AGENT_BY_ID[aid] for aid in agent_ids if aid in AGENT_BY_ID]

    if len(agents) < 2:
        return True, "insufficient_agents", {}

    # For each axis, check if there's opposing force
    axis_names = [
        "a1_constraint_density",
        "a2_directionality",
        "a3_ethical_charge",
        "a4_structural_formalism",
        "a5_frame_stability",
        "a6_coalition_cohesion",
        "a7_energy_expenditure",
    ]

    axis_summary: dict[str, float] = {}
    unstable_axes: list[str] = []

    for axis in axis_names:
        values = [getattr(agent.basin, axis) for agent in agents]
        axis_summary[axis] = sum(values) / len(values)

        # Check for uniform direction (all positive or all negative, abs > 0.3)
        all_positive = all(v > 0.3 for v in values)
        all_negative = all(v < -0.3 for v in values)

        if all_positive or all_negative:
            direction = "positive" if all_positive else "negative"
            unstable_axes.append(f"{axis}({direction})")

    if len(unstable_axes) >= 3:
        return (
            False,
            (
                f"axis_instability: {len(unstable_axes)} axes unchecked — "
                f"{', '.join(unstable_axes[:3])}"
            ),
            axis_summary,
        )

    return True, "axes_balanced", axis_summary


def singularity_check(
    reactor_state: dict[str, Any],
) -> tuple[bool, str, dict[str, Any]]:
    """Run all singularity prevention checks.

    Returns (safe, reason, details).
    safe=False if ANY check fails.
    """
    details: dict[str, Any] = {}

    # Check 1: Constraint singularity
    safe1, reason1 = check_constraint_singularity(reactor_state)
    details["constraint"] = {"safe": safe1, "reason": reason1}

    # Check 2: Directional singularity
    safe2, reason2 = check_directional_singularity(reactor_state)
    details["directional"] = {"safe": safe2, "reason": reason2}

    # Check 3: Reframe singularity
    safe3, reason3 = check_reframe_singularity(reactor_state)
    details["reframe"] = {"safe": safe3, "reason": reason3}

    # Check 4: Coalition shadow
    safe4, reason4 = check_coalition_shadow(reactor_state)
    details["coalition"] = {"safe": safe4, "reason": reason4}

    # Check 5: 7-axis stability
    safe5, reason5, axis_summary = check_axis_stability(reactor_state)
    details["axes"] = {"safe": safe5, "reason": reason5, "summary": axis_summary}

    # Aggregate
    all_safe = safe1 and safe2 and safe3 and safe4 and safe5
    if not all_safe:
        failed = [k for k, v in details.items() if not v["safe"]]
        reason = f"singularity_detected: {', '.join(failed)}"
    else:
        reason = "singularity_safe"

    return all_safe, reason, details
