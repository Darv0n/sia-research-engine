"""SIA Full Deliberation — All 7 agents on Opus, full entropy dynamics.

Runs the complete SIA cognitive lens array over a research report,
tracking thermodynamic entropy at every turn. Produces a deliberation
paper showing the agent dynamics from initial state to nexus position.

Usage:
    .venv/Scripts/python.exe scripts/sia_deliberation.py outputs/meta-agents-mcp-report.md
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env file if present
_env_path = PROJECT_ROOT / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, val = line.partition("=")
            val = val.strip().strip("\"'")
            os.environ.setdefault(key.strip(), val)

import anthropic

from deep_research_swarm.sia.agents import ALL_AGENTS, AGENT_BY_ID, SIAAgent
from deep_research_swarm.sia.kernel import (
    SIAKernel,
    VALID_IGNITION_PATTERNS,
    KERNEL_FRAME_TEMPLATE,
)
from deep_research_swarm.sia.entropy import compute_entropy, classify_band
from deep_research_swarm.contracts import TurnRecord


# ============================================================
# Configuration
# ============================================================

OPUS_MODEL = os.environ.get("OPUS_MODEL", "claude-opus-4-6")
MAX_TURNS = 14  # Extended: all 7 agents x 2 passes
TOKEN_BUDGET = 80000
INITIAL_ENTROPY = 0.55  # Start in turbulence — maximum generative tension


# ============================================================
# Entropy Tracker (standalone, doesn't need full ResearchState)
# ============================================================

class EntropyTracker:
    """Tracks entropy dynamics across deliberation turns."""

    def __init__(self, initial_e: float = 0.55):
        self.history: list[dict[str, Any]] = []
        self.e = initial_e
        self.e_amb = 0.5
        self.e_conf = 0.5
        self.e_nov = 0.6
        self.e_trust = 0.5

    def update(self, turn_record: TurnRecord, kernel: SIAKernel) -> dict[str, Any]:
        """Recompute entropy from deliberation dynamics."""
        constraints = len(kernel.constraints)
        challenges = len(turn_record.get("challenges", []))
        reframes = len(turn_record.get("reframes", []))
        unresolved = len(kernel._unresolved)

        # Constraint accumulation reduces ambiguity
        if constraints > 0:
            constraint_pressure = min(constraints / 15, 1.0)
            self.e_amb = max(0.05, self.e_amb - 0.04 * constraint_pressure)

        # Challenges increase conflict component
        if challenges > 0:
            self.e_conf = min(0.95, self.e_conf + 0.03 * challenges)
        else:
            self.e_conf = max(0.05, self.e_conf - 0.02)

        # Reframes spike novelty, then decay
        if reframes > 0:
            self.e_nov = min(0.95, self.e_nov + 0.06 * reframes)
        else:
            self.e_nov = max(0.05, self.e_nov - 0.03)

        # Trust improves when constraints resolve, degrades on unresolved growth
        if unresolved > 3:
            self.e_trust = min(0.95, self.e_trust + 0.02)
        elif constraints > unresolved:
            self.e_trust = max(0.05, self.e_trust - 0.03)

        # Scalar entropy
        self.e = round(
            0.30 * self.e_amb + 0.30 * self.e_conf + 0.20 * self.e_nov + 0.20 * self.e_trust,
            4,
        )
        band = classify_band(self.e)

        snapshot = {
            "turn": len(self.history) + 1,
            "agent": turn_record.get("agent", "unknown"),
            "e": self.e,
            "e_amb": round(self.e_amb, 4),
            "e_conf": round(self.e_conf, 4),
            "e_nov": round(self.e_nov, 4),
            "e_trust": round(self.e_trust, 4),
            "band": band,
            "constraints_total": constraints,
            "challenges_this_turn": challenges,
            "reframes_this_turn": reframes,
            "unresolved": unresolved,
        }
        self.history.append(snapshot)
        return snapshot


# ============================================================
# Opus Caller
# ============================================================

async def call_opus(
    client: anthropic.AsyncAnthropic,
    system: str,
    messages: list[dict],
    max_tokens: int = 4096,
) -> tuple[str, dict]:
    """Call Opus and return (text, usage_dict)."""
    response = await client.messages.create(
        model=OPUS_MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=messages,
    )
    text = response.content[0].text if response.content else ""
    usage = {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }
    return text, usage


# ============================================================
# Agent Sequencing
# ============================================================

def plan_agent_sequence(entropy_tracker: EntropyTracker) -> list[SIAAgent]:
    """Plan the full agent sequence — all 7 agents, 2 passes.

    Pass 1: Core 5 in entropy-aware order
    Pass 2: Reserve 2 (Shiro, Johan) + revisit key agents based on dynamics
    """
    band = classify_band(entropy_tracker.e)

    # Pass 1: entropy-aware ordering of core agents
    if band in ("runaway", "turbulence"):
        # High entropy: anchor first (Lawliet), then direction (Light),
        # then rupture (Rick), legitimacy (Makishima), prune (Shikamaru)
        pass1 = ["lawliet", "light", "rick", "makishima", "shikamaru"]
    elif band == "convergence":
        # Medium: direction first, then validate
        pass1 = ["light", "lawliet", "makishima", "rick", "shikamaru"]
    else:  # crystalline
        # Low entropy: rupture first to test if convergence is real
        pass1 = ["rick", "makishima", "lawliet", "light", "shikamaru"]

    # Pass 2: reserves + revisit
    pass2 = [
        "shiro",    # Formalize what emerged
        "johan",    # Coalition alignment audit
        "lawliet",  # Final constraint extraction
        "light",    # Final directional commitment
        "rick",     # Final frame check
        "makishima", # Final legitimacy check
        "shikamaru", # Final readiness assessment
    ]

    # Build full sequence (capped at MAX_TURNS)
    full = pass1 + pass2
    agents = []
    for aid in full[:MAX_TURNS]:
        agent = AGENT_BY_ID.get(aid)
        if agent:
            agents.append(agent)
    return agents


# ============================================================
# Main Deliberation
# ============================================================

async def run_deliberation(report_path: str) -> str:
    """Run full SIA deliberation over a research report. Returns formatted paper."""
    report_text = Path(report_path).read_text(encoding="utf-8")

    # Truncate to ~8K chars for source_summary (keeps Opus context manageable)
    # Extract the main body (skip YAML frontmatter and metadata sections)
    body_start = report_text.find("# ")
    if body_start > 0:
        body = report_text[body_start:]
    else:
        body = report_text

    # Find where metadata sections start
    for marker in ["## Confidence Assessment", "## Confidence Trends", "## Source Diversity"]:
        idx = body.find(marker)
        if idx > 0:
            body = body[:idx]
            break

    source_summary = body[:12000]
    research_question = "Multi-agent architectures that generate other agents and MCP servers"

    client = anthropic.AsyncAnthropic()

    # Initialize
    entropy = EntropyTracker(initial_e=INITIAL_ENTROPY)
    kernel = SIAKernel(
        max_turns=MAX_TURNS,
        token_budget=TOKEN_BUDGET,
        entropy_band=classify_band(INITIAL_ENTROPY),
        entropy_value=INITIAL_ENTROPY,
    )

    conversation: list[dict[str, str]] = []
    all_turns: list[dict[str, Any]] = []
    all_usage: list[dict] = []
    total_tokens = 0

    agents_sequence = plan_agent_sequence(entropy)

    print(f"SIA Deliberation — {len(agents_sequence)} turns planned", file=sys.stderr)
    print(f"Initial entropy: {entropy.e:.4f} ({classify_band(entropy.e)})", file=sys.stderr)
    print(f"Model: {OPUS_MODEL}", file=sys.stderr)
    print("---", file=sys.stderr)

    for turn_idx, agent in enumerate(agents_sequence):
        # Check budget
        if total_tokens >= TOKEN_BUDGET:
            print(f"  [Budget exhausted at turn {turn_idx + 1}]", file=sys.stderr)
            break

        # Build Kernel frame
        turn_instruction = _get_turn_instruction(agent, entropy, kernel, turn_idx)
        frame = KERNEL_FRAME_TEMPLATE.format(
            turn=turn_idx + 1,
            max_turns=len(agents_sequence),
            entropy_band=classify_band(entropy.e),
            entropy=entropy.e,
            constraint_count=len(kernel.constraints),
            unresolved_count=len(kernel._unresolved),
            rejected_count=len(kernel._rejected_branches),
            agent_name=agent.name,
            agent_archetype=agent.archetype,
            turn_instruction=turn_instruction,
        )

        # Build cognitive lens with template vars
        constraints_text = "\n".join(f"- {c}" for c in kernel.constraints) or "None yet."
        prior_summary = _build_prior(conversation)
        coverage_gaps = "See evidence gaps in source material."
        tensions_text = "See active tensions in source material."

        filled_lens = agent.cognitive_lens.format(
            research_question=research_question,
            source_summary=source_summary,
            entropy_band=classify_band(entropy.e),
            constraints=constraints_text,
            prior_summary=prior_summary,
            coverage_gaps=coverage_gaps,
            active_tensions=tensions_text,
        )

        system_prompt = frame + "\n\n" + filled_lens

        # Build messages: shared conversation thread (last 8 messages)
        recent = conversation[-8:] if len(conversation) > 8 else list(conversation)
        messages = recent + [
            {"role": "user", "content": "Analyze the evidence and respond according to your processing bias."}
        ]

        print(f"  [{agent.name}] ({agent.archetype}) — e={entropy.e:.4f} ...", end="", file=sys.stderr)
        start = time.monotonic()

        try:
            raw_output, usage = await call_opus(client, system_prompt, messages)
        except Exception as exc:
            print(f" ERROR: {exc}", file=sys.stderr)
            continue

        elapsed = time.monotonic() - start
        tokens_this = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
        total_tokens += tokens_this
        all_usage.append(usage)

        # Parse turn output via kernel
        record = kernel.parse_turn_output(agent, raw_output)
        kernel.update_state(record, tokens_used=tokens_this)

        # Update entropy
        entropy_snap = entropy.update(record, kernel)

        # Update kernel's entropy state
        kernel.entropy_band = entropy_snap["band"]
        kernel.entropy_value = entropy_snap["e"]

        # Grow conversation thread
        conversation.append({"role": "assistant", "content": raw_output})
        conversation.append({
            "role": "user",
            "content": (
                f"[Turn {turn_idx + 1} complete. Agent: {agent.name}. "
                f"Constraints: +{len(record.get('constraints', []))}={len(kernel.constraints)}. "
                f"Entropy: {entropy_snap['e']:.4f} ({entropy_snap['band']})]"
            ),
        })

        # Store turn data
        all_turns.append({
            "turn": turn_idx + 1,
            "agent": agent.name,
            "archetype": agent.archetype,
            "output": raw_output,
            "record": dict(record),
            "entropy": entropy_snap,
            "tokens": tokens_this,
            "elapsed": round(elapsed, 1),
        })

        print(
            f" {elapsed:.1f}s | +{len(record.get('constraints', []))}c "
            f"| e={entropy_snap['e']:.4f} ({entropy_snap['band']})",
            file=sys.stderr,
        )

    # Harvest from kernel
    reactor_state, reactor_trace = kernel.harvest()

    print("---", file=sys.stderr)
    print(f"Deliberation complete: {len(all_turns)} turns, {total_tokens} tokens", file=sys.stderr)

    # Generate the paper
    paper = _format_paper(
        research_question=research_question,
        all_turns=all_turns,
        entropy_tracker=entropy,
        reactor_state=dict(reactor_state),
        reactor_trace=dict(reactor_trace),
        total_tokens=total_tokens,
        all_usage=all_usage,
    )

    return paper


# ============================================================
# Turn Instructions (entropy-band aware)
# ============================================================

def _get_turn_instruction(
    agent: SIAAgent,
    entropy: EntropyTracker,
    kernel: SIAKernel,
    turn_idx: int,
) -> str:
    """Generate per-turn instruction based on entropy dynamics."""
    band = classify_band(entropy.e)

    if turn_idx == 0:
        return "IGNITION — Establish the opening frame. What does the evidence demand?"

    if band == "runaway":
        if agent.id in ("lawliet", "shikamaru"):
            return "ANCHOR — Entropy is dangerously high. Extract hard constraints to stabilize."
        return "COMPRESS — High entropy. Focus on what the evidence definitively establishes."

    if band == "turbulence":
        if agent.id == "rick":
            return "RUPTURE — Controlled frame break. Expose ONE hidden assumption, then anchor."
        if agent.id == "makishima":
            return "STRESS-TEST — Challenge the strongest claim. Find the value assumption."
        return "BUILD — Entropy is productive. Add structure without premature closure."

    if band == "convergence":
        if agent.id == "shikamaru":
            return "READINESS CHECK — Is convergence genuine or premature? Assess minimal path."
        return "REFINE — Evidence is settling. Sharpen claims, eliminate redundancy."

    # crystalline
    if agent.id == "rick":
        return "FINAL CHECK — Is this crystalline state earned or is it false convergence?"
    return "HARVEST — Extract final position. State the nexus conclusion."


# ============================================================
# Helpers
# ============================================================

def _build_prior(conversation: list[dict[str, str]]) -> str:
    if not conversation:
        return "No prior conversation."
    recent = conversation[-6:]
    parts = []
    for msg in recent:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")[:600]
        parts.append(f"[{role}] {content}")
    return "\n".join(parts)


# ============================================================
# Paper Formatter
# ============================================================

def _format_paper(
    research_question: str,
    all_turns: list[dict[str, Any]],
    entropy_tracker: EntropyTracker,
    reactor_state: dict,
    reactor_trace: dict,
    total_tokens: int,
    all_usage: list[dict],
) -> str:
    """Format the deliberation into a structured paper."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    total_cost = sum(
        u.get("input_tokens", 0) * 15 / 1_000_000 + u.get("output_tokens", 0) * 75 / 1_000_000
        for u in all_usage
    )

    lines: list[str] = []
    lines.append("---")
    lines.append(f"title: 'SIA Deliberation: {research_question}'")
    lines.append(f"generated: '{now}'")
    lines.append(f"model: '{OPUS_MODEL}'")
    lines.append(f"turns: {len(all_turns)}")
    lines.append(f"agents: {len(set(t['agent'] for t in all_turns))}")
    lines.append(f"total_tokens: {total_tokens}")
    lines.append(f"total_cost_usd: {total_cost:.4f}")
    lines.append(f"final_entropy: {entropy_tracker.e:.4f}")
    lines.append(f"final_band: '{classify_band(entropy_tracker.e)}'")
    lines.append(f"constraints_produced: {len(reactor_state.get('constraints', []))}")
    lines.append(f"termination: '{reactor_trace.get('termination_reason', 'complete')}'")
    lines.append("---")
    lines.append("")

    # Title
    lines.append(f"# SIA Deliberation: Dynamic to Nexus")
    lines.append("")
    lines.append(f"**Research question**: {research_question}")
    lines.append("")
    lines.append(
        "This document records a full Semantic Intelligence Architecture (SIA) deliberation "
        "over the research output. Seven cognitive lenses — each a distinct processing bias "
        "on the same Opus model — engage in multi-turn thermodynamic deliberation. Entropy "
        "tracks the system's information state: high entropy (turbulence/runaway) means "
        "unresolved tension; low entropy (convergence/crystalline) means the evidence has "
        "settled. The deliberation traces the path from initial dynamic state to final "
        "nexus position."
    )
    lines.append("")

    # Entropy Dynamics Table
    lines.append("## Entropy Dynamics")
    lines.append("")
    lines.append("| Turn | Agent | Band | e | e_amb | e_conf | e_nov | e_trust | Constraints | Challenges |")
    lines.append("|------|-------|------|---|-------|--------|-------|---------|-------------|------------|")
    for snap in entropy_tracker.history:
        lines.append(
            f"| {snap['turn']} | {snap['agent']} | {snap['band']} | "
            f"{snap['e']:.4f} | {snap['e_amb']:.4f} | {snap['e_conf']:.4f} | "
            f"{snap['e_nov']:.4f} | {snap['e_trust']:.4f} | "
            f"{snap['constraints_total']} | {snap['challenges_this_turn']} |"
        )
    lines.append("")

    # Entropy Sparkline
    lines.append("### Entropy Trajectory")
    lines.append("")
    bars = "▁▂▃▄▅▆▇█"
    sparkline = ""
    for snap in entropy_tracker.history:
        idx = min(int(snap["e"] * len(bars)), len(bars) - 1)
        sparkline += bars[idx]
    lines.append(f"`{sparkline}`")
    lines.append("")
    lines.append(
        f"Start: {entropy_tracker.history[0]['e']:.4f} ({entropy_tracker.history[0]['band']}) → "
        f"End: {entropy_tracker.history[-1]['e']:.4f} ({entropy_tracker.history[-1]['band']})"
        if entropy_tracker.history else "No turns executed."
    )
    lines.append("")

    # Agent Basin Profiles
    lines.append("## Agent Basin Profiles")
    lines.append("")
    lines.append("| Agent | Archetype | A1 | A2 | A3 | A4 | A5 | A6 | A7 |")
    lines.append("|-------|-----------|-----|-----|-----|-----|-----|-----|-----|")
    seen = set()
    for t in all_turns:
        if t["agent"] in seen:
            continue
        seen.add(t["agent"])
        agent = AGENT_BY_ID.get(t["agent"].lower())
        if agent:
            b = agent.basin
            lines.append(
                f"| {agent.name} | {agent.archetype} | "
                f"{b.a1_constraint_density:+.1f} | {b.a2_directionality:+.1f} | "
                f"{b.a3_ethical_charge:+.1f} | {b.a4_structural_formalism:+.1f} | "
                f"{b.a5_frame_stability:+.1f} | {b.a6_coalition_cohesion:+.1f} | "
                f"{b.a7_energy_expenditure:+.1f} |"
            )
    lines.append("")

    # Full Deliberation Transcript
    lines.append("## Deliberation Transcript")
    lines.append("")

    for t in all_turns:
        snap = t["entropy"]
        record = t["record"]
        lines.append(f"### Turn {t['turn']}: {t['agent']} ({t['archetype']})")
        lines.append("")
        lines.append(
            f"**Entropy**: {snap['e']:.4f} ({snap['band']}) | "
            f"**Tokens**: {t['tokens']:,} | **Time**: {t['elapsed']}s"
        )
        lines.append("")

        # Constraints extracted
        constraints = record.get("constraints", [])
        if constraints:
            lines.append(f"**Constraints extracted** ({len(constraints)}):")
            for c in constraints:
                lines.append(f"- {c}")
            lines.append("")

        # Challenges issued
        challenges = record.get("challenges", [])
        if challenges:
            lines.append(f"**Challenges issued** ({len(challenges)}):")
            for ch in challenges:
                lines.append(f"- {ch}")
            lines.append("")

        # Reframes proposed
        reframes = record.get("reframes", [])
        if reframes:
            lines.append(f"**Reframes proposed** ({len(reframes)}):")
            for rf in reframes:
                lines.append(f"- {rf}")
            lines.append("")

        # Full output
        lines.append("<details>")
        lines.append(f"<summary>Full {t['agent']} output ({len(t['output'])} chars)</summary>")
        lines.append("")
        lines.append(t["output"])
        lines.append("")
        lines.append("</details>")
        lines.append("")

    # Reactor Products
    lines.append("## Nexus Position: Reactor Products")
    lines.append("")

    constraints = reactor_state.get("constraints", [])
    if constraints:
        lines.append(f"### Constraints ({len(constraints)})")
        lines.append("")
        for i, c in enumerate(constraints, 1):
            lines.append(f"{i}. {c}")
        lines.append("")

    rejected = reactor_state.get("rejected_branches", [])
    if rejected:
        lines.append(f"### Rejected Branches ({len(rejected)})")
        lines.append("")
        for r in rejected:
            lines.append(f"- {r}")
        lines.append("")

    frames = reactor_state.get("active_frames", [])
    if frames:
        lines.append(f"### Active Frames ({len(frames)})")
        lines.append("")
        for f in frames:
            lines.append(f"- {f}")
        lines.append("")

    unresolved = reactor_state.get("unresolved", [])
    if unresolved:
        lines.append(f"### Unresolved Variables ({len(unresolved)})")
        lines.append("")
        for u in unresolved:
            lines.append(f"- {u}")
        lines.append("")

    # Coalition Map
    coalition = reactor_state.get("coalition_map", {})
    if coalition:
        lines.append("### Coalition Map")
        lines.append("")
        for position, agents in coalition.items():
            lines.append(f"- **{position}**: {', '.join(agents)}")
        lines.append("")

    # Reactor Trace
    lines.append("## Reactor Trace")
    lines.append("")
    lines.append(f"- **Turns executed**: {reactor_trace.get('turns_executed', 0)}")
    lines.append(f"- **Agents used**: {', '.join(reactor_trace.get('agents_used', []))}")
    lines.append(f"- **Constraints produced**: {reactor_trace.get('constraints_produced', 0)}")
    lines.append(f"- **Branches killed**: {reactor_trace.get('branches_killed', 0)}")
    lines.append(f"- **Challenges issued**: {reactor_trace.get('challenges_issued', 0)}")
    lines.append(f"- **Final entropy**: {reactor_trace.get('final_entropy', 0):.4f}")
    lines.append(f"- **Termination reason**: {reactor_trace.get('termination_reason', 'unknown')}")
    lines.append(f"- **Ignition pattern**: {reactor_trace.get('ignition_pattern', 'unknown')}")
    lines.append("")

    # Cost summary
    lines.append("## Cost Summary")
    lines.append("")
    lines.append(f"- **Model**: {OPUS_MODEL}")
    lines.append(f"- **Total tokens**: {total_tokens:,}")
    lines.append(f"- **Estimated cost**: ${total_cost:.4f}")
    lines.append(f"- **Turns**: {len(all_turns)}")
    lines.append(f"- **Unique agents**: {len(set(t['agent'] for t in all_turns))}")
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append(f"*SIA Deliberation generated {now} | {OPUS_MODEL} | {len(all_turns)} turns*")

    return "\n".join(lines)


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/sia_deliberation.py <report.md> [output.md]", file=sys.stderr)
        sys.exit(1)

    report_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "outputs/sia-deliberation.md"

    paper = asyncio.run(run_deliberation(report_path))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(paper, encoding="utf-8")
    print(f"\nPaper written to: {output_path}", file=sys.stderr)
