"""SIAKernel — thermodynamic steering engine for multi-turn deliberation.

The Kernel is the conductor: it selects speakers, frames turns, parses output,
tracks state, and decides when deliberation has converged. It does NOT generate
content — it orchestrates the agents who do.

Key responsibilities:
- Speaker selection via entropy-band steering + fatigue + anti-dominance
- Ignition doctrine: 4 valid opening patterns (never start with cooling sink)
- Turn framing: builds per-turn prompts with Kernel context injection
- Output parsing: extracts TurnRecord (constraints, challenges, reframes, int_type)
- State tracking: accumulates ReactorState across turns
- Termination: budget/turn/entropy/stagnation checks
- Harvest: produces deduplicated constraints, coalition map, rejected branches
"""

from __future__ import annotations

import json
import re
from typing import Any

from deep_research_swarm.contracts import (
    IntType,
    ReactorState,
    ReactorTrace,
    TurnRecord,
)
from deep_research_swarm.sia.agents import (
    AGENT_BY_ID,
    CORE_AGENTS,
    RESERVE_AGENTS,
    SIAAgent,
)
from deep_research_swarm.sia.covenants import get_covenant

# ============================================================
# Constants
# ============================================================

# Ignition patterns: valid first-speaker selections.
# Never start with a cooling sink (Shikamaru) — that kills entropy before it starts.
VALID_IGNITION_PATTERNS: tuple[tuple[str, str], ...] = (
    ("lawliet", "light"),  # Constraint extraction -> strategic direction
    ("light", "lawliet"),  # Strategic framing -> constraint validation
    ("lawliet", "rick"),  # Constraint extraction -> frame rupture (high-energy)
    ("light", "makishima"),  # Strategic direction -> legitimacy test
)

# Maximum consecutive turns for any single agent (anti-dominance)
MAX_CONSECUTIVE = 2

# Stagnation: consecutive turns with zero new constraints
STAGNATION_THRESHOLD = 3

# Minimum agents that must participate in a reactor run
MIN_DISTINCT_AGENTS = 3

# ============================================================
# Kernel Framing Template
# ============================================================

KERNEL_FRAME_TEMPLATE = """\
[KERNEL FRAME — Turn {turn}/{max_turns}]
Entropy band: {entropy_band} (e={entropy:.2f})
Constraints accumulated: {constraint_count}
Unresolved variables: {unresolved_count}
Rejected branches: {rejected_count}
Your role this turn: {agent_name} ({agent_archetype})
Instruction: {turn_instruction}
"""


# ============================================================
# SIAKernel
# ============================================================


class SIAKernel:
    """Thermodynamic steering engine for SIA reactor deliberation."""

    def __init__(
        self,
        max_turns: int = 6,
        token_budget: int = 20000,
        entropy_band: str = "convergence",
        entropy_value: float = 0.35,
    ) -> None:
        self.max_turns = max_turns
        self.token_budget = token_budget
        self.entropy_band = entropy_band
        self.entropy_value = entropy_value

        # Mutable state
        self._turn: int = 0
        self._tokens_used: int = 0
        self._turn_log: list[TurnRecord] = []
        self._constraints: list[str] = []
        self._rejected_branches: list[str] = []
        self._active_frames: list[str] = []
        self._unresolved: list[str] = []
        self._coalition_map: dict[str, list[str]] = {}
        self._challenges: list[str] = []

        # Anti-dominance tracking
        self._consecutive_count: int = 0
        self._last_agent_id: str = ""
        self._agent_fatigue: dict[str, int] = {}  # agent_id -> turns used
        self._stagnation_count: int = 0  # consecutive turns with 0 new constraints

        # RF tracking for covenant enforcement
        self._rf_count_since_anchor: int = 0

    # ----------------------------------------------------------
    # Speaker Selection
    # ----------------------------------------------------------

    def select_speaker(self, turn: int | None = None) -> SIAAgent:
        """Select the next speaker based on entropy band, fatigue, and anti-dominance.

        Returns the SIAAgent to speak next.
        """
        effective_turn = turn if turn is not None else self._turn

        # Ignition: first two turns use ignition pattern
        if effective_turn < 2:
            pattern = self._select_ignition_pattern()
            agent_id = pattern[effective_turn]
            return AGENT_BY_ID[agent_id]

        # Build candidate pool
        candidates = self._build_candidate_pool()

        # Score candidates by entropy-band relevance
        scored = []
        for agent in candidates:
            score = self._score_candidate(agent)
            scored.append((score, agent))

        scored.sort(key=lambda x: x[0], reverse=True)

        # Return the highest-scoring candidate
        if scored:
            return scored[0][1]

        # Absolute fallback: Lawliet (always safe)
        return AGENT_BY_ID["lawliet"]

    def _select_ignition_pattern(self) -> tuple[str, str]:
        """Select ignition pattern based on entropy band.

        Uses VALID_IGNITION_PATTERNS as the source of truth.
        """
        if self.entropy_band == "runaway":
            return VALID_IGNITION_PATTERNS[0]  # lawliet -> light
        if self.entropy_band == "turbulence":
            return VALID_IGNITION_PATTERNS[3]  # light -> makishima
        if self.entropy_band == "crystalline":
            return VALID_IGNITION_PATTERNS[2]  # lawliet -> rick
        # Default (convergence): standard pattern
        return VALID_IGNITION_PATTERNS[0]  # lawliet -> light

    def _build_candidate_pool(self) -> list[SIAAgent]:
        """Build the pool of eligible agents for this turn."""
        candidates: list[SIAAgent] = []

        for agent in CORE_AGENTS:
            # Anti-dominance: skip if agent just spoke MAX_CONSECUTIVE times
            if agent.id == self._last_agent_id and self._consecutive_count >= MAX_CONSECUTIVE:
                continue

            # Rick: only eligible when entropy > 0.40 or stagnation
            if agent.id == "rick":
                if self.entropy_value <= 0.40 and self._stagnation_count < STAGNATION_THRESHOLD:
                    continue

            candidates.append(agent)

        # Reserve agents: conditional activation
        for agent in RESERVE_AGENTS:
            if self._should_activate_reserve(agent):
                candidates.append(agent)

        # If no candidates (all blocked), reset anti-dominance and use Lawliet
        if not candidates:
            self._consecutive_count = 0
            candidates = [AGENT_BY_ID["lawliet"]]

        return candidates

    def _should_activate_reserve(self, agent: SIAAgent) -> bool:
        """Check if a reserve agent should be activated."""
        if agent.id == "shiro":
            # Activate when structural formalism is low (few constraints, many turns)
            return self._turn >= 4 and len(self._constraints) < 2
        if agent.id == "johan":
            # Activate when coalition cohesion is dangerously unexamined
            return self._turn >= 4 and len(self._coalition_map) == 0 and self.entropy_value > 0.45
        return False

    def _score_candidate(self, agent: SIAAgent) -> float:
        """Score a candidate agent for the current entropy band."""
        score = 1.0

        # Entropy-band scoring
        if self.entropy_band == "runaway":
            # Runaway: favor compression (Lawliet, Shikamaru), penalize rupture
            if agent.id in ("lawliet", "shikamaru"):
                score += 2.0
            if agent.id == "rick":
                score -= 1.0
        elif self.entropy_band == "turbulence":
            # Turbulence: anchor after rupture
            if self._last_agent_id == "rick":
                # After Rick, anchor with Lawliet
                if agent.id == "lawliet":
                    score += 3.0
            if agent.id in ("lawliet", "light"):
                score += 1.0
        elif self.entropy_band == "convergence":
            # Convergence: build and compress
            if agent.id in ("light", "lawliet"):
                score += 1.5
            if agent.id == "shikamaru":
                score += 0.5
        elif self.entropy_band == "crystalline":
            # Crystalline: harvest — Shikamaru to prune, Light to direct
            if agent.id == "shikamaru":
                score += 2.0
            if agent.id == "light":
                score += 1.0

        # Fatigue penalty: agents used more get penalized
        fatigue = self._agent_fatigue.get(agent.id, 0)
        score -= fatigue * 0.3

        # RF covenant enforcement: if too many reframes without anchor, penalize RF agents
        if self._rf_count_since_anchor >= 2 and agent.id == "rick":
            score -= 2.0

        # Covenant-aware scoring: check pairwise dynamics with last speaker
        # Only boost for explicit pairwise covenants, not wildcards (*, agent)
        if self._last_agent_id:
            cov = get_covenant(self._last_agent_id, agent.id)
            if cov is not None and cov.agent_b != "*" and cov.agent_a != "*":
                # Favor productive pairings
                score += 0.5

        return score

    # ----------------------------------------------------------
    # Turn Framing
    # ----------------------------------------------------------

    def frame_turn(self, agent: SIAAgent) -> str:
        """Build the Kernel framing prefix for this turn.

        This is prepended to the agent's cognitive_lens prompt to give
        the agent situational awareness of the reactor state.
        """
        # Select turn instruction based on entropy band + context
        instruction = self._turn_instruction(agent)

        return KERNEL_FRAME_TEMPLATE.format(
            turn=self._turn + 1,
            max_turns=self.max_turns,
            entropy_band=self.entropy_band,
            entropy=self.entropy_value,
            constraint_count=len(self._constraints),
            unresolved_count=len(self._unresolved),
            rejected_count=len(self._rejected_branches),
            agent_name=agent.name,
            agent_archetype=agent.archetype,
            turn_instruction=instruction,
        )

    def _turn_instruction(self, agent: SIAAgent) -> str:
        """Generate turn-specific instruction based on context."""
        if self._turn == 0:
            return "IGNITION — establish initial constraints from the evidence."

        if self.entropy_band == "runaway":
            return (
                "CONTAINMENT — entropy is dangerously high. "
                "Extract constraints, reject weak branches, compress."
            )

        if self.entropy_band == "turbulence" and self._last_agent_id == "rick":
            return (
                "ANCHOR — Rick just ruptured. "
                "Re-establish grounding. Extract new constraints from the reframe."
            )

        if self.entropy_band == "crystalline":
            return (
                "HARVEST — entropy is very low. "
                "Prune redundant branches, confirm final constraints, assess readiness."
            )

        if self._stagnation_count >= STAGNATION_THRESHOLD:
            return (
                "STAGNATION DETECTED — no new constraints in "
                f"{self._stagnation_count} turns. "
                "Change approach: reframe, challenge, or declare convergence."
            )

        # Default: build and expand
        return "BUILD — expand on prior analysis, extract new constraints."

    # ----------------------------------------------------------
    # Output Parsing
    # ----------------------------------------------------------

    def parse_turn_output(self, agent: SIAAgent, raw_output: str) -> TurnRecord:
        """Parse agent output into a structured TurnRecord.

        Attempts JSON parse first, falls back to section-header extraction.
        """
        constraints: list[str] = []
        challenges: list[str] = []
        reframes: list[str] = []
        response_to_prior: list[str] = []
        int_type = self._detect_int_type(agent, raw_output)

        # Try JSON parse first
        json_data = self._try_json_parse(raw_output)
        if json_data:
            constraints = json_data.get("constraints", json_data.get("new_constraints", []))
            challenges = json_data.get("challenges", [])
            reframes = json_data.get("reframes", json_data.get("reframe", []))
            response_to_prior = json_data.get("response_to_prior", [])
            # Normalize single strings to lists
            if isinstance(constraints, str):
                constraints = [constraints]
            if isinstance(challenges, str):
                challenges = [challenges]
            if isinstance(reframes, str):
                reframes = [reframes]
            if isinstance(response_to_prior, str):
                response_to_prior = [response_to_prior]
        else:
            # Fallback: section-header extraction
            constraints = self._extract_section(raw_output, "CONSTRAINTS", "NEW CONSTRAINTS")
            challenges = self._extract_section(raw_output, "CHALLENGE", "CHALLENGES")
            reframes = self._extract_section(raw_output, "REFRAME", "HIDDEN ASSUMPTIONS")
            response_to_prior = self._extract_section(
                raw_output, "RESPONSE", "COALITION", "CONTINGENCY"
            )

        return TurnRecord(
            turn=self._turn,
            agent=agent.id,
            int_type=int_type,
            constraints=constraints,
            challenges=challenges,
            reframes=reframes,
            response_to_prior=response_to_prior,
            raw_output=raw_output,
        )

    def _detect_int_type(self, agent: SIAAgent, raw_output: str) -> str:
        """Detect the interaction type from agent output."""
        if self._turn == 0:
            return IntType.INIT.value

        lower = raw_output.lower()

        # Check for explicit markers
        if any(kw in lower for kw in ("reframe", "alternative frame", "what if")):
            return IntType.RF.value
        if any(kw in lower for kw in ("challenge", "vulnerable", "question")):
            return IntType.C.value
        if any(kw in lower for kw in ("clarif", "unresolved", "variable")):
            return IntType.CL.value
        if any(kw in lower for kw in ("coalition", "align", "support")):
            return IntType.CO.value
        if any(kw in lower for kw in ("agree", "confirm", "endorse")):
            return IntType.A.value

        # Default to agent's preferred first type
        if agent.preferred_int_types:
            return agent.preferred_int_types[0]
        return IntType.B.value

    def _try_json_parse(self, text: str) -> dict[str, Any] | None:
        """Try to extract JSON from agent output."""
        # Try direct parse
        try:
            return json.loads(text)
        except (json.JSONDecodeError, ValueError):
            pass

        # Try to find JSON block in text — scan for balanced braces
        depth = 0
        start = -1
        for i, ch in enumerate(text):
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start >= 0:
                    candidate = text[start : i + 1]
                    try:
                        return json.loads(candidate)
                    except (json.JSONDecodeError, ValueError):
                        start = -1  # reset, try next block

        return None

    def _extract_section(self, text: str, *headers: str) -> list[str]:
        """Extract items under section headers (bullets, numbered, or plain lines)."""
        items: list[str] = []
        for header in headers:
            # Match header, then capture everything until next header or end
            pattern = (
                rf"(?:^|\n)\d*\.?\s*\**{header}\**[:\s]*\n?"
                r"((?:(?:[-*•]|\d+[.)]\s)\s*.*\n?|(?!\d*\.?\s*\**[A-Z]).+\n?)*)"
            )
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                block = match.group(1)
                for line in block.split("\n"):
                    # Strip bullet markers, numbered prefixes, leading whitespace
                    cleaned = re.sub(r"^(?:[-*•]|\d+[.)]\s)\s*", "", line.strip())
                    if cleaned:
                        items.append(cleaned)
        return items

    # ----------------------------------------------------------
    # State Update
    # ----------------------------------------------------------

    def update_state(self, record: TurnRecord, tokens_used: int = 0) -> None:
        """Accumulate reactor state from a turn record."""
        self._turn_log.append(record)
        self._tokens_used += tokens_used

        # Track anti-dominance
        if record["agent"] == self._last_agent_id:
            self._consecutive_count += 1
        else:
            self._consecutive_count = 1
        self._last_agent_id = record["agent"]

        # Track fatigue
        self._agent_fatigue[record["agent"]] = self._agent_fatigue.get(record["agent"], 0) + 1

        # Accumulate constraints (deduplicated)
        new_constraints = 0
        for c in record["constraints"]:
            if c and c not in self._constraints:
                self._constraints.append(c)
                new_constraints += 1

        # Track stagnation
        if new_constraints == 0:
            self._stagnation_count += 1
        else:
            self._stagnation_count = 0

        # Accumulate challenges and track unresolved
        for ch in record["challenges"]:
            if ch and ch not in self._challenges:
                self._challenges.append(ch)
            # Challenge not yet addressed by a constraint -> unresolved
            if ch and ch not in self._constraints and ch not in self._unresolved:
                self._unresolved.append(ch)

        # Re-check: resolved challenges (matched by a new constraint) leave unresolved
        if new_constraints > 0:
            self._unresolved = [u for u in self._unresolved if u not in self._constraints]

        # Accumulate reframes -> active frames
        for rf in record["reframes"]:
            if rf and rf not in self._active_frames:
                self._active_frames.append(rf)

        # Track RF for covenant enforcement
        if record["int_type"] == IntType.RF.value:
            self._rf_count_since_anchor += 1
        elif record["int_type"] in (IntType.CL.value, IntType.B.value, IntType.C.value):
            # Anchor types reset RF counter
            self._rf_count_since_anchor = 0

        # Build coalition map from response_to_prior
        if record["response_to_prior"]:
            self._coalition_map.setdefault(record["agent"], [])
            for ref in record["response_to_prior"]:
                if ref and ref not in self._coalition_map[record["agent"]]:
                    self._coalition_map[record["agent"]].append(ref)

        self._turn += 1

    # ----------------------------------------------------------
    # Termination
    # ----------------------------------------------------------

    def should_terminate(self) -> tuple[bool, str]:
        """Check if the reactor should stop deliberating.

        Returns (should_stop, reason).
        """
        # Hard limit: max turns
        if self._turn >= self.max_turns:
            return True, "max_turns_reached"

        # Budget exhaustion
        if self._tokens_used >= self.token_budget:
            return True, "token_budget_exhausted"

        # Crystalline entropy + sufficient constraints
        if self.entropy_band == "crystalline" and len(self._constraints) >= 3 and self._turn >= 3:
            return True, "crystalline_convergence"

        # Stagnation: too many turns with no new constraints
        if self._stagnation_count >= STAGNATION_THRESHOLD and self._turn >= 3:
            return True, "stagnation"

        # Minimum turns: always run at least 3 turns
        if self._turn < 3:
            return False, ""

        return False, ""

    # ----------------------------------------------------------
    # Harvest
    # ----------------------------------------------------------

    def harvest(self) -> tuple[ReactorState, ReactorTrace]:
        """Harvest the final reactor products.

        Returns (reactor_state, reactor_trace) for consumption by the synthesizer.
        """
        # Deduplicate unresolved from turn records
        unresolved: list[str] = []
        for record in self._turn_log:
            for c in record.get("challenges", []):
                # Challenges that were never resolved become unresolved
                if c and c not in self._constraints and c not in unresolved:
                    unresolved.append(c)

        # Build rejected branches from reframes that didn't become constraints
        rejected: list[str] = []
        for record in self._turn_log:
            for rf in record.get("reframes", []):
                if rf and rf not in self._active_frames:
                    if rf not in rejected:
                        rejected.append(rf)

        # Key claims: constraints + active frames
        key_claims = list(dict.fromkeys(self._constraints + self._active_frames))

        state = ReactorState(
            constraints=list(self._constraints),
            rejected_branches=rejected + list(self._rejected_branches),
            active_frames=list(self._active_frames),
            key_claims=key_claims,
            coalition_map=dict(self._coalition_map),
            unresolved=unresolved,
            turn_log=list(self._turn_log),
        )

        agents_used = list(dict.fromkeys(r["agent"] for r in self._turn_log))

        trace = ReactorTrace(
            turns_executed=self._turn,
            agents_used=agents_used,
            constraints_produced=len(self._constraints),
            branches_killed=len(rejected) + len(self._rejected_branches),
            challenges_issued=len(self._challenges),
            final_entropy=self.entropy_value,
            termination_reason=self.should_terminate()[1] or "incomplete",
            ignition_pattern=f"{self._turn_log[0]['agent']}->{self._turn_log[1]['agent']}"
            if len(self._turn_log) >= 2
            else "incomplete",
        )

        return state, trace

    # ----------------------------------------------------------
    # Properties
    # ----------------------------------------------------------

    @property
    def turn(self) -> int:
        return self._turn

    @property
    def tokens_used(self) -> int:
        return self._tokens_used

    @property
    def constraints(self) -> list[str]:
        return list(self._constraints)

    @property
    def distinct_agents_used(self) -> int:
        return len(set(r["agent"] for r in self._turn_log))

    def get_reactor_state(self) -> ReactorState:
        """Get the current reactor state (before harvest)."""
        return ReactorState(
            constraints=list(self._constraints),
            rejected_branches=list(self._rejected_branches),
            active_frames=list(self._active_frames),
            key_claims=list(dict.fromkeys(self._constraints + self._active_frames)),
            coalition_map=dict(self._coalition_map),
            unresolved=list(self._unresolved),
            turn_log=list(self._turn_log),
        )
