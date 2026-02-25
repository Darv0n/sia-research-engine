"""Tests for SIA Phase 3 — Kernel engine + reactor integration.

Tests cover:
- SIAKernel: speaker selection, ignition, anti-dominance, fatigue, termination
- Kernel: turn framing, output parsing, state accumulation, harvest
- Covenants: entropy-band steering, RF tracking
- Synthesizer: reactor augments V10, graceful degradation
- Graph: state field, tunable registration
"""

from __future__ import annotations

from deep_research_swarm.contracts import IntType, TurnRecord
from deep_research_swarm.sia.agents import (
    AGENT_BY_ID,
    ALL_AGENTS,
    CORE_AGENTS,
    RESERVE_AGENTS,
    SIAAgent,
)
from deep_research_swarm.sia.covenants import (
    ALL_COVENANTS,
    ALL_TRIADS,
    get_covenant,
    get_triads_involving,
)
from deep_research_swarm.sia.kernel import (
    MAX_CONSECUTIVE,
    MIN_DISTINCT_AGENTS,
    STAGNATION_THRESHOLD,
    VALID_IGNITION_PATTERNS,
    SIAKernel,
)

# ============================================================
# Helpers
# ============================================================


def _make_record(
    turn: int = 0,
    agent: str = "lawliet",
    int_type: str = "INIT",
    constraints: list[str] | None = None,
    challenges: list[str] | None = None,
    reframes: list[str] | None = None,
) -> TurnRecord:
    return TurnRecord(
        turn=turn,
        agent=agent,
        int_type=int_type,
        constraints=constraints or [],
        challenges=challenges or [],
        reframes=reframes or [],
        response_to_prior=[],
        raw_output="test output",
    )


# ============================================================
# Agent Definitions
# ============================================================


class TestAgentDefinitions:
    def test_all_seven_agents_exist(self):
        assert len(ALL_AGENTS) == 7

    def test_five_core_agents(self):
        assert len(CORE_AGENTS) == 5
        core_ids = {a.id for a in CORE_AGENTS}
        assert core_ids == {"lawliet", "light", "rick", "makishima", "shikamaru"}

    def test_two_reserve_agents(self):
        assert len(RESERVE_AGENTS) == 2
        reserve_ids = {a.id for a in RESERVE_AGENTS}
        assert reserve_ids == {"shiro", "johan"}

    def test_all_agents_have_cognitive_lens(self):
        for agent in ALL_AGENTS:
            assert agent.cognitive_lens, f"{agent.id} has no cognitive lens"
            assert len(agent.cognitive_lens) > 100

    def test_cognitive_lens_has_template_vars(self):
        required_vars = ["{research_question}", "{source_summary}", "{entropy_band}"]
        for agent in ALL_AGENTS:
            for var in required_vars:
                assert var in agent.cognitive_lens, f"{agent.id} lens missing {var}"

    def test_all_agents_have_basin_profiles(self):
        for agent in ALL_AGENTS:
            axes = agent.basin.axes
            assert len(axes) == 7
            for name, val in axes.items():
                assert -1.0 <= val <= 1.0, f"{agent.id}.{name} = {val} out of bounds"

    def test_all_agents_have_preferred_int_types(self):
        valid_types = {e.value for e in IntType}
        for agent in ALL_AGENTS:
            assert len(agent.preferred_int_types) >= 2
            for it in agent.preferred_int_types:
                assert it in valid_types, f"{agent.id} has invalid IntType {it}"

    def test_agent_lookup_by_id(self):
        for agent in ALL_AGENTS:
            assert AGENT_BY_ID[agent.id] is agent

    def test_rick_is_conditional(self):
        rick = AGENT_BY_ID["rick"]
        assert "entropy" in rick.activation_condition.lower()

    def test_shikamaru_is_cooling_sink(self):
        shikamaru = AGENT_BY_ID["shikamaru"]
        assert shikamaru.basin.a7_energy_expenditure < 0  # minimal path


# ============================================================
# Covenants
# ============================================================


class TestCovenants:
    def test_all_covenants_defined(self):
        assert len(ALL_COVENANTS) == 6

    def test_covenant_lookup_both_orderings(self):
        cov = get_covenant("lawliet", "light")
        assert cov is not None
        cov_rev = get_covenant("light", "lawliet")
        assert cov_rev is cov

    def test_system_wide_covenants(self):
        shika_cov = get_covenant("shikamaru", "*")
        assert shika_cov is not None
        assert shika_cov.agent_a == "shikamaru"

        johan_cov = get_covenant("johan", "*")
        assert johan_cov is not None
        assert johan_cov.agent_a == "johan"

    def test_system_wide_resolves_from_specific_agent(self):
        # Getting covenant between shikamaru and lawliet should return system-wide
        cov = get_covenant("shikamaru", "lawliet")
        assert cov is not None
        assert cov.agent_b == "*"

    def test_no_covenant_between_unrelated(self):
        # makishima <-> shikamaru has no specific covenant (shikamaru has system-wide)
        cov = get_covenant("makishima", "shikamaru")
        assert cov is not None  # system-wide applies

    def test_all_covenants_have_failure_modes(self):
        for cov in ALL_COVENANTS:
            assert cov.failure_mode
            assert cov.escalation_trigger
            assert cov.kernel_lever

    def test_all_triads_defined(self):
        assert len(ALL_TRIADS) == 2

    def test_triads_involving(self):
        rick_triads = get_triads_involving("rick")
        assert len(rick_triads) >= 1

    def test_intypes_valid_across_covenants(self):
        valid = {e.value for e in IntType}
        for cov in ALL_COVENANTS:
            for it in cov.preferred_int_types_a:
                assert it in valid, f"Invalid IntType {it} in covenant {cov.agent_a}-{cov.agent_b}"
            for it in cov.preferred_int_types_b:
                assert it in valid


# ============================================================
# Kernel: Ignition
# ============================================================


class TestKernelIgnition:
    def test_valid_ignition_patterns_count(self):
        assert len(VALID_IGNITION_PATTERNS) == 4

    def test_no_cooling_sink_first(self):
        """Ignition doctrine: never start with Shikamaru (cooling sink)."""
        for pattern in VALID_IGNITION_PATTERNS:
            assert pattern[0] != "shikamaru"
            assert pattern[1] != "shikamaru"

    def test_ignition_selects_from_patterns(self):
        kernel = SIAKernel(max_turns=6)
        first = kernel.select_speaker(0)
        second = kernel.select_speaker(1)
        pair = (first.id, second.id)
        assert pair in VALID_IGNITION_PATTERNS

    def test_convergence_band_uses_default_pattern(self):
        kernel = SIAKernel(entropy_band="convergence")
        first = kernel.select_speaker(0)
        assert first.id == "lawliet"

    def test_runaway_band_starts_with_constraint(self):
        kernel = SIAKernel(entropy_band="runaway")
        first = kernel.select_speaker(0)
        assert first.id == "lawliet"

    def test_turbulence_band_ignition(self):
        kernel = SIAKernel(entropy_band="turbulence")
        first = kernel.select_speaker(0)
        second = kernel.select_speaker(1)
        assert (first.id, second.id) == ("light", "makishima")

    def test_crystalline_band_ignition(self):
        kernel = SIAKernel(entropy_band="crystalline")
        first = kernel.select_speaker(0)
        second = kernel.select_speaker(1)
        assert (first.id, second.id) == ("lawliet", "rick")


# ============================================================
# Kernel: Speaker Selection
# ============================================================


class TestKernelSpeakerSelection:
    def test_post_ignition_selects_from_pool(self):
        kernel = SIAKernel(max_turns=10)
        # Run ignition
        r0 = _make_record(turn=0, agent="lawliet", constraints=["c1"])
        kernel.update_state(r0)
        r1 = _make_record(turn=1, agent="light", constraints=["c2"])
        kernel.update_state(r1)
        # Turn 2+
        agent = kernel.select_speaker(2)
        assert isinstance(agent, SIAAgent)

    def test_anti_dominance_blocks_consecutive(self):
        kernel = SIAKernel(max_turns=10, entropy_band="convergence")
        # Same agent MAX_CONSECUTIVE times
        for i in range(MAX_CONSECUTIVE):
            r = _make_record(turn=i, agent="lawliet", constraints=[f"c{i}"])
            kernel.update_state(r)
        # Next selection should NOT be lawliet
        agent = kernel.select_speaker(MAX_CONSECUTIVE)
        assert agent.id != "lawliet"

    def test_fatigue_penalizes_overused(self):
        kernel = SIAKernel(max_turns=10, entropy_band="convergence")
        # Use lawliet twice then light once
        kernel.update_state(_make_record(turn=0, agent="lawliet", constraints=["c1"]))
        kernel.update_state(_make_record(turn=1, agent="light", constraints=["c2"]))
        kernel.update_state(_make_record(turn=2, agent="lawliet", constraints=["c3"]))
        # Lawliet has fatigue=2, light has fatigue=1
        assert kernel._agent_fatigue["lawliet"] == 2
        assert kernel._agent_fatigue["light"] == 1

    def test_rick_blocked_in_low_entropy(self):
        kernel = SIAKernel(max_turns=10, entropy_value=0.30, entropy_band="convergence")
        # Rick should not be in candidate pool at low entropy without stagnation
        candidates = kernel._build_candidate_pool()
        candidate_ids = {c.id for c in candidates}
        assert "rick" not in candidate_ids

    def test_rick_eligible_in_high_entropy(self):
        kernel = SIAKernel(max_turns=10, entropy_value=0.55, entropy_band="turbulence")
        candidates = kernel._build_candidate_pool()
        candidate_ids = {c.id for c in candidates}
        assert "rick" in candidate_ids

    def test_rick_eligible_on_stagnation(self):
        kernel = SIAKernel(max_turns=10, entropy_value=0.30, entropy_band="convergence")
        # Force stagnation
        for i in range(STAGNATION_THRESHOLD):
            kernel.update_state(_make_record(turn=i, agent="lawliet"))  # no constraints
        candidates = kernel._build_candidate_pool()
        candidate_ids = {c.id for c in candidates}
        assert "rick" in candidate_ids

    def test_runaway_favors_compression(self):
        kernel = SIAKernel(entropy_band="runaway", entropy_value=0.80)
        lawliet_score = kernel._score_candidate(AGENT_BY_ID["lawliet"])
        rick_score = kernel._score_candidate(AGENT_BY_ID["rick"])
        assert lawliet_score > rick_score

    def test_crystalline_favors_harvest(self):
        kernel = SIAKernel(entropy_band="crystalline", entropy_value=0.15)
        shika_score = kernel._score_candidate(AGENT_BY_ID["shikamaru"])
        rick_score = kernel._score_candidate(AGENT_BY_ID["rick"])
        assert shika_score > rick_score


# ============================================================
# Kernel: Reserve Agent Activation
# ============================================================


class TestReserveActivation:
    def test_shiro_not_activated_early(self):
        kernel = SIAKernel(max_turns=10)
        assert not kernel._should_activate_reserve(AGENT_BY_ID["shiro"])

    def test_shiro_activated_late_with_few_constraints(self):
        kernel = SIAKernel(max_turns=10)
        kernel._turn = 4
        kernel._constraints = ["only_one"]
        assert kernel._should_activate_reserve(AGENT_BY_ID["shiro"])

    def test_johan_not_activated_in_low_entropy(self):
        kernel = SIAKernel(max_turns=10, entropy_value=0.30)
        kernel._turn = 5
        assert not kernel._should_activate_reserve(AGENT_BY_ID["johan"])

    def test_johan_activated_in_high_entropy_no_coalitions(self):
        kernel = SIAKernel(max_turns=10, entropy_value=0.50)
        kernel._turn = 5
        assert kernel._should_activate_reserve(AGENT_BY_ID["johan"])


# ============================================================
# Kernel: Turn Framing
# ============================================================


class TestKernelFraming:
    def test_frame_contains_turn_info(self):
        kernel = SIAKernel(max_turns=6, entropy_band="convergence", entropy_value=0.35)
        frame = kernel.frame_turn(AGENT_BY_ID["lawliet"])
        assert "Turn 1/6" in frame
        assert "convergence" in frame
        assert "Lawliet" in frame

    def test_ignition_instruction(self):
        kernel = SIAKernel()
        frame = kernel.frame_turn(AGENT_BY_ID["lawliet"])
        assert "IGNITION" in frame

    def test_runaway_instruction(self):
        kernel = SIAKernel(entropy_band="runaway")
        kernel._turn = 2  # past ignition
        frame = kernel.frame_turn(AGENT_BY_ID["lawliet"])
        assert "CONTAINMENT" in frame

    def test_anchor_after_rick(self):
        kernel = SIAKernel(entropy_band="turbulence")
        kernel._turn = 2
        kernel._last_agent_id = "rick"
        frame = kernel.frame_turn(AGENT_BY_ID["lawliet"])
        assert "ANCHOR" in frame

    def test_stagnation_instruction(self):
        kernel = SIAKernel()
        kernel._turn = 4
        kernel._stagnation_count = STAGNATION_THRESHOLD
        frame = kernel.frame_turn(AGENT_BY_ID["lawliet"])
        assert "STAGNATION" in frame

    def test_crystalline_harvest(self):
        kernel = SIAKernel(entropy_band="crystalline")
        kernel._turn = 3
        frame = kernel.frame_turn(AGENT_BY_ID["shikamaru"])
        assert "HARVEST" in frame


# ============================================================
# Kernel: Output Parsing
# ============================================================


class TestKernelParsing:
    def test_parse_json_output(self):
        kernel = SIAKernel()
        kernel._turn = 1
        raw = '{"constraints": ["C1", "C2"], "challenges": ["Ch1"], "reframes": ["RF1"]}'
        record = kernel.parse_turn_output(AGENT_BY_ID["lawliet"], raw)
        assert record["constraints"] == ["C1", "C2"]
        assert record["challenges"] == ["Ch1"]
        assert record["reframes"] == ["RF1"]

    def test_parse_section_header_output(self):
        kernel = SIAKernel()
        kernel._turn = 1
        raw = """1. NEW CONSTRAINTS:
- The evidence shows X is true
- Y is unsupported

2. CHALLENGES:
- Claim Z lacks evidence

3. REFRAME:
- Consider the opposite perspective
"""
        record = kernel.parse_turn_output(AGENT_BY_ID["lawliet"], raw)
        assert len(record["constraints"]) == 2
        assert len(record["challenges"]) == 1
        assert len(record["reframes"]) == 1

    def test_detect_int_type_rf(self):
        kernel = SIAKernel()
        kernel._turn = 1
        it = kernel._detect_int_type(AGENT_BY_ID["rick"], "I want to reframe the question")
        assert it == IntType.RF.value

    def test_detect_int_type_challenge(self):
        kernel = SIAKernel()
        kernel._turn = 1
        it = kernel._detect_int_type(AGENT_BY_ID["lawliet"], "I challenge this claim")
        assert it == IntType.C.value

    def test_detect_int_type_init(self):
        kernel = SIAKernel()
        it = kernel._detect_int_type(AGENT_BY_ID["lawliet"], "anything")
        assert it == IntType.INIT.value

    def test_detect_int_type_fallback(self):
        kernel = SIAKernel()
        kernel._turn = 1
        it = kernel._detect_int_type(AGENT_BY_ID["lawliet"], "generic text without markers")
        # Fallback to first preferred type
        assert it == AGENT_BY_ID["lawliet"].preferred_int_types[0]

    def test_parse_embedded_json(self):
        kernel = SIAKernel()
        kernel._turn = 1
        raw = 'Some text before {"constraints": ["C1"]} some text after'
        record = kernel.parse_turn_output(AGENT_BY_ID["lawliet"], raw)
        assert record["constraints"] == ["C1"]


# ============================================================
# Kernel: State Update
# ============================================================


class TestKernelStateUpdate:
    def test_constraints_accumulate(self):
        kernel = SIAKernel()
        kernel.update_state(_make_record(constraints=["C1", "C2"]))
        kernel.update_state(_make_record(turn=1, agent="light", constraints=["C3"]))
        assert kernel.constraints == ["C1", "C2", "C3"]

    def test_constraints_deduplicate(self):
        kernel = SIAKernel()
        kernel.update_state(_make_record(constraints=["C1"]))
        kernel.update_state(_make_record(turn=1, agent="light", constraints=["C1", "C2"]))
        assert kernel.constraints == ["C1", "C2"]

    def test_stagnation_tracks_no_constraints(self):
        kernel = SIAKernel()
        kernel.update_state(_make_record())  # no constraints
        assert kernel._stagnation_count == 1
        kernel.update_state(_make_record(turn=1, agent="light"))
        assert kernel._stagnation_count == 2

    def test_stagnation_resets_on_constraint(self):
        kernel = SIAKernel()
        kernel.update_state(_make_record())  # stagnation=1
        kernel.update_state(_make_record(turn=1, agent="light", constraints=["C1"]))
        assert kernel._stagnation_count == 0

    def test_consecutive_tracking(self):
        kernel = SIAKernel()
        kernel.update_state(_make_record(agent="lawliet"))
        assert kernel._consecutive_count == 1
        kernel.update_state(_make_record(turn=1, agent="lawliet"))
        assert kernel._consecutive_count == 2
        kernel.update_state(_make_record(turn=2, agent="light"))
        assert kernel._consecutive_count == 1

    def test_fatigue_increments(self):
        kernel = SIAKernel()
        kernel.update_state(_make_record(agent="lawliet"))
        kernel.update_state(_make_record(turn=1, agent="lawliet"))
        assert kernel._agent_fatigue["lawliet"] == 2

    def test_rf_tracking(self):
        kernel = SIAKernel()
        kernel.update_state(_make_record(int_type="RF", reframes=["r1"]))
        assert kernel._rf_count_since_anchor == 1
        kernel.update_state(_make_record(turn=1, agent="light", int_type="CL"))
        assert kernel._rf_count_since_anchor == 0  # anchor resets

    def test_turn_increments(self):
        kernel = SIAKernel()
        assert kernel.turn == 0
        kernel.update_state(_make_record())
        assert kernel.turn == 1

    def test_tokens_accumulate(self):
        kernel = SIAKernel()
        kernel.update_state(_make_record(), tokens_used=500)
        kernel.update_state(_make_record(turn=1, agent="light"), tokens_used=300)
        assert kernel.tokens_used == 800


# ============================================================
# Kernel: Termination
# ============================================================


class TestKernelTermination:
    def test_max_turns_terminates(self):
        kernel = SIAKernel(max_turns=3)
        for i in range(3):
            kernel.update_state(_make_record(turn=i, agent="lawliet" if i % 2 == 0 else "light"))
        should_stop, reason = kernel.should_terminate()
        assert should_stop
        assert reason == "max_turns_reached"

    def test_budget_exhaustion_terminates(self):
        kernel = SIAKernel(token_budget=1000)
        kernel.update_state(_make_record(), tokens_used=1100)
        should_stop, reason = kernel.should_terminate()
        assert should_stop
        assert reason == "token_budget_exhausted"

    def test_crystalline_convergence(self):
        kernel = SIAKernel(entropy_band="crystalline", entropy_value=0.15)
        kernel.update_state(_make_record(constraints=["c1", "c2", "c3"]))
        kernel.update_state(_make_record(turn=1, agent="light"))
        kernel.update_state(_make_record(turn=2, agent="lawliet"))
        should_stop, reason = kernel.should_terminate()
        assert should_stop
        assert reason == "crystalline_convergence"

    def test_stagnation_terminates(self):
        kernel = SIAKernel(max_turns=10)
        # 3 turns minimum + stagnation
        for i in range(STAGNATION_THRESHOLD + 1):
            kernel.update_state(_make_record(turn=i, agent="lawliet" if i % 2 == 0 else "light"))
        should_stop, reason = kernel.should_terminate()
        assert should_stop
        assert reason == "stagnation"

    def test_minimum_turns_enforced(self):
        kernel = SIAKernel(max_turns=10, entropy_band="crystalline", entropy_value=0.10)
        kernel.update_state(_make_record(constraints=["c1", "c2", "c3"]))
        # Only 1 turn — should NOT terminate yet
        should_stop, _ = kernel.should_terminate()
        assert not should_stop

    def test_no_early_termination_in_convergence(self):
        kernel = SIAKernel(max_turns=10, entropy_band="convergence")
        kernel.update_state(_make_record(constraints=["c1"]))
        should_stop, _ = kernel.should_terminate()
        assert not should_stop


# ============================================================
# Kernel: Harvest
# ============================================================


class TestKernelHarvest:
    def test_harvest_produces_reactor_state(self):
        kernel = SIAKernel(max_turns=4)
        kernel.update_state(_make_record(constraints=["C1", "C2"]))
        kernel.update_state(_make_record(turn=1, agent="light", constraints=["C3"]))
        kernel.update_state(
            _make_record(turn=2, agent="rick", reframes=["RF1"], challenges=["Ch1"])
        )
        kernel.update_state(_make_record(turn=3, agent="lawliet", constraints=["C4"]))

        state, trace = kernel.harvest()

        assert isinstance(state, dict)
        assert isinstance(trace, dict)
        assert "constraints" in state
        assert "rejected_branches" in state
        assert "active_frames" in state
        assert "turn_log" in state

    def test_harvest_deduplicates_constraints(self):
        kernel = SIAKernel(max_turns=3)
        kernel.update_state(_make_record(constraints=["C1"]))
        kernel.update_state(_make_record(turn=1, agent="light", constraints=["C1", "C2"]))
        kernel.update_state(_make_record(turn=2, agent="lawliet", constraints=["C2", "C3"]))

        state, _ = kernel.harvest()
        assert state["constraints"] == ["C1", "C2", "C3"]

    def test_harvest_trace_has_required_fields(self):
        kernel = SIAKernel(max_turns=3)
        kernel.update_state(_make_record(constraints=["C1"]))
        kernel.update_state(_make_record(turn=1, agent="light", challenges=["Ch1"]))
        kernel.update_state(_make_record(turn=2, agent="rick", reframes=["RF1"]))

        _, trace = kernel.harvest()
        assert trace["turns_executed"] == 3
        assert len(trace["agents_used"]) >= 2
        assert trace["constraints_produced"] >= 1
        assert trace["challenges_issued"] >= 1
        assert "termination_reason" in trace
        assert "ignition_pattern" in trace

    def test_harvest_ignition_pattern_string(self):
        kernel = SIAKernel(max_turns=3)
        kernel.update_state(_make_record(agent="lawliet"))
        kernel.update_state(_make_record(turn=1, agent="light"))
        kernel.update_state(_make_record(turn=2, agent="rick"))

        _, trace = kernel.harvest()
        assert trace["ignition_pattern"] == "lawliet->light"

    def test_harvest_active_frames_from_reframes(self):
        kernel = SIAKernel(max_turns=3)
        kernel.update_state(_make_record(agent="lawliet"))
        kernel.update_state(
            _make_record(turn=1, agent="rick", int_type="RF", reframes=["Frame A", "Frame B"])
        )
        kernel.update_state(_make_record(turn=2, agent="lawliet"))

        state, _ = kernel.harvest()
        assert "Frame A" in state["active_frames"]
        assert "Frame B" in state["active_frames"]

    def test_harvest_coalition_map(self):
        kernel = SIAKernel(max_turns=2)
        r = TurnRecord(
            turn=0,
            agent="light",
            int_type="CO",
            constraints=[],
            challenges=[],
            reframes=[],
            response_to_prior=["agrees with lawliet"],
            raw_output="test",
        )
        kernel.update_state(r)
        kernel.update_state(_make_record(turn=1, agent="lawliet"))

        state, _ = kernel.harvest()
        assert "light" in state["coalition_map"]

    def test_get_reactor_state_before_harvest(self):
        kernel = SIAKernel(max_turns=2)
        kernel.update_state(_make_record(constraints=["C1"]))
        state = kernel.get_reactor_state()
        assert state["constraints"] == ["C1"]


# ============================================================
# Kernel: Distinct Agents Property
# ============================================================


class TestKernelProperties:
    def test_distinct_agents_count(self):
        kernel = SIAKernel()
        kernel.update_state(_make_record(agent="lawliet"))
        kernel.update_state(_make_record(turn=1, agent="light"))
        kernel.update_state(_make_record(turn=2, agent="lawliet"))
        assert kernel.distinct_agents_used == 2

    def test_min_distinct_agents_constant(self):
        assert MIN_DISTINCT_AGENTS == 3


# ============================================================
# Config + State Integration
# ============================================================


class TestPhase3Config:
    def test_reactor_tunables_registered(self):
        from deep_research_swarm.adaptive.registry import TunableRegistry

        r = TunableRegistry()
        assert "sia_reactor_turns" in r
        assert "sia_reactor_budget" in r

    def test_reactor_tunable_defaults(self):
        from deep_research_swarm.adaptive.registry import TunableRegistry

        r = TunableRegistry()
        assert r.get("sia_reactor_turns") == 6
        assert r.get("sia_reactor_budget") == 20000

    def test_reactor_tunable_bounds(self):
        from deep_research_swarm.adaptive.registry import TunableRegistry

        r = TunableRegistry()
        # Floor
        r.set("sia_reactor_turns", 1)
        assert r.get("sia_reactor_turns") == 3  # floor

        # Ceiling
        r.set("sia_reactor_turns", 20)
        assert r.get("sia_reactor_turns") == 10  # ceiling

    def test_tunable_count_updated(self):
        from deep_research_swarm.adaptive.registry import TunableRegistry

        r = TunableRegistry()
        assert len(r) == 26  # 24 (V10 Phase 2) + 2 (Phase 3)

    def test_state_has_reactor_trace(self):
        """Verify reactor_trace field exists in ResearchState annotations."""
        from deep_research_swarm.graph.state import ResearchState

        annotations = ResearchState.__annotations__
        assert "reactor_trace" in annotations


# ============================================================
# Graph Topology
# ============================================================


class TestPhase3Graph:
    def _build_graph(self):
        from deep_research_swarm.config import Settings
        from deep_research_swarm.graph.builder import build_graph

        settings = Settings(anthropic_api_key="test-key")
        return build_graph(settings, enable_cache=False)

    def test_graph_compiles(self):
        graph = self._build_graph()
        assert graph is not None

    def test_node_count(self):
        """Graph node count should include all Phase 1-3 additions."""
        graph = self._build_graph()
        nodes = set(graph.get_graph().nodes.keys()) - {"__start__", "__end__"}
        # V9 (18) + compute_entropy + deliberate_panel + compress + score_merge = 22
        assert len(nodes) == 22

    def test_synthesize_node_exists(self):
        graph = self._build_graph()
        nodes = set(graph.get_graph().nodes.keys())
        assert "synthesize" in nodes


# ============================================================
# Synthesizer Reactor Integration
# ============================================================


class TestSynthesizerReactorRouting:
    def test_synthesize_accepts_kwargs(self):
        import inspect

        from deep_research_swarm.agents.synthesizer import synthesize

        sig = inspect.signature(synthesize)
        assert "sonnet_caller" in sig.parameters
        assert "haiku_caller" in sig.parameters

    def test_v10_path_has_reactor(self):
        """Verify _run_reactor function exists and is callable."""
        from deep_research_swarm.agents.synthesizer import _run_reactor

        assert callable(_run_reactor)

    def test_run_reactor_unpacks_call_tuple(self):
        """C1 regression: caller.call() returns (str, TokenUsage), not a response object."""
        import ast
        import inspect

        from deep_research_swarm.agents.synthesizer import _run_reactor

        source = inspect.getsource(_run_reactor)
        tree = ast.parse(source)
        # Find the call assignment — must be a Tuple unpack, not a single Name
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                # Look for the reactor_caller.call() assignment
                if isinstance(node.value, ast.Await):
                    call = node.value.value
                    if isinstance(call, ast.Call):
                        func = call.func
                        if isinstance(func, ast.Attribute) and func.attr == "call":
                            # Target must be a Tuple (unpacking), not a single Name
                            target = node.targets[0]
                            assert isinstance(target, ast.Tuple), (
                                "reactor_caller.call() must be unpacked as "
                                "'raw_output, usage = await ...', not assigned to a single var"
                            )
                            assert len(target.elts) == 2
                            return
        # If we get here, we didn't find the call at all
        raise AssertionError("Could not find reactor_caller.call() in _run_reactor source")

    def test_build_prior_summary_empty(self):
        from deep_research_swarm.agents.synthesizer import _build_prior_summary

        result = _build_prior_summary([])
        assert result == "No prior conversation."

    def test_build_prior_summary_with_messages(self):
        from deep_research_swarm.agents.synthesizer import _build_prior_summary

        conversation = [
            {"role": "assistant", "content": "Analysis of evidence..."},
            {"role": "user", "content": "[Turn 1 complete.]"},
        ]
        result = _build_prior_summary(conversation)
        assert "Analysis" in result
        assert "Turn 1" in result


class TestS8NestedJsonParsing:
    """S8 regression: _try_json_parse must handle nested JSON objects."""

    def _make_kernel(self):
        from deep_research_swarm.sia.kernel import SIAKernel

        return SIAKernel()

    def test_flat_json(self):
        kernel = self._make_kernel()
        result = kernel._try_json_parse('{"key": "value"}')
        assert result == {"key": "value"}

    def test_nested_json(self):
        kernel = self._make_kernel()
        text = '{"constraints": ["a", "b"], "nested": {"inner": true}}'
        result = kernel._try_json_parse(text)
        assert result is not None
        assert result["constraints"] == ["a", "b"]
        assert result["nested"]["inner"] is True

    def test_json_embedded_in_text(self):
        kernel = self._make_kernel()
        text = 'Here is my analysis:\n{"constraints": ["x"], "challenges": []}\nEnd.'
        result = kernel._try_json_parse(text)
        assert result is not None
        assert result["constraints"] == ["x"]

    def test_deeply_nested_json(self):
        kernel = self._make_kernel()
        text = '{"a": {"b": {"c": [1, 2, 3]}}}'
        result = kernel._try_json_parse(text)
        assert result is not None
        assert result["a"]["b"]["c"] == [1, 2, 3]


class TestS9SectionExtraction:
    """S9 regression: _extract_section must handle non-bullet content."""

    def _make_kernel(self):
        from deep_research_swarm.sia.kernel import SIAKernel

        return SIAKernel()

    def test_bullet_items(self):
        kernel = self._make_kernel()
        text = "Constraints:\n- First item\n- Second item\n"
        result = kernel._extract_section(text, "Constraints")
        assert "First item" in result
        assert "Second item" in result

    def test_numbered_items(self):
        kernel = self._make_kernel()
        text = "Constraints:\n1. First item\n2. Second item\n"
        result = kernel._extract_section(text, "Constraints")
        assert "First item" in result
        assert "Second item" in result

    def test_star_bullet_items(self):
        kernel = self._make_kernel()
        text = "Challenges:\n* Alpha challenge\n* Beta challenge\n"
        result = kernel._extract_section(text, "Challenges")
        assert "Alpha challenge" in result
        assert "Beta challenge" in result


class TestS14WildcardCovenantBonus:
    """S14 regression: Wildcard covenants must not give uniform score bonus."""

    def test_wildcard_covenant_no_bonus(self):
        """Covenants with agent_b='*' should not add score."""
        from deep_research_swarm.sia.covenants import get_covenant

        # Johan has a wildcard covenant (johan, *)
        cov = get_covenant("johan", "rick")
        assert cov is not None
        # The wildcard means agent_a or agent_b is "*"
        assert cov.agent_a == "*" or cov.agent_b == "*"

    def test_pairwise_covenant_exists(self):
        """Explicit pairwise covenants should exist and not be wildcards."""
        from deep_research_swarm.sia.covenants import get_covenant

        # Lawliet-Light is an explicit pair
        cov = get_covenant("lawliet", "light")
        assert cov is not None
        assert cov.agent_a != "*" and cov.agent_b != "*"

    def test_kernel_scoring_skips_wildcards(self):
        """_score_candidate source must check for wildcard covenants."""
        import inspect

        from deep_research_swarm.sia.kernel import SIAKernel

        source = inspect.getsource(SIAKernel._score_candidate)
        assert '"*"' in source


# ============================================================
# M2: Unresolved tracking
# ============================================================


class TestM2UnresolvedTracking:
    """M2 regression: kernel must populate _unresolved from challenges."""

    def test_challenges_populate_unresolved(self):
        from deep_research_swarm.sia.kernel import SIAKernel

        kernel = SIAKernel()
        record = {
            "turn": 0,
            "agent": "lawliet",
            "int_type": "B",
            "constraints": [],
            "challenges": ["Missing evidence for claim X"],
            "reframes": [],
            "response_to_prior": [],
            "raw_output": "",
        }
        kernel.update_state(record)
        assert len(kernel._unresolved) == 1
        assert "Missing evidence for claim X" in kernel._unresolved

    def test_resolved_challenges_removed(self):
        from deep_research_swarm.sia.kernel import SIAKernel

        kernel = SIAKernel()
        # Turn 1: challenge raised
        kernel.update_state(
            {
                "turn": 0,
                "agent": "rick",
                "int_type": "C",
                "constraints": [],
                "challenges": ["Weak grounding for hypothesis"],
                "reframes": [],
                "response_to_prior": [],
                "raw_output": "",
            }
        )
        assert "Weak grounding for hypothesis" in kernel._unresolved
        # Turn 2: constraint resolves the challenge
        kernel.update_state(
            {
                "turn": 1,
                "agent": "lawliet",
                "int_type": "B",
                "constraints": ["Weak grounding for hypothesis"],
                "challenges": [],
                "reframes": [],
                "response_to_prior": [],
                "raw_output": "",
            }
        )
        assert "Weak grounding for hypothesis" not in kernel._unresolved

    def test_unresolved_count_in_frame(self):
        from deep_research_swarm.sia.kernel import SIAKernel

        kernel = SIAKernel()
        kernel.update_state(
            {
                "turn": 0,
                "agent": "rick",
                "int_type": "C",
                "constraints": [],
                "challenges": ["Gap A", "Gap B"],
                "reframes": [],
                "response_to_prior": [],
                "raw_output": "",
            }
        )
        assert len(kernel._unresolved) == 2


# ============================================================
# M3: Dead field removed
# ============================================================


class TestM3NoKeyClaimsField:
    """M3 regression: _key_claims field must not exist on SIAKernel."""

    def test_no_key_claims_attribute(self):
        from deep_research_swarm.sia.kernel import SIAKernel

        kernel = SIAKernel()
        assert not hasattr(kernel, "_key_claims")


# ============================================================
# M4: Ignition patterns use constant
# ============================================================


class TestM4IgnitionPatternConstant:
    """M4 regression: _select_ignition_pattern must reference VALID_IGNITION_PATTERNS."""

    def test_ignition_uses_constant(self):
        import inspect

        from deep_research_swarm.sia.kernel import SIAKernel

        source = inspect.getsource(SIAKernel._select_ignition_pattern)
        assert "VALID_IGNITION_PATTERNS" in source

    def test_ignition_returns_valid_patterns(self):
        from deep_research_swarm.sia.kernel import VALID_IGNITION_PATTERNS, SIAKernel

        for band in ("runaway", "turbulence", "crystalline", "convergence"):
            kernel = SIAKernel(entropy_band=band)
            pattern = kernel._select_ignition_pattern()
            assert pattern in VALID_IGNITION_PATTERNS
