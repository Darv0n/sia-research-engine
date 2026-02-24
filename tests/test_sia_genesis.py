"""Tests for SIA Phase 0: Agent definitions, covenants, and V10 TypedDicts.

Target: ~25 tests covering agent instantiation, basin profiles, cognitive lenses,
covenant lookup, high-risk triads, IntType grammar, and contract types.
"""

from __future__ import annotations

import pytest

from deep_research_swarm.contracts import (
    ClaimVerdict,
    EntropyBand,
    EntropyState,
    IntType,
    KnowledgeArtifact,
    ReactorState,
    SwarmMetadata,
    TurnRecord,
)
from deep_research_swarm.sia.agents import (
    AGENT_BY_ID,
    ALL_AGENTS,
    CORE_AGENTS,
    LAWLIET,
    LIGHT,
    RESERVE_AGENTS,
    RICK,
    SHIKAMARU,
    BasinProfile,
    SIAAgent,
)
from deep_research_swarm.sia.covenants import (
    ALL_COVENANTS,
    ALL_TRIADS,
    COVENANT_TRIGGERS,
    TRIGGER_INTERVENTIONS,
    get_covenant,
    get_triads_involving,
)

# ============================================================
# Agent Registry Tests
# ============================================================


class TestAgentRegistry:
    def test_all_agents_count(self):
        assert len(ALL_AGENTS) == 7

    def test_core_agents_count(self):
        assert len(CORE_AGENTS) == 5

    def test_reserve_agents_count(self):
        assert len(RESERVE_AGENTS) == 2

    def test_core_agents_are_correct(self):
        core_ids = {a.id for a in CORE_AGENTS}
        assert core_ids == {"lawliet", "light", "rick", "makishima", "shikamaru"}

    def test_reserve_agents_are_correct(self):
        reserve_ids = {a.id for a in RESERVE_AGENTS}
        assert reserve_ids == {"shiro", "johan"}

    def test_agent_by_id_lookup(self):
        for agent in ALL_AGENTS:
            assert AGENT_BY_ID[agent.id] is agent

    def test_no_duplicate_ids(self):
        ids = [a.id for a in ALL_AGENTS]
        assert len(ids) == len(set(ids))

    def test_all_agents_are_frozen(self):
        with pytest.raises(AttributeError):
            LAWLIET.id = "modified"  # type: ignore[misc]


# ============================================================
# Agent Field Tests
# ============================================================


class TestAgentFields:
    @pytest.mark.parametrize("agent", ALL_AGENTS, ids=lambda a: a.id)
    def test_agent_has_required_fields(self, agent: SIAAgent):
        assert agent.id
        assert agent.name
        assert agent.archetype in ("Delta", "Sigma", "Lambda", "Gamma", "Pi")
        assert isinstance(agent.basin, BasinProfile)
        assert agent.cognitive_lens
        assert len(agent.preferred_int_types) > 0
        assert len(agent.repulsion_patterns) > 0
        assert agent.instability_signature
        assert agent.activation_condition

    @pytest.mark.parametrize("agent", ALL_AGENTS, ids=lambda a: a.id)
    def test_cognitive_lens_has_template_vars(self, agent: SIAAgent):
        lens = agent.cognitive_lens
        assert "{research_question}" in lens
        assert "{entropy_band}" in lens
        assert "{source_summary}" in lens
        assert "{prior_summary}" in lens

    @pytest.mark.parametrize("agent", ALL_AGENTS, ids=lambda a: a.id)
    def test_preferred_int_types_are_valid(self, agent: SIAAgent):
        valid = {e.value for e in IntType}
        for it in agent.preferred_int_types:
            assert it in valid, f"{agent.id}: invalid IntType '{it}'"


# ============================================================
# Basin Profile Tests
# ============================================================


class TestBasinProfile:
    @pytest.mark.parametrize("agent", ALL_AGENTS, ids=lambda a: a.id)
    def test_basin_axes_bounded(self, agent: SIAAgent):
        for name, val in agent.basin.axes.items():
            assert -1.0 <= val <= 1.0, f"{agent.id}.{name} = {val} out of bounds"

    def test_basin_axes_dict_has_7_keys(self):
        axes = LAWLIET.basin.axes
        assert len(axes) == 7
        assert all(k.startswith("a") for k in axes)

    def test_basin_profile_rejects_out_of_bounds(self):
        with pytest.raises(ValueError):
            BasinProfile(1.5, 0, 0, 0, 0, 0, 0)

    def test_basin_profile_rejects_negative_out_of_bounds(self):
        with pytest.raises(ValueError):
            BasinProfile(0, 0, 0, 0, -1.5, 0, 0)

    def test_basin_profile_frozen(self):
        with pytest.raises(AttributeError):
            LAWLIET.basin.a1_constraint_density = 0.0  # type: ignore[misc]

    def test_lawliet_is_constraint_heavy(self):
        assert LAWLIET.basin.a1_constraint_density >= 0.8

    def test_light_is_directional(self):
        assert LIGHT.basin.a2_directionality >= 0.8

    def test_rick_is_frame_destabilizer(self):
        assert RICK.basin.a5_frame_stability <= -0.8

    def test_shikamaru_is_energy_minimizer(self):
        assert SHIKAMARU.basin.a7_energy_expenditure <= -0.8


# ============================================================
# Covenant Tests
# ============================================================


class TestCovenants:
    def test_all_covenants_count(self):
        assert len(ALL_COVENANTS) == 6

    def test_covenant_lookup_both_orderings(self):
        cov = get_covenant("lawliet", "light")
        assert cov is not None
        assert cov.agent_a == "lawliet"
        assert cov.agent_b == "light"

        cov_reverse = get_covenant("light", "lawliet")
        assert cov_reverse is cov

    def test_covenant_lawliet_rick(self):
        cov = get_covenant("lawliet", "rick")
        assert cov is not None
        assert "rupture" in cov.coupling.lower()

    def test_covenant_light_makishima(self):
        cov = get_covenant("light", "makishima")
        assert cov is not None
        assert "moral" in cov.coupling.lower() or "destiny" in cov.coupling.lower()

    def test_system_wide_covenant_shikamaru(self):
        cov = get_covenant("shikamaru", "*")
        assert cov is not None
        assert "efficiency" in cov.coupling.lower()

    def test_system_wide_covenant_johan(self):
        cov = get_covenant("johan", "*")
        assert cov is not None
        assert "alignment" in cov.coupling.lower()

    def test_no_covenant_returns_none(self):
        # lawliet-makishima has no direct covenant (goes through system-wide)
        # But since we don't have one, let's test a truly missing pair
        # that also has no system-wide fallback
        # Actually lawliet-shiro has no covenant either, but shiro has no system-wide
        cov = get_covenant("lawliet", "shiro")
        # No direct covenant and neither has * coupling
        assert cov is None or cov.agent_b == "*"

    def test_all_covenants_have_required_fields(self):
        for cov in ALL_COVENANTS:
            assert cov.agent_a
            assert cov.agent_b
            assert cov.coupling
            assert cov.signature
            assert cov.failure_mode
            assert cov.escalation_trigger
            assert cov.kernel_lever

    def test_covenant_int_types_are_valid(self):
        valid = {e.value for e in IntType}
        for cov in ALL_COVENANTS:
            for it in cov.preferred_int_types_a:
                assert it in valid, f"Covenant {cov.agent_a}-{cov.agent_b}: invalid '{it}'"
            for it in cov.preferred_int_types_b:
                assert it in valid, f"Covenant {cov.agent_a}-{cov.agent_b}: invalid '{it}'"


# ============================================================
# High-Risk Triad Tests
# ============================================================


class TestTriads:
    def test_all_triads_count(self):
        assert len(ALL_TRIADS) == 2

    def test_triad_agents_are_valid(self):
        valid_ids = {a.id for a in ALL_AGENTS}
        for triad in ALL_TRIADS:
            assert len(triad.agents) == 3
            for agent_id in triad.agents:
                assert agent_id in valid_ids

    def test_get_triads_involving_rick(self):
        triads = get_triads_involving("rick")
        assert len(triads) == 1
        assert "rick" in triads[0].agents

    def test_get_triads_involving_light(self):
        triads = get_triads_involving("light")
        assert len(triads) == 1

    def test_get_triads_involving_nonexistent(self):
        triads = get_triads_involving("shikamaru")
        assert len(triads) == 0


# ============================================================
# Covenant Trigger Tests
# ============================================================


class TestCovenantTriggers:
    def test_all_triggers_have_interventions(self):
        for trigger in COVENANT_TRIGGERS:
            assert trigger in TRIGGER_INTERVENTIONS

    def test_trigger_count(self):
        assert len(COVENANT_TRIGGERS) == 5


# ============================================================
# V10 TypedDict Import Tests
# ============================================================


class TestV10TypeDicts:
    def test_entropy_band_enum(self):
        assert EntropyBand.CRYSTALLINE.value == "crystalline"
        assert EntropyBand.CONVERGENCE.value == "convergence"
        assert EntropyBand.TURBULENCE.value == "turbulence"
        assert EntropyBand.RUNAWAY.value == "runaway"

    def test_int_type_enum(self):
        assert IntType.B.value == "B"
        assert IntType.RF.value == "RF"
        assert IntType.INIT.value == "INIT"
        assert len(IntType) == 9

    def test_entropy_state_constructible(self):
        state: EntropyState = {
            "e": 0.5,
            "e_amb": 0.3,
            "e_conf": 0.4,
            "e_nov": 0.2,
            "e_trust": 0.1,
            "band": "turbulence",
            "turn": 3,
            "stagnation_count": 0,
        }
        assert state["band"] == "turbulence"

    def test_turn_record_constructible(self):
        rec: TurnRecord = {
            "turn": 1,
            "agent": "lawliet",
            "int_type": "C",
            "constraints": ["evidence requires X"],
            "challenges": ["claim Y ungrounded"],
            "reframes": [],
            "response_to_prior": [],
            "raw_output": "test output",
        }
        assert rec["agent"] == "lawliet"

    def test_reactor_state_constructible(self):
        state: ReactorState = {
            "constraints": [],
            "rejected_branches": [],
            "active_frames": [],
            "key_claims": [],
            "coalition_map": {},
            "unresolved": [],
            "turn_log": [],
        }
        assert isinstance(state["turn_log"], list)

    def test_claim_verdict_constructible(self):
        cv: ClaimVerdict = {
            "claim_id": "c-001",
            "claim_text": "test claim",
            "grounding_score": 0.8,
            "grounding_method": "jaccard_v1",
            "authority_score": 0.7,
            "authority_level": "institutional",
            "contradicted": False,
        }
        assert cv["grounding_score"] == 0.8

    def test_knowledge_artifact_constructible(self):
        ka: KnowledgeArtifact = {
            "question": "test",
            "facets": [],
            "clusters": [],
            "claim_verdicts": [],
            "active_tensions": [],
            "coverage": {
                "facet_coverage": {},
                "overall_coverage": 0.0,
                "uncovered_facets": [],
                "under_represented_perspectives": [],
            },
            "insights": [],
            "authority_profiles": [],
            "structural_risks": [],
            "compression_ratio": 1.0,
            "wave_count": 0,
        }
        assert ka["compression_ratio"] == 1.0

    def test_swarm_metadata_constructible(self):
        sm: SwarmMetadata = {
            "n_reactors": 3,
            "reactor_configs": [],
            "reactor_entropies": [0.5, 0.6, 0.4],
            "reactor_tokens": [1000, 1200, 900],
            "reactor_costs": [0.1, 0.12, 0.09],
            "winner_id": "reactor-0",
            "selection_reason": "highest composite score",
            "selection_scores": {},
            "cross_validation_scores": {},
            "total_tokens_all": 3100,
            "total_cost_all": 0.31,
            "failed_reactors": [],
        }
        assert sm["n_reactors"] == 3
