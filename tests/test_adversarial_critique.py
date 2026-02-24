"""Tests for SIA Phase 4 — Adversarial Critique + Singularity Prevention.

Tests cover:
- Adversarial critique: sequence, conditional activation, score adjustments
- Singularity prevention: 4 checks + 7-axis stability
- Five-way convergence: entropy + singularity veto
- Critic mode switch: SIA enabled -> adversarial, disabled -> classic
- State fields: adversarial_findings, critique_trace
"""

from __future__ import annotations

import pytest

from deep_research_swarm.contracts import (
    AdversarialFinding,
    Confidence,
    CritiqueTrace,
    GraderScores,
    SectionDraft,
    TurnRecord,
)
from deep_research_swarm.sia.adversarial_critique import (
    CRITIQUE_SEQUENCE,
    PROCESSING_BIASES,
    SEVERITY_ADJUSTMENTS,
    _apply_score_adjustments,
    _build_sections_text,
    _parse_critique_output,
    _should_activate_agent,
)
from deep_research_swarm.sia.singularity_prevention import (
    check_axis_stability,
    check_coalition_shadow,
    check_constraint_singularity,
    check_directional_singularity,
    check_reframe_singularity,
    singularity_check,
)

# ============================================================
# Helpers
# ============================================================


def _make_section(
    sid: str = "sec-001",
    heading: str = "Test Section",
    score: float = 0.85,
) -> SectionDraft:
    return SectionDraft(
        id=sid,
        heading=heading,
        content="Test content [1] with citations [2].",
        citation_ids=["[1]", "[2]"],
        confidence_score=score,
        confidence_level=Confidence.HIGH if score >= 0.8 else Confidence.MEDIUM,
        grader_scores=GraderScores(relevance=score, hallucination=score, quality=score),
    )


def _make_finding(
    agent: str = "makishima",
    severity: str = "minor",
    target: str = "sec-001",
) -> AdversarialFinding:
    return AdversarialFinding(
        agent=agent,
        int_type="C",
        target_section=target,
        finding="Test finding",
        severity=severity,
        actionable=True,
        response_to="",
    )


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


def _make_reactor_state(
    turn_log: list[TurnRecord] | None = None,
    constraints: list[str] | None = None,
    active_frames: list[str] | None = None,
    coalition_map: dict[str, list[str]] | None = None,
) -> dict:
    return {
        "turn_log": turn_log or [],
        "constraints": constraints or [],
        "rejected_branches": [],
        "active_frames": active_frames or [],
        "key_claims": [],
        "coalition_map": coalition_map or {},
        "unresolved": [],
    }


# ============================================================
# Adversarial Critique Sequence
# ============================================================


class TestCritiqueSequence:
    """Test the adversarial critique agent sequence."""

    def test_sequence_has_5_entries(self):
        assert len(CRITIQUE_SEQUENCE) == 5

    def test_sequence_order(self):
        agent_ids = [agent_id for agent_id, _ in CRITIQUE_SEQUENCE]
        assert agent_ids == ["makishima", "lawliet", "rick", "shikamaru", "light"]

    def test_makishima_is_required(self):
        assert CRITIQUE_SEQUENCE[0] == ("makishima", True)

    def test_lawliet_is_required(self):
        assert CRITIQUE_SEQUENCE[1] == ("lawliet", True)

    def test_rick_is_conditional(self):
        assert CRITIQUE_SEQUENCE[2] == ("rick", False)

    def test_shikamaru_is_required(self):
        assert CRITIQUE_SEQUENCE[3] == ("shikamaru", True)

    def test_light_is_conditional(self):
        assert CRITIQUE_SEQUENCE[4] == ("light", False)


class TestConditionalActivation:
    """Test conditional agent activation logic."""

    def test_rick_activates_high_entropy(self):
        assert _should_activate_agent("rick", 0.50, False) is True

    def test_rick_inactive_low_entropy(self):
        assert _should_activate_agent("rick", 0.30, False) is False

    def test_rick_boundary_exact_040(self):
        assert _should_activate_agent("rick", 0.40, False) is False

    def test_rick_above_boundary(self):
        assert _should_activate_agent("rick", 0.41, False) is True

    def test_light_activates_on_replan(self):
        assert _should_activate_agent("light", 0.30, True) is True

    def test_light_inactive_no_replan(self):
        assert _should_activate_agent("light", 0.30, False) is False

    def test_required_agent_always_active(self):
        assert _should_activate_agent("makishima", 0.10, False) is True
        assert _should_activate_agent("lawliet", 0.10, False) is True
        assert _should_activate_agent("shikamaru", 0.10, False) is True


# ============================================================
# Score Adjustments
# ============================================================


class TestScoreAdjustments:
    """Test severity-based score adjustments."""

    def test_severity_values(self):
        assert SEVERITY_ADJUSTMENTS["critical"] == -0.15
        assert SEVERITY_ADJUSTMENTS["significant"] == -0.08
        assert SEVERITY_ADJUSTMENTS["minor"] == -0.03

    def test_critical_finding_reduces_score(self):
        sections = [_make_section(score=0.90)]
        findings = [_make_finding(severity="critical", target="sec-001")]
        updated = _apply_score_adjustments(sections, findings)
        assert updated[0]["confidence_score"] == pytest.approx(0.75, abs=0.01)

    def test_significant_finding_reduces_score(self):
        sections = [_make_section(score=0.90)]
        findings = [_make_finding(severity="significant", target="sec-001")]
        updated = _apply_score_adjustments(sections, findings)
        assert updated[0]["confidence_score"] == pytest.approx(0.82, abs=0.01)

    def test_minor_finding_reduces_score(self):
        sections = [_make_section(score=0.90)]
        findings = [_make_finding(severity="minor", target="sec-001")]
        updated = _apply_score_adjustments(sections, findings)
        assert updated[0]["confidence_score"] == pytest.approx(0.87, abs=0.01)

    def test_global_finding_applies_to_all_sections(self):
        sections = [
            _make_section(sid="sec-001", score=0.90),
            _make_section(sid="sec-002", score=0.90),
        ]
        findings = [_make_finding(severity="critical", target="global")]
        updated = _apply_score_adjustments(sections, findings)
        # Global applies at half impact
        for sec in updated:
            assert sec["confidence_score"] < 0.90

    def test_multiple_findings_stack(self):
        sections = [_make_section(score=0.90)]
        findings = [
            _make_finding(severity="critical", target="sec-001"),
            _make_finding(severity="significant", target="sec-001"),
        ]
        updated = _apply_score_adjustments(sections, findings)
        # 0.90 - 0.15 - 0.08 = 0.67
        assert updated[0]["confidence_score"] == pytest.approx(0.67, abs=0.01)

    def test_score_never_below_zero(self):
        sections = [_make_section(score=0.10)]
        findings = [
            _make_finding(severity="critical", target="sec-001"),
            _make_finding(severity="critical", target="sec-001"),
        ]
        updated = _apply_score_adjustments(sections, findings)
        assert updated[0]["confidence_score"] >= 0.0

    def test_confidence_level_updated(self):
        sections = [_make_section(score=0.85)]
        findings = [_make_finding(severity="critical", target="sec-001")]
        updated = _apply_score_adjustments(sections, findings)
        # 0.85 - 0.15 = 0.70 -> MEDIUM
        assert updated[0]["confidence_level"] == Confidence.MEDIUM


# ============================================================
# Output Parsing
# ============================================================


class TestCritiqueOutputParsing:
    """Test critique output parsing."""

    def test_parse_valid_json(self):
        output = (
            '{"findings": [{"target_section": "sec-001", "finding": "test",'
            ' "severity": "minor", "actionable": true}],'
            ' "recommendation": "converge", "constraints": [],'
            ' "missing_variables": [], "alternative_frames": []}'
        )
        result = _parse_critique_output("makishima", output)
        assert len(result["findings"]) == 1
        assert result["recommendation"] == "converge"

    def test_parse_json_in_text(self):
        output = (
            "Here is my analysis:\n"
            '{"findings": [{"target_section": "global", "finding": "issue",'
            ' "severity": "significant", "actionable": true}],'
            ' "recommendation": "replan"}\nEnd.'
        )
        result = _parse_critique_output("lawliet", output)
        assert len(result["findings"]) == 1

    def test_parse_fallback_text(self):
        output = "Some analysis with a critical: issue found here\n"
        result = _parse_critique_output("rick", output)
        assert len(result["findings"]) >= 1

    def test_parse_empty_output(self):
        result = _parse_critique_output("makishima", "")
        assert result["findings"] == []
        assert result["recommendation"] == "converge"


# ============================================================
# Processing Biases
# ============================================================


class TestProcessingBiases:
    """Test that all critique sequence agents have processing biases."""

    def test_all_agents_have_biases(self):
        for agent_id, _ in CRITIQUE_SEQUENCE:
            assert agent_id in PROCESSING_BIASES, f"Missing bias for {agent_id}"

    def test_biases_are_nonempty(self):
        for agent_id, bias in PROCESSING_BIASES.items():
            assert len(bias) > 20, f"Bias too short for {agent_id}"


# ============================================================
# Sections Text Builder
# ============================================================


class TestSectionsText:
    """Test section text formatting."""

    def test_builds_text_with_ids(self):
        sections = [_make_section(sid="sec-001", heading="Introduction")]
        text = _build_sections_text(sections)
        assert "sec-001" in text
        assert "Introduction" in text

    def test_multiple_sections(self):
        sections = [
            _make_section(sid="sec-001", heading="First"),
            _make_section(sid="sec-002", heading="Second"),
        ]
        text = _build_sections_text(sections)
        assert "First" in text
        assert "Second" in text


# ============================================================
# Singularity Prevention — Constraint Singularity
# ============================================================


class TestConstraintSingularity:
    """Test constraint singularity detection."""

    def test_no_turn_log_is_safe(self):
        state = _make_reactor_state()
        safe, reason = check_constraint_singularity(state)
        assert safe is True

    def test_insufficient_constraints_is_safe(self):
        state = _make_reactor_state(
            turn_log=[_make_record(constraints=["c1"])],
            constraints=["c1"],
        )
        safe, reason = check_constraint_singularity(state)
        assert safe is True

    def test_all_from_one_agent_is_unsafe(self):
        records = [
            _make_record(turn=0, agent="lawliet", constraints=["c1", "c2", "c3"]),
            _make_record(turn=1, agent="light", constraints=[]),
            _make_record(turn=2, agent="rick", constraints=[]),
        ]
        state = _make_reactor_state(
            turn_log=records,
            constraints=["c1", "c2", "c3"],
        )
        safe, reason = check_constraint_singularity(state)
        assert safe is False
        assert "constraint_singularity" in reason or "near_singularity" in reason

    def test_distributed_constraints_is_safe(self):
        records = [
            _make_record(turn=0, agent="lawliet", constraints=["c1"]),
            _make_record(turn=1, agent="light", constraints=["c2"]),
            _make_record(turn=2, agent="rick", constraints=["c3"]),
        ]
        state = _make_reactor_state(
            turn_log=records,
            constraints=["c1", "c2", "c3"],
        )
        safe, reason = check_constraint_singularity(state)
        assert safe is True

    def test_dominant_agent_over_80pct_is_unsafe(self):
        records = [
            _make_record(
                turn=0,
                agent="lawliet",
                constraints=["c1", "c2", "c3", "c4", "c5"],
            ),
            _make_record(turn=1, agent="light", constraints=["c6"]),
        ]
        state = _make_reactor_state(
            turn_log=records,
            constraints=["c1", "c2", "c3", "c4", "c5", "c6"],
        )
        safe, reason = check_constraint_singularity(state)
        assert safe is False
        assert "near_singularity" in reason


# ============================================================
# Singularity Prevention — Directional Singularity
# ============================================================


class TestDirectionalSingularity:
    """Test directional singularity detection."""

    def test_insufficient_turns_is_safe(self):
        state = _make_reactor_state(turn_log=[_make_record()])
        safe, reason = check_directional_singularity(state)
        assert safe is True

    def test_no_reframes_in_many_turns_is_unsafe(self):
        records = [_make_record(turn=i, agent="lawliet", int_type="B") for i in range(5)]
        state = _make_reactor_state(turn_log=records)
        safe, reason = check_directional_singularity(state)
        assert safe is False
        assert "directional_singularity" in reason

    def test_reframes_present_is_safe(self):
        records = [
            _make_record(turn=0, agent="lawliet", int_type="B"),
            _make_record(turn=1, agent="light", int_type="B"),
            _make_record(turn=2, agent="rick", int_type="RF", reframes=["frame1"]),
            _make_record(turn=3, agent="lawliet", int_type="B"),
        ]
        state = _make_reactor_state(turn_log=records, active_frames=["frame1"])
        safe, reason = check_directional_singularity(state)
        assert safe is True

    def test_reframes_attempted_but_not_accepted(self):
        records = [
            _make_record(turn=0, agent="lawliet", int_type="B"),
            _make_record(turn=1, agent="rick", int_type="RF"),
            _make_record(turn=2, agent="lawliet", int_type="B"),
            _make_record(turn=3, agent="light", int_type="B"),
        ]
        state = _make_reactor_state(turn_log=records, active_frames=[])
        safe, reason = check_directional_singularity(state)
        assert safe is False
        assert "none accepted" in reason


# ============================================================
# Singularity Prevention — Reframe Singularity
# ============================================================


class TestReframeSingularity:
    """Test reframe singularity detection."""

    def test_insufficient_turns_is_safe(self):
        state = _make_reactor_state(turn_log=[_make_record()])
        safe, reason = check_reframe_singularity(state)
        assert safe is True

    def test_many_frames_no_constraints_is_unsafe(self):
        records = [_make_record(turn=i) for i in range(4)]
        state = _make_reactor_state(
            turn_log=records,
            active_frames=["f1", "f2", "f3"],
            constraints=[],
        )
        safe, reason = check_reframe_singularity(state)
        assert safe is False
        assert "reframe_singularity" in reason

    def test_balanced_frames_and_constraints_is_safe(self):
        records = [_make_record(turn=i) for i in range(4)]
        state = _make_reactor_state(
            turn_log=records,
            active_frames=["f1", "f2"],
            constraints=["c1", "c2"],
        )
        safe, reason = check_reframe_singularity(state)
        assert safe is True

    def test_high_frame_to_constraint_ratio_is_unsafe(self):
        records = [_make_record(turn=i) for i in range(5)]
        state = _make_reactor_state(
            turn_log=records,
            active_frames=["f1", "f2", "f3", "f4"],
            constraints=["c1"],
        )
        safe, reason = check_reframe_singularity(state)
        assert safe is False
        assert "near_singularity" in reason


# ============================================================
# Singularity Prevention — Coalition Shadow
# ============================================================


class TestCoalitionShadow:
    """Test coalition shadow detection."""

    def test_insufficient_turns_is_safe(self):
        state = _make_reactor_state(turn_log=[_make_record()])
        safe, reason = check_coalition_shadow(state)
        assert safe is True

    def test_agreement_without_coalitions_is_unsafe(self):
        records = [
            _make_record(turn=0, agent="lawliet", constraints=["c1"]),
            _make_record(turn=1, agent="light", constraints=["c2"]),
            _make_record(turn=2, agent="shikamaru"),
            _make_record(turn=3, agent="makishima"),
        ]
        state = _make_reactor_state(
            turn_log=records,
            constraints=["c1", "c2"],
            coalition_map={},
        )
        safe, reason = check_coalition_shadow(state)
        assert safe is False
        assert "coalition_shadow" in reason

    def test_explicit_coalitions_is_safe(self):
        records = [
            _make_record(turn=0, agent="lawliet", constraints=["c1"]),
            _make_record(turn=1, agent="light", constraints=["c2"]),
            _make_record(turn=2, agent="shikamaru"),
            _make_record(turn=3, agent="makishima"),
        ]
        state = _make_reactor_state(
            turn_log=records,
            constraints=["c1", "c2"],
            coalition_map={"light": ["lawliet"]},
        )
        safe, reason = check_coalition_shadow(state)
        assert safe is True

    def test_challenges_present_is_safe(self):
        records = [
            _make_record(turn=0, agent="lawliet", constraints=["c1"]),
            _make_record(turn=1, agent="rick", challenges=["challenge1"]),
            _make_record(turn=2, agent="shikamaru"),
            _make_record(turn=3, agent="makishima"),
        ]
        state = _make_reactor_state(
            turn_log=records,
            constraints=["c1"],
        )
        safe, reason = check_coalition_shadow(state)
        assert safe is True


# ============================================================
# Singularity Prevention — 7-Axis Stability
# ============================================================


class TestAxisStability:
    """Test 7-axis stability check."""

    def test_no_turn_log_is_safe(self):
        state = _make_reactor_state()
        safe, reason, axes = check_axis_stability(state)
        assert safe is True

    def test_single_agent_is_safe(self):
        state = _make_reactor_state(turn_log=[_make_record(agent="lawliet")])
        safe, reason, axes = check_axis_stability(state)
        assert safe is True

    def test_diverse_agents_balanced(self):
        records = [
            _make_record(turn=0, agent="lawliet"),
            _make_record(turn=1, agent="rick"),
            _make_record(turn=2, agent="shikamaru"),
        ]
        state = _make_reactor_state(turn_log=records)
        safe, reason, axes = check_axis_stability(state)
        assert safe is True
        assert len(axes) == 7

    def test_axis_summary_has_all_7(self):
        records = [
            _make_record(turn=0, agent="lawliet"),
            _make_record(turn=1, agent="light"),
        ]
        state = _make_reactor_state(turn_log=records)
        _, _, axes = check_axis_stability(state)
        assert "a1_constraint_density" in axes
        assert "a7_energy_expenditure" in axes


# ============================================================
# Singularity Prevention — Full Check
# ============================================================


class TestSingularityCheck:
    """Test the aggregate singularity check."""

    def test_empty_state_is_safe(self):
        state = _make_reactor_state()
        safe, reason, details = singularity_check(state)
        assert safe is True

    def test_all_checks_in_details(self):
        state = _make_reactor_state()
        _, _, details = singularity_check(state)
        assert "constraint" in details
        assert "directional" in details
        assert "reframe" in details
        assert "coalition" in details
        assert "axes" in details

    def test_one_failure_makes_unsafe(self):
        # Trigger directional singularity
        records = [_make_record(turn=i, agent="lawliet", int_type="B") for i in range(5)]
        state = _make_reactor_state(turn_log=records)
        safe, reason, details = singularity_check(state)
        assert safe is False
        assert "singularity_detected" in reason

    def test_healthy_reactor_is_safe(self):
        records = [
            _make_record(turn=0, agent="lawliet", int_type="INIT", constraints=["c1"]),
            _make_record(turn=1, agent="light", int_type="B", constraints=["c2"]),
            _make_record(
                turn=2,
                agent="rick",
                int_type="RF",
                reframes=["frame1"],
                challenges=["ch1"],
            ),
            _make_record(turn=3, agent="makishima", int_type="C", constraints=["c3"]),
        ]
        state = _make_reactor_state(
            turn_log=records,
            constraints=["c1", "c2", "c3"],
            active_frames=["frame1"],
            coalition_map={"light": ["lawliet"]},
        )
        safe, reason, details = singularity_check(state)
        assert safe is True


# ============================================================
# Critic Mode Switch
# ============================================================


class TestCriticModeSwitch:
    """Test the SIA mode switch in critic.py."""

    def test_is_sia_enabled_with_both_fields(self):
        from deep_research_swarm.agents.critic import _is_sia_enabled

        state = {
            "entropy_state": {"e": 0.35, "band": "convergence"},
            "reactor_trace": {"turns_executed": 6},
        }
        assert _is_sia_enabled(state) is True

    def test_is_sia_disabled_without_entropy(self):
        from deep_research_swarm.agents.critic import _is_sia_enabled

        state = {
            "entropy_state": {},
            "reactor_trace": {"turns_executed": 6},
        }
        assert _is_sia_enabled(state) is False

    def test_is_sia_disabled_without_reactor(self):
        from deep_research_swarm.agents.critic import _is_sia_enabled

        state = {
            "entropy_state": {"e": 0.35, "band": "convergence"},
            "reactor_trace": {},
        }
        assert _is_sia_enabled(state) is False

    def test_is_sia_disabled_empty_state(self):
        from deep_research_swarm.agents.critic import _is_sia_enabled

        state = {}
        assert _is_sia_enabled(state) is False


# ============================================================
# Adversarial Critique Full Integration (async)
# ============================================================


class TestAdversarialCritiqueIntegration:
    """Integration tests for the adversarial_critique function."""

    @pytest.mark.asyncio
    async def test_returns_expected_keys(self):
        from unittest.mock import AsyncMock

        caller = AsyncMock()
        caller.call_json = AsyncMock(
            return_value=(
                {
                    "findings": [
                        {
                            "target_section": "sec-001",
                            "finding": "test",
                            "severity": "minor",
                            "actionable": True,
                        }
                    ],
                    "recommendation": "converge",
                    "constraints": [],
                    "missing_variables": [],
                    "alternative_frames": [],
                },
                {
                    "agent": "adversarial_makishima",
                    "model": "test",
                    "input_tokens": 100,
                    "output_tokens": 50,
                    "cost_usd": 0.01,
                    "timestamp": "2026-01-01T00:00:00Z",
                },
            )
        )

        state = {
            "research_question": "test question",
            "section_drafts": [_make_section()],
            "current_iteration": 1,
            "max_iterations": 3,
            "total_tokens_used": 1000,
            "token_budget": 200000,
            "entropy_state": {"e": 0.35, "band": "convergence"},
            "iteration_history": [],
            "sub_queries": [],
            "search_results": [],
            "extracted_contents": [],
            "tunable_snapshot": {},
        }

        from deep_research_swarm.sia.adversarial_critique import (
            adversarial_critique,
        )

        result = await adversarial_critique(state, caller)

        assert "section_drafts" in result
        assert "converged" in result
        assert "convergence_reason" in result
        assert "iteration_history" in result
        assert "token_usage" in result
        assert "adversarial_findings" in result
        assert "critique_trace" in result

    @pytest.mark.asyncio
    async def test_required_agents_called(self):
        from unittest.mock import AsyncMock

        call_count = 0

        async def mock_call_json(**kwargs):
            nonlocal call_count
            call_count += 1
            return (
                {
                    "findings": [],
                    "recommendation": "converge",
                    "constraints": [],
                    "missing_variables": [],
                    "alternative_frames": [],
                },
                {
                    "agent": kwargs.get("agent_name", "test"),
                    "model": "test",
                    "input_tokens": 100,
                    "output_tokens": 50,
                    "cost_usd": 0.01,
                    "timestamp": "2026-01-01T00:00:00Z",
                },
            )

        caller = AsyncMock()
        caller.call_json = mock_call_json

        state = {
            "research_question": "test",
            "section_drafts": [_make_section()],
            "current_iteration": 1,
            "max_iterations": 3,
            "total_tokens_used": 0,
            "token_budget": 200000,
            "entropy_state": {"e": 0.30},  # below 0.40, Rick skipped
            "iteration_history": [],
            "sub_queries": [],
            "search_results": [],
            "extracted_contents": [],
            "tunable_snapshot": {},
        }

        from deep_research_swarm.sia.adversarial_critique import (
            adversarial_critique,
        )

        await adversarial_critique(state, caller)

        # 3 required agents (makishima, lawliet, shikamaru)
        # Rick skipped (entropy 0.30 <= 0.40)
        # Light skipped (no replan recommendation)
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_rick_included_high_entropy(self):
        from unittest.mock import AsyncMock

        agent_names_called = []

        async def mock_call_json(**kwargs):
            agent_names_called.append(kwargs.get("agent_name", ""))
            return (
                {
                    "findings": [],
                    "recommendation": "converge",
                    "constraints": [],
                    "missing_variables": [],
                    "alternative_frames": [],
                },
                {
                    "agent": "test",
                    "model": "test",
                    "input_tokens": 100,
                    "output_tokens": 50,
                    "cost_usd": 0.01,
                    "timestamp": "2026-01-01T00:00:00Z",
                },
            )

        caller = AsyncMock()
        caller.call_json = mock_call_json

        state = {
            "research_question": "test",
            "section_drafts": [_make_section()],
            "current_iteration": 1,
            "max_iterations": 3,
            "total_tokens_used": 0,
            "token_budget": 200000,
            "entropy_state": {"e": 0.50},  # above 0.40, Rick included
            "iteration_history": [],
            "sub_queries": [],
            "search_results": [],
            "extracted_contents": [],
            "tunable_snapshot": {},
        }

        from deep_research_swarm.sia.adversarial_critique import (
            adversarial_critique,
        )

        await adversarial_critique(state, caller)

        assert "adversarial_rick" in agent_names_called

    @pytest.mark.asyncio
    async def test_empty_sections_returns_converged(self):
        from unittest.mock import AsyncMock

        caller = AsyncMock()
        state = {
            "research_question": "test",
            "section_drafts": [],
            "entropy_state": {"e": 0.35},
        }

        from deep_research_swarm.sia.adversarial_critique import (
            adversarial_critique,
        )

        result = await adversarial_critique(state, caller)
        assert result["converged"] is True
        assert result["convergence_reason"] == "no_sections_to_evaluate"

    @pytest.mark.asyncio
    async def test_max_iterations_forces_convergence(self):
        from unittest.mock import AsyncMock

        caller = AsyncMock()
        caller.call_json = AsyncMock(
            return_value=(
                {
                    "findings": [
                        {
                            "target_section": "sec-001",
                            "finding": "critical issue",
                            "severity": "critical",
                            "actionable": True,
                        }
                    ],
                    "recommendation": "replan",
                    "constraints": [],
                    "missing_variables": [],
                    "alternative_frames": [],
                },
                {
                    "agent": "test",
                    "model": "test",
                    "input_tokens": 100,
                    "output_tokens": 50,
                    "cost_usd": 0.01,
                    "timestamp": "2026-01-01T00:00:00Z",
                },
            )
        )

        state = {
            "research_question": "test",
            "section_drafts": [_make_section()],
            "current_iteration": 3,
            "max_iterations": 3,
            "total_tokens_used": 0,
            "token_budget": 200000,
            "entropy_state": {"e": 0.35},
            "iteration_history": [],
            "sub_queries": [],
            "search_results": [],
            "extracted_contents": [],
            "tunable_snapshot": {},
        }

        from deep_research_swarm.sia.adversarial_critique import (
            adversarial_critique,
        )

        result = await adversarial_critique(state, caller)
        assert result["converged"] is True
        assert "max_iterations" in result["convergence_reason"]


# ============================================================
# State Fields
# ============================================================


class TestStateFields:
    """Test that Phase 4 state fields exist."""

    def test_adversarial_findings_field(self):
        from deep_research_swarm.graph.state import ResearchState

        assert "adversarial_findings" in ResearchState.__annotations__

    def test_critique_trace_field(self):
        from deep_research_swarm.graph.state import ResearchState

        assert "critique_trace" in ResearchState.__annotations__


# ============================================================
# Graph Integration
# ============================================================


class TestGraphIntegration:
    """Test that Phase 4 integrates with the graph."""

    def test_graph_compiles(self):
        from deep_research_swarm.config import Settings

        settings = Settings(
            anthropic_api_key="test-key",
            searxng_url="http://localhost:8080",
        )
        graph = __import__(
            "deep_research_swarm.graph.builder", fromlist=["build_graph"]
        ).build_graph(settings, enable_cache=False)
        assert graph is not None

    def test_singularity_check_importable_from_builder(self):
        from deep_research_swarm.graph.builder import singularity_check

        assert callable(singularity_check)

    def test_critique_trace_contract(self):
        """CritiqueTrace has all expected fields."""
        fields = CritiqueTrace.__annotations__
        assert "turns" in fields
        assert "findings_count" in fields
        assert "critical_findings" in fields
        assert "constraints_extracted" in fields
        assert "missing_variables" in fields
        assert "alternative_frames" in fields
        assert "recommendation" in fields

    def test_adversarial_finding_contract(self):
        """AdversarialFinding has all expected fields."""
        fields = AdversarialFinding.__annotations__
        assert "agent" in fields
        assert "int_type" in fields
        assert "target_section" in fields
        assert "finding" in fields
        assert "severity" in fields
        assert "actionable" in fields
        assert "response_to" in fields
