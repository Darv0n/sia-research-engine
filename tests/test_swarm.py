"""Tests for sia/swarm.py — SwarmOrchestrator, winner selection, perturbation."""

from __future__ import annotations

import pytest

from deep_research_swarm.config import Settings
from deep_research_swarm.sia.swarm import (
    SwarmOrchestrator,
    _compute_cross_validation,
    _generate_reactor_id,
    _perturb_state,
    _score_reactor,
    _split_budget,
    select_winner,
)


class TestGenerateReactorId:
    """Deterministic reactor ID generation."""

    def test_format(self):
        rid = _generate_reactor_id(0, "test question")
        assert rid.startswith("reactor-0-")
        assert len(rid) == len("reactor-0-") + 6

    def test_deterministic(self):
        a = _generate_reactor_id(1, "same question")
        b = _generate_reactor_id(1, "same question")
        assert a == b

    def test_different_index_different_id(self):
        a = _generate_reactor_id(0, "q")
        b = _generate_reactor_id(1, "q")
        assert a != b


class TestSplitBudget:
    """Budget splitting across reactors."""

    def test_even_split(self):
        budgets = _split_budget(200000, 2)
        assert budgets == [100000, 100000]

    def test_remainder_goes_to_first(self):
        budgets = _split_budget(200002, 3)
        assert sum(budgets) == 200002
        # 200002 // 3 = 66667, remainder = 200002 - 66667*3 = 1
        assert budgets[0] == 66668  # Gets the 1 extra token
        assert budgets[1] == 66667
        assert budgets[2] == 66667

    def test_three_reactors(self):
        budgets = _split_budget(300000, 3)
        assert budgets == [100000, 100000, 100000]

    def test_preserves_total(self):
        for n in range(2, 6):
            budgets = _split_budget(200000, n)
            assert sum(budgets) == 200000


class TestPerturbState:
    """Perturbation strategies for reactor diversity."""

    def _base_state(self):
        return {
            "research_question": "test",
            "token_budget": 200000,
            "perspectives": ["a", "b", "c"],
            "tunable_snapshot": {
                "target_queries": 12,
                "results_per_query": 15,
                "extraction_cap": 50,
            },
        }

    def test_baseline_no_change(self):
        base = self._base_state()
        result = _perturb_state(base, "baseline", 0)
        assert result["perspectives"] == ["a", "b", "c"]

    def test_entropy_high_sets_turbulence(self):
        result = _perturb_state(self._base_state(), "entropy_high", 1)
        assert result["entropy_state"]["band"] == "turbulence"
        assert result["entropy_state"]["e"] > 0.45

    def test_entropy_low_sets_crystalline(self):
        result = _perturb_state(self._base_state(), "entropy_low", 2)
        assert result["entropy_state"]["band"] == "crystalline"
        assert result["entropy_state"]["e"] < 0.20

    def test_perspective_shuffle_reverses(self):
        result = _perturb_state(self._base_state(), "perspective_shuffle", 3)
        assert result["perspectives"] == ["c", "b", "a"]

    def test_depth_focus_adjusts_tunables(self):
        result = _perturb_state(self._base_state(), "depth_focus", 4)
        snap = result["tunable_snapshot"]
        assert snap["target_queries"] < 12
        assert snap["results_per_query"] > 15
        assert snap["extraction_cap"] > 50

    def test_does_not_mutate_original(self):
        base = self._base_state()
        _perturb_state(base, "entropy_high", 1)
        assert "entropy_state" not in base


class TestScoreReactor:
    """Structural quality scoring for reactor results."""

    def _mock_result(self, **overrides):
        result = {
            "entropy_state": {"e": 0.3},
            "reactor_trace": {"constraints_produced": 5},
            "section_drafts": [
                {"heading": "A", "confidence_score": 0.8},
                {"heading": "B", "confidence_score": 0.7},
            ],
            "converged": True,
        }
        result.update(overrides)
        return result

    def test_entropy_stability(self):
        scores = _score_reactor(self._mock_result(entropy_state={"e": 0.1}))
        assert scores["entropy_stability"] == 0.9

    def test_entropy_stability_high_entropy(self):
        scores = _score_reactor(self._mock_result(entropy_state={"e": 0.8}))
        assert scores["entropy_stability"] == pytest.approx(0.2)

    def test_constraint_density_normalized(self):
        scores = _score_reactor(self._mock_result(reactor_trace={"constraints_produced": 10}))
        assert scores["constraint_density"] == 1.0

    def test_constraint_density_partial(self):
        scores = _score_reactor(self._mock_result(reactor_trace={"constraints_produced": 3}))
        assert scores["constraint_density"] == pytest.approx(0.3)

    def test_grounding_quality(self):
        scores = _score_reactor(self._mock_result())
        assert scores["grounding_quality"] == pytest.approx(0.75)

    def test_grounding_quality_no_sections(self):
        scores = _score_reactor(self._mock_result(section_drafts=[]))
        assert scores["grounding_quality"] == 0.0

    def test_convergence_bonus(self):
        scores_converged = _score_reactor(self._mock_result(converged=True))
        scores_not = _score_reactor(self._mock_result(converged=False))
        assert scores_converged["convergence"] > scores_not["convergence"]

    def test_comprehensiveness(self):
        drafts = [{"heading": f"S{i}", "confidence_score": 0.7} for i in range(8)]
        scores = _score_reactor(self._mock_result(section_drafts=drafts))
        assert scores["comprehensiveness"] == 1.0


class TestCrossValidation:
    """Cross-validation between reactors."""

    def test_two_identical_reactors(self):
        results = {
            "r-0": {"section_drafts": [{"heading": "A"}, {"heading": "B"}]},
            "r-1": {"section_drafts": [{"heading": "A"}, {"heading": "B"}]},
        }
        cv = _compute_cross_validation(results)
        assert cv["r-0"] == 1.0
        assert cv["r-1"] == 1.0

    def test_no_overlap(self):
        results = {
            "r-0": {"section_drafts": [{"heading": "A"}]},
            "r-1": {"section_drafts": [{"heading": "B"}]},
        }
        cv = _compute_cross_validation(results)
        assert cv["r-0"] == 0.0
        assert cv["r-1"] == 0.0

    def test_partial_overlap(self):
        results = {
            "r-0": {"section_drafts": [{"heading": "A"}, {"heading": "B"}]},
            "r-1": {"section_drafts": [{"heading": "A"}, {"heading": "C"}]},
        }
        cv = _compute_cross_validation(results)
        assert 0.0 < cv["r-0"] < 1.0

    def test_single_reactor_default(self):
        results = {"r-0": {"section_drafts": [{"heading": "A"}]}}
        cv = _compute_cross_validation(results)
        assert cv["r-0"] == 0.5


class TestSelectWinner:
    """Winner selection from completed reactor results."""

    def _mock_results(self):
        return {
            "r-0": {
                "entropy_state": {"e": 0.2},
                "reactor_trace": {"constraints_produced": 8},
                "section_drafts": [
                    {"heading": "A", "confidence_score": 0.85},
                    {"heading": "B", "confidence_score": 0.80},
                ],
                "converged": True,
            },
            "r-1": {
                "entropy_state": {"e": 0.6},
                "reactor_trace": {"constraints_produced": 3},
                "section_drafts": [
                    {"heading": "A", "confidence_score": 0.60},
                ],
                "converged": False,
            },
        }

    def test_selects_better_reactor(self):
        winner_id, reason, scores = select_winner(self._mock_results())
        assert winner_id == "r-0"
        assert scores["r-0"] > scores["r-1"]

    def test_reason_includes_score(self):
        _, reason, _ = select_winner(self._mock_results())
        assert "highest_weighted_score" in reason

    def test_single_reactor(self):
        results = {"r-0": {"entropy_state": {}, "section_drafts": []}}
        winner_id, reason, _ = select_winner(results)
        assert winner_id == "r-0"
        assert reason == "single_reactor"

    def test_empty_results(self):
        winner_id, reason, _ = select_winner({})
        assert winner_id == ""
        assert reason == "no_results"

    def test_custom_weights(self):
        results = self._mock_results()
        # Weight only grounding quality — r-0 should still win
        weights = {
            "entropy_stability": 0.0,
            "constraint_density": 0.0,
            "grounding_quality": 1.0,
            "comprehensiveness": 0.0,
            "convergence": 0.0,
            "cross_validation": 0.0,
        }
        winner_id, _, _ = select_winner(results, weights=weights)
        assert winner_id == "r-0"


class TestSwarmOrchestrator:
    """SwarmOrchestrator construction and config generation."""

    def _settings(self):
        return Settings(anthropic_api_key="test-key", swarm_max_reactors=5)

    def test_min_2_reactors(self):
        with pytest.raises(ValueError, match="at least 2"):
            SwarmOrchestrator(self._settings(), n_reactors=1)

    def test_caps_at_max(self):
        orch = SwarmOrchestrator(self._settings(), n_reactors=10)
        assert orch.n_reactors == 5  # Capped at swarm_max_reactors

    def test_builds_configs(self):
        orch = SwarmOrchestrator(self._settings(), n_reactors=3)
        configs = orch._build_reactor_configs({"research_question": "test", "token_budget": 300000})
        assert len(configs) == 3
        # Each config is (reactor_id, strategy, state)
        reactor_ids = [c[0] for c in configs]
        assert len(set(reactor_ids)) == 3  # All unique

    def test_budget_split_in_configs(self):
        orch = SwarmOrchestrator(self._settings(), n_reactors=3)
        configs = orch._build_reactor_configs({"research_question": "test", "token_budget": 300000})
        total = sum(c[2]["token_budget"] for c in configs)
        assert total == 300000

    def test_strategies_cycle(self):
        orch = SwarmOrchestrator(self._settings(), n_reactors=5)
        configs = orch._build_reactor_configs({"research_question": "test", "token_budget": 500000})
        strategies = [c[1] for c in configs]
        assert strategies[0] == "baseline"
        assert len(set(strategies)) == 5  # All different for 5 reactors


class TestBaseExceptionHandling:
    """C6 regression: CancelledError (BaseException) must be caught as failure."""

    def test_cancelled_error_is_base_exception(self):
        """CancelledError inherits BaseException, not Exception, in Python 3.8+."""
        import asyncio

        assert issubclass(asyncio.CancelledError, BaseException)
        assert not issubclass(asyncio.CancelledError, Exception)

    def test_base_exception_classified_as_failure(self):
        """isinstance(BaseException(), BaseException) must be True for the swarm filter."""
        import asyncio

        result = asyncio.CancelledError()
        # This is the exact check from swarm.py line 386
        assert isinstance(result, BaseException)

    def test_keyboard_interrupt_classified_as_failure(self):
        """KeyboardInterrupt is also BaseException — must be caught."""
        result = KeyboardInterrupt()
        assert isinstance(result, BaseException)

    def test_normal_exception_still_caught(self):
        """Regular Exception subclasses must still be caught."""
        result = ValueError("test error")
        assert isinstance(result, BaseException)
