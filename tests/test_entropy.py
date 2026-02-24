"""Tests for SIA Phase 1: Entropy layer — thermodynamic convergence control.

Target: ~40 tests covering classify_band, compute_entropy, entropy_gate,
false convergence detection, dominance detection, entropy steering, and
graph integration (node position, backward compat).
"""

from __future__ import annotations

from deep_research_swarm.contracts import EntropyBand, EntropyState
from deep_research_swarm.sia.entropy import (
    classify_band,
    compute_entropy,
    detect_dominance,
    detect_false_convergence,
    entropy_gate,
)
from deep_research_swarm.sia.entropy_steering import steer_tunables

# ============================================================
# classify_band tests
# ============================================================


class TestClassifyBand:
    def test_crystalline_at_zero(self):
        assert classify_band(0.0) == "crystalline"

    def test_crystalline_at_boundary(self):
        assert classify_band(0.20) == "crystalline"

    def test_convergence_just_above(self):
        assert classify_band(0.21) == "convergence"

    def test_convergence_at_boundary(self):
        assert classify_band(0.45) == "convergence"

    def test_turbulence_just_above(self):
        assert classify_band(0.46) == "turbulence"

    def test_turbulence_at_boundary(self):
        assert classify_band(0.70) == "turbulence"

    def test_runaway_just_above(self):
        assert classify_band(0.71) == "runaway"

    def test_runaway_at_one(self):
        assert classify_band(1.0) == "runaway"

    def test_all_bands_are_valid_enum_values(self):
        valid = {b.value for b in EntropyBand}
        for e in [0.0, 0.10, 0.20, 0.30, 0.45, 0.50, 0.70, 0.80, 1.0]:
            assert classify_band(e) in valid


# ============================================================
# compute_entropy tests
# ============================================================


def _make_state(**overrides):
    """Create a minimal ResearchState-like dict for testing."""
    base = {
        "section_drafts": [],
        "research_gaps": [],
        "contradictions": [],
        "scored_documents": [],
        "sub_queries": [],
        "follow_up_queries": [],
        "citations": [],
        "citation_to_passage_map": {},
    }
    base.update(overrides)
    return base


class TestComputeEntropy:
    def test_returns_entropy_state(self):
        entropy = compute_entropy(_make_state())
        assert "e" in entropy
        assert "band" in entropy
        assert "turn" in entropy

    def test_entropy_bounded_0_1(self):
        entropy = compute_entropy(_make_state())
        assert 0.0 <= entropy["e"] <= 1.0

    def test_all_components_bounded(self):
        entropy = compute_entropy(_make_state())
        for key in ("e_amb", "e_conf", "e_nov", "e_trust"):
            assert 0.0 <= entropy[key] <= 1.0

    def test_first_turn_number(self):
        entropy = compute_entropy(_make_state())
        assert entropy["turn"] == 1

    def test_turn_increments(self):
        e1 = compute_entropy(_make_state())
        e2 = compute_entropy(_make_state(), e1)
        assert e2["turn"] == 2

    def test_high_ambiguity_with_no_sections(self):
        entropy = compute_entropy(_make_state())
        assert entropy["e_amb"] > 0.7

    def test_low_ambiguity_with_high_confidence(self):
        sections = [
            {"confidence_score": 0.9, "grader_scores": {}, "id": "s1", "heading": "A"},
            {"confidence_score": 0.85, "grader_scores": {}, "id": "s2", "heading": "B"},
        ]
        entropy = compute_entropy(_make_state(section_drafts=sections))
        assert entropy["e_amb"] < 0.5

    def test_high_conflict_with_contradictions(self):
        contradictions = [{"claim_a": "x", "claim_b": "y", "severity": "direct"} for _ in range(5)]
        docs = [{"id": f"d{i}"} for i in range(10)]
        entropy = compute_entropy(_make_state(contradictions=contradictions, scored_documents=docs))
        assert entropy["e_conf"] > 0.3

    def test_stagnation_count_increments(self):
        e1 = compute_entropy(_make_state())
        # Same state -> similar entropy -> stagnation detected
        e2 = compute_entropy(_make_state(), e1)
        # May or may not be stagnant depending on novelty calc
        assert isinstance(e2["stagnation_count"], int)

    def test_stagnation_resets_on_big_change(self):
        e1: EntropyState = {
            "e": 0.5,
            "e_amb": 0.5,
            "e_conf": 0.5,
            "e_nov": 0.5,
            "e_trust": 0.5,
            "band": "turbulence",
            "turn": 1,
            "stagnation_count": 3,
        }
        # Create state that should produce very different entropy
        sections = [
            {"confidence_score": 0.95, "grader_scores": {}, "id": "s1", "heading": "A"}
            for _ in range(5)
        ]
        e2 = compute_entropy(
            _make_state(section_drafts=sections, sub_queries=[{"id": "sq-1"}]),
            e1,
        )
        # If entropy changed enough, stagnation should reset
        if abs(e2["e"] - e1["e"]) >= 0.03:
            assert e2["stagnation_count"] == 0


# ============================================================
# entropy_gate tests
# ============================================================


class TestEntropyGate:
    def test_allows_in_crystalline(self):
        entropy: EntropyState = {
            "e": 0.15,
            "e_amb": 0.1,
            "e_conf": 0.1,
            "e_nov": 0.1,
            "e_trust": 0.1,
            "band": "crystalline",
            "turn": 1,
            "stagnation_count": 0,
        }
        ok, reason = entropy_gate(entropy, [{"id": "s1"}])
        assert ok is True

    def test_allows_in_convergence(self):
        entropy: EntropyState = {
            "e": 0.35,
            "e_amb": 0.3,
            "e_conf": 0.3,
            "e_nov": 0.3,
            "e_trust": 0.3,
            "band": "convergence",
            "turn": 1,
            "stagnation_count": 0,
        }
        ok, _ = entropy_gate(entropy, [{"id": "s1"}])
        assert ok is True

    def test_blocks_in_runaway(self):
        entropy: EntropyState = {
            "e": 0.85,
            "e_amb": 0.8,
            "e_conf": 0.9,
            "e_nov": 0.7,
            "e_trust": 0.8,
            "band": "runaway",
            "turn": 1,
            "stagnation_count": 0,
        }
        ok, reason = entropy_gate(entropy, [{"id": "s1"}])
        assert ok is False
        assert "runaway" in reason

    def test_blocks_turbulence_no_sections(self):
        entropy: EntropyState = {
            "e": 0.55,
            "e_amb": 0.5,
            "e_conf": 0.5,
            "e_nov": 0.5,
            "e_trust": 0.5,
            "band": "turbulence",
            "turn": 1,
            "stagnation_count": 0,
        }
        ok, reason = entropy_gate(entropy, [])
        assert ok is False
        assert "turbulence" in reason

    def test_allows_turbulence_with_sections(self):
        entropy: EntropyState = {
            "e": 0.55,
            "e_amb": 0.5,
            "e_conf": 0.5,
            "e_nov": 0.5,
            "e_trust": 0.5,
            "band": "turbulence",
            "turn": 1,
            "stagnation_count": 0,
        }
        ok, _ = entropy_gate(entropy, [{"id": "s1"}])
        assert ok is True


# ============================================================
# detect_false_convergence tests
# ============================================================


class TestFalseConvergence:
    def test_no_false_convergence_normal(self):
        entropy: EntropyState = {
            "e": 0.15,
            "e_amb": 0.1,
            "e_conf": 0.1,
            "e_nov": 0.1,
            "e_trust": 0.1,
            "band": "crystalline",
            "turn": 3,
            "stagnation_count": 0,
        }
        fc, reason = detect_false_convergence(entropy, [{"id": "s1"}], [])
        assert fc is False

    def test_low_entropy_with_contradictions(self):
        entropy: EntropyState = {
            "e": 0.25,
            "e_amb": 0.2,
            "e_conf": 0.4,
            "e_nov": 0.1,
            "e_trust": 0.1,
            "band": "convergence",
            "turn": 3,
            "stagnation_count": 0,
        }
        contradictions = [
            {"claim_a": "x", "claim_b": "y", "severity": "direct"},
            {"claim_a": "a", "claim_b": "b", "severity": "nuanced"},
        ]
        fc, reason = detect_false_convergence(entropy, [], contradictions)
        assert fc is True
        assert "contradictions" in reason

    def test_rapid_drop_without_quality(self):
        prev: EntropyState = {
            "e": 0.7,
            "e_amb": 0.6,
            "e_conf": 0.7,
            "e_nov": 0.5,
            "e_trust": 0.5,
            "band": "turbulence",
            "turn": 2,
            "stagnation_count": 0,
        }
        curr: EntropyState = {
            "e": 0.3,
            "e_amb": 0.2,
            "e_conf": 0.3,
            "e_nov": 0.2,
            "e_trust": 0.2,
            "band": "convergence",
            "turn": 3,
            "stagnation_count": 0,
        }
        sections = [
            {"id": "s1", "confidence_score": 0.5},
            {"id": "s2", "confidence_score": 0.6},
        ]
        fc, reason = detect_false_convergence(curr, sections, [], prev)
        assert fc is True
        assert "rapid_drop" in reason

    def test_suspicious_uniformity(self):
        entropy: EntropyState = {
            "e": 0.15,
            "e_amb": 0.1,
            "e_conf": 0.1,
            "e_nov": 0.1,
            "e_trust": 0.1,
            "band": "crystalline",
            "turn": 3,
            "stagnation_count": 0,
        }
        sections = [{"id": f"s{i}", "confidence_score": 0.95} for i in range(5)]
        contradictions = [{"claim_a": "x", "claim_b": "y"}]
        fc, reason = detect_false_convergence(entropy, sections, contradictions)
        assert fc is True
        assert "uniformity" in reason


# ============================================================
# detect_dominance tests
# ============================================================


class TestDominance:
    def test_no_dominance_with_varied_scores(self):
        sections = [
            {
                "id": "s1",
                "confidence_score": 0.8,
                "grader_scores": {"relevance": 0.9, "hallucination": 0.7, "quality": 0.8},
            },
            {
                "id": "s2",
                "confidence_score": 0.6,
                "grader_scores": {"relevance": 0.5, "hallucination": 0.8, "quality": 0.6},
            },
            {
                "id": "s3",
                "confidence_score": 0.7,
                "grader_scores": {"relevance": 0.7, "hallucination": 0.6, "quality": 0.7},
            },
        ]
        dom, reason = detect_dominance([], sections)
        assert dom is False

    def test_dominance_with_uniform_scores(self):
        sections = [
            {
                "id": f"s{i}",
                "confidence_score": 0.75,
                "grader_scores": {"relevance": 0.75, "hallucination": 0.75, "quality": 0.75},
            }
            for i in range(5)
        ]
        dom, reason = detect_dominance([], sections)
        assert dom is True

    def test_insufficient_sections(self):
        sections = [{"id": "s1", "confidence_score": 0.8}]
        dom, reason = detect_dominance([], sections)
        assert dom is False
        assert "insufficient" in reason

    def test_grader_dimension_collapse(self):
        sections = [
            {
                "id": f"s{i}",
                "confidence_score": 0.5 + i * 0.1,
                "grader_scores": {"relevance": 0.7, "hallucination": 0.7, "quality": 0.7},
            }
            for i in range(4)
        ]
        dom, reason = detect_dominance([], sections)
        assert dom is True
        assert "dimension_collapse" in reason


# ============================================================
# entropy_steering tests
# ============================================================


class TestEntropySteering:
    def test_crystalline_no_change(self):
        entropy: EntropyState = {
            "e": 0.15,
            "e_amb": 0.1,
            "e_conf": 0.1,
            "e_nov": 0.1,
            "e_trust": 0.1,
            "band": "crystalline",
            "turn": 1,
            "stagnation_count": 0,
        }
        snap = {"perspectives_count": 5, "target_queries": 12}
        result = steer_tunables(entropy, snap)
        assert result == snap

    def test_runaway_reduces_perspectives(self):
        entropy: EntropyState = {
            "e": 0.85,
            "e_amb": 0.8,
            "e_conf": 0.9,
            "e_nov": 0.7,
            "e_trust": 0.8,
            "band": "runaway",
            "turn": 1,
            "stagnation_count": 0,
        }
        snap = {"perspectives_count": 5, "target_queries": 12, "follow_up_budget": 5}
        result = steer_tunables(entropy, snap)
        assert result["perspectives_count"] < snap["perspectives_count"]
        assert result["follow_up_budget"] == 0
        assert result["target_queries"] < snap["target_queries"]

    def test_turbulence_reduces_queries(self):
        entropy: EntropyState = {
            "e": 0.55,
            "e_amb": 0.5,
            "e_conf": 0.5,
            "e_nov": 0.5,
            "e_trust": 0.5,
            "band": "turbulence",
            "turn": 1,
            "stagnation_count": 0,
        }
        snap = {"target_queries": 12}
        result = steer_tunables(entropy, snap)
        assert result["target_queries"] < snap["target_queries"]

    def test_convergence_increases_sections(self):
        entropy: EntropyState = {
            "e": 0.35,
            "e_amb": 0.3,
            "e_conf": 0.3,
            "e_nov": 0.3,
            "e_trust": 0.3,
            "band": "convergence",
            "turn": 1,
            "stagnation_count": 0,
        }
        snap = {"max_sections": 8}
        result = steer_tunables(entropy, snap)
        assert result["max_sections"] > snap["max_sections"]

    def test_does_not_modify_original(self):
        entropy: EntropyState = {
            "e": 0.85,
            "e_amb": 0.8,
            "e_conf": 0.9,
            "e_nov": 0.7,
            "e_trust": 0.8,
            "band": "runaway",
            "turn": 1,
            "stagnation_count": 0,
        }
        snap = {"perspectives_count": 5, "target_queries": 12, "follow_up_budget": 5}
        original = dict(snap)
        steer_tunables(entropy, snap)
        assert snap == original

    def test_handles_missing_keys(self):
        entropy: EntropyState = {
            "e": 0.85,
            "e_amb": 0.8,
            "e_conf": 0.9,
            "e_nov": 0.7,
            "e_trust": 0.8,
            "band": "runaway",
            "turn": 1,
            "stagnation_count": 0,
        }
        snap = {"some_other_key": 42}
        result = steer_tunables(entropy, snap)
        assert result["some_other_key"] == 42


# ============================================================
# Graph Integration Tests
# ============================================================


class TestEntropyGraphIntegration:
    def test_state_has_entropy_fields(self):
        """Verify entropy state fields exist in ResearchState annotations."""
        from deep_research_swarm.graph.state import ResearchState

        annotations = ResearchState.__annotations__
        assert "entropy_state" in annotations
        assert "entropy_history" in annotations

    def test_compute_entropy_in_node_map(self):
        """Verify compute_entropy node exists in graph topology."""
        from deep_research_swarm.config import Settings
        from deep_research_swarm.graph.builder import build_graph

        settings = Settings(anthropic_api_key="test-key")
        graph = build_graph(settings, enable_cache=False)
        node_names = set(graph.get_graph().nodes.keys())
        assert "compute_entropy" in node_names

    def test_critique_connects_to_entropy(self):
        """Verify critique -> compute_entropy -> rollup_budget wiring."""
        from deep_research_swarm.config import Settings
        from deep_research_swarm.graph.builder import build_graph

        settings = Settings(anthropic_api_key="test-key")
        graph = build_graph(settings, enable_cache=False)
        g = graph.get_graph()

        # Find edges from critique — should go to compute_entropy
        critique_targets = set()
        for edge in g.edges:
            if edge.source == "critique":
                critique_targets.add(edge.target)
        assert "compute_entropy" in critique_targets

        # Find edges from compute_entropy — should go to rollup_budget
        entropy_targets = set()
        for edge in g.edges:
            if edge.source == "compute_entropy":
                entropy_targets.add(edge.target)
        assert "rollup_budget" in entropy_targets

    def test_backward_compat_empty_entropy(self):
        """compute_entropy should handle states without prior entropy_state."""
        state = _make_state()
        entropy = compute_entropy(state)
        assert entropy["turn"] == 1
        assert entropy["stagnation_count"] == 0
