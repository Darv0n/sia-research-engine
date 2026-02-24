"""Tests for V8 types (Tunable, ComplexityProfile, AdaptationEvent) and state fields."""

from __future__ import annotations

from deep_research_swarm.contracts import AdaptationEvent, ComplexityProfile, Tunable


class TestTunableType:
    def test_create_tunable(self):
        t = Tunable(
            name="test_param",
            default=10,
            floor=1,
            ceiling=100,
            category="test",
        )
        assert t["name"] == "test_param"
        assert t["default"] == 10
        assert t["floor"] == 1
        assert t["ceiling"] == 100
        assert t["category"] == "test"

    def test_tunable_float(self):
        t = Tunable(
            name="threshold",
            default=0.3,
            floor=0.1,
            ceiling=0.9,
            category="grounding",
        )
        assert isinstance(t["default"], float)


class TestComplexityProfile:
    def test_create_profile(self):
        p = ComplexityProfile(
            result_count=500,
            backends_used=3,
            iteration=1,
            extraction_success_rate=0.85,
            mean_grounding_score=0.72,
            token_spend_rate=0.3,
            scored_doc_count=120,
            citation_chain_yield=25,
            volume_factor=1.35,
            backend_factor=1.1,
            iter_factor=1.0,
            multiplier=1.0,
        )
        assert p["result_count"] == 500
        assert p["multiplier"] == 1.0

    def test_profile_all_fields_present(self):
        """Verify ComplexityProfile has all expected metric fields."""
        fields = ComplexityProfile.__annotations__
        expected = {
            "result_count",
            "backends_used",
            "iteration",
            "extraction_success_rate",
            "mean_grounding_score",
            "token_spend_rate",
            "scored_doc_count",
            "citation_chain_yield",
            "volume_factor",
            "backend_factor",
            "iter_factor",
            "multiplier",
        }
        assert set(fields.keys()) == expected


class TestAdaptationEvent:
    def test_create_event(self):
        e = AdaptationEvent(
            tunable_name="extraction_cap",
            old_value=30,
            new_value=75,
            reason="High result volume (2800 results)",
            trigger="adapt_extraction",
            iteration=1,
        )
        assert e["tunable_name"] == "extraction_cap"
        assert e["old_value"] == 30
        assert e["new_value"] == 75
        assert e["trigger"] == "adapt_extraction"

    def test_event_all_fields_present(self):
        fields = AdaptationEvent.__annotations__
        expected = {
            "tunable_name",
            "old_value",
            "new_value",
            "reason",
            "trigger",
            "iteration",
        }
        assert set(fields.keys()) == expected


class TestStateFieldsExist:
    """Verify V8 state fields are declared in ResearchState."""

    def test_state_has_tunable_snapshot(self):
        from deep_research_swarm.graph.state import ResearchState

        assert "tunable_snapshot" in ResearchState.__annotations__

    def test_state_has_adaptation_events(self):
        from deep_research_swarm.graph.state import ResearchState

        assert "adaptation_events" in ResearchState.__annotations__

    def test_state_has_complexity_profile(self):
        from deep_research_swarm.graph.state import ResearchState

        assert "complexity_profile" in ResearchState.__annotations__


class TestConfigV8Settings:
    """Verify V8 config fields exist with correct defaults."""

    def test_adaptive_mode_default_true(self):
        import os

        # Clear env var to test default
        old = os.environ.pop("ADAPTIVE_MODE", None)
        try:
            from deep_research_swarm.config import Settings

            s = Settings()
            assert s.adaptive_mode is True
        finally:
            if old is not None:
                os.environ["ADAPTIVE_MODE"] = old

    def test_embedding_model_default(self):
        from deep_research_swarm.config import Settings

        s = Settings()
        assert s.embedding_model == "BAAI/bge-small-en-v1.5"

    def test_grobid_url_default_empty(self):
        from deep_research_swarm.config import Settings

        s = Settings()
        assert s.grobid_url == ""
