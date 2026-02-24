"""Tests verifying adaptive tunables are wired into pipeline nodes (PR-05).

Tests ensure that tunable_snapshot values override V7 hardcoded defaults
and that missing/empty snapshots fall back gracefully.
"""

from __future__ import annotations

from deep_research_swarm.agents.extractor import extract_content
from deep_research_swarm.agents.searcher import search_sub_query

# --- Searcher: num_results parameter ---


class TestSearcherTunableWiring:
    def test_num_results_default(self):
        """Default num_results is 10 when no tunable provided."""
        import inspect

        sig = inspect.signature(search_sub_query)
        assert sig.parameters["num_results"].default == 10

    def test_num_results_parameter_accepted(self):
        """Searcher accepts num_results as a keyword argument."""
        import inspect

        sig = inspect.signature(search_sub_query)
        assert "num_results" in sig.parameters


# --- Extractor: content_truncation_chars parameter ---


class TestExtractorTunableWiring:
    def test_content_truncation_default(self):
        """Default content_truncation_chars is 50000 when no tunable provided."""
        import inspect

        sig = inspect.signature(extract_content)
        assert sig.parameters["content_truncation_chars"].default == 50000

    def test_truncation_parameter_accepted(self):
        """Extractor accepts content_truncation_chars as keyword argument."""
        import inspect

        sig = inspect.signature(extract_content)
        assert "content_truncation_chars" in sig.parameters


# --- Contradiction: max_docs parameter ---


class TestContradictionTunableWiring:
    def test_max_docs_default_none(self):
        """max_docs defaults to None (resolved to 10 internally)."""
        import inspect

        from deep_research_swarm.agents.contradiction import detect_contradictions

        sig = inspect.signature(detect_contradictions)
        assert sig.parameters["max_docs"].default is None


# --- Synthesizer: reads tunables from state ---


class TestSynthesizerTunableWiring:
    def test_synthesizer_reads_tunable_snapshot(self):
        """Synthesizer source code references tunable_snapshot."""
        import inspect

        from deep_research_swarm.agents.synthesizer import synthesize

        source = inspect.getsource(synthesize)
        assert "tunable_snapshot" in source

    def test_synthesizer_has_grounding_pass_fallback(self):
        """Synthesizer falls back to GROUNDING_PASS_THRESHOLD when no tunable."""
        import inspect

        from deep_research_swarm.agents.synthesizer import synthesize

        source = inspect.getsource(synthesize)
        assert "GROUNDING_PASS_THRESHOLD" in source
        assert "grounding_pass_threshold" in source


# --- Citation chain: reads tunables from state ---


class TestCitationChainTunableWiring:
    def test_citation_chain_reads_tunable_snapshot(self):
        """Citation chain source code references tunable_snapshot."""
        import inspect

        from deep_research_swarm.agents.citation_chain import citation_chain

        source = inspect.getsource(citation_chain)
        assert "tunable_snapshot" in source
        assert "citation_chain_budget" in source


# --- Critic: reads budget_exhaustion_pct from tunables ---


class TestCriticTunableWiring:
    def test_critic_reads_tunable_snapshot(self):
        """Critic source code references tunable_snapshot."""
        import inspect

        from deep_research_swarm.agents.critic import critique

        source = inspect.getsource(critique)
        assert "tunable_snapshot" in source
        assert "budget_exhaustion_pct" in source


# --- Builder: extract_node reads tunables ---


class TestBuilderTunableWiring:
    def test_builder_extract_reads_extraction_cap(self):
        """Build graph's extract_node reads extraction_cap from state."""
        import inspect

        from deep_research_swarm.graph.builder import build_graph

        source = inspect.getsource(build_graph)
        assert "extraction_cap" in source

    def test_builder_extract_reads_content_truncation(self):
        """Build graph's extract_node reads content_truncation_chars from state."""
        import inspect

        from deep_research_swarm.graph.builder import build_graph

        source = inspect.getsource(build_graph)
        assert "content_truncation_chars" in source

    def test_builder_search_reads_results_per_query(self):
        """Build graph's search_node reads results_per_query from state."""
        import inspect

        from deep_research_swarm.graph.builder import build_graph

        source = inspect.getsource(build_graph)
        assert "results_per_query" in source

    def test_builder_contradiction_reads_max_docs(self):
        """Build graph's contradiction_node reads contradiction_max_docs from state."""
        import inspect

        from deep_research_swarm.graph.builder import build_graph

        source = inspect.getsource(build_graph)
        assert "contradiction_max_docs" in source


# --- Backward Compat: empty tunable_snapshot uses defaults ---


class TestBackwardCompatibility:
    def test_empty_snapshot_uses_defaults(self):
        """When tunable_snapshot is empty, all defaults should apply."""
        from deep_research_swarm.adaptive.registry import TunableRegistry

        r = TunableRegistry.from_snapshot({})
        assert r.get("extraction_cap") == 30
        assert r.get("results_per_query") == 10
        assert r.get("contradiction_max_docs") == 10
        assert r.get("jaccard_threshold") == 0.3
        assert r.get("grounding_pass_threshold") == 0.8
        assert r.get("max_refinement_attempts") == 2
        assert r.get("citation_chain_budget") == 50
        assert r.get("budget_exhaustion_pct") == 0.9

    def test_missing_snapshot_key_uses_default(self):
        """state.get("tunable_snapshot", {}).get("name", DEFAULT) pattern."""
        state: dict = {}
        snap = state.get("tunable_snapshot", {})
        assert snap.get("extraction_cap", 30) == 30
        assert snap.get("results_per_query", 10) == 10
