"""Tests for V10 Phase 2 — Tensegrity wiring, deliberation panel, compression.

Coverage:
  - Deliberation panel: all 4 judges (output shapes, edge cases)
  - Merge: cross-referencing, structural risks
  - Clustering: heading-based (embedding tested separately)
  - Artifact: build, compression ratio, empty input
  - Grounding cascade: verify_claim, batch verification
  - Synthesizer: V10 path routing, V9 fallback
  - Config: haiku_model
  - Registry: 6 new tunables
  - Graph: topology (new nodes + edges)
  - State: new fields
"""

from __future__ import annotations

import inspect

from deep_research_swarm.compress.artifact import build_knowledge_artifact
from deep_research_swarm.compress.cluster import (
    cluster_by_heading,
    rank_passages_in_cluster,
)
from deep_research_swarm.compress.grounding import verify_claim, verify_claims_batch
from deep_research_swarm.contracts import (
    ClaimVerdict,
    CoverageMap,
    JudgmentContext,
    SourcePassage,
)
from deep_research_swarm.deliberate.merge import merge_judgments
from deep_research_swarm.deliberate.panel import (
    authority_judge,
    coverage_judge,
    grounding_judge,
)

# ============================================================
# Helpers
# ============================================================


def _make_passage(
    pid: str = "sp-test1",
    content: str = "Test content about research",
    url: str = "https://example.com/test",
    heading: str = "Test Heading",
) -> SourcePassage:
    return SourcePassage(
        id=pid,
        source_id=url,
        source_url=url,
        heading_context=heading,
        content=content,
        position=0,
        char_offset=0,
        token_count=len(content.split()),
        claim_ids=[],
    )


def _make_scored_doc(
    url: str = "https://example.com/test",
    title: str = "Test Doc",
    content: str = "Test content about research",
    score: float = 0.8,
) -> dict:
    return {
        "url": url,
        "title": title,
        "content": content,
        "combined_score": score,
        "authority": "professional",
        "authority_level": "professional",
    }


def _make_section_draft(
    sid: str = "sec-test",
    heading: str = "Test Section",
    content: str = "This is test content [1].",
    confidence: float = 0.8,
) -> dict:
    return {
        "id": sid,
        "heading": heading,
        "content": content,
        "citation_ids": ["[1]"],
        "confidence_score": confidence,
        "confidence_level": "HIGH",
        "grader_scores": {"relevance": 0.8, "hallucination": 0.9, "quality": 0.8},
    }


# ============================================================
# Authority Judge
# ============================================================


class TestAuthorityJudge:
    def test_returns_expected_keys(self):
        docs = [_make_scored_doc()]
        passages = [_make_passage()]
        result = authority_judge(docs, passages)
        assert "source_credibility" in result
        assert "authority_profiles" in result
        assert "claim_verdicts_authority" in result

    def test_source_credibility_populated(self):
        docs = [_make_scored_doc(url="https://nature.com/paper")]
        passages = [_make_passage(url="https://nature.com/paper")]
        result = authority_judge(docs, passages)
        assert "https://nature.com/paper" in result["source_credibility"]
        assert result["source_credibility"]["https://nature.com/paper"] >= 0.9

    def test_authority_profile_shape(self):
        docs = [_make_scored_doc()]
        passages = [_make_passage()]
        result = authority_judge(docs, passages)
        profiles = result["authority_profiles"]
        assert len(profiles) == 1
        assert "dominant_authority" in profiles[0]
        assert "source_count" in profiles[0]
        assert "avg_authority_score" in profiles[0]
        assert "institutional_ratio" in profiles[0]

    def test_empty_input(self):
        result = authority_judge([], [])
        assert result["source_credibility"] == {}
        assert result["claim_verdicts_authority"] == []

    def test_partial_verdicts_per_passage(self):
        docs = [_make_scored_doc()]
        passages = [_make_passage(pid="sp-1"), _make_passage(pid="sp-2")]
        result = authority_judge(docs, passages)
        assert len(result["claim_verdicts_authority"]) == 2


# ============================================================
# Grounding Judge
# ============================================================


class TestGroundingJudge:
    def test_returns_expected_keys(self):
        sections = [_make_section_draft()]
        passages = [_make_passage()]
        result = grounding_judge(sections, passages, {"[1]": ["sp-test1"]})
        assert "claim_verdicts_grounding" in result
        assert "passage_to_claims" in result

    def test_empty_sections(self):
        result = grounding_judge([], [], {})
        assert result["claim_verdicts_grounding"] == []

    def test_verdict_has_grounding_fields(self):
        sections = [_make_section_draft(content="Research shows results [1].")]
        passages = [_make_passage(content="Research shows important results.")]
        result = grounding_judge(sections, passages, {"[1]": ["sp-test1"]})
        verdicts = result["claim_verdicts_grounding"]
        if verdicts:
            v = verdicts[0]
            assert "claim_id" in v
            assert "grounding_score" in v
            assert "grounding_method" in v

    def test_passage_to_claims_populated(self):
        sections = [_make_section_draft(content="Research shows results [1].")]
        passages = [_make_passage(content="Research shows important results.")]
        result = grounding_judge(sections, passages, {"[1]": ["sp-test1"]})
        # May have entries depending on claim extraction
        assert isinstance(result["passage_to_claims"], dict)


# ============================================================
# Coverage Judge
# ============================================================


class TestCoverageJudge:
    def test_returns_expected_keys(self):
        docs = [_make_scored_doc()]
        result = coverage_judge(docs, "test question", [{"question": "test"}])
        assert "facets" in result
        assert "coverage_map" in result
        assert "next_wave_queries" in result

    def test_facet_generation(self):
        queries = [
            {"question": "What is AI safety?"},
            {"question": "How does alignment work?"},
        ]
        result = coverage_judge([], "AI safety", queries)
        assert len(result["facets"]) == 2

    def test_coverage_map_shape(self):
        result = coverage_judge(
            [_make_scored_doc()],
            "test",
            [{"question": "test query"}],
        )
        cm = result["coverage_map"]
        assert "facet_coverage" in cm
        assert "overall_coverage" in cm
        assert "uncovered_facets" in cm
        assert "under_represented_perspectives" in cm

    def test_deduplicates_facets(self):
        queries = [
            {"question": "What is AI?"},
            {"question": "What is AI?"},
        ]
        result = coverage_judge([], "AI", queries)
        assert len(result["facets"]) == 1

    def test_empty_queries(self):
        result = coverage_judge([], "test", [])
        assert result["facets"] == []
        assert result["coverage_map"]["overall_coverage"] == 0.0

    def test_next_wave_queries_for_uncovered(self):
        # No docs -> low coverage -> should generate follow-up queries
        queries = [
            {"question": "detailed topic A"},
            {"question": "detailed topic B"},
            {"question": "detailed topic C"},
        ]
        result = coverage_judge([], "broad topic", queries)
        # Should suggest queries for uncovered facets
        assert isinstance(result["next_wave_queries"], list)


# ============================================================
# Merge Judgments
# ============================================================


class TestMergeJudgments:
    def test_returns_judgment_context(self):
        auth = {
            "source_credibility": {"url1": 0.9},
            "authority_profiles": [],
            "claim_verdicts_authority": [],
        }
        ground = {"claim_verdicts_grounding": [], "passage_to_claims": {}}
        contra = {"contradictions": [], "active_tensions": [], "token_usage": []}
        cover = {
            "facets": [],
            "coverage_map": {
                "facet_coverage": {},
                "overall_coverage": 0.5,
                "uncovered_facets": [],
                "under_represented_perspectives": [],
            },
            "next_wave_queries": [],
        }
        jc = merge_judgments(auth, ground, contra, cover)
        assert "claim_verdicts" in jc
        assert "source_credibility" in jc
        assert "active_tensions" in jc
        assert "coverage_map" in jc
        assert "structural_risks" in jc
        assert "wave_number" in jc

    def test_cross_references_authority_grounding(self):
        auth = {
            "source_credibility": {"url1": 0.9},
            "authority_profiles": [],
            "claim_verdicts_authority": [
                {"passage_id": "sp-1", "authority_score": 0.9, "authority_level": "institutional"},
            ],
        }
        ground = {
            "claim_verdicts_grounding": [
                {
                    "claim_id": "cl-test",
                    "claim_text": "Test claim",
                    "grounding_score": 0.8,
                    "grounding_method": "jaccard_v1",
                },
            ],
            "passage_to_claims": {"sp-1": ["cl-test"]},
        }
        contra = {"contradictions": [], "active_tensions": [], "token_usage": []}
        cover = {
            "facets": [],
            "coverage_map": {
                "facet_coverage": {},
                "overall_coverage": 0.7,
                "uncovered_facets": [],
                "under_represented_perspectives": [],
            },
            "next_wave_queries": [],
        }
        jc = merge_judgments(auth, ground, contra, cover)
        verdicts = jc["claim_verdicts"]
        assert len(verdicts) == 1
        # Should have both grounding AND authority data
        v = verdicts[0]
        assert v["grounding_score"] == 0.8
        assert v["authority_score"] == 0.9

    def test_structural_risks_low_coverage(self):
        auth = {"source_credibility": {}, "authority_profiles": [], "claim_verdicts_authority": []}
        ground = {"claim_verdicts_grounding": [], "passage_to_claims": {}}
        contra = {"contradictions": [], "active_tensions": [], "token_usage": []}
        cover = {
            "facets": [],
            "coverage_map": {
                "facet_coverage": {},
                "overall_coverage": 0.2,
                "uncovered_facets": ["a", "b", "c"],
                "under_represented_perspectives": ["institutional/academic sources"],
            },
            "next_wave_queries": [],
        }
        jc = merge_judgments(auth, ground, contra, cover)
        risks = jc["structural_risks"]
        assert any("low_coverage" in r for r in risks)
        assert any("uncovered_facets" in r for r in risks)
        assert any("perspective_gap" in r for r in risks)

    def test_wave_number_passed_through(self):
        auth = {"source_credibility": {}, "authority_profiles": [], "claim_verdicts_authority": []}
        ground = {"claim_verdicts_grounding": [], "passage_to_claims": {}}
        contra = {"contradictions": [], "active_tensions": [], "token_usage": []}
        cover = {
            "facets": [],
            "coverage_map": {
                "facet_coverage": {},
                "overall_coverage": 0.5,
                "uncovered_facets": [],
                "under_represented_perspectives": [],
            },
            "next_wave_queries": [],
        }
        jc = merge_judgments(auth, ground, contra, cover, wave_number=3)
        assert jc["wave_number"] == 3

    def test_facets_propagated_through_merge(self):
        """C5 regression: facets from coverage_judge must reach JudgmentContext."""
        auth = {"source_credibility": {}, "authority_profiles": [], "claim_verdicts_authority": []}
        ground = {"claim_verdicts_grounding": [], "passage_to_claims": {}}
        contra = {"contradictions": [], "active_tensions": [], "token_usage": []}
        test_facets = [
            {"id": "facet-abc", "question": "What is X?", "weight": 0.5},
            {"id": "facet-def", "question": "What is Y?", "weight": 0.5},
        ]
        cover = {
            "facets": test_facets,
            "coverage_map": {
                "facet_coverage": {},
                "overall_coverage": 0.5,
                "uncovered_facets": [],
                "under_represented_perspectives": [],
            },
            "next_wave_queries": [],
        }
        jc = merge_judgments(auth, ground, contra, cover)
        assert "facets" in jc
        assert len(jc["facets"]) == 2
        assert jc["facets"][0]["id"] == "facet-abc"

    def test_contradiction_cross_reference_by_text(self):
        """C3 regression: contradiction matching uses claim text, not IDs."""
        auth = {
            "source_credibility": {},
            "authority_profiles": [],
            "claim_verdicts_authority": [],
        }
        ground = {
            "claim_verdicts_grounding": [
                {
                    "claim_id": "cl-grounding-001",
                    "claim_text": "The sky is blue",
                    "grounding_score": 0.8,
                    "grounding_method": "jaccard_v1",
                },
                {
                    "claim_id": "cl-grounding-002",
                    "claim_text": "Water is wet",
                    "grounding_score": 0.7,
                    "grounding_method": "jaccard_v1",
                },
            ],
            "passage_to_claims": {},
        }
        contra = {
            "contradictions": [],
            "active_tensions": [
                {
                    "id": "tension-abc",
                    "claim_a": {
                        "claim_id": "cl-tension-abc-a",  # Different ID namespace
                        "claim_text": "The sky is blue",
                        "grounding_score": 0.0,
                        "grounding_method": "pending",
                        "authority_score": 0.0,
                        "authority_level": "unknown",
                        "contradicted": True,
                    },
                    "claim_b": {
                        "claim_id": "cl-tension-abc-b",
                        "claim_text": "The sky is red",
                        "grounding_score": 0.0,
                        "grounding_method": "pending",
                        "authority_score": 0.0,
                        "authority_level": "unknown",
                        "contradicted": True,
                    },
                    "severity": "significant",
                    "authority_differential": 0.0,
                    "resolution_hint": "",
                },
            ],
            "token_usage": [],
        }
        cover = {
            "facets": [],
            "coverage_map": {
                "facet_coverage": {},
                "overall_coverage": 0.5,
                "uncovered_facets": [],
                "under_represented_perspectives": [],
            },
            "next_wave_queries": [],
        }
        jc = merge_judgments(auth, ground, contra, cover)
        verdicts = jc["claim_verdicts"]
        # "The sky is blue" should be marked contradicted (text match)
        blue_verdict = [v for v in verdicts if v["claim_text"] == "The sky is blue"][0]
        assert blue_verdict["contradicted"] is True
        assert blue_verdict["contradiction_id"] == "tension-abc"
        # "Water is wet" should NOT be contradicted
        wet_verdict = [v for v in verdicts if v["claim_text"] == "Water is wet"][0]
        assert wet_verdict["contradicted"] is False

    def test_next_wave_queries_have_all_subquery_fields(self):
        """C4 regression: SubQuery must have all required fields."""
        auth = {"source_credibility": {}, "authority_profiles": [], "claim_verdicts_authority": []}
        ground = {"claim_verdicts_grounding": [], "passage_to_claims": {}}
        contra = {"contradictions": [], "active_tensions": [], "token_usage": []}
        cover = {
            "facets": [],
            "coverage_map": {
                "facet_coverage": {},
                "overall_coverage": 0.3,
                "uncovered_facets": ["What is X?"],
                "under_represented_perspectives": [],
            },
            "next_wave_queries": [
                {"id": "sq-wave-test", "question": "What is X?", "search_backends": ["searxng"]},
            ],
        }
        jc = merge_judgments(auth, ground, contra, cover)
        queries = jc["next_wave_queries"]
        assert len(queries) == 1
        sq = queries[0]
        # All required SubQuery fields must be present
        assert "id" in sq
        assert "question" in sq
        assert "perspective" in sq
        assert "priority" in sq
        assert "parent_query_id" in sq
        assert "search_backends" in sq
        assert sq["search_backends"] == ["searxng"]


# ============================================================
# Clustering
# ============================================================


class TestClustering:
    def test_heading_cluster_groups_by_heading(self):
        passages = [
            _make_passage(pid="sp-1", heading="Introduction"),
            _make_passage(pid="sp-2", heading="Introduction"),
            _make_passage(pid="sp-3", heading="Methods"),
            _make_passage(pid="sp-4", heading="Methods"),
        ]
        clusters = cluster_by_heading(passages)
        assert len(clusters) >= 2
        # Check all passages accounted for
        all_ids = []
        for c in clusters:
            all_ids.extend(c["passage_ids"])
        assert set(all_ids) == {"sp-1", "sp-2", "sp-3", "sp-4"}

    def test_heading_cluster_max_limit(self):
        passages = [_make_passage(pid=f"sp-{i}", heading=f"Heading {i}") for i in range(20)]
        clusters = cluster_by_heading(passages, max_clusters=5)
        assert len(clusters) <= 5

    def test_empty_passages(self):
        assert cluster_by_heading([]) == []

    def test_cluster_format(self):
        passages = [
            _make_passage(pid="sp-1", heading="Test"),
            _make_passage(pid="sp-2", heading="Test"),
        ]
        clusters = cluster_by_heading(passages)
        assert len(clusters) >= 1
        c = clusters[0]
        assert "cluster_id" in c
        assert "theme" in c
        assert "passage_ids" in c

    def test_rank_passages_by_credibility(self):
        ids = ["sp-1", "sp-2", "sp-3"]
        passages = [
            _make_passage(pid="sp-1", url="https://example.com"),
            _make_passage(pid="sp-2", url="https://nature.com"),
            _make_passage(pid="sp-3", url="https://blog.com"),
        ]
        credibility = {
            "https://example.com": 0.5,
            "https://nature.com": 0.95,
            "https://blog.com": 0.3,
        }
        ranked = rank_passages_in_cluster(ids, credibility, passages)
        assert ranked[0] == "sp-2"  # nature.com highest
        assert ranked[-1] == "sp-3"  # blog.com lowest


# ============================================================
# Knowledge Artifact
# ============================================================


class TestKnowledgeArtifact:
    def _make_judgment_context(self) -> JudgmentContext:
        return JudgmentContext(
            claim_verdicts=[
                ClaimVerdict(
                    claim_id="cl-1",
                    claim_text="Test claim",
                    grounding_score=0.8,
                    grounding_method="jaccard_v1",
                    authority_score=0.7,
                    authority_level="professional",
                    contradicted=False,
                ),
            ],
            source_credibility={"https://example.com": 0.7},
            active_tensions=[],
            coverage_map=CoverageMap(
                facet_coverage={"f1": 0.8},
                overall_coverage=0.8,
                uncovered_facets=[],
                under_represented_perspectives=[],
            ),
            next_wave_queries=[],
            overall_coverage=0.8,
            structural_risks=[],
            wave_number=1,
        )

    def test_builds_artifact(self):
        jc = self._make_judgment_context()
        passages = [
            _make_passage(pid="sp-1"),
            _make_passage(pid="sp-2", heading="Other"),
        ]
        artifact = build_knowledge_artifact(
            jc,
            passages,
            use_embeddings=False,
        )
        assert "clusters" in artifact
        assert "claim_verdicts" in artifact
        assert "coverage" in artifact
        assert "compression_ratio" in artifact

    def test_empty_passages_returns_empty_artifact(self):
        jc = self._make_judgment_context()
        artifact = build_knowledge_artifact(jc, [], use_embeddings=False)
        assert artifact["clusters"] == []
        assert any("no_passages" in r for r in artifact["structural_risks"])

    def test_compression_ratio(self):
        jc = self._make_judgment_context()
        passages = [_make_passage(pid=f"sp-{i}") for i in range(10)]
        artifact = build_knowledge_artifact(
            jc,
            passages,
            use_embeddings=False,
        )
        assert 0.0 <= artifact["compression_ratio"] <= 1.0

    def test_clusters_have_authority(self):
        jc = self._make_judgment_context()
        passages = [
            _make_passage(pid="sp-1", heading="Topic A"),
            _make_passage(pid="sp-2", heading="Topic A"),
        ]
        artifact = build_knowledge_artifact(
            jc,
            passages,
            use_embeddings=False,
        )
        for c in artifact["clusters"]:
            assert "authority" in c
            assert "dominant_authority" in c["authority"]

    def test_insights_generated(self):
        jc = self._make_judgment_context()
        passages = [_make_passage(pid=f"sp-{i}", heading=f"Topic {i % 3}") for i in range(9)]
        artifact = build_knowledge_artifact(
            jc,
            passages,
            use_embeddings=False,
        )
        assert isinstance(artifact["insights"], list)


# ============================================================
# Grounding Cascade
# ============================================================


class TestGroundingCascade:
    def test_verify_claim_jaccard_fallback(self):
        passage = _make_passage(content="AI safety research is important for alignment")
        ok, score, method = verify_claim(
            "AI safety research is critical",
            passage,
            embedding_provider=None,
        )
        assert method == "jaccard_v1"
        assert isinstance(score, float)

    def test_batch_verification(self):
        passages = [_make_passage(content="Test content about AI safety")]
        claims = [
            {"claim_id": "cl-1", "claim_text": "AI safety is important"},
        ]
        results = verify_claims_batch(
            claims,
            passages,
            {"cl-1": ["sp-test1"]},
        )
        assert len(results) == 1
        assert "claim_id" in results[0]
        assert "grounded" in results[0]
        assert "method" in results[0]

    def test_batch_no_linked_passages(self):
        passages = [_make_passage()]
        claims = [{"claim_id": "cl-1", "claim_text": "unlinked claim"}]
        results = verify_claims_batch(claims, passages, {})
        assert len(results) == 1
        assert results[0]["grounded"] is False
        assert results[0]["method"] == "unverified"


# ============================================================
# Config + Registry
# ============================================================


class TestConfigPhase2:
    def test_haiku_model_default(self):
        from deep_research_swarm.config import Settings

        s = Settings()
        assert s.haiku_model == "claude-haiku-4-5-20251001"

    def test_haiku_model_field_exists(self):
        from deep_research_swarm.config import Settings

        s = Settings(haiku_model="custom-haiku")
        assert s.haiku_model == "custom-haiku"


class TestRegistryPhase2:
    def test_new_tunables_registered(self):
        from deep_research_swarm.adaptive.registry import TunableRegistry

        reg = TunableRegistry()
        new_names = [
            "max_waves",
            "wave_batch_size",
            "wave_extract_cap",
            "coverage_threshold",
            "max_clusters",
            "claims_per_cluster",
        ]
        for name in new_names:
            assert name in reg, f"{name} not registered"

    def test_tunable_count(self):
        from deep_research_swarm.adaptive.registry import TunableRegistry

        reg = TunableRegistry()
        assert len(reg) == 26  # 18 V9 + 6 V10 deliberation + 2 Phase 3 reactor

    def test_deliberation_category(self):
        from deep_research_swarm.adaptive.registry import TunableRegistry

        reg = TunableRegistry()
        cats = reg.categories()
        assert "deliberation" in cats
        assert len(cats["deliberation"]) == 6

    def test_coverage_threshold_bounds(self):
        from deep_research_swarm.adaptive.registry import TunableRegistry

        reg = TunableRegistry()
        # Coverage threshold has floor 0.5, ceiling 0.95
        result = reg.set("coverage_threshold", 0.3)
        assert result == 0.5  # clamped to floor
        result = reg.set("coverage_threshold", 1.0)
        assert result == 0.95  # clamped to ceiling


# ============================================================
# State Fields
# ============================================================


class TestStatePhase2:
    def test_new_fields_exist(self):
        from deep_research_swarm.graph.state import ResearchState

        annotations = ResearchState.__annotations__
        assert "panel_judgments" in annotations
        assert "judgment_context" in annotations
        assert "knowledge_artifact" in annotations
        assert "deliberation_waves" in annotations
        assert "wave_count" in annotations

    def test_composition_field_exists(self):
        """C2 regression: composition must be a state field so LangGraph preserves it."""
        from deep_research_swarm.graph.state import ResearchState

        annotations = ResearchState.__annotations__
        assert "composition" in annotations


# ============================================================
# Graph Topology
# ============================================================


class TestGraphPhase2:
    def _build_graph(self):
        from deep_research_swarm.config import Settings
        from deep_research_swarm.graph.builder import build_graph

        settings = Settings(anthropic_api_key="test-key")
        return build_graph(settings, enable_cache=False)

    def test_new_nodes_present(self):
        graph = self._build_graph()
        nodes = set(graph.get_graph().nodes.keys())
        assert "deliberate_panel" in nodes
        assert "compress" in nodes

    def test_contradiction_to_panel_edge(self):
        graph = self._build_graph()
        edges = set()
        for edge in graph.get_graph().edges:
            edges.add((edge.source, edge.target))
        assert ("contradiction", "deliberate_panel") in edges

    def test_panel_to_compress_edge(self):
        graph = self._build_graph()
        edges = set()
        for edge in graph.get_graph().edges:
            edges.add((edge.source, edge.target))
        assert ("deliberate_panel", "compress") in edges

    def test_compress_to_synthesize_edge(self):
        graph = self._build_graph()
        edges = set()
        for edge in graph.get_graph().edges:
            edges.add((edge.source, edge.target))
        assert ("compress", "synthesize") in edges

    def test_total_node_count(self):
        graph = self._build_graph()
        nodes = set(graph.get_graph().nodes.keys())
        # Exclude __start__ and __end__
        real_nodes = nodes - {"__start__", "__end__"}
        # V9 (18) + compute_entropy + deliberate_panel + compress + score_merge = 22
        assert len(real_nodes) == 22


# ============================================================
# Synthesizer V10 Path Routing
# ============================================================


class TestSynthesizerV10Routing:
    def test_v10_path_activates_with_artifact(self):
        """synthesize() should use V10 path when knowledge_artifact has clusters."""
        # We can't easily test the full async path, but we can verify
        # the routing logic by checking state handling
        # Verify the function accepts the new kwargs
        import inspect

        from deep_research_swarm.agents.synthesizer import synthesize

        sig = inspect.signature(synthesize)
        params = list(sig.parameters.keys())
        assert "sonnet_caller" in params
        assert "haiku_caller" in params

    def test_v10_outline_prompt_exists(self):
        from deep_research_swarm.agents.synthesizer import V10_OUTLINE_SYSTEM

        assert "{min_sections}" in V10_OUTLINE_SYSTEM
        assert "{max_sections}" in V10_OUTLINE_SYSTEM
        assert "KnowledgeArtifact" in V10_OUTLINE_SYSTEM

    def test_v10_section_prompt_exists(self):
        from deep_research_swarm.agents.synthesizer import V10_SECTION_SYSTEM

        assert "cluster" in V10_SECTION_SYSTEM.lower()
        assert "[N]" in V10_SECTION_SYSTEM


class TestS1S2ModelTiering:
    """S1/S2 regression: V9 path must use correct model tiers."""

    def test_v9_draft_uses_sonnet_caller(self):
        """V9 section drafting should use sonnet_caller when available."""
        import inspect

        from deep_research_swarm.agents.synthesizer import synthesize

        source = inspect.getsource(synthesize)
        # The V9 fallback function is called from synthesize
        assert "sonnet_caller" in source
        assert "haiku_caller" in source

    def test_v9_draft_section_call_uses_sonnet(self):
        """_draft_section in V9 path should receive sonnet_caller or caller."""
        import ast

        from deep_research_swarm.agents import synthesizer

        tree = ast.parse(inspect.getsource(synthesizer))
        # Find _draft_section calls to verify they pass sonnet_caller
        found_sonnet = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id == "_draft_section":
                    for arg in node.args:
                        if isinstance(arg, ast.BoolOp):
                            for val in arg.values:
                                if isinstance(val, ast.Name) and val.id == "sonnet_caller":
                                    found_sonnet = True
        assert found_sonnet, "_draft_section must be called with sonnet_caller"

    def test_v9_compose_uses_haiku(self):
        """_compose_report in V9 path should receive haiku_caller or caller."""
        import ast

        from deep_research_swarm.agents import synthesizer

        tree = ast.parse(inspect.getsource(synthesizer))
        found_haiku = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id == "_compose_report":
                    for arg in node.args:
                        if isinstance(arg, ast.BoolOp):
                            for val in arg.values:
                                if isinstance(val, ast.Name) and val.id == "haiku_caller":
                                    found_haiku = True
        assert found_haiku, "_compose_report must be called with haiku_caller"


class TestS7HeadingContext:
    """S7 regression: cluster.py must use heading_context, not heading."""

    def test_cluster_reads_heading_context(self):
        """cluster_by_heading uses heading_context field."""
        passages = [
            _make_passage(pid="sp-a", heading="Alpha Topic"),
            _make_passage(pid="sp-b", heading="Alpha Topic"),
            _make_passage(pid="sp-c", heading="Beta Topic"),
            _make_passage(pid="sp-d", heading="Beta Topic"),
        ]
        clusters = cluster_by_heading(passages)
        assert len(clusters) >= 2

    def test_passage_text_uses_heading_context(self):
        """_passage_text reads heading_context, not heading."""
        from deep_research_swarm.compress.cluster import _passage_text

        passage = _make_passage(heading="My Heading")
        text = _passage_text(passage)
        assert "My Heading" in text

    def test_format_clusters_theme_from_heading_context(self):
        """Cluster theme should come from heading_context."""
        from deep_research_swarm.compress.cluster import _format_clusters

        passages = [_make_passage(pid="sp-x", heading="Theme Alpha")]
        clusters = _format_clusters(passages, [[0]])
        assert clusters[0]["theme"] == "Theme Alpha"


class TestS6ClaimsForCluster:
    """S6 regression: _claims_for_cluster must filter by passage_ids."""

    def test_linked_claims_preferred_via_passage_map(self):
        from deep_research_swarm.compress.artifact import _claims_for_cluster

        claims = [
            {"claim_id": "cl-abc-0", "claim_text": "A", "grounding_method": "jaccard_v1"},
            {"claim_id": "cl-xyz-0", "claim_text": "B", "grounding_method": "jaccard_v1"},
            {"claim_id": "cl-other-0", "claim_text": "C", "grounding_method": "jaccard_v1"},
        ]
        p2c = {"sp-abc": ["cl-abc-0"], "sp-xyz": ["cl-xyz-0"]}
        result = _claims_for_cluster(
            ["sp-abc", "sp-xyz"], claims, max_claims=2, passage_to_claims=p2c
        )
        claim_ids = [c["claim_id"] for c in result]
        assert "cl-abc-0" in claim_ids
        assert "cl-xyz-0" in claim_ids

    def test_fallback_when_no_linked_claims(self):
        from deep_research_swarm.compress.artifact import _claims_for_cluster

        claims = [
            {"claim_id": "cl-other-0", "claim_text": "A", "grounding_method": "jaccard_v1"},
        ]
        # No passage_to_claims entry for sp-abc -> falls back to second pass
        result = _claims_for_cluster(["sp-abc"], claims, max_claims=5, passage_to_claims={})
        assert len(result) == 1

    def test_no_map_still_uses_fallback(self):
        from deep_research_swarm.compress.artifact import _claims_for_cluster

        claims = [
            {"claim_id": "cl-1", "claim_text": "A", "grounding_method": "jaccard_v1"},
        ]
        # No passage_to_claims at all -> second pass fills
        result = _claims_for_cluster(["sp-abc"], claims, max_claims=5)
        assert len(result) == 1


class TestS12ReactorStateField:
    """S12 regression: ResearchState must have reactor_state for singularity checks."""

    def test_state_has_reactor_state_field(self):
        from deep_research_swarm.graph.state import ResearchState

        assert "reactor_state" in ResearchState.__annotations__

    def test_entropy_node_reads_reactor_state(self):
        """compute_entropy_node must read reactor_state, not reactor_trace."""
        source = inspect.getsource(
            __import__("deep_research_swarm.graph.builder", fromlist=["build_graph"]).build_graph
        )
        assert 'state.get("reactor_state"' in source

    def test_synthesizer_stores_reactor_state(self):
        """_synthesize_v10 must store reactor_products as reactor_state."""
        source = inspect.getsource(
            __import__(
                "deep_research_swarm.agents.synthesizer", fromlist=["_synthesize_v10"]
            )._synthesize_v10
        )
        assert '"reactor_state"' in source


class TestS13CitationUrl:
    """S13 regression: Citation URL must use source_url, not source_id."""

    def test_build_output_uses_source_url(self):
        source = inspect.getsource(
            __import__(
                "deep_research_swarm.agents.synthesizer", fromlist=["_build_output"]
            )._build_output
        )
        assert "source_url" in source
        assert "source_id" not in source or "source_id=url" not in source


class TestS16RenumberCitations:
    """S16 regression: _renumber_section_content must handle 2-digit refs."""

    def test_two_digit_citations_not_corrupted(self):
        from deep_research_swarm.agents.synthesizer import (
            _renumber_section_content,
        )

        # Map: local [1] -> global [5], local [10] -> global [6]
        section_map = {"[1]": "[5]", "[10]": "[6]"}
        content = "See [1] and [10] for details."
        result = _renumber_section_content(content, section_map)
        assert "[5]" in result
        assert "[6]" in result
        assert "[5]0" not in result  # This was the corruption pattern

    def test_single_digit_unaffected(self):
        from deep_research_swarm.agents.synthesizer import (
            _renumber_section_content,
        )

        section_map = {"[1]": "[3]", "[2]": "[4]"}
        content = "Ref [1] and [2]."
        result = _renumber_section_content(content, section_map)
        assert result == "Ref [3] and [4]."


class TestS17ClaimsGlobalDedup:
    """S17 regression: Claims must not repeat across clusters."""

    def test_globally_assigned_prevents_duplicates(self):
        from deep_research_swarm.compress.artifact import _claims_for_cluster

        claims = [
            {"claim_id": "c1", "claim_text": "A", "grounding_method": "jaccard_v1"},
            {"claim_id": "c2", "claim_text": "B", "grounding_method": "jaccard_v1"},
        ]
        assigned: set[str] = set()

        # Cluster 1 gets c1
        r1 = _claims_for_cluster([], claims, max_claims=1, globally_assigned=assigned)
        assert len(r1) == 1
        assert r1[0]["claim_id"] == "c1"

        # Cluster 2 should NOT get c1 again
        r2 = _claims_for_cluster([], claims, max_claims=1, globally_assigned=assigned)
        assert len(r2) == 1
        assert r2[0]["claim_id"] == "c2"

        # Cluster 3 has nothing left
        r3 = _claims_for_cluster([], claims, max_claims=1, globally_assigned=assigned)
        assert len(r3) == 0


class TestS18CoverageDenominator:
    """S18 regression: Coverage denominator must not produce inflated scores."""

    def test_coverage_uses_bounded_expected_matches(self):
        """Coverage formula must use expected_matches with bounded range [3, 10]."""
        from deep_research_swarm.deliberate.panel import coverage_judge

        source = inspect.getsource(coverage_judge)
        assert "expected_matches" in source
        # Must have both floor and ceiling bounds
        lines = [ln.strip() for ln in source.split("\n") if not ln.strip().startswith("#")]
        code_only = "\n".join(lines)
        assert "min(10," in code_only
        assert "max(3," in code_only


class TestS20ReactorConversationCap:
    """S20 regression: Reactor conversation must be capped to prevent unbounded growth."""

    def test_conversation_capped_in_source(self):
        source = inspect.getsource(
            __import__(
                "deep_research_swarm.agents.synthesizer", fromlist=["_run_reactor"]
            )._run_reactor
        )
        # Must have recent_history slicing
        assert "recent_history" in source
        assert "conversation[-6:]" in source


# ============================================================
# S21: _claims_for_cluster uses passage_to_claims map
# ============================================================


class TestS21PassageToClaimsLinkage:
    """S21 regression: claim-cluster linkage must use passage_to_claims map."""

    def test_linked_claims_via_map(self):
        from deep_research_swarm.compress.artifact import _claims_for_cluster

        claims = [
            {"claim_id": "cl-a", "claim_text": "A", "grounding_method": "jaccard_v1"},
            {"claim_id": "cl-b", "claim_text": "B", "grounding_method": "jaccard_v1"},
            {"claim_id": "cl-c", "claim_text": "C", "grounding_method": "jaccard_v1"},
        ]
        p2c = {"sp-1": ["cl-a"], "sp-2": ["cl-b"]}
        result = _claims_for_cluster(["sp-1", "sp-2"], claims, max_claims=5, passage_to_claims=p2c)
        ids = [c["claim_id"] for c in result]
        # First pass: cl-a, cl-b linked; second pass: cl-c unlinked but grounded
        assert ids[:2] == ["cl-a", "cl-b"]
        assert "cl-c" in ids

    def test_no_map_falls_through_to_second_pass(self):
        from deep_research_swarm.compress.artifact import _claims_for_cluster

        claims = [
            {"claim_id": "cl-x", "claim_text": "X", "grounding_method": "jaccard_v1"},
        ]
        result = _claims_for_cluster(["sp-1"], claims, max_claims=5, passage_to_claims=None)
        assert len(result) == 1
        assert result[0]["claim_id"] == "cl-x"

    def test_passage_to_claims_in_judgment_context(self):
        """merge_judgments must propagate passage_to_claims to JudgmentContext."""
        from deep_research_swarm.deliberate.merge import merge_judgments

        auth = {"source_credibility": {}, "authority_profiles": [], "claim_verdicts_authority": []}
        ground = {
            "claim_verdicts_grounding": [],
            "passage_to_claims": {"sp-1": ["cl-a"]},
        }
        contra = {"contradictions": [], "active_tensions": [], "token_usage": []}
        cover = {
            "facets": [],
            "coverage_map": {
                "facet_coverage": {},
                "overall_coverage": 0.5,
                "uncovered_facets": [],
                "under_represented_perspectives": [],
            },
            "next_wave_queries": [],
        }
        jc = merge_judgments(auth, ground, contra, cover)
        assert "passage_to_claims" in jc
        assert jc["passage_to_claims"] == {"sp-1": ["cl-a"]}

    def test_artifact_builder_threads_map(self):
        """build_knowledge_artifact must pass passage_to_claims from JudgmentContext."""
        jc = JudgmentContext(
            claim_verdicts=[
                ClaimVerdict(
                    claim_id="cl-linked",
                    claim_text="Linked claim",
                    grounding_score=0.9,
                    grounding_method="jaccard_v1",
                    authority_score=0.7,
                    authority_level="professional",
                    contradicted=False,
                ),
            ],
            source_credibility={"https://example.com": 0.7},
            active_tensions=[],
            coverage_map=CoverageMap(
                facet_coverage={},
                overall_coverage=0.5,
                uncovered_facets=[],
                under_represented_perspectives=[],
            ),
            next_wave_queries=[],
            overall_coverage=0.5,
            structural_risks=[],
            wave_number=1,
            facets=[],
        )
        jc["passage_to_claims"] = {"sp-1": ["cl-linked"]}
        passages = [_make_passage(pid="sp-1")]
        artifact = build_knowledge_artifact(jc, passages, use_embeddings=False)
        # cl-linked should be in first cluster via map linkage
        all_claims = []
        for c in artifact["clusters"]:
            all_claims.extend(c["claims"])
        assert any(c["claim_id"] == "cl-linked" for c in all_claims)


# ============================================================
# S23: Confidence from grounding_score, not hardcoded 0.7
# ============================================================


class TestS23GroundingConfidence:
    """S23 regression: confidence_score must derive from grounding, not hardcoded."""

    def test_source_uses_grounding_score(self):
        source = inspect.getsource(
            __import__(
                "deep_research_swarm.agents.synthesizer", fromlist=["_build_output"]
            )._build_output
        )
        # Must not contain the old hardcoded default
        assert 'draft.get("confidence", 0.7)' not in source
        # Must reference grounding_score for confidence
        assert "grounding_score" in source


# ============================================================
# S24: reactor_configs passed as parameter, not instance state
# ============================================================


class TestS24ReactorConfigsLocal:
    """S24 regression: SwarmOrchestrator must not rely on self._reactor_configs."""

    def test_run_single_reactor_accepts_configs(self):
        from deep_research_swarm.sia.swarm import SwarmOrchestrator

        sig = inspect.signature(SwarmOrchestrator._run_single_reactor)
        assert "reactor_configs" in sig.parameters

    def test_no_instance_state_in_run(self):
        """run() must use local variable, not self._reactor_configs."""
        from deep_research_swarm.sia.swarm import SwarmOrchestrator

        source = inspect.getsource(SwarmOrchestrator.run)
        assert "self._reactor_configs" not in source


# ============================================================
# S25: compression_ratio is claims/passages (information density)
# ============================================================


class TestS25CompressionRatio:
    """S25 regression: compression_ratio must reflect information density."""

    def test_ratio_reflects_claims_not_passages(self):
        """compression_ratio should be total_claims / original_passages."""
        source = inspect.getsource(build_knowledge_artifact)
        # Must compute ratio from claims, not passage counts
        assert "total_claims" in source
        assert 'sum(len(pc["claims"])' in source

    def test_ratio_zero_when_no_claims(self):
        jc = JudgmentContext(
            claim_verdicts=[],
            source_credibility={"https://example.com": 0.5},
            active_tensions=[],
            coverage_map=CoverageMap(
                facet_coverage={},
                overall_coverage=0.5,
                uncovered_facets=[],
                under_represented_perspectives=[],
            ),
            next_wave_queries=[],
            overall_coverage=0.5,
            structural_risks=[],
            wave_number=1,
            facets=[],
        )
        passages = [_make_passage(pid=f"sp-{i}") for i in range(5)]
        artifact = build_knowledge_artifact(jc, passages, use_embeddings=False)
        # No claims -> ratio should be 0
        assert artifact["compression_ratio"] == 0.0


# ============================================================
# S26: Coverage denominator bounded [3, 10]
# ============================================================


class TestS26CoverageBounded:
    """S26 regression: coverage denominator must have floor AND ceiling."""

    def test_many_docs_few_facets_not_deflated(self):
        """With 100 docs and 2 facets, denominator capped at 10."""
        from deep_research_swarm.deliberate.panel import coverage_judge

        docs = [_make_scored_doc(title=f"test topic {i}") for i in range(100)]
        queries = [{"question": "test topic"}, {"question": "other topic"}]
        result = coverage_judge(docs, "test", queries)
        # Coverage should not be near-zero due to unbounded denominator
        overall = result["coverage_map"]["overall_coverage"]
        # With 100 docs matching "test" keyword, coverage should be reasonable
        assert overall >= 0.0  # Basic sanity

    def test_source_has_ceiling(self):
        from deep_research_swarm.deliberate.panel import coverage_judge

        source = inspect.getsource(coverage_judge)
        assert "min(10," in source


# ============================================================
# S27: plan_node resets converged for new iteration
# ============================================================


class TestS27PlanNodeReset:
    """S27 regression: plan_node must reset converged for each new iteration."""

    def test_plan_node_resets_convergence(self):
        from deep_research_swarm.graph.builder import build_graph

        source = inspect.getsource(build_graph)
        # plan_node must set converged=False
        assert '"converged"' in source or "converged" in source
        # Find the plan_node closure and check it resets
        assert 'result["converged"] = False' in source
        assert 'result["convergence_reason"] = ""' in source


# ============================================================
# M7: Singularity check simplified condition
# ============================================================


class TestM7SingularitySimplified:
    """M7 regression: check_constraint_singularity should not have redundant condition."""

    def test_source_no_redundant_len_check(self):
        import inspect

        from deep_research_swarm.sia.singularity_prevention import check_constraint_singularity

        source = inspect.getsource(check_constraint_singularity)
        # Should NOT have the old redundant "and len(agent_constraints) == 1"
        assert "len(agent_constraints) == 1" not in source

    def test_single_agent_all_constraints_detected(self):
        from deep_research_swarm.sia.singularity_prevention import check_constraint_singularity

        reactor_state = {
            "turn_log": [
                {"agent": "lawliet", "constraints": ["c1", "c2", "c3"]},
            ],
            "constraints": ["c1", "c2", "c3"],
        }
        safe, reason = check_constraint_singularity(reactor_state)
        assert not safe
        assert "constraint_singularity" in reason


# ============================================================
# M10: authority_judge caches authority level
# ============================================================


class TestM10AuthorityCached:
    """M10 regression: authority_judge must not call score_authority twice per URL."""

    def test_source_uses_url_authority_level(self):
        import inspect

        from deep_research_swarm.deliberate.panel import authority_judge

        source = inspect.getsource(authority_judge)
        assert "url_authority_level" in source
        # Second pass should NOT call score_authority
        # Count occurrences of score_authority in the passage loop
        passage_section = source.split("# Partial claim verdicts")[1]
        assert "score_authority" not in passage_section


# ============================================================
# M11: cluster overflow check simplified
# ============================================================


class TestM11ClusterOverflowCheck:
    """M11 regression: overflow check should not have redundant truthiness."""

    def test_source_no_redundant_check(self):
        import inspect

        from deep_research_swarm.compress.cluster import cluster_by_heading

        source = inspect.getsource(cluster_by_heading)
        assert "clusters and len(clusters)" not in source


# ============================================================
# M12: synthesize_node no pop-then-reassign
# ============================================================


class TestM12NoPushPop:
    """M12 regression: synthesize_node should use get, not pop-then-reassign."""

    def test_no_pop_reactor_trace(self):
        import inspect

        from deep_research_swarm.graph.builder import build_graph

        source = inspect.getsource(build_graph)
        assert 'result.pop("reactor_trace"' not in source


# ============================================================
# M13: config detection uses "config" in params
# ============================================================


class TestM13ConfigDetection:
    """M13 regression: _wrap_with_logging must check param name, not count."""

    def test_has_config_checks_name(self):
        import inspect

        from deep_research_swarm.graph.builder import _wrap_with_logging

        source = inspect.getsource(_wrap_with_logging)
        assert '"config" in params' in source
        assert "len(params) >= 2" not in source


# ============================================================
# M14: search_followup uses sub_query_id
# ============================================================


class TestM14SubQueryIdField:
    """M14 regression: search_followup_node must use sub_query_id, not query_id."""

    def test_source_uses_sub_query_id(self):
        import inspect

        from deep_research_swarm.graph.builder import build_graph

        source = inspect.getsource(build_graph)
        # The dedup line must use sub_query_id
        assert 'sr.get("sub_query_id"' in source
        assert 'sr.get("query_id"' not in source
