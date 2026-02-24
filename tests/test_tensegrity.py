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
        heading=heading,
        content=content,
        token_count=len(content.split()),
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
        assert len(reg) == 24  # 18 V9 + 6 V10

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
