"""Tests for contracts.py — TypedDict construction, enum values."""

from deep_research_swarm.contracts import (
    Citation,
    Confidence,
    ExtractedContent,
    GraderScores,
    ResearchGap,
    ScoredDocument,
    SearchBackend,
    SearchResult,
    SectionDraft,
    SourceAuthority,
    SubQuery,
    TokenUsage,
)


class TestEnums:
    def test_confidence_values(self):
        assert Confidence.HIGH == "HIGH"
        assert Confidence.MEDIUM == "MEDIUM"
        assert Confidence.LOW == "LOW"

    def test_confidence_is_str(self):
        assert isinstance(Confidence.HIGH, str)

    def test_source_authority_values(self):
        assert SourceAuthority.INSTITUTIONAL == "institutional"
        assert SourceAuthority.PROFESSIONAL == "professional"
        assert SourceAuthority.COMMUNITY == "community"
        assert SourceAuthority.PROMOTIONAL == "promotional"
        assert SourceAuthority.UNKNOWN == "unknown"


class TestTypedDicts:
    def test_sub_query_construction(self, sample_sub_query):
        assert sample_sub_query["id"] == "sq-001"
        assert sample_sub_query["priority"] == 1
        assert sample_sub_query["search_backends"] == ["searxng"]

    def test_search_result_construction(self, sample_search_results):
        sr = sample_search_results[0]
        assert sr["backend"] == "searxng"
        assert sr["rank"] == 1
        assert isinstance(sr["authority"], SourceAuthority)

    def test_grader_scores_construction(self):
        gs = GraderScores(relevance=0.9, hallucination=0.85, quality=0.8)
        assert gs["relevance"] == 0.9

    def test_section_draft_construction(self, sample_section_drafts):
        sec = sample_section_drafts[0]
        assert sec["confidence_level"] == Confidence.HIGH
        assert sec["confidence_score"] == 0.85

    def test_token_usage_construction(self, sample_token_usage):
        assert sample_token_usage["agent"] == "planner"
        assert sample_token_usage["cost_usd"] == 0.06


class TestProtocol:
    def test_search_backend_is_runtime_checkable(self):
        """Verify SearchBackend can be used with isinstance."""

        class FakeBackend:
            name = "fake"

            async def search(self, query, *, num_results=10, category=None):
                return []

            async def health_check(self):
                return True

        assert isinstance(FakeBackend(), SearchBackend)

    def test_non_conformant_is_not_backend(self):
        class NotABackend:
            pass

        assert not isinstance(NotABackend(), SearchBackend)
