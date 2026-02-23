"""Tests for contracts.py — TypedDict construction, enum values."""

from deep_research_swarm.contracts import (
    Confidence,
    GraderScores,
    SearchBackend,
    SearchResult,
    SourceAuthority,
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


class TestV7TypedDicts:
    """Tests for V7 foundation types (PR-01)."""

    def test_provenance_record_construction(self, sample_provenance_record):
        pr = sample_provenance_record
        assert pr["source_kind"] == "web"
        assert pr["content_hash"] == "a1b2c3d4e5f6"
        assert pr["capture_timestamp"] == ""  # Non-archive

    def test_scholarly_metadata_construction(self, sample_scholarly_metadata):
        sm = sample_scholarly_metadata
        assert sm["doi"] == "10.1038/s41586-023-06768-0"
        assert sm["citation_count"] == 150
        assert sm["is_open_access"] is True
        assert len(sm["authors"]) == 2

    def test_archive_capture_construction(self, sample_archive_capture):
        ac = sample_archive_capture
        assert ac["status_code"] == 200
        assert ac["capture_timestamp"] == "20240101000000"
        assert "web.archive.org" in ac["archive_url"]

    def test_source_passage_construction(self, sample_source_passage):
        sp = sample_source_passage
        assert sp["id"].startswith("sp-")
        assert sp["position"] == 0
        assert sp["claim_ids"] == []  # OE1: empty in V7
        assert sp["heading_context"] == "Introduction"

    def test_section_outline_construction(self, sample_section_outline):
        so = sample_section_outline
        assert so["narrative_role"] == "introduction"
        assert len(so["key_claims"]) == 2
        assert len(so["source_ids"]) == 1

    def test_indexed_content_v8_stub(self, sample_indexed_content):
        ic = sample_indexed_content
        assert ic["embedding"] == []  # V8 stub: empty in V7
        assert ic["index_backend"] == ""  # V8 stub: empty in V7

    def test_search_result_backward_compat(self):
        """SearchResult without V7 fields must still work (critical)."""
        sr = SearchResult(
            id="sr-test",
            sub_query_id="sq-001",
            url="https://example.com",
            title="Test",
            snippet="A test result",
            backend="searxng",
            rank=1,
            score=0.5,
            authority=SourceAuthority.UNKNOWN,
            timestamp="2026-02-23T00:00:00Z",
        )
        assert sr["id"] == "sr-test"
        # NotRequired fields should not be present
        assert "provenance" not in sr
        assert "scholarly_metadata" not in sr

    def test_search_result_with_v7_fields(
        self, sample_provenance_record, sample_scholarly_metadata
    ):
        """SearchResult with V7 optional fields."""
        sr = SearchResult(
            id="sr-v7",
            sub_query_id="sq-001",
            url="https://example.com",
            title="Test",
            snippet="A test result",
            backend="openalex",
            rank=1,
            score=0.8,
            authority=SourceAuthority.INSTITUTIONAL,
            timestamp="2026-02-23T00:00:00Z",
            provenance=sample_provenance_record,
            scholarly_metadata=sample_scholarly_metadata,
        )
        assert sr["provenance"]["content_hash"] == "a1b2c3d4e5f6"
        assert sr["scholarly_metadata"]["doi"] == "10.1038/s41586-023-06768-0"

    def test_section_draft_backward_compat(self):
        """SectionDraft without V7 grounding fields must still work."""
        from deep_research_swarm.contracts import Confidence, GraderScores, SectionDraft

        sd = SectionDraft(
            id="sec-test",
            heading="Test",
            content="Content [1]",
            citation_ids=["[1]"],
            confidence_score=0.85,
            confidence_level=Confidence.HIGH,
            grader_scores=GraderScores(relevance=0.9, hallucination=0.9, quality=0.9),
        )
        assert sd["id"] == "sec-test"
        assert "grounding_score" not in sd
        assert "claim_details" not in sd

    def test_section_draft_with_grounding(self):
        """SectionDraft with V7 grounding fields (OE3, D4)."""
        from deep_research_swarm.contracts import Confidence, GraderScores, SectionDraft

        sd = SectionDraft(
            id="sec-grounded",
            heading="Test",
            content="Content [1]",
            citation_ids=["[1]"],
            confidence_score=0.85,
            confidence_level=Confidence.HIGH,
            grader_scores=GraderScores(relevance=0.9, hallucination=0.9, quality=0.9),
            grounding_score=0.72,
            claim_details=[{"claim": "test", "grounded": True, "similarity": 0.72}],
        )
        assert sd["grounding_score"] == 0.72
        assert len(sd["claim_details"]) == 1
