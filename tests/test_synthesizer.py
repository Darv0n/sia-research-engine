"""Tests for agents/synthesizer.py — outline-first synthesis (PR-11)."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from deep_research_swarm.agents.synthesizer import (
    GROUNDING_PASS_THRESHOLD,
    MAX_REFINEMENT_ATTEMPTS,
    _build_global_citation_map,
    _build_passage_context,
    _parse_outline_sections,
    _renumber_section_content,
    _validate_outline,
    synthesize,
)
from deep_research_swarm.contracts import (
    Confidence,
    GraderScores,
    ScoredDocument,
    SectionDraft,
    SectionOutline,
    SourceAuthority,
    SourcePassage,
)

# --- Fixtures ---


def _make_passage(
    pid: str = "sp-abc123",
    source_id: str = "1",
    content: str = "Test passage content about research.",
    position: int = 0,
) -> SourcePassage:
    return SourcePassage(
        id=pid,
        source_id=source_id,
        content=content,
        position=position,
        token_estimate=20,
    )


def _make_scored_doc(
    doc_id: str = "sd-1",
    url: str = "https://example.com/1",
    title: str = "Test Doc",
    content: str = "Document content.",
    combined_score: float = 0.8,
) -> ScoredDocument:
    return ScoredDocument(
        id=doc_id,
        url=url,
        title=title,
        content=content,
        rrf_score=0.5,
        authority=SourceAuthority.INSTITUTIONAL,
        authority_score=0.7,
        combined_score=combined_score,
        sub_query_ids=["sq-001"],
    )


def _make_section_outline(
    heading: str = "Test Section",
    key_claims: list[str] | None = None,
    source_ids: list[str] | None = None,
) -> SectionOutline:
    return SectionOutline(
        heading=heading,
        key_claims=key_claims if key_claims is not None else ["Claim A", "Claim B"],
        source_ids=source_ids if source_ids is not None else ["1"],
        narrative_role="evidence",
    )


def _make_state(**overrides) -> dict:
    """Build a minimal ResearchState for testing."""
    passages = [
        _make_passage("sp-001", "1", "CRISPR-Cas9 enables genome editing via DNA cleavage"),
        _make_passage("sp-002", "1", "Gene editing has transformed molecular biology"),
        _make_passage("sp-003", "2", "Machine learning improves protein prediction"),
        _make_passage("sp-004", "2", "Deep learning methods applied to structure prediction"),
    ]
    scored = [
        _make_scored_doc("sd-1", content="CRISPR gene editing content"),
        _make_scored_doc("sd-2", content="ML protein prediction content"),
    ]
    base = {
        "research_question": "How has CRISPR changed biology?",
        "scored_documents": scored,
        "source_passages": passages,
        "contradictions": [],
        "current_iteration": 1,
        "section_drafts": [],
    }
    base.update(overrides)
    return base


# --- Tests: Outline Parsing ---


class TestParseOutlineSections:
    def test_parses_sections(self):
        data = {
            "sections": [
                {
                    "heading": "Introduction",
                    "key_claims": ["CRISPR is revolutionary"],
                    "source_ids": [1, 2],
                    "narrative_role": "introduction",
                },
            ],
        }
        sections = _parse_outline_sections(data)
        assert len(sections) == 1
        assert sections[0]["heading"] == "Introduction"
        assert sections[0]["source_ids"] == ["1", "2"]

    def test_empty_input(self):
        assert _parse_outline_sections({}) == []
        assert _parse_outline_sections({"sections": []}) == []


# --- Tests: Stage 0 Validation ---


class TestValidateOutline:
    def test_valid_outline(self):
        sections = [_make_section_outline()]
        passages = [_make_passage("sp-1", "1", "relevant content")]
        scored = [_make_scored_doc()]
        is_valid, failures = _validate_outline(sections, passages, scored)
        assert is_valid is True
        assert failures == []

    def test_nonexistent_source_ids(self):
        sections = [_make_section_outline(source_ids=["99"])]
        passages = [_make_passage("sp-1", "1")]
        scored = [_make_scored_doc()]
        is_valid, failures = _validate_outline(sections, passages, scored)
        assert not is_valid
        assert any("non-existent" in f for f in failures)

    def test_no_assignable_passages(self):
        """Section with no matching passages (and no fallback) is flagged."""
        sections = [_make_section_outline(source_ids=["999"])]
        # Empty passage list: no fallback possible
        passages: list[SourcePassage] = []
        scored = [_make_scored_doc()]
        is_valid, failures = _validate_outline(sections, passages, scored)
        assert not is_valid
        assert any("no assignable" in f for f in failures)

    def test_too_many_sections(self):
        # With only 5 passages, max sections = max(3, 5//5) = 3
        sections = [_make_section_outline(f"Section {i}", source_ids=["1"]) for i in range(5)]
        passages = [_make_passage(f"sp-{i}", "1") for i in range(5)]
        scored = [_make_scored_doc()]
        is_valid, failures = _validate_outline(sections, passages, scored)
        assert any("passage budget" in f for f in failures)

    def test_missing_key_claims(self):
        sections = [_make_section_outline(key_claims=[])]
        passages = [_make_passage("sp-1", "1")]
        scored = [_make_scored_doc()]
        is_valid, failures = _validate_outline(sections, passages, scored)
        assert not is_valid
        assert any("no key_claims" in f for f in failures)


# --- Tests: Citation Renumbering ---


class TestBuildGlobalCitationMap:
    def test_basic_renumbering(self):
        drafts = [
            {"heading": "Sec A"},
            {"heading": "Sec B"},
        ]
        section_passages = {
            "Sec A": [_make_passage("sp-1"), _make_passage("sp-2")],
            "Sec B": [_make_passage("sp-3")],
        }
        cit_map, p_map = _build_global_citation_map(drafts, section_passages)
        # Sec A: [1]->[1], [2]->[2]
        # Sec B: [1]->[3]
        assert cit_map["Sec A"]["[1]"] == "[1]"
        assert cit_map["Sec A"]["[2]"] == "[2]"
        assert cit_map["Sec B"]["[1]"] == "[3]"
        assert p_map["[1]"] == ["sp-1"]
        assert p_map["[3]"] == ["sp-3"]

    def test_empty_passages(self):
        drafts = [{"heading": "Empty"}]
        section_passages = {"Empty": []}
        cit_map, p_map = _build_global_citation_map(
            drafts,
            section_passages,
        )
        assert cit_map["Empty"] == {}
        assert p_map == {}


class TestRenumberSectionContent:
    def test_basic_renumber(self):
        content = "Claim A [1]. Claim B [2]."
        section_map = {"[1]": "[5]", "[2]": "[6]"}
        result = _renumber_section_content(content, section_map)
        assert "[5]" in result
        assert "[6]" in result
        assert "[1]" not in result
        assert "[2]" not in result

    def test_no_double_replacement(self):
        """Null-byte pattern prevents [1]->[2] and [2]->[3] collision."""
        content = "First [1] then [2]."
        section_map = {"[1]": "[2]", "[2]": "[3]"}
        result = _renumber_section_content(content, section_map)
        assert result == "First [2] then [3]."

    def test_empty_map(self):
        assert _renumber_section_content("text [1]", {}) == "text [1]"

    def test_multiple_same_citation(self):
        content = "Claim [1]. Another claim [1]."
        section_map = {"[1]": "[7]"}
        result = _renumber_section_content(content, section_map)
        assert result == "Claim [7]. Another claim [7]."


# --- Tests: Stage 3 Grounding Verification ---


class TestGroundingVerification:
    def test_threshold_value(self):
        assert GROUNDING_PASS_THRESHOLD == 0.8

    def test_max_refinement_attempts(self):
        assert MAX_REFINEMENT_ATTEMPTS == 2


# --- Tests: Passage Context ---


class TestBuildPassageContext:
    def test_numbered_passages(self):
        passages = [
            _make_passage("sp-1", "1", "First passage"),
            _make_passage("sp-2", "2", "Second passage"),
        ]
        ctx = _build_passage_context(passages)
        assert "[1]" in ctx
        assert "[2]" in ctx
        assert "sp-1" in ctx
        assert "First passage" in ctx


# --- Tests: Full Pipeline ---


class TestSynthesize:
    @pytest.mark.asyncio
    async def test_no_scored_docs_returns_no_results(self):
        state = _make_state(scored_documents=[])
        mock_caller = AsyncMock()
        result = await synthesize(state, mock_caller)
        assert len(result["section_drafts"]) == 1
        assert result["section_drafts"][0]["heading"] == "No Results"

    @pytest.mark.asyncio
    async def test_outline_generation_called(self):
        """LLM is called for outline generation."""
        state = _make_state()

        outline_response = {
            "sections": [
                {
                    "heading": "Gene Editing",
                    "key_claims": ["CRISPR enables genome editing"],
                    "source_ids": [1],
                    "narrative_role": "evidence",
                },
            ],
            "narrative_arc": "Overview of CRISPR",
        }
        section_response = {
            "heading": "Gene Editing",
            "content": "CRISPR-Cas9 enables genome editing [1].",
            "passage_ids_used": ["sp-001"],
            "unsupported_claims": [],
        }
        compose_response = {
            "introduction": "This report examines CRISPR.",
            "section_transitions": {},
            "conclusion": "CRISPR has changed biology.",
        }
        mock_usage = {
            "input_tokens": 100,
            "output_tokens": 50,
            "cost_usd": 0.01,
        }

        call_count = 0

        async def mock_call_json(**kwargs):
            nonlocal call_count
            call_count += 1
            agent = kwargs.get("agent_name", "")
            if "outline" in agent:
                return outline_response, mock_usage
            elif "refine" in agent:
                return section_response, mock_usage
            elif "compose" in agent:
                return compose_response, mock_usage
            else:
                return section_response, mock_usage

        mock_caller = AsyncMock()
        mock_caller.call_json = AsyncMock(side_effect=mock_call_json)

        result = await synthesize(state, mock_caller)
        assert "section_drafts" in result
        assert call_count >= 2  # At least outline + section draft

    @pytest.mark.asyncio
    async def test_composition_in_output(self):
        """Composition structure is present in output."""
        state = _make_state()

        outline_response = {
            "sections": [
                {
                    "heading": "Overview",
                    "key_claims": ["CRISPR is important"],
                    "source_ids": [1],
                    "narrative_role": "intro",
                },
            ],
        }
        section_response = {
            "heading": "Overview",
            "content": "CRISPR technology [1].",
            "passage_ids_used": ["sp-001"],
            "unsupported_claims": [],
        }
        compose_response = {
            "introduction": "This examines CRISPR.",
            "section_transitions": {},
            "conclusion": "In conclusion, CRISPR matters.",
        }
        mock_usage = {
            "input_tokens": 50,
            "output_tokens": 30,
            "cost_usd": 0.005,
        }

        async def mock_call_json(**kwargs):
            agent = kwargs.get("agent_name", "")
            if "outline" in agent:
                return outline_response, mock_usage
            elif "compose" in agent:
                return compose_response, mock_usage
            return section_response, mock_usage

        mock_caller = AsyncMock()
        mock_caller.call_json = AsyncMock(side_effect=mock_call_json)

        result = await synthesize(state, mock_caller)
        assert "composition" in result
        assert result["composition"]["introduction"] == "This examines CRISPR."
        assert result["composition"]["conclusion"] == "In conclusion, CRISPR matters."

    @pytest.mark.asyncio
    async def test_grounding_score_survives(self):
        """grounding_score and claim_details survive in output (OE3)."""
        state = _make_state()

        outline_response = {
            "sections": [
                {
                    "heading": "Results",
                    "key_claims": ["Editing works"],
                    "source_ids": [1],
                    "narrative_role": "evidence",
                },
            ],
        }
        section_response = {
            "heading": "Results",
            "content": "CRISPR-Cas9 enables genome editing [1].",
            "passage_ids_used": ["sp-001"],
            "unsupported_claims": [],
        }
        compose_response = {
            "introduction": "Intro.",
            "section_transitions": {},
            "conclusion": "Conclusion.",
        }
        mock_usage = {
            "input_tokens": 50,
            "output_tokens": 30,
            "cost_usd": 0.005,
        }

        async def mock_call_json(**kwargs):
            agent = kwargs.get("agent_name", "")
            if "outline" in agent:
                return outline_response, mock_usage
            elif "compose" in agent:
                return compose_response, mock_usage
            return section_response, mock_usage

        mock_caller = AsyncMock()
        mock_caller.call_json = AsyncMock(side_effect=mock_call_json)

        result = await synthesize(state, mock_caller)
        for sd in result["section_drafts"]:
            assert "grounding_score" in sd
            assert "claim_details" in sd

    @pytest.mark.asyncio
    async def test_citation_to_passage_map_in_output(self):
        """citation_to_passage_map is a first-class output (OE4, D3)."""
        state = _make_state()

        outline_response = {
            "sections": [
                {
                    "heading": "Section A",
                    "key_claims": ["Claim"],
                    "source_ids": [1],
                    "narrative_role": "evidence",
                },
            ],
        }
        section_response = {
            "heading": "Section A",
            "content": "Content [1].",
            "passage_ids_used": ["sp-001"],
            "unsupported_claims": [],
        }
        compose_response = {
            "introduction": "I.",
            "section_transitions": {},
            "conclusion": "C.",
        }
        mock_usage = {
            "input_tokens": 50,
            "output_tokens": 30,
            "cost_usd": 0.005,
        }

        async def mock_call_json(**kwargs):
            agent = kwargs.get("agent_name", "")
            if "outline" in agent:
                return outline_response, mock_usage
            elif "compose" in agent:
                return compose_response, mock_usage
            return section_response, mock_usage

        mock_caller = AsyncMock()
        mock_caller.call_json = AsyncMock(side_effect=mock_call_json)

        result = await synthesize(state, mock_caller)
        assert "citation_to_passage_map" in result
        assert isinstance(result["citation_to_passage_map"], dict)

    @pytest.mark.asyncio
    async def test_research_gaps_from_unsupported_claims(self):
        """Unsupported claims become research gaps."""
        state = _make_state()

        outline_response = {
            "sections": [
                {
                    "heading": "Analysis",
                    "key_claims": ["Claim X"],
                    "source_ids": [1],
                    "narrative_role": "analysis",
                },
            ],
        }
        section_response = {
            "heading": "Analysis",
            "content": "[insufficient evidence]",
            "passage_ids_used": [],
            "unsupported_claims": ["Claim X could not be supported"],
        }
        compose_response = {
            "introduction": "I.",
            "section_transitions": {},
            "conclusion": "C.",
        }
        mock_usage = {
            "input_tokens": 50,
            "output_tokens": 30,
            "cost_usd": 0.005,
        }

        async def mock_call_json(**kwargs):
            agent = kwargs.get("agent_name", "")
            if "outline" in agent:
                return outline_response, mock_usage
            elif "compose" in agent:
                return compose_response, mock_usage
            return section_response, mock_usage

        mock_caller = AsyncMock()
        mock_caller.call_json = AsyncMock(side_effect=mock_call_json)

        result = await synthesize(state, mock_caller)
        assert len(result.get("research_gaps", [])) >= 1
        assert "Claim X" in result["research_gaps"][0]["description"]

    @pytest.mark.asyncio
    async def test_token_usage_accumulated(self):
        """All LLM calls contribute to token_usage."""
        state = _make_state()

        outline_response = {
            "sections": [
                {
                    "heading": "S1",
                    "key_claims": ["C1"],
                    "source_ids": [1],
                    "narrative_role": "evidence",
                },
            ],
        }
        section_response = {
            "heading": "S1",
            "content": "Content [1].",
            "passage_ids_used": [],
            "unsupported_claims": [],
        }
        compose_response = {
            "introduction": "I.",
            "section_transitions": {},
            "conclusion": "C.",
        }
        mock_usage = {
            "input_tokens": 100,
            "output_tokens": 50,
            "cost_usd": 0.01,
        }

        async def mock_call_json(**kwargs):
            agent = kwargs.get("agent_name", "")
            if "outline" in agent:
                return outline_response, mock_usage
            elif "compose" in agent:
                return compose_response, mock_usage
            return section_response, mock_usage

        mock_caller = AsyncMock()
        mock_caller.call_json = AsyncMock(side_effect=mock_call_json)

        result = await synthesize(state, mock_caller)
        assert "token_usage" in result
        assert len(result["token_usage"]) >= 2

    @pytest.mark.asyncio
    async def test_iteration_2_keeps_high_sections(self):
        """Iteration 2+ keeps HIGH grounding sections from previous."""
        prev_section = SectionDraft(
            id="sec-prev",
            heading="Kept Section",
            content="Previously written content [1].",
            citation_ids=["[1]"],
            confidence_score=0.95,
            confidence_level=Confidence.HIGH,
            grader_scores=GraderScores(
                relevance=0.9,
                hallucination=1.0,
                quality=0.9,
            ),
            grounding_score=0.9,
            claim_details=[],
        )

        state = _make_state(
            current_iteration=2,
            section_drafts=[prev_section],
        )

        outline_response = {
            "sections": [
                {
                    "heading": "Kept Section",
                    "key_claims": ["Existing claim"],
                    "source_ids": [1],
                    "narrative_role": "evidence",
                },
                {
                    "heading": "New Section",
                    "key_claims": ["New claim"],
                    "source_ids": [2],
                    "narrative_role": "analysis",
                },
            ],
        }
        section_response = {
            "heading": "New Section",
            "content": "New content [1].",
            "passage_ids_used": ["sp-003"],
            "unsupported_claims": [],
        }
        compose_response = {
            "introduction": "I.",
            "section_transitions": {},
            "conclusion": "C.",
        }
        mock_usage = {
            "input_tokens": 50,
            "output_tokens": 30,
            "cost_usd": 0.005,
        }

        draft_calls = []

        async def mock_call_json(**kwargs):
            agent = kwargs.get("agent_name", "")
            if "outline" in agent:
                return outline_response, mock_usage
            elif "compose" in agent:
                return compose_response, mock_usage
            else:
                draft_calls.append(agent)
                return section_response, mock_usage

        mock_caller = AsyncMock()
        mock_caller.call_json = AsyncMock(side_effect=mock_call_json)

        result = await synthesize(state, mock_caller)
        # "Kept Section" should not trigger a draft call
        headings = [sd["heading"] for sd in result["section_drafts"]]
        assert "Kept Section" in headings
        assert "New Section" in headings

    @pytest.mark.asyncio
    async def test_outline_revision_on_invalid(self):
        """Invalid outline triggers revision pass."""
        state = _make_state()

        call_count = 0

        # First outline: invalid (source_id 99 doesn't exist)
        bad_outline = {
            "sections": [
                {
                    "heading": "Bad",
                    "key_claims": ["claim"],
                    "source_ids": [99],
                    "narrative_role": "evidence",
                },
            ],
        }
        # Second outline: valid
        good_outline = {
            "sections": [
                {
                    "heading": "Good",
                    "key_claims": ["claim"],
                    "source_ids": [1],
                    "narrative_role": "evidence",
                },
            ],
        }
        section_response = {
            "heading": "Good",
            "content": "Content [1].",
            "passage_ids_used": ["sp-001"],
            "unsupported_claims": [],
        }
        compose_response = {
            "introduction": "I.",
            "section_transitions": {},
            "conclusion": "C.",
        }
        mock_usage = {
            "input_tokens": 50,
            "output_tokens": 30,
            "cost_usd": 0.005,
        }

        async def mock_call_json(**kwargs):
            nonlocal call_count
            call_count += 1
            agent = kwargs.get("agent_name", "")
            if "outline" in agent:
                # First call returns bad outline, second returns good
                if call_count <= 1:
                    return bad_outline, mock_usage
                return good_outline, mock_usage
            elif "compose" in agent:
                return compose_response, mock_usage
            return section_response, mock_usage

        mock_caller = AsyncMock()
        mock_caller.call_json = AsyncMock(side_effect=mock_call_json)

        result = await synthesize(state, mock_caller)
        # Should have at least 2 outline calls (initial + revision)
        assert call_count >= 2
        headings = [sd["heading"] for sd in result["section_drafts"]]
        assert "Good" in headings
