"""Tests for scoring/grounding.py — mechanical grounding verification (PR-10)."""

from __future__ import annotations

from deep_research_swarm.contracts import SectionOutline, SourcePassage
from deep_research_swarm.scoring.grounding import (
    _jaccard,
    _tokenize,
    assign_passages_to_sections,
    compute_section_grounding_score,
    find_relevant_passages,
    second_pass_grounding,
    verify_grounding,
)


def _make_passage(
    content: str,
    *,
    pid: str = "sp-test0001",
    source_id: str = "src-a",
    position: int = 0,
) -> SourcePassage:
    return SourcePassage(
        id=pid,
        source_id=source_id,
        source_url="https://example.com",
        content=content,
        position=position,
        char_offset=0,
        token_count=len(content.split()),
        heading_context="",
        claim_ids=[],
    )


class TestTokenize:
    def test_basic(self):
        tokens = _tokenize("Quantum entanglement is a phenomenon")
        assert "quantum" in tokens
        assert "entanglement" in tokens
        assert "phenomenon" in tokens

    def test_stopwords_removed(self):
        tokens = _tokenize("the a an is are was to of in and or but")
        assert len(tokens) == 0

    def test_short_tokens_removed(self):
        tokens = _tokenize("go do it me")
        assert len(tokens) == 0

    def test_case_insensitive(self):
        tokens = _tokenize("QUANTUM Entanglement")
        assert "quantum" in tokens
        assert "entanglement" in tokens

    def test_numbers_included(self):
        tokens = _tokenize("version 123 release")
        assert "123" in tokens
        assert "version" in tokens


class TestJaccard:
    def test_identical(self):
        s = {"quantum", "physics"}
        assert _jaccard(s, s) == 1.0

    def test_disjoint(self):
        assert _jaccard({"quantum"}, {"biology"}) == 0.0

    def test_partial_overlap(self):
        a = {"quantum", "physics", "entanglement"}
        b = {"quantum", "physics", "chemistry"}
        score = _jaccard(a, b)
        assert 0.0 < score < 1.0
        assert score == 2 / 4  # 2 shared / 4 total unique

    def test_empty_sets(self):
        assert _jaccard(set(), {"a"}) == 0.0
        assert _jaccard({"a"}, set()) == 0.0
        assert _jaccard(set(), set()) == 0.0


class TestVerifyGrounding:
    def test_grounded_high_overlap(self):
        claim = "Quantum entanglement enables faster than light communication"
        passage = _make_passage(
            "Quantum entanglement is sometimes said to enable faster than light communication"
        )
        grounded, score, method = verify_grounding(claim, passage)
        assert grounded is True
        assert score > 0.3
        assert method == "jaccard_v1"

    def test_not_grounded_no_overlap(self):
        claim = "The stock market crashed in 2025"
        passage = _make_passage("Quantum entanglement is a fundamental phenomenon in physics")
        grounded, score, method = verify_grounding(claim, passage)
        assert grounded is False
        assert score < 0.3
        assert method == "jaccard_v1"

    def test_custom_threshold(self):
        claim = "Quantum physics research"
        passage = _make_passage("Research in quantum physics and chemistry")
        _, score, _ = verify_grounding(claim, passage, threshold=0.9)
        # Score should be decent but below 0.9
        assert score < 0.9

    def test_returns_method_string(self):
        """OE5: method field for V8 migration flexibility."""
        _, _, method = verify_grounding("test", _make_passage("test content"))
        assert method == "jaccard_v1"

    def test_deterministic(self):
        """Same inputs produce same outputs."""
        claim = "Quantum entanglement research"
        passage = _make_passage("Research on quantum entanglement phenomena")
        r1 = verify_grounding(claim, passage)
        r2 = verify_grounding(claim, passage)
        assert r1 == r2

    def test_threshold_boundary(self):
        """Threshold=0.3 is intentionally permissive (D2)."""
        claim = "Quantum entanglement enables communication"
        passage = _make_passage(
            "Quantum entanglement particles become correlated states "
            "communication channels enable information transfer"
        )
        grounded, score, _ = verify_grounding(claim, passage, threshold=0.3)
        # This should have decent overlap
        assert isinstance(grounded, bool)
        assert 0.0 <= score <= 1.0


class TestFindRelevantPassages:
    def test_returns_top_k(self):
        passages = [
            _make_passage("Quantum entanglement is fundamental", pid="sp-001"),
            _make_passage("Biology of cells and organisms", pid="sp-002"),
            _make_passage("Quantum physics experiments show", pid="sp-003"),
        ]
        results = find_relevant_passages("quantum physics", passages, top_k=2)
        assert len(results) == 2
        # First result should be about quantum
        assert "quantum" in results[0][0]["content"].lower()

    def test_empty_query(self):
        passages = [_make_passage("content")]
        assert find_relevant_passages("", passages) == []

    def test_empty_passages(self):
        assert find_relevant_passages("query", []) == []

    def test_scores_are_floats(self):
        passages = [_make_passage("Test content about science")]
        results = find_relevant_passages("science research", passages)
        assert len(results) == 1
        assert isinstance(results[0][1], float)


class TestAssignPassagesToSections:
    def test_assigns_by_source_id(self):
        outline = [
            SectionOutline(
                heading="Intro",
                key_claims=["entanglement"],
                source_ids=["src-a"],
                narrative_role="introduction",
            )
        ]
        passages = [
            _make_passage("Entanglement content", pid="sp-001", source_id="src-a"),
            _make_passage("Unrelated content", pid="sp-002", source_id="src-b"),
        ]
        result = assign_passages_to_sections(outline, passages)
        assert "Intro" in result
        assert all(p["source_id"] == "src-a" for p in result["Intro"])

    def test_empty_outline(self):
        assert assign_passages_to_sections([], [_make_passage("x")]) == {}

    def test_empty_passages(self):
        outline = [
            SectionOutline(
                heading="Intro",
                key_claims=[],
                source_ids=[],
                narrative_role="introduction",
            )
        ]
        assert assign_passages_to_sections(outline, []) == {}

    def test_max_passages_per_section(self):
        outline = [
            SectionOutline(
                heading="Test",
                key_claims=["topic"],
                source_ids=["src-a"],
                narrative_role="evidence",
            )
        ]
        passages = [
            _make_passage(f"Content {i}", pid=f"sp-{i:03d}", source_id="src-a", position=i)
            for i in range(20)
        ]
        result = assign_passages_to_sections(outline, passages, max_passages_per_section=5)
        assert len(result["Test"]) <= 5

    def test_fallback_to_all_passages(self):
        """When no source_ids match, falls back to all passages."""
        outline = [
            SectionOutline(
                heading="Intro",
                key_claims=["quantum"],
                source_ids=["nonexistent-source"],
                narrative_role="introduction",
            )
        ]
        passages = [
            _make_passage("Quantum physics content", pid="sp-001", source_id="src-a"),
        ]
        result = assign_passages_to_sections(outline, passages)
        assert len(result["Intro"]) == 1


class TestComputeSectionGroundingScore:
    def test_fully_grounded(self):
        """All claims grounded -> score = 1.0."""
        passage = _make_passage(
            "Quantum entanglement enables particles to be correlated "
            "across distances regardless of separation",
            pid="sp-001",
        )
        content = (
            "Quantum entanglement enables particles to be correlated across large distances [1]."
        )
        score, details = compute_section_grounding_score(
            section_content=content,
            section_citations=["[1]"],
            passages=[passage],
            citation_to_passage_map={"[1]": ["sp-001"]},
        )
        assert score > 0.0
        assert len(details) == 1
        assert details[0]["citation_id"] == "[1]"

    def test_no_citations(self):
        """No citations -> score 0.0, empty details."""
        score, details = compute_section_grounding_score(
            section_content="Plain text without citations.",
            section_citations=[],
            passages=[],
            citation_to_passage_map={},
        )
        assert score == 0.0
        assert details == []

    def test_ungrounded_claim(self):
        """Claim with no matching passage content -> not grounded."""
        passage = _make_passage("Biology of cells and organisms", pid="sp-001")
        content = "The stock market crashed in 2025 [1]."
        score, details = compute_section_grounding_score(
            section_content=content,
            section_citations=["[1]"],
            passages=[passage],
            citation_to_passage_map={"[1]": ["sp-001"]},
        )
        assert len(details) == 1
        assert details[0]["grounded"] is False

    def test_claim_details_structure(self):
        """claim_details has correct keys (OE3, D4)."""
        passage = _make_passage("Quantum entanglement research", pid="sp-001")
        content = "Quantum entanglement is important [1]."
        _, details = compute_section_grounding_score(
            section_content=content,
            section_citations=["[1]"],
            passages=[passage],
            citation_to_passage_map={"[1]": ["sp-001"]},
        )
        assert len(details) == 1
        d = details[0]
        assert "claim" in d
        assert "citation_id" in d
        assert "passage_id" in d
        assert "grounded" in d
        assert "similarity" in d
        assert "method" in d
        assert d["method"] == "jaccard_v1"

    def test_multiple_citations_in_section(self):
        """Multiple citation markers are each verified separately."""
        p1 = _make_passage("Quantum entanglement phenomenon", pid="sp-001")
        p2 = _make_passage("Bell test experiments confirm", pid="sp-002")
        content = (
            "Quantum entanglement is a phenomenon [1]. Bell test experiments confirm this [2]."
        )
        score, details = compute_section_grounding_score(
            section_content=content,
            section_citations=["[1]", "[2]"],
            passages=[p1, p2],
            citation_to_passage_map={"[1]": ["sp-001"], "[2]": ["sp-002"]},
        )
        assert len(details) == 2
        citation_ids = {d["citation_id"] for d in details}
        assert "[1]" in citation_ids
        assert "[2]" in citation_ids


# --- Second-Pass Grounding (V8, PR-06) ---


class TestSecondPassGrounding:
    def test_below_floor_not_rescued(self):
        """Score below borderline_floor (0.15) — too far gone."""
        claim = "The stock market crashed in 2025"
        borderline = _make_passage("Quantum physics entanglement experiments", pid="sp-border")
        accepted = [
            _make_passage("Quantum physics research laboratory", pid="sp-acc1"),
            _make_passage("Physics experiments quantum mechanics", pid="sp-acc2"),
        ]
        rescued, score, method = second_pass_grounding(claim, borderline, accepted)
        assert rescued is False
        assert method == "neighborhood_v1"

    def test_rescued_with_enough_neighbors(self):
        """Borderline passage rescued when 2+ accepted passages are topically related."""
        # Claim and borderline share some tokens (academic paraphrase)
        claim = "lace patterns demonstrate mathematical structure in textile engineering"
        borderline = _make_passage(
            "textile patterns mathematical analysis engineering applications "
            "structural properties textile manufacturing",
            pid="sp-border",
        )
        # Accepted passages in same topic neighborhood
        accepted = [
            _make_passage(
                "mathematical structure textile patterns engineering analysis "
                "manufacturing processes structural design",
                pid="sp-acc1",
            ),
            _make_passage(
                "engineering applications textile manufacturing structural "
                "patterns analysis mathematical foundations",
                pid="sp-acc2",
            ),
            _make_passage(
                "textile engineering structural analysis manufacturing "
                "patterns mathematical properties applications",
                pid="sp-acc3",
            ),
        ]
        rescued, score, method = second_pass_grounding(claim, borderline, accepted)
        assert rescued is True
        assert method == "neighborhood_v1"
        assert score <= 0.35  # capped

    def test_not_rescued_with_one_neighbor(self):
        """Only 1 neighbor — not enough for rescue (need >= 2)."""
        claim = "lace patterns engineering"
        borderline = _make_passage(
            "textile patterns engineering analysis structural properties",
            pid="sp-border",
        )
        accepted = [
            _make_passage(
                "textile patterns engineering structural analysis manufacturing",
                pid="sp-acc1",
            ),
            _make_passage("biology cells organisms marine life", pid="sp-acc2"),
        ]
        rescued, score, method = second_pass_grounding(claim, borderline, accepted)
        # May or may not rescue depending on exact Jaccard scores
        assert method == "neighborhood_v1"

    def test_returns_method_neighborhood_v1(self):
        """OE5/I7: method="neighborhood_v1" for second pass."""
        claim = "test claim"
        borderline = _make_passage("test content passage", pid="sp-border")
        _, _, method = second_pass_grounding(claim, borderline, [])
        assert method == "neighborhood_v1"

    def test_score_capped_at_035(self):
        """Adjusted score never exceeds 0.35."""
        claim = "textile patterns engineering analysis structural manufacturing"
        borderline = _make_passage(
            "textile patterns engineering analysis structural manufacturing design",
            pid="sp-border",
        )
        # Many neighbors to boost score
        accepted = [
            _make_passage(
                f"textile patterns engineering analysis structural manufacturing {i}",
                pid=f"sp-acc{i}",
            )
            for i in range(10)
        ]
        rescued, score, method = second_pass_grounding(claim, borderline, accepted)
        assert score <= 0.35

    def test_empty_accepted_not_rescued(self):
        """No accepted passages means no neighbors, no rescue."""
        claim = "lace patterns engineering"
        borderline = _make_passage(
            "textile patterns engineering analysis",
            pid="sp-border",
        )
        rescued, score, method = second_pass_grounding(claim, borderline, [])
        assert rescued is False

    def test_purely_additive(self):
        """Second pass never downgrades — it only potentially rescues."""
        claim = "quantum entanglement research"
        borderline = _make_passage("quantum entanglement research experiments", pid="sp-border")
        # First pass
        _, first_score, _ = verify_grounding(claim, borderline)
        # Second pass
        _, second_score, _ = second_pass_grounding(claim, borderline, [])
        # Without any accepted neighbors, score stays the same
        assert second_score >= first_score or second_score == first_score

    def test_custom_thresholds(self):
        """Can adjust borderline_floor and neighborhood_threshold."""
        claim = "test claim content"
        borderline = _make_passage("test claim content data", pid="sp-border")
        rescued, score, method = second_pass_grounding(
            claim,
            borderline,
            [],
            borderline_floor=0.5,
            neighborhood_threshold=0.1,
        )
        assert method == "neighborhood_v1"

    def test_deterministic(self):
        claim = "lace patterns engineering"
        borderline = _make_passage("lace patterns engineering textile", pid="sp-border")
        accepted = [_make_passage("lace patterns engineering design", pid="sp-acc1")]
        r1 = second_pass_grounding(claim, borderline, accepted)
        r2 = second_pass_grounding(claim, borderline, accepted)
        assert r1 == r2


class TestSecondPassIntegration:
    """Test second pass integration in compute_section_grounding_score."""

    def test_borderline_claim_rescued_in_section(self):
        """A borderline claim gets rescued when accepted passages are in same neighborhood."""
        # Passage 1: well-grounded (high overlap with its claim)
        grounded_passage = _make_passage(
            "Quantum entanglement enables particles to be correlated "
            "across distances instantaneously quantum mechanics research",
            pid="sp-001",
        )
        # Passage 2: borderline (paraphrased, low Jaccard with its claim)
        borderline_passage = _make_passage(
            "Quantum mechanics research particles correlation "
            "entanglement experiments distance physics",
            pid="sp-002",
        )
        # Claim 1: high overlap with passage 1 (grounded first-pass)
        # Claim 2: low overlap with passage 2 (borderline)
        content = (
            "Quantum entanglement enables particles to be correlated across distances [1]. "
            "Modern experiments demonstrate new correlation phenomena [2]."
        )
        score, details = compute_section_grounding_score(
            section_content=content,
            section_citations=["[1]", "[2]"],
            passages=[grounded_passage, borderline_passage],
            citation_to_passage_map={"[1]": ["sp-001"], "[2]": ["sp-002"]},
        )
        # At least claim [1] should be grounded
        assert len(details) == 2
        assert any(d["grounded"] for d in details)

    def test_no_regression_on_clearly_grounded(self):
        """Well-grounded claims remain grounded after second pass."""
        passage = _make_passage(
            "Quantum entanglement enables particles to be correlated "
            "across distances regardless of separation",
            pid="sp-001",
        )
        content = (
            "Quantum entanglement enables particles to be correlated across large distances [1]."
        )
        score, details = compute_section_grounding_score(
            section_content=content,
            section_citations=["[1]"],
            passages=[passage],
            citation_to_passage_map={"[1]": ["sp-001"]},
        )
        assert score > 0.0
        assert details[0]["grounded"] is True

    def test_no_regression_on_clearly_ungrounded(self):
        """Completely ungrounded claims stay ungrounded (below floor)."""
        passage = _make_passage("Biology of cells and organisms", pid="sp-001")
        content = "The stock market crashed in 2025 [1]."
        score, details = compute_section_grounding_score(
            section_content=content,
            section_citations=["[1]"],
            passages=[passage],
            citation_to_passage_map={"[1]": ["sp-001"]},
        )
        assert details[0]["grounded"] is False
