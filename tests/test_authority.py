"""Tests for source authority classification."""

from deep_research_swarm.contracts import SourceAuthority
from deep_research_swarm.scoring.authority import (
    authority_score,
    classify_authority,
    score_authority,
)

INST = SourceAuthority.INSTITUTIONAL
PROF = SourceAuthority.PROFESSIONAL
COMM = SourceAuthority.COMMUNITY
PROM = SourceAuthority.PROMOTIONAL
UNKN = SourceAuthority.UNKNOWN


class TestClassifyAuthority:
    def test_edu_is_institutional(self):
        assert classify_authority("https://mit.edu/research") == INST

    def test_gov_is_institutional(self):
        assert classify_authority("https://www.nih.gov/health") == INST

    def test_nature_is_institutional(self):
        assert classify_authority("https://www.nature.com/articles/123") == INST

    def test_arxiv_is_institutional(self):
        assert classify_authority("https://arxiv.org/abs/2401.12345") == INST

    def test_bbc_is_professional(self):
        assert classify_authority("https://www.bbc.com/news") == PROF

    def test_stackoverflow_is_professional(self):
        url = "https://stackoverflow.com/questions/123"
        assert classify_authority(url) == PROF

    def test_reddit_is_community(self):
        assert classify_authority("https://www.reddit.com/r/science") == COMM

    def test_wikipedia_is_community(self):
        assert classify_authority("https://en.wikipedia.org/wiki/Test") == COMM

    def test_unknown_domain(self):
        assert classify_authority("https://randomsite123.com/page") == UNKN

    def test_empty_url(self):
        assert classify_authority("") == UNKN

    def test_invalid_url(self):
        assert classify_authority("not-a-url") == UNKN

    def test_promotional_pattern(self):
        assert classify_authority("https://shop.example.com/buy") == PROM


class TestAuthorityScore:
    def test_institutional_is_highest(self):
        assert authority_score(INST) > authority_score(PROF)

    def test_professional_above_community(self):
        assert authority_score(PROF) > authority_score(COMM)

    def test_promotional_is_lowest(self):
        assert authority_score(PROM) < authority_score(UNKN)

    def test_scores_in_range(self):
        for auth in SourceAuthority:
            score = authority_score(auth)
            assert 0.0 <= score <= 1.0


class TestClassifyAuthorityV7Domains:
    """V7: scholarly domains are classified as institutional (PR-09)."""

    def test_doi_org_is_institutional(self):
        assert classify_authority("https://doi.org/10.1234/test") == INST

    def test_semanticscholar_is_institutional(self):
        assert classify_authority("https://www.semanticscholar.org/paper/123") == INST

    def test_biorxiv_is_institutional(self):
        assert classify_authority("https://www.biorxiv.org/content/123") == INST

    def test_plos_is_institutional(self):
        assert classify_authority("https://journals.plos.org/article") == INST

    def test_researchgate_is_institutional(self):
        assert classify_authority("https://www.researchgate.net/publication/123") == INST


class TestScoreAuthority:
    """V7: score_authority() with scholarly metadata (PR-09)."""

    def test_fallback_without_metadata(self):
        """Without metadata, behaves identically to classify + authority_score."""
        auth, score = score_authority("https://www.nature.com/articles/123")
        assert auth == INST
        assert score == authority_score(INST)

    def test_fallback_unknown_domain(self):
        auth, score = score_authority("https://randomsite.com/page")
        assert auth == UNKN
        assert score == authority_score(UNKN)

    def test_citation_bonus(self):
        """High citations boost the score."""
        meta = {"citation_count": 500, "is_open_access": False, "venue": ""}
        _, base = score_authority("https://doi.org/10.1234/test")
        _, boosted = score_authority("https://doi.org/10.1234/test", scholarly_metadata=meta)
        assert boosted > base

    def test_open_access_bonus(self):
        meta = {"citation_count": 0, "is_open_access": True, "venue": ""}
        _, base = score_authority("https://doi.org/10.1234/test")
        _, boosted = score_authority("https://doi.org/10.1234/test", scholarly_metadata=meta)
        assert boosted > base

    def test_venue_bonus(self):
        meta = {"citation_count": 0, "is_open_access": False, "venue": "Nature"}
        _, base = score_authority("https://doi.org/10.1234/test")
        _, boosted = score_authority("https://doi.org/10.1234/test", scholarly_metadata=meta)
        assert boosted > base

    def test_unknown_url_promoted_with_metadata(self):
        """Unknown-domain URL with scholarly metadata gets promoted to INSTITUTIONAL."""
        meta = {"citation_count": 10, "is_open_access": False, "venue": "ICML"}
        auth, score = score_authority("https://randomsite.com/paper.pdf", scholarly_metadata=meta)
        assert auth == INST
        assert score >= authority_score(INST)

    def test_score_capped_at_one(self):
        """Even with massive citations + all bonuses, score doesn't exceed 1.0."""
        meta = {"citation_count": 10000, "is_open_access": True, "venue": "Nature"}
        _, score = score_authority("https://doi.org/10.1234/test", scholarly_metadata=meta)
        assert score <= 1.0

    def test_classify_authority_unchanged(self):
        """Original classify_authority() behavior is preserved."""
        assert classify_authority("https://mit.edu/research") == INST
        assert classify_authority("https://www.bbc.com/news") == PROF
        assert classify_authority("https://reddit.com/r/test") == COMM
