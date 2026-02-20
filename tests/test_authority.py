"""Tests for source authority classification."""

from deep_research_swarm.contracts import SourceAuthority
from deep_research_swarm.scoring.authority import authority_score, classify_authority


class TestClassifyAuthority:
    def test_edu_is_institutional(self):
        assert classify_authority("https://mit.edu/research") == SourceAuthority.INSTITUTIONAL

    def test_gov_is_institutional(self):
        assert classify_authority("https://www.nih.gov/health") == SourceAuthority.INSTITUTIONAL

    def test_nature_is_institutional(self):
        assert classify_authority("https://www.nature.com/articles/123") == SourceAuthority.INSTITUTIONAL

    def test_arxiv_is_institutional(self):
        assert classify_authority("https://arxiv.org/abs/2401.12345") == SourceAuthority.INSTITUTIONAL

    def test_bbc_is_professional(self):
        assert classify_authority("https://www.bbc.com/news") == SourceAuthority.PROFESSIONAL

    def test_stackoverflow_is_professional(self):
        assert classify_authority("https://stackoverflow.com/questions/123") == SourceAuthority.PROFESSIONAL

    def test_reddit_is_community(self):
        assert classify_authority("https://www.reddit.com/r/science") == SourceAuthority.COMMUNITY

    def test_wikipedia_is_community(self):
        assert classify_authority("https://en.wikipedia.org/wiki/Test") == SourceAuthority.COMMUNITY

    def test_unknown_domain(self):
        assert classify_authority("https://randomsite123.com/page") == SourceAuthority.UNKNOWN

    def test_empty_url(self):
        assert classify_authority("") == SourceAuthority.UNKNOWN

    def test_invalid_url(self):
        assert classify_authority("not-a-url") == SourceAuthority.UNKNOWN

    def test_promotional_pattern(self):
        assert classify_authority("https://shop.example.com/buy") == SourceAuthority.PROMOTIONAL


class TestAuthorityScore:
    def test_institutional_is_highest(self):
        assert authority_score(SourceAuthority.INSTITUTIONAL) > authority_score(SourceAuthority.PROFESSIONAL)

    def test_professional_above_community(self):
        assert authority_score(SourceAuthority.PROFESSIONAL) > authority_score(SourceAuthority.COMMUNITY)

    def test_promotional_is_lowest(self):
        assert authority_score(SourceAuthority.PROMOTIONAL) < authority_score(SourceAuthority.UNKNOWN)

    def test_scores_in_range(self):
        for auth in SourceAuthority:
            score = authority_score(auth)
            assert 0.0 <= score <= 1.0
