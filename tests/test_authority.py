"""Tests for source authority classification."""

from deep_research_swarm.contracts import SourceAuthority
from deep_research_swarm.scoring.authority import authority_score, classify_authority

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
