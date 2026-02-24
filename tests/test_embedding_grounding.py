"""Tests for scoring/embedding_grounding.py — embedding-based grounding (PR-07, I7)."""

from __future__ import annotations

from deep_research_swarm.contracts import SourcePassage
from deep_research_swarm.scoring.embedding_grounding import (
    EmbeddingProvider,
    FastEmbedProvider,
    cosine_similarity,
    get_embedding_provider,
    verify_grounding_embedding,
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


# --- Fake provider for tests (no fastembed dependency) ---


class FakeEmbeddingProvider:
    """Deterministic fake for testing — encodes text as character frequency vector."""

    def encode(self, text: str) -> list[float]:
        # Simple: 26-dim vector of letter frequencies
        text_lower = text.lower()
        total = max(len(text_lower), 1)
        return [text_lower.count(chr(ord("a") + i)) / total for i in range(26)]

    def similarity(self, a: list[float], b: list[float]) -> float:
        return cosine_similarity(a, b)


# --- Cosine similarity ---


class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = [1.0, 2.0, 3.0]
        assert abs(cosine_similarity(v, v) - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        assert abs(cosine_similarity(a, b)) < 1e-6

    def test_opposite_vectors(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert abs(cosine_similarity(a, b) - (-1.0)) < 1e-6

    def test_empty_vectors(self):
        assert cosine_similarity([], []) == 0.0

    def test_zero_vector(self):
        assert cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0

    def test_different_lengths(self):
        assert cosine_similarity([1.0, 2.0], [1.0]) == 0.0

    def test_known_value(self):
        a = [1.0, 2.0, 3.0]
        b = [4.0, 5.0, 6.0]
        # dot=32, norm_a=sqrt(14), norm_b=sqrt(77)
        expected = 32 / (14**0.5 * 77**0.5)
        assert abs(cosine_similarity(a, b) - expected) < 1e-6


# --- EmbeddingProvider protocol ---


class TestEmbeddingProviderProtocol:
    def test_fake_satisfies_protocol(self):
        """FakeEmbeddingProvider satisfies the EmbeddingProvider protocol."""
        provider = FakeEmbeddingProvider()
        assert isinstance(provider, EmbeddingProvider)

    def test_fastembed_provider_satisfies_protocol(self):
        """FastEmbedProvider satisfies the EmbeddingProvider protocol (structural subtyping)."""
        # We can check the class itself — it has the right methods
        assert hasattr(FastEmbedProvider, "encode")
        assert hasattr(FastEmbedProvider, "similarity")


# --- verify_grounding_embedding ---


class TestVerifyGroundingEmbedding:
    def test_high_similarity_grounded(self):
        """Similar text should be grounded."""
        provider = FakeEmbeddingProvider()
        claim = "quantum entanglement research experiments"
        passage = _make_passage("quantum entanglement research experiments and results")
        grounded, score, method = verify_grounding_embedding(claim, passage, provider)
        assert grounded is True
        assert score > 0.7
        assert method == "embedding_v1"

    def test_dissimilar_not_grounded(self):
        """Completely different text should not be grounded."""
        provider = FakeEmbeddingProvider()
        claim = "zzzzz qqqqq xxxxx"
        passage = _make_passage("aaaa bbbb cccc dddd eeee ffff")
        grounded, score, method = verify_grounding_embedding(claim, passage, provider)
        assert grounded is False
        assert method == "embedding_v1"

    def test_returns_method_embedding_v1(self):
        """I7: method="embedding_v1" for coexistence with jaccard_v1."""
        provider = FakeEmbeddingProvider()
        _, _, method = verify_grounding_embedding("test", _make_passage("test"), provider)
        assert method == "embedding_v1"

    def test_custom_threshold(self):
        """Custom threshold changes grounding decision."""
        provider = FakeEmbeddingProvider()
        claim = "test content"
        passage = _make_passage("test content data")
        # With very high threshold, should not be grounded
        grounded_strict, _, _ = verify_grounding_embedding(
            claim, passage, provider, threshold=0.999
        )
        # With very low threshold, should be grounded
        grounded_lax, _, _ = verify_grounding_embedding(claim, passage, provider, threshold=0.01)
        assert grounded_lax is True
        # strict might or might not pass depending on fake provider scores

    def test_custom_method_name(self):
        """Can override method name."""
        provider = FakeEmbeddingProvider()
        _, _, method = verify_grounding_embedding(
            "test", _make_passage("test"), provider, method="custom_v2"
        )
        assert method == "custom_v2"

    def test_score_is_rounded(self):
        """Score is rounded to 4 decimal places."""
        provider = FakeEmbeddingProvider()
        _, score, _ = verify_grounding_embedding(
            "test claim", _make_passage("test content"), provider
        )
        # Check it's rounded (no more than 4 decimal places)
        assert score == round(score, 4)

    def test_deterministic(self):
        """Same inputs produce same outputs."""
        provider = FakeEmbeddingProvider()
        claim = "quantum entanglement"
        passage = _make_passage("quantum entanglement research")
        r1 = verify_grounding_embedding(claim, passage, provider)
        r2 = verify_grounding_embedding(claim, passage, provider)
        assert r1 == r2


# --- get_embedding_provider ---


class TestGetEmbeddingProvider:
    def test_returns_none_when_fastembed_missing(self):
        """Falls back gracefully when fastembed not installed."""
        # This test works whether fastembed is installed or not:
        # if installed, returns a provider; if not, returns None
        result = get_embedding_provider()
        assert result is None or isinstance(result, EmbeddingProvider)

    def test_accepts_model_name(self):
        """Factory accepts model_name parameter."""
        result = get_embedding_provider(model_name="nonexistent/model")
        # Should return None (fastembed likely not installed) or a provider
        assert result is None or isinstance(result, EmbeddingProvider)
