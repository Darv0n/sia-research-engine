"""Embedding-based grounding verification (V8, I7).

EmbeddingProvider protocol + FastEmbedProvider implementation.
Returns method="embedding_v1" — coexists with "jaccard_v1" via OE5.

fastembed is an optional dependency: MIT, ONNX-based, ~50MB, no PyTorch.
Falls back to Jaccard if fastembed is not installed.
"""

from __future__ import annotations

import math
from typing import Protocol, runtime_checkable

from deep_research_swarm.contracts import SourcePassage

# --- Protocol ---


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Protocol for embedding providers (structural subtyping)."""

    def encode(self, text: str) -> list[float]: ...

    def similarity(self, a: list[float], b: list[float]) -> float: ...


# --- Cosine similarity (standalone, no numpy) ---


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors without numpy."""
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


# --- FastEmbed provider ---


class FastEmbedProvider:
    """Embedding provider backed by fastembed (ONNX, MIT license).

    Lazy-loads the model on first encode() call to avoid import-time cost.
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        self._model_name = model_name
        self._model = None

    def _ensure_model(self) -> None:
        if self._model is None:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(model_name=self._model_name)

    def encode(self, text: str) -> list[float]:
        """Encode a single text string into a float vector."""
        self._ensure_model()
        # fastembed returns a generator of numpy arrays
        embeddings = list(self._model.embed([text]))
        if embeddings:
            return embeddings[0].tolist()
        return []

    def similarity(self, a: list[float], b: list[float]) -> float:
        """Cosine similarity between two embedding vectors."""
        return cosine_similarity(a, b)


# --- Embedding grounding verification ---


def verify_grounding_embedding(
    claim: str,
    cited_passage: SourcePassage,
    provider: EmbeddingProvider,
    *,
    threshold: float = 0.7,
    method: str = "embedding_v1",
) -> tuple[bool, float, str]:
    """Verify grounding using embedding similarity (I7).

    Returns (is_grounded, similarity_score, method).
    threshold=0.7 for cosine similarity (much stricter scale than Jaccard).
    """
    claim_emb = provider.encode(claim)
    passage_emb = provider.encode(cited_passage["content"])
    score = provider.similarity(claim_emb, passage_emb)
    return (score >= threshold, round(score, 4), method)


# --- Provider factory ---


def get_embedding_provider(model_name: str = "BAAI/bge-small-en-v1.5") -> EmbeddingProvider | None:
    """Try to create an embedding provider. Returns None if fastembed unavailable."""
    try:
        import fastembed  # noqa: F401

        return FastEmbedProvider(model_name=model_name)
    except ImportError:
        return None
