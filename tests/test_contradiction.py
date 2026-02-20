"""Tests for contradiction detection."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from deep_research_swarm.agents.contradiction import _safe_url, detect_contradictions
from deep_research_swarm.contracts import ScoredDocument, SourceAuthority


def _doc(url: str, content: str = "Content") -> ScoredDocument:
    """Helper to build a minimal ScoredDocument."""
    return ScoredDocument(
        id="sd-test",
        url=url,
        title="Test",
        content=content,
        rrf_score=0.01,
        authority=SourceAuthority.UNKNOWN,
        authority_score=0.5,
        combined_score=0.1,
        sub_query_ids=["sq-001"],
    )


class TestDetectContradictions:
    @pytest.mark.asyncio
    async def test_no_docs_returns_empty(self):
        """Fewer than 2 docs returns empty list."""
        caller = MagicMock()
        result, usage = await detect_contradictions([_doc("http://a.com")], caller)
        assert result == []
        assert usage == {}

    @pytest.mark.asyncio
    async def test_llm_parse(self):
        """Correctly parses LLM response into Contradiction objects."""
        caller = MagicMock()
        caller.call_json = AsyncMock(
            return_value=(
                {
                    "contradictions": [
                        {
                            "claim_a": "Earth is flat",
                            "source_a_index": 1,
                            "claim_b": "Earth is round",
                            "source_b_index": 2,
                            "topic": "Earth shape",
                            "severity": "direct",
                        }
                    ]
                },
                {
                    "agent": "contradiction_detector",
                    "model": "test",
                    "input_tokens": 100,
                    "output_tokens": 50,
                    "cost_usd": 0.001,
                    "timestamp": "2026-01-01T00:00:00Z",
                },
            )
        )

        docs = [_doc("http://a.com"), _doc("http://b.com")]
        result, usage = await detect_contradictions(docs, caller)

        assert len(result) == 1
        assert result[0]["claim_a"] == "Earth is flat"
        assert result[0]["source_a_url"] == "http://a.com"
        assert result[0]["source_b_url"] == "http://b.com"
        assert result[0]["severity"] == "direct"

    def test_index_mapping(self):
        """_safe_url maps 1-based indices to correct docs."""
        docs = [_doc("http://a.com"), _doc("http://b.com"), _doc("http://c.com")]
        assert _safe_url(docs, 1) == "http://a.com"
        assert _safe_url(docs, 2) == "http://b.com"
        assert _safe_url(docs, 3) == "http://c.com"

    def test_out_of_range_index(self):
        """Out-of-range indices return empty string."""
        docs = [_doc("http://a.com")]
        assert _safe_url(docs, 0) == ""
        assert _safe_url(docs, 2) == ""
        assert _safe_url(docs, -1) == ""
