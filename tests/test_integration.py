"""Integration tests — mocked LLM, real graph execution.

These tests verify the full pipeline with canned responses,
without making real API calls.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

# Canned LLM responses
PLANNER_RESPONSE = json.dumps(
    {
        "perspectives": ["physics fundamentals", "experimental evidence", "applications"],
        "sub_queries": [
            {
                "question": "What is quantum entanglement in simple terms?",
                "perspective": "physics fundamentals",
                "priority": 1,
            },
            {
                "question": "Bell test experiments quantum entanglement proof",
                "perspective": "experimental evidence",
                "priority": 2,
            },
            {
                "question": "Quantum entanglement applications in computing and communication",
                "perspective": "applications",
                "priority": 3,
            },
        ],
    }
)

SYNTHESIZER_RESPONSE = json.dumps(
    {
        "sections": [
            {
                "heading": "What is Quantum Entanglement?",
                "content": "Quantum entanglement is a phenomenon "
                "where two particles become correlated [1].",
                "source_ids": [1],
                "confidence": 0.85,
            },
            {
                "heading": "Experimental Verification",
                "content": "Bell test experiments have confirmed entanglement [2].",
                "source_ids": [2],
                "confidence": 0.80,
            },
        ],
        "gaps": [],
    }
)

CRITIC_RESPONSE = json.dumps(
    {
        "evaluations": [
            {
                "section_id": "placeholder",  # Will be replaced
                "relevance": 0.90,
                "hallucination": 0.85,
                "quality": 0.85,
            },
            {
                "section_id": "placeholder2",
                "relevance": 0.85,
                "hallucination": 0.80,
                "quality": 0.80,
            },
        ]
    }
)


class MockUsage:
    def __init__(self):
        self.input_tokens = 100
        self.output_tokens = 50


class MockTextBlock:
    def __init__(self, text):
        self.type = "text"
        self.text = text


class MockResponse:
    def __init__(self, text):
        self.content = [MockTextBlock(text)]
        self.usage = MockUsage()


@pytest.mark.asyncio
async def test_planner_produces_sub_queries():
    """Test that the planner agent produces valid sub-queries."""
    from deep_research_swarm.agents.base import AgentCaller
    from deep_research_swarm.agents.planner import plan

    # Mock the Anthropic client
    with patch("deep_research_swarm.agents.base.anthropic") as mock_anthropic:
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=MockResponse(PLANNER_RESPONSE))
        mock_anthropic.AsyncAnthropic.return_value = mock_client

        caller = AgentCaller(api_key="test", model="claude-opus-4-6")

        state = {
            "research_question": "What is quantum entanglement?",
            "current_iteration": 0,
            "sub_queries": [],
            "research_gaps": [],
            "search_backends": ["searxng"],
        }

        result = await plan(state, caller)

        assert "perspectives" in result
        assert len(result["perspectives"]) == 3
        assert "sub_queries" in result
        assert len(result["sub_queries"]) == 3
        expected_q = "What is quantum entanglement in simple terms?"
        assert result["sub_queries"][0]["question"] == expected_q
        assert result["current_iteration"] == 1
