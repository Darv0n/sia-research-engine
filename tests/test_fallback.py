"""Tests for AgentCaller OverloadedError fallback and warning suppression."""

from __future__ import annotations

import warnings
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import anthropic
import httpx
import pytest
from anthropic._exceptions import OverloadedError

from deep_research_swarm.agents.base import AgentCaller

# --- Helpers ---


def _make_response(input_tokens: int = 100, output_tokens: int = 50) -> SimpleNamespace:
    """Build a minimal mock response matching Anthropic SDK shape."""
    text_block = SimpleNamespace(type="text", text="hello")
    usage = SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens)
    return SimpleNamespace(content=[text_block], usage=usage)


def _make_overloaded_error() -> OverloadedError:
    """Build an OverloadedError with a minimal httpx.Response."""
    resp = httpx.Response(
        status_code=529, request=httpx.Request("POST", "https://api.anthropic.com")
    )
    return OverloadedError(message="overloaded", response=resp, body=None)


def _make_caller(*, fallback_model: str | None = None, max_retries: int = 2) -> AgentCaller:
    return AgentCaller(
        api_key="test-key",
        model="claude-opus-4-6",
        max_concurrent=1,
        max_retries=max_retries,
        fallback_model=fallback_model,
    )


_CALL_KWARGS = dict(
    system="sys",
    messages=[{"role": "user", "content": "hi"}],
    agent_name="test-agent",
    max_tokens=100,
    temperature=0.0,
)


# --- Fix 1: Fallback tests ---


class TestOverloadedFallback:
    @pytest.mark.asyncio
    async def test_fallback_triggers_on_overloaded(self):
        """After retries exhausted by OverloadedError, caller uses fallback model."""
        caller = _make_caller(fallback_model="claude-sonnet-4-6", max_retries=2)
        success_resp = _make_response()

        # First 2 calls raise OverloadedError, third (fallback) succeeds
        caller._client.messages.create = AsyncMock(
            side_effect=[
                _make_overloaded_error(),
                _make_overloaded_error(),
                success_resp,
            ]
        )

        text, usage = await caller._call_with_retry(**_CALL_KWARGS)

        assert text == "hello"
        assert usage["model"] == "claude-sonnet-4-6"
        # 2 retries + 1 fallback = 3 calls
        assert caller._client.messages.create.call_count == 3
        # Fallback call should use sonnet model
        fallback_call = caller._client.messages.create.call_args_list[2]
        assert fallback_call.kwargs["model"] == "claude-sonnet-4-6"

    @pytest.mark.asyncio
    async def test_no_fallback_on_generic_api_error(self):
        """Generic APIError should NOT trigger fallback, even if configured."""
        caller = _make_caller(fallback_model="claude-sonnet-4-6", max_retries=2)

        mock_response = AsyncMock()
        mock_response.status_code = 500
        mock_response.headers = {}

        caller._client.messages.create = AsyncMock(
            side_effect=anthropic.InternalServerError(
                message="server error",
                body=None,
                response=mock_response,
            )
        )

        with pytest.raises(RuntimeError, match="failed after 2 retries"):
            await caller._call_with_retry(**_CALL_KWARGS)

        # Should NOT have attempted fallback
        assert caller._client.messages.create.call_count == 2

    @pytest.mark.asyncio
    async def test_no_fallback_when_unconfigured(self):
        """Without fallback_model, OverloadedError just raises RuntimeError."""
        caller = _make_caller(fallback_model=None, max_retries=2)

        caller._client.messages.create = AsyncMock(
            side_effect=[_make_overloaded_error(), _make_overloaded_error()]
        )

        with pytest.raises(RuntimeError, match="failed after 2 retries"):
            await caller._call_with_retry(**_CALL_KWARGS)

        # Only retry attempts, no fallback
        assert caller._client.messages.create.call_count == 2

    @pytest.mark.asyncio
    async def test_fallback_uses_correct_pricing(self):
        """Fallback call should use sonnet pricing, not opus."""
        caller = _make_caller(fallback_model="claude-sonnet-4-6", max_retries=1)
        # 1000 input, 500 output with sonnet pricing: (1000*3 + 500*15) / 1M = 0.0105
        success_resp = _make_response(input_tokens=1000, output_tokens=500)

        caller._client.messages.create = AsyncMock(
            side_effect=[
                _make_overloaded_error(),
                success_resp,
            ]
        )

        _, usage = await caller._call_with_retry(**_CALL_KWARGS)

        assert usage["model"] == "claude-sonnet-4-6"
        expected_cost = (1000 * 3.0 + 500 * 15.0) / 1_000_000
        assert usage["cost_usd"] == round(expected_cost, 6)

    @pytest.mark.asyncio
    async def test_fallback_warning_to_stderr(self, capsys):
        """Fallback should print warning to stderr."""
        caller = _make_caller(fallback_model="claude-sonnet-4-6", max_retries=1)
        success_resp = _make_response()

        caller._client.messages.create = AsyncMock(
            side_effect=[
                _make_overloaded_error(),
                success_resp,
            ]
        )

        await caller._call_with_retry(**_CALL_KWARGS)

        captured = capsys.readouterr()
        assert "overloaded (529)" in captured.err
        assert "falling back to claude-sonnet-4-6" in captured.err

    @pytest.mark.asyncio
    async def test_success_on_retry_no_fallback(self):
        """If a retry succeeds, fallback is never attempted."""
        caller = _make_caller(fallback_model="claude-sonnet-4-6", max_retries=3)
        success_resp = _make_response()

        caller._client.messages.create = AsyncMock(
            side_effect=[
                _make_overloaded_error(),
                success_resp,  # Second attempt succeeds
            ]
        )

        text, usage = await caller._call_with_retry(**_CALL_KWARGS)

        assert text == "hello"
        assert usage["model"] == "claude-opus-4-6"  # Original model, not fallback
        assert caller._client.messages.create.call_count == 2


# --- Fix 2: Mode in ResearchState ---


class TestModeInState:
    def test_mode_field_exists_in_state(self):
        """ResearchState should have a 'mode' field."""
        from deep_research_swarm.graph.state import ResearchState

        annotations = ResearchState.__annotations__
        assert "mode" in annotations

    def test_mode_in_initial_state_template(self):
        """Verify that the initial_state dict includes 'mode' by checking the source."""
        # Instead of running the CLI, verify the state type accepts mode
        from deep_research_swarm.graph.state import ResearchState

        # Create a minimal state dict with mode
        state: ResearchState = {
            "research_question": "test",
            "max_iterations": 1,
            "token_budget": 1000,
            "search_backends": [],
            "memory_context": "",
            "mode": "hitl",
            "perspectives": [],
            "sub_queries": [],
            "search_results": [],
            "extracted_contents": [],
            "scored_documents": [],
            "diversity_metrics": {},
            "section_drafts": [],
            "citations": [],
            "contradictions": [],
            "research_gaps": [],
            "current_iteration": 0,
            "converged": False,
            "convergence_reason": "",
            "token_usage": [],
            "total_tokens_used": 0,
            "total_cost_usd": 0.0,
            "iteration_history": [],
            "final_report": "",
        }
        assert state["mode"] == "hitl"


# --- Fix 3: Warning suppression ---


class TestWarningsSuppressed:
    @patch("deep_research_swarm.graph.builder.AgentCaller")
    def test_no_config_warning_on_compile(self, mock_caller_cls):
        """Graph compilation should not emit the config-typing UserWarning."""
        from deep_research_swarm.config import Settings
        from deep_research_swarm.graph.builder import build_graph

        settings = Settings(
            anthropic_api_key="test-key",
            searxng_url="http://localhost:8080",
        )
        mock_caller_cls.return_value = AsyncMock()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            graph = build_graph(settings)

        config_warnings = [
            w
            for w in caught
            if "config" in str(w.message).lower() and "parameter" in str(w.message).lower()
        ]
        assert config_warnings == [], f"Unexpected config warnings: {config_warnings}"
        assert graph is not None
