"""Tests for AgentCaller.call_json JSON retry logic."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from deep_research_swarm.agents.base import AgentCaller


# --- Helpers ---


def _make_response(
    text: str, input_tokens: int = 100, output_tokens: int = 50
) -> SimpleNamespace:
    """Build a minimal mock response matching Anthropic SDK shape."""
    text_block = SimpleNamespace(type="text", text=text)
    usage = SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens)
    return SimpleNamespace(content=[text_block], usage=usage)


def _make_caller() -> AgentCaller:
    return AgentCaller(
        api_key="test-key",
        model="claude-sonnet-4-6",
        max_concurrent=1,
        max_retries=1,
    )


_CALL_KWARGS = dict(
    system="Return JSON only.",
    messages=[{"role": "user", "content": "hi"}],
    agent_name="test-agent",
    max_tokens=4096,
    temperature=0.0,
)


# --- Tests: happy path (no retry needed) ---


class TestCallJsonHappyPath:
    @pytest.mark.asyncio
    async def test_valid_json_returns_immediately(self):
        """When LLM returns valid JSON, no retry occurs."""
        caller = _make_caller()
        caller._client.messages.create = AsyncMock(
            return_value=_make_response('{"heading": "Test", "content": "Hello"}')
        )
        data, usage = await caller.call_json(**_CALL_KWARGS)
        assert data == {"heading": "Test", "content": "Hello"}
        assert caller._client.messages.create.call_count == 1

    @pytest.mark.asyncio
    async def test_json_in_markdown_fence(self):
        """JSON wrapped in ```json fences should parse without retry."""
        caller = _make_caller()
        caller._client.messages.create = AsyncMock(
            return_value=_make_response(
                '```json\n{"heading": "Test", "content": "ok"}\n```'
            )
        )
        data, usage = await caller.call_json(**_CALL_KWARGS)
        assert data["heading"] == "Test"
        assert caller._client.messages.create.call_count == 1

    @pytest.mark.asyncio
    async def test_json_embedded_in_prose(self):
        """JSON embedded in surrounding prose should be extracted without retry."""
        caller = _make_caller()
        caller._client.messages.create = AsyncMock(
            return_value=_make_response(
                'Here is the result: {"heading": "Test", "content": "ok"} Hope that helps!'
            )
        )
        data, usage = await caller.call_json(**_CALL_KWARGS)
        assert data["heading"] == "Test"
        assert caller._client.messages.create.call_count == 1


# --- Tests: retry on malformed JSON ---


class TestCallJsonRetry:
    @pytest.mark.asyncio
    async def test_retry_on_prose_then_json(self):
        """When first response is prose, retry should get valid JSON."""
        caller = _make_caller()
        prose_resp = _make_response(
            "I can see the issue. The citations need fixing. Let me rewrite..."
        )
        json_resp = _make_response('{"heading": "Fixed", "content": "Revised text"}')

        caller._client.messages.create = AsyncMock(
            side_effect=[prose_resp, json_resp]
        )
        data, usage = await caller.call_json(**_CALL_KWARGS)
        assert data == {"heading": "Fixed", "content": "Revised text"}
        assert caller._client.messages.create.call_count == 2

    @pytest.mark.asyncio
    async def test_retry_on_malformed_json(self):
        """When first response has broken JSON, retry should recover."""
        caller = _make_caller()
        broken_resp = _make_response('{"heading": "Test", "content": "missing quote}')
        fixed_resp = _make_response('{"heading": "Test", "content": "fixed"}')

        caller._client.messages.create = AsyncMock(
            side_effect=[broken_resp, fixed_resp]
        )
        data, usage = await caller.call_json(**_CALL_KWARGS)
        assert data["content"] == "fixed"
        assert caller._client.messages.create.call_count == 2

    @pytest.mark.asyncio
    async def test_retry_message_contains_failed_output(self):
        """Retry call should include the failed response as assistant context."""
        caller = _make_caller()
        bad_text = "Not JSON at all, just thinking out loud."
        good_json = '{"heading": "ok", "content": "ok"}'

        caller._client.messages.create = AsyncMock(
            side_effect=[
                _make_response(bad_text),
                _make_response(good_json),
            ]
        )
        await caller.call_json(**_CALL_KWARGS)

        # Second call should have the failed text as assistant message
        retry_call = caller._client.messages.create.call_args_list[1]
        retry_messages = retry_call.kwargs["messages"]

        # Original user message + assistant (failed) + user (retry instruction)
        assert len(retry_messages) == 3
        assert retry_messages[1]["role"] == "assistant"
        assert retry_messages[1]["content"] == bad_text
        assert retry_messages[2]["role"] == "user"
        assert "not valid JSON" in retry_messages[2]["content"]

    @pytest.mark.asyncio
    async def test_retry_uses_temperature_zero(self):
        """Retry call should force temperature=0.0 for determinism."""
        caller = _make_caller()
        caller._client.messages.create = AsyncMock(
            side_effect=[
                _make_response("not json"),
                _make_response('{"ok": true}'),
            ]
        )
        # Original call uses temperature=0.5
        kwargs = {**_CALL_KWARGS, "temperature": 0.5}
        await caller.call_json(**kwargs)

        retry_call = caller._client.messages.create.call_args_list[1]
        assert retry_call.kwargs["temperature"] == 0.0


# --- Tests: token usage merging ---


class TestCallJsonTokenMerge:
    @pytest.mark.asyncio
    async def test_combined_usage_on_retry(self):
        """Token usage from both attempts should be merged."""
        caller = _make_caller()
        resp1 = _make_response("not json", input_tokens=500, output_tokens=200)
        resp2 = _make_response('{"ok": true}', input_tokens=600, output_tokens=100)

        caller._client.messages.create = AsyncMock(side_effect=[resp1, resp2])
        data, usage = await caller.call_json(**_CALL_KWARGS)

        assert data == {"ok": True}
        assert usage["input_tokens"] == 1100  # 500 + 600
        assert usage["output_tokens"] == 300  # 200 + 100
        assert usage["agent"] == "test-agent"  # Original agent name, not _json_retry

    @pytest.mark.asyncio
    async def test_no_retry_preserves_single_usage(self):
        """When no retry is needed, usage is from single call only."""
        caller = _make_caller()
        caller._client.messages.create = AsyncMock(
            return_value=_make_response(
                '{"ok": true}', input_tokens=300, output_tokens=50
            )
        )
        data, usage = await caller.call_json(**_CALL_KWARGS)
        assert usage["input_tokens"] == 300
        assert usage["output_tokens"] == 50


# --- Tests: failure after retry ---


class TestCallJsonRetryExhausted:
    @pytest.mark.asyncio
    async def test_raises_after_both_attempts_fail(self):
        """If both attempts return unparseable output, raise ValueError."""
        caller = _make_caller()
        caller._client.messages.create = AsyncMock(
            side_effect=[
                _make_response("totally not json"),
                _make_response("still not json"),
            ]
        )
        with pytest.raises(ValueError, match="after retry"):
            await caller.call_json(**_CALL_KWARGS)

    @pytest.mark.asyncio
    async def test_truncation_detection_on_retry(self):
        """Truncated JSON on retry should report truncation error."""
        caller = _make_caller()
        # First: prose. Second: truncated JSON (no closing brace)
        caller._client.messages.create = AsyncMock(
            side_effect=[
                _make_response("not json"),
                _make_response('{"heading": "Test", "content": "cut off here'),
            ]
        )
        with pytest.raises(ValueError, match="Truncated JSON"):
            await caller.call_json(**_CALL_KWARGS)

    @pytest.mark.asyncio
    async def test_error_message_includes_both_attempts(self):
        """Error message should show raw output from both attempts."""
        caller = _make_caller()
        caller._client.messages.create = AsyncMock(
            side_effect=[
                _make_response("first bad output"),
                _make_response("second bad output"),
            ]
        )
        with pytest.raises(ValueError) as exc_info:
            await caller.call_json(**_CALL_KWARGS)

        msg = str(exc_info.value)
        assert "attempt 1" in msg
        assert "attempt 2" in msg
        assert "first bad" in msg
        assert "second bad" in msg
