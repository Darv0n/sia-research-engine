"""AgentCaller — Anthropic SDK wrapper with token tracking and rate limiting."""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime, timezone

import anthropic
from anthropic._exceptions import OverloadedError

from deep_research_swarm.contracts import TokenUsage

# Pricing per million tokens (as of 2025)
_PRICING: dict[str, dict[str, float]] = {
    "claude-opus-4-6": {"input": 15.0, "output": 75.0},
    "claude-sonnet-4-6": {"input": 3.0, "output": 15.0},
}


def _extract_json(text: str) -> str:
    """Extract JSON from model response, handling fenced blocks and prose wrapping."""
    cleaned = text.strip()

    # Handle ```json fenced blocks
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = lines[1:]  # Remove opening fence
        json_lines = []
        for line in lines:
            if line.strip() == "```":
                break
            json_lines.append(line)
        cleaned = "\n".join(json_lines).strip()

    # If it already looks like JSON, return as-is
    if cleaned.startswith("{") or cleaned.startswith("["):
        return cleaned

    # Try to find a JSON object embedded in prose
    start = cleaned.find("{")
    if start != -1:
        # Find the matching closing brace by tracking depth
        depth = 0
        for i in range(start, len(cleaned)):
            if cleaned[i] == "{":
                depth += 1
            elif cleaned[i] == "}":
                depth -= 1
                if depth == 0:
                    return cleaned[start : i + 1]
        # No matching close — return from start (will trigger truncation error)
        return cleaned[start:]

    # No JSON found at all — return empty string (will trigger parse error)
    return ""


class AgentCaller:
    """Wraps Anthropic API calls with token tracking and concurrency control."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        max_concurrent: int = 5,
        max_retries: int = 3,
        fallback_model: str | None = None,
    ) -> None:
        self.model = model
        self.fallback_model = fallback_model
        self._client = anthropic.AsyncAnthropic(api_key=api_key)
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._max_retries = max_retries
        self._usage_log: list[TokenUsage] = []

    async def call(
        self,
        *,
        system: str,
        messages: list[dict],
        agent_name: str,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> tuple[str, TokenUsage]:
        """Make an API call with retry and token tracking.

        Returns (response_text, token_usage).
        """
        async with self._semaphore:
            return await self._call_with_retry(
                system=system,
                messages=messages,
                agent_name=agent_name,
                max_tokens=max_tokens,
                temperature=temperature,
            )

    async def _call_with_retry(
        self,
        *,
        system: str,
        messages: list[dict],
        agent_name: str,
        max_tokens: int,
        temperature: float,
    ) -> tuple[str, TokenUsage]:
        last_error = None
        overloaded = False

        for attempt in range(self._max_retries):
            try:
                response = await self._client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=system,
                    messages=messages,
                )

                text = ""
                for block in response.content:
                    if block.type == "text":
                        text += block.text

                usage = self._track_usage(response, agent_name)
                return text, usage

            except OverloadedError:
                overloaded = True
                wait = 2 ** (attempt + 1)
                print(
                    f"WARNING: {self.model} overloaded (529), "
                    f"retry {attempt + 1}/{self._max_retries} in {wait}s",
                    file=sys.stderr,
                )
                await asyncio.sleep(wait)
                last_error = "overloaded"
            except anthropic.RateLimitError:
                wait = 2 ** (attempt + 1)
                await asyncio.sleep(wait)
                last_error = "rate_limit"
            except anthropic.APIError as e:
                last_error = str(e)
                if attempt < self._max_retries - 1:
                    await asyncio.sleep(1)

        # Fallback: if overloaded and a fallback model is configured, try once
        if overloaded and self.fallback_model:
            print(
                f"WARNING: {self.model} exhausted retries, "
                f"falling back to {self.fallback_model} for {agent_name}",
                file=sys.stderr,
            )
            try:
                response = await self._client.messages.create(
                    model=self.fallback_model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=system,
                    messages=messages,
                )

                text = ""
                for block in response.content:
                    if block.type == "text":
                        text += block.text

                usage = self._track_usage(response, agent_name, model_override=self.fallback_model)
                return text, usage
            except anthropic.APIError as e:
                last_error = f"fallback ({self.fallback_model}) also failed: {e}"

        raise RuntimeError(f"AgentCaller failed after {self._max_retries} retries: {last_error}")

    def _track_usage(
        self, response, agent_name: str, *, model_override: str | None = None
    ) -> TokenUsage:
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens

        model = model_override or self.model
        pricing = _PRICING.get(model, {"input": 3.0, "output": 15.0})
        cost = (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000

        usage = TokenUsage(
            agent=agent_name,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=round(cost, 6),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        self._usage_log.append(usage)
        return usage

    @property
    def total_tokens(self) -> int:
        return sum(u["input_tokens"] + u["output_tokens"] for u in self._usage_log)

    @property
    def total_cost(self) -> float:
        return sum(u["cost_usd"] for u in self._usage_log)

    async def call_json(
        self,
        *,
        system: str,
        messages: list[dict],
        agent_name: str,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> tuple[dict, TokenUsage]:
        """Make an API call expecting JSON response. Retries once on parse failure."""
        text, usage = await self.call(
            system=system,
            messages=messages,
            agent_name=agent_name,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        # Try to extract JSON from response
        cleaned = _extract_json(text)

        try:
            data = json.loads(cleaned)
            return data, usage
        except json.JSONDecodeError:
            pass  # Fall through to retry

        # --- Single retry: feed the failed output back and demand strict JSON ---
        retry_messages = messages + [
            {"role": "assistant", "content": text},
            {
                "role": "user",
                "content": (
                    "Your previous response was not valid JSON. "
                    "Return ONLY the JSON object — no commentary, no markdown fences, "
                    "no explanation. Start with { and end with }."
                ),
            },
        ]
        text2, usage2 = await self.call(
            system=system,
            messages=retry_messages,
            agent_name=f"{agent_name}_json_retry",
            max_tokens=max_tokens,
            temperature=0.0,
        )

        # Merge token usage
        combined_usage = TokenUsage(
            agent=agent_name,
            model=usage["model"],
            input_tokens=usage["input_tokens"] + usage2["input_tokens"],
            output_tokens=usage["output_tokens"] + usage2["output_tokens"],
            cost_usd=round(usage["cost_usd"] + usage2["cost_usd"], 6),
            timestamp=usage2["timestamp"],
        )

        cleaned2 = _extract_json(text2)
        try:
            data = json.loads(cleaned2)
            return data, combined_usage
        except json.JSONDecodeError as e:
            # Detect truncated responses (JSON cut off mid-stream)
            stripped = cleaned2.rstrip()
            if stripped and stripped[-1] not in ("}", "]"):
                raise ValueError(
                    f"Truncated JSON from {agent_name} (likely hit max_tokens). "
                    f"Response ends at char {len(cleaned2)}. "
                    f"Increase max_tokens for this agent.\nRaw tail: ...{text2[-200:]}"
                )
            raise ValueError(
                f"Failed to parse JSON from {agent_name} after retry: {e}\n"
                f"Raw (attempt 1): {text[:300]}\n"
                f"Raw (attempt 2): {text2[:300]}"
            )
