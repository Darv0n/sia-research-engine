"""AgentCaller — Anthropic SDK wrapper with token tracking and rate limiting."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone

import anthropic

from deep_research_swarm.contracts import TokenUsage

# Pricing per million tokens (as of 2025)
_PRICING: dict[str, dict[str, float]] = {
    "claude-opus-4-6": {"input": 15.0, "output": 75.0},
    "claude-sonnet-4-6": {"input": 3.0, "output": 15.0},
}


class AgentCaller:
    """Wraps Anthropic API calls with token tracking and concurrency control."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        max_concurrent: int = 5,
        max_retries: int = 3,
    ) -> None:
        self.model = model
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

            except anthropic.RateLimitError:
                wait = 2 ** (attempt + 1)
                await asyncio.sleep(wait)
                last_error = "rate_limit"
            except anthropic.APIError as e:
                last_error = str(e)
                if attempt < self._max_retries - 1:
                    await asyncio.sleep(1)

        raise RuntimeError(f"AgentCaller failed after {self._max_retries} retries: {last_error}")

    def _track_usage(self, response, agent_name: str) -> TokenUsage:
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens

        pricing = _PRICING.get(self.model, {"input": 3.0, "output": 15.0})
        cost = (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000

        usage = TokenUsage(
            agent=agent_name,
            model=self.model,
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
        """Make an API call expecting JSON response. Parses the response."""
        text, usage = await self.call(
            system=system,
            messages=messages,
            agent_name=agent_name,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        # Try to extract JSON from response (handles ```json blocks)
        cleaned = text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            # Remove first and last lines (``` markers)
            lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            cleaned = "\n".join(lines)

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON from {agent_name}: {e}\nRaw: {text[:500]}")

        return data, usage
