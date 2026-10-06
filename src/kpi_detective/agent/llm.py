"""LLM clients with token and cost accounting.

* ``anthropic``: Claude via the Anthropic API (``ANTHROPIC_API_KEY``).
* ``ollama``: any local model through Ollama's OpenAI-compatible endpoint,
  for cheap development and the cost comparison (PRD section 10).

Both run at temperature 0 for reproducibility.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Protocol

#: USD per million tokens (input, output). Check your provider's current
#: price list before quoting costs; override with --price-in/--price-out.
PRICES: dict[str, tuple[float, float]] = {
    "claude-haiku-4-5-20251001": (1.0, 5.0),
}


@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0
    price_in: float | None = None
    price_out: float | None = None
    by_node: dict[str, int] = field(default_factory=dict)

    @property
    def cost_usd(self) -> float | None:
        if self.price_in is None or self.price_out is None:
            return None
        return round((self.input_tokens * self.price_in + self.output_tokens * self.price_out) / 1e6, 6)

    def to_dict(self) -> dict:
        return {
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "llm_seconds": round(self.seconds, 3),
            "cost_usd": self.cost_usd,
        }


class LLMClient(Protocol):
    model: str
    usage: Usage

    def complete(self, system: str, user: str, node: str = "") -> str: ...


class LLMError(RuntimeError):
    pass


class AnthropicClient:
    def __init__(
        self,
        model: str | None = None,
        max_tokens: int = 2500,
        price_in: float | None = None,
        price_out: float | None = None,
    ):
        import anthropic

        self.model = model or os.environ.get("KPI_LLM_MODEL", "claude-sonnet-5-5")
        self.max_tokens = max_tokens
        self._client = anthropic.Anthropic()
        p = PRICES.get(self.model, (None, None))
        self.usage = Usage(
            price_in=price_in if price_in is not None else p[0], price_out=price_out if price_out is not None else p[1]
        )

    def complete(self, system: str, user: str, node: str = "") -> str:
        t0 = time.perf_counter()
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=0,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        self.usage.calls += 1
        self.usage.input_tokens += msg.usage.input_tokens
        self.usage.output_tokens += msg.usage.output_tokens
        self.usage.seconds += time.perf_counter() - t0
        self.usage.by_node[node] = self.usage.by_node.get(node, 0) + 1
        return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")


class OllamaClient:
    """OpenAI-compatible endpoint (Ollama by default)."""

    def __init__(self, model: str | None = None, base_url: str | None = None, max_tokens: int = 2500):
        from openai import OpenAI

        self.model = model or os.environ.get("KPI_LLM_MODEL", "qwen2.5:14b")
        self.max_tokens = max_tokens
        self._client = OpenAI(
            base_url=base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"), api_key="ollama"
        )
        self.usage = Usage(price_in=0.0, price_out=0.0)  # local: no API cost

    def complete(self, system: str, user: str, node: str = "") -> str:
        t0 = time.perf_counter()
        resp = self._client.chat.completions.create(
            model=self.model,
            temperature=0,
            max_tokens=self.max_tokens,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        self.usage.calls += 1
        if resp.usage:
            self.usage.input_tokens += resp.usage.prompt_tokens
            self.usage.output_tokens += resp.usage.completion_tokens
        self.usage.seconds += time.perf_counter() - t0
        self.usage.by_node[node] = self.usage.by_node.get(node, 0) + 1
        return resp.choices[0].message.content or ""


def make_client(provider: str, model: str | None = None, **kw) -> LLMClient:
    if provider == "anthropic":
        return AnthropicClient(model, **kw)
    if provider == "ollama":
        return OllamaClient(model)
    raise ValueError(f"unknown provider {provider!r} (use anthropic or ollama)")


_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def parse_json(text: str) -> dict:
    """Extract the first JSON object from a model response."""
    m = _FENCE.search(text)
    candidate = m.group(1) if m else text
    start = candidate.find("{")
    if start < 0:
        raise LLMError(f"no JSON object in response: {text[:200]!r}")
    depth, in_str, esc = 0, False, False
    for i, ch in enumerate(candidate[start:], start):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(candidate[start : i + 1])
    raise LLMError("unterminated JSON object in response")
