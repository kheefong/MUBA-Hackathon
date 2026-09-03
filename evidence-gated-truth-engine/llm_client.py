"""
llm_client.py — Thin, uniform async client over the three configured models.

All three providers (DeepSeek, Kimi, MiniMax) expose OpenAI-compatible
`/chat/completions` endpoints, so a single adapter covers them. If a
provider's API diverges, add a branch keyed on `ModelConfig.api_style`.
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

import httpx

from config import LLM_MAX_RETRIES, LLM_TEMPERATURE, LLM_TIMEOUT_SECONDS, ModelConfig

logger = logging.getLogger("truth_engine.llm_client")


class LLMCallError(RuntimeError):
    """Raised when a model call fails after all retries."""


class MissingAPIKeyError(LLMCallError):
    """Raised when the environment variable for a model's API key is unset."""


def _extract_json(text: str) -> Any:
    """Best-effort extraction of a JSON object/array from an LLM response.

    Handles the common cases: pure JSON, JSON wrapped in ```json fences,
    or JSON embedded in surrounding prose.
    """
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fallback: find the first balanced {...} or [...] block.
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start = text.find(open_ch)
        if start == -1:
            continue
        depth = 0
        for i in range(start, len(text)):
            if text[i] == open_ch:
                depth += 1
            elif text[i] == close_ch:
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break
    raise ValueError(f"Could not extract JSON from model output: {text[:200]!r}")


async def call_model_json(
    model: ModelConfig,
    system_prompt: str,
    user_prompt: str,
    *,
    max_tokens: int = 1024,
) -> dict | list:
    """Call `model` and parse its reply as JSON.

    Raises MissingAPIKeyError if the relevant env var isn't set, and
    LLMCallError for network/parse failures after retries are exhausted.
    """
    api_key = os.getenv(model.env_key)
    if not api_key:
        raise MissingAPIKeyError(
            f"Environment variable {model.env_key} is not set for {model.name}."
        )

    payload = {
        "model": model.model_id,
        "temperature": LLM_TEMPERATURE,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    last_err: Exception | None = None
    async with httpx.AsyncClient(timeout=LLM_TIMEOUT_SECONDS) as client:
        for attempt in range(LLM_MAX_RETRIES + 1):
            try:
                resp = await client.post(
                    f"{model.base_url.rstrip('/')}/chat/completions",
                    json=payload,
                    headers=headers,
                )
                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                return _extract_json(content)
            except (httpx.HTTPError, KeyError, IndexError, ValueError) as e:
                last_err = e
                logger.warning(
                    "Call to %s failed (attempt %d/%d): %s",
                    model.name, attempt + 1, LLM_MAX_RETRIES + 1, e,
                )

    raise LLMCallError(f"All calls to {model.name} failed: {last_err}") from last_err
