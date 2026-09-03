"""
model_interrogator.py — Step 2: send the claim to all three models in
parallel and normalize their responses.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

from config import ALL_MODELS, ModelConfig
from llm_client import LLMCallError, call_model_json

logger = logging.getLogger("truth_engine.model_interrogator")

SYSTEM_PROMPT = "You are a fact-checking assistant. Return JSON only, no prose outside the JSON."

USER_PROMPT_TEMPLATE = """\
Claim: "{claim_text}"

Provide:
1. Verdict: TRUE / FALSE / NOT_ENOUGH_INFO
2. Confidence: a number from 0 to 100 indicating how certain you are.
3. Reasoning: a step-by-step explanation, max 5 sentences.
4. Cited evidence: list any sources or factual premises you used.

Return JSON only:
{{
  "verdict": "TRUE",
  "confidence": 87,
  "reasoning": "...",
  "cited_evidence": ["...", "..."]
}}
"""

VALID_VERDICTS = {"TRUE", "FALSE", "NOT_ENOUGH_INFO"}


@dataclass
class ModelResponse:
    model: ModelConfig
    verdict: str
    confidence: float  # 0-100 raw, as returned by the model
    reasoning: str
    cited_evidence: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


def _parse_response(model: ModelConfig, raw: dict) -> ModelResponse:
    verdict = str(raw.get("verdict", "NOT_ENOUGH_INFO")).upper()
    if verdict not in VALID_VERDICTS:
        verdict = "NOT_ENOUGH_INFO"

    try:
        confidence = float(raw.get("confidence", 0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(100.0, confidence))

    cited = raw.get("cited_evidence") or []
    if isinstance(cited, str):
        cited = [cited]

    return ModelResponse(
        model=model,
        verdict=verdict,
        confidence=confidence,
        reasoning=str(raw.get("reasoning", "")),
        cited_evidence=[str(c) for c in cited],
    )


async def _interrogate_one(model: ModelConfig, claim_text: str) -> ModelResponse:
    try:
        raw = await call_model_json(
            model,
            SYSTEM_PROMPT,
            USER_PROMPT_TEMPLATE.format(claim_text=claim_text.replace('"', "'")),
            max_tokens=600,
        )
        if not isinstance(raw, dict):
            raise ValueError("Expected a JSON object from model.")
        return _parse_response(model, raw)
    except (LLMCallError, ValueError) as e:
        logger.error("Interrogation of %s failed: %s", model.name, e)
        return ModelResponse(
            model=model,
            verdict="NOT_ENOUGH_INFO",
            confidence=0.0,
            reasoning="",
            cited_evidence=[],
            error=str(e),
        )


async def interrogate_all(claim_text: str, models: list[ModelConfig] | None = None) -> list[ModelResponse]:
    """Fan out the claim to every configured model concurrently."""
    models = models or ALL_MODELS
    results = await asyncio.gather(*(_interrogate_one(m, claim_text) for m in models))
    return list(results)
