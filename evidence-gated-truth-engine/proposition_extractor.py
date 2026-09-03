"""
proposition_extractor.py — Step 3: turn each model's free-text reasoning +
cited evidence into atomic, canonicalized factual propositions.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass

from config import UTILITY_MODEL
from llm_client import LLMCallError, call_model_json
from model_interrogator import ModelResponse

SYSTEM_PROMPT = (
    "You extract atomic factual propositions from text for a fact-checking "
    "pipeline. Return JSON only."
)

USER_PROMPT_TEMPLATE = """\
Extract every atomic factual proposition from the text below.
An atomic proposition is a single checkable statement about one entity, property, time, or relationship.

Rules:
- Normalize dates to ISO format (YYYY-MM-DD).
- Normalize numbers with units (e.g., "RM2,000" -> 2000 MYR).
- Resolve pronouns to explicit entities.
- Split compound claims into separate propositions.

Text:
{text}

Return JSON list:
[
  {{
    "proposition": "The minimum wage in Malaysia is 1700 MYR per month as of 2026-01-01.",
    "subject": "minimum wage Malaysia",
    "predicate": "is_equal_to",
    "object": "1700 MYR/month",
    "time": "2026-01-01"
  }}
]
"""


@dataclass
class Proposition:
    proposition: str
    subject: str
    predicate: str
    object: str
    time: str | None
    source_model: str

    @property
    def canonical_key(self) -> str:
        norm = lambda s: (s or "").strip().lower()
        return f"{norm(self.subject)}|{norm(self.predicate)}|{norm(self.object)}|{norm(self.time)}"


def _parse_propositions(raw: object, source_model: str) -> list[Proposition]:
    if not isinstance(raw, list):
        return []
    props: list[Proposition] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        text = item.get("proposition")
        if not text:
            continue
        props.append(
            Proposition(
                proposition=str(text),
                subject=str(item.get("subject", "")),
                predicate=str(item.get("predicate", "")),
                object=str(item.get("object", "")),
                time=item.get("time"),
                source_model=source_model,
            )
        )
    return props


async def _extract_for_response(response: ModelResponse) -> list[Proposition]:
    if not response.ok or not (response.reasoning or response.cited_evidence):
        return []
    text = response.reasoning + "\n" + "\n".join(response.cited_evidence)
    try:
        raw = await call_model_json(
            UTILITY_MODEL,
            SYSTEM_PROMPT,
            USER_PROMPT_TEMPLATE.format(text=text),
            max_tokens=1200,
        )
    except LLMCallError:
        return []
    return _parse_propositions(raw, response.model.name)


async def extract_all_propositions(responses: list[ModelResponse]) -> list[Proposition]:
    """Extract propositions for every model response concurrently."""
    results = await asyncio.gather(*(_extract_for_response(r) for r in responses))
    props: list[Proposition] = []
    for group in results:
        props.extend(group)
    return props
