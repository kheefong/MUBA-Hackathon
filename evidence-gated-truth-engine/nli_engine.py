"""
nli_engine.py — Step 4: build the pairwise contradiction/entailment matrix
across propositions sharing a canonical key but coming from different
models. Also reused by evidence_retriever.py for evidence<->proposition
entailment (SUPPORT/CONTRADICT/NOT_COVERED framing).
"""
from __future__ import annotations

import asyncio
import itertools
import logging
from dataclasses import dataclass
from functools import lru_cache

from config import NLI_BACKEND, UTILITY_MODEL
from llm_client import LLMCallError, call_model_json
from proposition_extractor import Proposition

logger = logging.getLogger("truth_engine.nli_engine")

RELATION_SCORE = {"ENTAILMENT": 1.0, "NEUTRAL": 0.5, "CONTRADICTION": 0.0}

SYSTEM_PROMPT = "You are a precise natural-language-inference engine. Return JSON only."

USER_PROMPT_TEMPLATE = """\
Given Proposition A and Proposition B, decide if A entails B, contradicts B, or is neutral to B.

Proposition A: {prop_a}
Proposition B: {prop_b}

Return JSON:
{{
  "relation": "ENTAILMENT" | "CONTRADICTION" | "NEUTRAL",
  "explanation": "one sentence"
}}
"""


@dataclass
class PairRelation:
    prop_a: Proposition
    prop_b: Proposition
    relation: str
    score: float
    explanation: str = ""


# ---------------------------------------------------------------------------
# Backend A (default): prompt-based NLI via the utility LLM. Zero extra deps.
# ---------------------------------------------------------------------------
async def _llm_entailment(text_a: str, text_b: str) -> tuple[str, str]:
    try:
        raw = await call_model_json(
            UTILITY_MODEL,
            SYSTEM_PROMPT,
            USER_PROMPT_TEMPLATE.format(prop_a=text_a, prop_b=text_b),
            max_tokens=200,
        )
    except LLMCallError as e:
        logger.warning("NLI call failed, defaulting to NEUTRAL: %s", e)
        return "NEUTRAL", str(e)

    if not isinstance(raw, dict):
        return "NEUTRAL", "Malformed NLI output."
    relation = str(raw.get("relation", "NEUTRAL")).upper()
    if relation not in RELATION_SCORE:
        relation = "NEUTRAL"
    return relation, str(raw.get("explanation", ""))


# ---------------------------------------------------------------------------
# Backend B (optional): HuggingFace DeBERTa-v3-large-mnli, loaded lazily so
# `transformers`/`torch` are only required if NLI_BACKEND=hf.
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _get_hf_pipeline():
    from transformers import pipeline  # local import: optional dependency

    from config import HF_NLI_MODEL

    return pipeline("text-classification", model=HF_NLI_MODEL, top_k=None)


def _hf_entailment(text_a: str, text_b: str) -> tuple[str, str]:
    clf = _get_hf_pipeline()
    result = clf(f"{text_a} [SEP] {text_b}")
    # HF DeBERTa-mnli labels are typically ENTAILMENT/NEUTRAL/CONTRADICTION.
    scores = {r["label"].upper(): r["score"] for r in result[0]}
    best_label = max(scores, key=scores.get)
    if best_label not in RELATION_SCORE:
        best_label = "NEUTRAL"
    return best_label, f"HF model score={scores.get(best_label, 0):.3f}"


async def entailment(text_a: str, text_b: str) -> tuple[str, str]:
    """Returns (relation, explanation). Relation in RELATION_SCORE.keys()."""
    if NLI_BACKEND == "hf":
        try:
            return await asyncio.to_thread(_hf_entailment, text_a, text_b)
        except Exception as e:  # noqa: BLE001 - fall back gracefully
            logger.warning("HF NLI backend failed (%s), falling back to LLM.", e)
    return await _llm_entailment(text_a, text_b)


async def build_contradiction_matrix(propositions: list[Proposition]) -> list[PairRelation]:
    """Pairs propositions that share a canonical key but come from
    different models, and scores their relation.
    """
    by_key: dict[str, list[Proposition]] = {}
    for p in propositions:
        by_key.setdefault(p.canonical_key, []).append(p)

    pairs: list[tuple[Proposition, Proposition]] = []
    for _key, group in by_key.items():
        for a, b in itertools.combinations(group, 2):
            if a.source_model != b.source_model:
                pairs.append((a, b))

    if not pairs:
        return []

    async def _score_pair(a: Proposition, b: Proposition) -> PairRelation:
        relation, explanation = await entailment(a.proposition, b.proposition)
        return PairRelation(a, b, relation, RELATION_SCORE[relation], explanation)

    results = await asyncio.gather(*(_score_pair(a, b) for a, b in pairs))
    return list(results)
