"""
claim_classifier.py — Step 1 of the pipeline: decide whether a claim is
even worth running through the full evidence-gated engine.
"""
from __future__ import annotations

from dataclasses import dataclass

from config import UTILITY_MODEL
from llm_client import LLMCallError, call_model_json

SYSTEM_PROMPT = "You are a claim classifier for a fact-verification system. Return JSON only."

USER_PROMPT_TEMPLATE = """\
Classify the claim into one of:
- FACTUAL_CHECKABLE: A claim about past or present states of the world that can be verified with evidence.
- PREDICTION: A claim about the future.
- OPINION: A normative, subjective, or value-based statement.
- AMBIGUOUS: Unclear reference, vague terms, or cannot be resolved without additional context.
- MIXED: Contains both checkable and non-checkable components.

Claim: "{claim_text}"

Return JSON:
{{
  "category": "FACTUAL_CHECKABLE" | "PREDICTION" | "OPINION" | "AMBIGUOUS" | "MIXED",
  "checkable_part": "extracted checkable sub-claim if MIXED, else null",
  "reason": "one-sentence explanation"
}}
"""

VALID_CATEGORIES = {"FACTUAL_CHECKABLE", "PREDICTION", "OPINION", "AMBIGUOUS", "MIXED"}


@dataclass
class ClassificationResult:
    category: str
    checkable_part: str | None
    reason: str

    @property
    def should_run_engine(self) -> bool:
        return self.category in ("FACTUAL_CHECKABLE", "MIXED")

    @property
    def text_to_verify(self) -> str | None:
        if self.category == "FACTUAL_CHECKABLE":
            return None  # caller should use the original claim text
        if self.category == "MIXED":
            return self.checkable_part
        return None


async def classify_claim(claim_text: str) -> ClassificationResult:
    try:
        raw = await call_model_json(
            UTILITY_MODEL,
            SYSTEM_PROMPT,
            USER_PROMPT_TEMPLATE.format(claim_text=claim_text.replace('"', "'")),
            max_tokens=300,
        )
    except LLMCallError as e:
        # If the classifier itself fails, fail safe: treat as ambiguous
        # rather than silently running the (expensive) full pipeline.
        return ClassificationResult(
            category="AMBIGUOUS",
            checkable_part=None,
            reason=f"Classifier call failed: {e}",
        )

    if not isinstance(raw, dict):
        return ClassificationResult("AMBIGUOUS", None, "Malformed classifier output.")

    category = raw.get("category", "AMBIGUOUS")
    if category not in VALID_CATEGORIES:
        category = "AMBIGUOUS"

    return ClassificationResult(
        category=category,
        checkable_part=raw.get("checkable_part"),
        reason=raw.get("reason", ""),
    )
