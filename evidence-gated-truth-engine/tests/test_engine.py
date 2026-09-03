"""
test_engine.py — End-to-end scenario tests (spec: at least 5 claims) plus
focused unit tests for the scoring formulas.

External calls (LLM + search) are monkeypatched with deterministic canned
responses so the suite runs without real API keys or network access.
"""
from __future__ import annotations

import asyncio

import pytest

import calibration
import config
import evidence_retriever
import llm_client
import scoring
from model_interrogator import ModelResponse
from nli_engine import PairRelation
from proposition_extractor import Proposition

# asyncio_mode = auto (see pytest.ini) handles async test functions
# automatically; no module-level pytestmark needed, since it would
# incorrectly tag the synchronous unit tests below too.

# ---------------------------------------------------------------------------
# Scenario registry: each scenario configures what the mocked LLM/search
# calls return, keyed by a marker string in the claim text.
# ---------------------------------------------------------------------------
SCENARIO_MARKER = "__scenario__"


def _classifier_reply(category: str, checkable_part: str | None = None) -> dict:
    return {"category": category, "checkable_part": checkable_part, "reason": "test fixture"}


def _interrogation_reply(verdict: str, confidence: float) -> dict:
    return {
        "verdict": verdict,
        "confidence": confidence,
        "reasoning": "Test reasoning.",
        "cited_evidence": ["Ministry press release states the figure."],
    }


def _proposition_reply(value: str) -> list[dict]:
    return [
        {
            "proposition": f"The value is {value}.",
            "subject": "test subject",
            "predicate": "is_equal_to",
            "object": value,
            "time": "2026-01-01",
        }
    ]


async def _fake_call_model_json_factory(scenario: str):
    async def _fake_call_model_json(model, system_prompt, user_prompt, max_tokens=1024):
        # --- Claim classifier ---
        if "claim classifier" in system_prompt.lower():
            if scenario == "opinion":
                return _classifier_reply("OPINION")
            if scenario == "prediction":
                return _classifier_reply("PREDICTION")
            if scenario == "mixed":
                return _classifier_reply("MIXED", checkable_part="Country X raised its minimum wage in 2026.")
            return _classifier_reply("FACTUAL_CHECKABLE")

        # --- Multi-model interrogation ---
        if "fact-checking assistant" in system_prompt.lower():
            if scenario in ("agree_supported", "mixed"):
                return _interrogation_reply("TRUE", 90)
            if scenario == "agree_contradicted":
                return _interrogation_reply("TRUE", 88)
            return _interrogation_reply("TRUE", 80)

        # --- Proposition extraction ---
        if "atomic factual proposition" in system_prompt.lower():
            if scenario == "agree_contradicted":
                return _proposition_reply("2000 MYR/month")
            return _proposition_reply("1700 MYR/month")

        # --- NLI entailment (model vs model) ---
        if "natural-language-inference" in system_prompt.lower():
            return {"relation": "ENTAILMENT", "explanation": "Both state the same figure."}

        # --- Evidence entailment (evidence vs proposition) ---
        if "support" in system_prompt.lower() and "contradict" in system_prompt.lower():
            if scenario == "agree_supported" or scenario == "mixed":
                return {"relation": "SUPPORT", "explanation": "Matches official source."}
            if scenario == "agree_contradicted":
                return {"relation": "CONTRADICT", "explanation": "Official source states a different figure."}
            return {"relation": "NOT_COVERED", "explanation": "No matching evidence."}

        raise AssertionError(f"Unexpected prompt in scenario {scenario}: {system_prompt[:80]}")

    return _fake_call_model_json


async def _fake_search_factory(scenario: str):
    async def _fake_search(client, query):
        if scenario in ("agree_supported", "agree_contradicted", "mixed"):
            return [
                {
                    "title": "Ministry of Human Resources Order",
                    "url": "https://www.mohr.gov.my/order-2026",
                    "text": "The official order text.",
                }
            ]
        return []  # no_evidence scenarios (opinion/prediction never reach retrieval)

    return _fake_search


@pytest.fixture
def patch_scenario(monkeypatch):
    async def _apply(scenario: str):
        monkeypatch.setattr(llm_client, "call_model_json", await _fake_call_model_json_factory(scenario))
        # Also patch the name imported into each module (they do `from llm_client import call_model_json`)
        for mod in ("claim_classifier", "model_interrogator", "proposition_extractor", "nli_engine", "evidence_retriever"):
            import importlib

            m = importlib.import_module(mod)
            if hasattr(m, "call_model_json"):
                monkeypatch.setattr(m, "call_model_json", await _fake_call_model_json_factory(scenario))
        monkeypatch.setattr(evidence_retriever, "_search", await _fake_search_factory(scenario))
        # Force a search key so retrieval doesn't short-circuit.
        monkeypatch.setattr(config, "SEARCH_API_KEY", "test-search-key")
        monkeypatch.setattr(evidence_retriever, "SEARCH_API_KEY", "test-search-key")

    return _apply


# ---------------------------------------------------------------------------
# Scenario 1: all models agree, evidence supports -> high truth & consensus
# ---------------------------------------------------------------------------
async def test_scenario_agree_and_supported(patch_scenario):
    await patch_scenario("agree_supported")
    from pipeline import verify_claim

    result = await verify_claim("Malaysia raised its minimum wage to RM1700 in 2026.")
    assert result["status"] == "CHECKED"
    assert result["truth_score"] is not None and result["truth_score"] > 0.6
    assert result["consensus_score"] > 0.6
    assert result["verdict"] == "TRUE"


# ---------------------------------------------------------------------------
# Scenario 2: all models agree, evidence contradicts -> low truth, high consensus
# ---------------------------------------------------------------------------
async def test_scenario_agree_but_contradicted(patch_scenario):
    await patch_scenario("agree_contradicted")
    from pipeline import verify_claim

    result = await verify_claim("Malaysia raised its minimum wage to RM2000 in 2026.")
    assert result["status"] == "CHECKED"
    assert result["truth_score"] == 0.0
    assert result["consensus_score"] > 0.6  # models still agreed with each other
    assert result["evidence_alignment"] == 0.0


# ---------------------------------------------------------------------------
# Scenario 3: opinion -> NOT_VERIFIABLE
# ---------------------------------------------------------------------------
async def test_scenario_opinion(patch_scenario):
    await patch_scenario("opinion")
    from pipeline import verify_claim

    result = await verify_claim("Malaysia's minimum wage policy is unfair.")
    assert result["status"] == "NOT_VERIFIABLE"
    assert result["truth_score"] is None


# ---------------------------------------------------------------------------
# Scenario 4: prediction -> NOT_VERIFIABLE (spec calls this UNVERIFIABLE due
# to future tense; our classifier maps PREDICTION -> NOT_VERIFIABLE status,
# consistent with "stop and return NOT_VERIFIABLE" for non-checkable claims)
# ---------------------------------------------------------------------------
async def test_scenario_prediction(patch_scenario):
    await patch_scenario("prediction")
    from pipeline import verify_claim

    result = await verify_claim("Malaysia will raise its minimum wage again in 2028.")
    assert result["status"] == "NOT_VERIFIABLE"
    assert result["truth_score"] is None


# ---------------------------------------------------------------------------
# Scenario 5: mixed claim -> engine runs on the checkable part only
# ---------------------------------------------------------------------------
async def test_scenario_mixed(patch_scenario):
    await patch_scenario("mixed")
    from pipeline import verify_claim

    result = await verify_claim(
        "Country X raised its minimum wage in 2026, which was a terrible economic decision."
    )
    assert result["status"] == "CHECKED"
    assert result["classification"] == "MIXED"
    assert result["truth_score"] is not None


# ---------------------------------------------------------------------------
# Unit tests: scoring formulas in isolation
# ---------------------------------------------------------------------------
def _resp(name, verdict, confidence=80):
    return ModelResponse(
        model=next(m for m in config.ALL_MODELS if m.name == name),
        verdict=verdict,
        confidence=confidence,
        reasoning="",
        cited_evidence=[],
    )


def test_weighted_verdict_agreement_perfect():
    responses = [_resp(m.name, "TRUE") for m in config.ALL_MODELS]
    weights = {m.name: 1 / 3 for m in config.ALL_MODELS}
    assert scoring.weighted_verdict_agreement(responses, weights) == pytest.approx(1.0)


def test_weighted_verdict_agreement_full_disagreement():
    names = [m.name for m in config.ALL_MODELS]
    responses = [_resp(names[0], "TRUE"), _resp(names[1], "FALSE"), _resp(names[2], "NOT_ENOUGH_INFO")]
    weights = {n: 1 / 3 for n in names}
    kappa = scoring.weighted_verdict_agreement(responses, weights)
    assert kappa == pytest.approx(0.0, abs=1e-6)


def test_evidence_alignment_gated_by_contradiction():
    prop = Proposition("x is 5", "x", "is", "5", "2026-01-01", "DeepSeek-V4-Flash")
    supported = evidence_retriever.PropositionEvidence(
        proposition=prop,
        snippets=[
            evidence_retriever.EvidenceSnippet("t", "https://gov.my/a", "text", "official_government", 0.95, relation="SUPPORT"),
            evidence_retriever.EvidenceSnippet("t2", "https://news.com/b", "text", "established_national_news", 0.75, relation="CONTRADICT"),
        ],
    )
    # Even with strong support, a credible (>0.2) contradiction zeroes alignment.
    assert scoring.evidence_alignment([supported]) == 0.0


def test_evidence_coverage_gate():
    prop = Proposition("x is 5", "x", "is", "5", "2026-01-01", "DeepSeek-V4-Flash")
    no_coverage = evidence_retriever.PropositionEvidence(proposition=prop, snippets=[])
    assert scoring.evidence_coverage([no_coverage]) == 0.0


def test_reasoning_consistency_no_pairs_returns_zero():
    assert scoring.reasoning_consistency([], {}) == 0.0
