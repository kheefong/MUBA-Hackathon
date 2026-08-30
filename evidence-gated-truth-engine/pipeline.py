"""
pipeline.py — Orchestrates the full Evidence-Gated Truth Engine flow and
assembles the final JSON output described in the spec.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from calibration import calibrate_all
from claim_classifier import classify_claim
from database import log_calibration_sample, log_verification_result
from evidence_retriever import retrieve_all_evidence
from model_interrogator import interrogate_all
from nli_engine import build_contradiction_matrix
from proposition_extractor import extract_all_propositions
from scoring import score_claim

logger = logging.getLogger("truth_engine.pipeline")


def _build_not_verifiable_result(claim_id: str, claim_text: str, classification) -> dict:
    return {
        "claim_id": claim_id,
        "claim_text": claim_text,
        "status": "NOT_VERIFIABLE",
        "classification": classification.category,
        "truth_score": None,
        "consensus_score": None,
        "confidence_score": None,
        "evidence_coverage": None,
        "evidence_alignment": None,
        "verdict": "NOT_ENOUGH_INFO",
        "explanation": classification.reason or "Claim is not a checkable factual statement.",
        "model_details": [],
        "contradiction_flags": [],
        "sources_used": [],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _build_explanation(result, prop_evidences, pair_relations) -> str:
    parts = []
    if result.status == "UNVERIFIABLE":
        parts.append(
            f"Evidence coverage was only {result.evidence_coverage:.0%}, below the "
            "50% threshold required to compute a Truth Score."
        )
    else:
        parts.append(
            f"Truth Score {result.truth_score:.2f} reflects evidence alignment "
            f"({result.evidence_alignment:.2f}) gating a base score built from "
            f"reasoning consistency ({result.reasoning_consistency:.2f}) and "
            f"weighted verdict agreement ({result.weighted_verdict_agreement:.2f})."
        )

    contradictory_props = [pe for pe in prop_evidences if pe.contradiction_score > 0.2]
    if contradictory_props:
        parts.append(
            f"{len(contradictory_props)} proposition(s) had credible contradicting evidence."
        )

    contradicting_pairs = [pr for pr in pair_relations if pr.relation == "CONTRADICTION"]
    if contradicting_pairs:
        parts.append(
            f"Models contradicted each other on {len(contradicting_pairs)} shared proposition(s)."
        )

    return " ".join(parts)


async def verify_claim(claim_text: str) -> dict:
    claim_id = str(uuid.uuid4())

    # Step 1: classify
    classification = await classify_claim(claim_text)
    if not classification.should_run_engine:
        result = _build_not_verifiable_result(claim_id, claim_text, classification)
        await log_verification_result(claim_id, claim_text, result)
        return result

    text_to_verify = classification.text_to_verify or claim_text

    # Step 2: multi-model interrogation
    responses = await interrogate_all(text_to_verify)

    # Step 3: proposition extraction
    propositions = await extract_all_propositions(responses)

    # Step 4: NLI contradiction matrix (model-vs-model)
    pair_relations = await build_contradiction_matrix(propositions)

    # Step 5: evidence retrieval + credibility + entailment
    prop_evidences = await retrieve_all_evidence(propositions)

    # Step 6: calibration + dynamic trust weights
    raw_confidences = {r.model.name: r.confidence for r in responses}
    calibrations = await calibrate_all(raw_confidences)

    # Step 7: evidence-gated scoring
    result = score_claim(responses, pair_relations, prop_evidences, calibrations)

    # Log calibration samples for future runs. `correct` is left NULL here
    # because we don't have ground truth for this claim yet; a separate
    # ground-truth-labeling job (or evidence-alignment self-training) should
    # come back and update `correct` once resolvable. As an interim proxy,
    # we mark a model "correct" if its verdict matches the evidence-gated
    # final verdict AND evidence coverage was usable.
    for r in responses:
        if not r.ok:
            continue
        correct = None
        if result.status == "CHECKED":
            correct = 1 if r.verdict == result.final_verdict else 0
        await log_calibration_sample(r.model.name, claim_id, r.confidence, correct)

    model_details = [
        {
            "model": r.model.name,
            "verdict": r.verdict,
            "raw_confidence": r.confidence,
            "calibrated_confidence": calibrations[r.model.name].calibrated_confidence,
            "trust_weight": calibrations[r.model.name].trust_weight,
            "atomic_propositions": [p.proposition for p in propositions if p.source_model == r.model.name],
        }
        for r in responses
    ]

    contradiction_flags = [
        {
            "models": [pr.prop_a.source_model, pr.prop_b.source_model],
            "proposition": f"{pr.prop_a.proposition}  <->  {pr.prop_b.proposition}",
            "relation": pr.relation,
            "evidence": pr.explanation,
        }
        for pr in pair_relations
        if pr.relation == "CONTRADICTION"
    ]

    sources_used = [
        {
            "title": s.title,
            "url": s.url,
            "credibility": s.credibility,
            "relation": s.relation,
        }
        for pe in prop_evidences
        for s in pe.snippets
        if s.relation != "NOT_COVERED"
    ]

    output = {
        "claim_id": claim_id,
        "claim_text": claim_text,
        "status": result.status,
        "classification": classification.category,
        "truth_score": result.truth_score,
        "consensus_score": result.consensus_score,
        "confidence_score": result.confidence_score,
        "evidence_coverage": result.evidence_coverage,
        "evidence_alignment": result.evidence_alignment,
        "verdict": result.final_verdict,
        "explanation": _build_explanation(result, prop_evidences, pair_relations),
        "model_details": model_details,
        "contradiction_flags": contradiction_flags,
        "sources_used": sources_used,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    await log_verification_result(claim_id, claim_text, output)
    return output
