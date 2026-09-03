"""
scoring.py — Step 7: combine model verdicts, reasoning consistency, and
evidence alignment into Truth / Consensus / Confidence scores.

All formulas here follow the spec exactly:

  Weighted Verdict Agreement (chance-corrected, trust-weighted kappa)
  Reasoning Consistency        (from NLI pair scores)
  Evidence Coverage            (fraction of propositions with usable evidence)
  Evidence Alignment           (support, gated to 0 by any credible contradiction)
  Truth Score  = Evidence Alignment * (0.4 + 0.3*ReasoningConsistency + 0.3*WVA)
  Consensus    = 0.5*WVA + 0.5*ReasoningConsistency
  Confidence   = sum_m trust_weight_m * calibrated_confidence_m   (None if coverage < 0.5)
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

from calibration import ModelCalibration
from evidence_retriever import PropositionEvidence
from model_interrogator import ModelResponse
from nli_engine import PairRelation

LABELS = ("TRUE", "FALSE", "NOT_ENOUGH_INFO")

COVERAGE_GATE_THRESHOLD = 0.5
CREDIBILITY_CONTRADICTION_THRESHOLD = 0.2


@dataclass
class ScoringResult:
    truth_score: float | None
    consensus_score: float
    confidence_score: float | None
    evidence_coverage: float
    evidence_alignment: float | None
    weighted_verdict_agreement: float
    reasoning_consistency: float
    final_verdict: str
    status: str  # CHECKED | UNVERIFIABLE


def weighted_verdict_agreement(
    responses: list[ModelResponse], weights: dict[str, float]
) -> float:
    """Chance-corrected, trust-weighted kappa over verdict labels."""
    ok_responses = [r for r in responses if r.ok]
    if len(ok_responses) < 2:
        return 0.0

    w = {r.model.name: weights.get(r.model.name, 1.0 / len(ok_responses)) for r in ok_responses}
    total_w = sum(w.values())
    if total_w == 0:
        return 0.0

    # Weighted label prevalence p_l = sum(w_m * I[v_m=l]) / sum(w_m)
    prevalence = {
        label: sum(w[r.model.name] for r in ok_responses if r.verdict == label) / total_w
        for label in LABELS
    }

    pair_weight_sum = 0.0
    disagreement_weight_sum = 0.0
    for a, b in itertools.combinations(ok_responses, 2):
        pw = w[a.model.name] * w[b.model.name]
        pair_weight_sum += pw
        if a.verdict != b.verdict:
            disagreement_weight_sum += pw

    if pair_weight_sum == 0:
        return 0.0

    d_observed = disagreement_weight_sum / pair_weight_sum
    d_expected = 1 - sum(p**2 for p in prevalence.values())

    if d_expected == 0:
        # No expected disagreement under prevalence (e.g. all one label) ->
        # perfect agreement is trivially achieved, so kappa = 1.
        return 1.0

    kappa = 1 - (d_observed / d_expected)
    return max(0.0, min(1.0, kappa))


def reasoning_consistency(pair_relations: list[PairRelation], weights: dict[str, float]) -> float:
    """Weighted mean of NLI pair scores across propositions shared by
    different models, weighted by w_i * w_j.
    """
    if not pair_relations:
        return 0.0

    numerator = 0.0
    denominator = 0.0
    for pr in pair_relations:
        wi = weights.get(pr.prop_a.source_model, 0.0)
        wj = weights.get(pr.prop_b.source_model, 0.0)
        pw = wi * wj
        numerator += pw * pr.score
        denominator += pw

    if denominator == 0:
        return 0.0
    return numerator / denominator


def evidence_coverage(prop_evidences: list[PropositionEvidence]) -> float:
    if not prop_evidences:
        return 0.0
    return sum(pe.coverage for pe in prop_evidences) / len(prop_evidences)


def evidence_alignment(prop_evidences: list[PropositionEvidence]) -> float:
    """Mean alignment over *covered* propositions. A credible contradiction
    (credibility > 0.2) zeroes out alignment for that proposition
    regardless of support strength.
    """
    covered = [pe for pe in prop_evidences if pe.coverage == 1.0]
    if not covered:
        return 0.0

    alignments = []
    for pe in covered:
        if pe.contradiction_score > CREDIBILITY_CONTRADICTION_THRESHOLD:
            alignments.append(0.0)
        else:
            alignments.append(pe.support_score)
    return sum(alignments) / len(alignments)


def _majority_verdict(responses: list[ModelResponse], weights: dict[str, float]) -> str:
    ok_responses = [r for r in responses if r.ok]
    if not ok_responses:
        return "NOT_ENOUGH_INFO"
    tally: dict[str, float] = {label: 0.0 for label in LABELS}
    for r in ok_responses:
        tally[r.verdict] += weights.get(r.model.name, 1.0)
    return max(tally, key=tally.get)


def score_claim(
    responses: list[ModelResponse],
    pair_relations: list[PairRelation],
    prop_evidences: list[PropositionEvidence],
    calibrations: dict[str, ModelCalibration],
) -> ScoringResult:
    weights = {name: c.trust_weight for name, c in calibrations.items()}

    wva = weighted_verdict_agreement(responses, weights)
    rc = reasoning_consistency(pair_relations, weights)
    coverage = evidence_coverage(prop_evidences)

    consensus = 0.5 * wva + 0.5 * rc

    if coverage < COVERAGE_GATE_THRESHOLD:
        return ScoringResult(
            truth_score=None,
            consensus_score=consensus,
            confidence_score=None,
            evidence_coverage=coverage,
            evidence_alignment=None,
            weighted_verdict_agreement=wva,
            reasoning_consistency=rc,
            final_verdict="NOT_ENOUGH_INFO",
            status="UNVERIFIABLE",
        )

    alignment = evidence_alignment(prop_evidences)
    truth_score = alignment * (0.4 + 0.3 * rc + 0.3 * wva) if alignment > 0 else 0.0

    confidence_score = sum(
        c.trust_weight * c.calibrated_confidence for c in calibrations.values()
    ) / 100.0  # normalize to 0-1 scale to match other scores

    final_verdict = _majority_verdict(responses, weights)
    # If evidence flatly contradicts, override an optimistic majority verdict.
    if alignment == 0.0 and any(pe.contradiction_score > CREDIBILITY_CONTRADICTION_THRESHOLD for pe in prop_evidences):
        final_verdict = "FALSE" if final_verdict == "TRUE" else final_verdict

    return ScoringResult(
        truth_score=truth_score,
        consensus_score=consensus,
        confidence_score=confidence_score,
        evidence_coverage=coverage,
        evidence_alignment=alignment,
        weighted_verdict_agreement=wva,
        reasoning_consistency=rc,
        final_verdict=final_verdict,
        status="CHECKED",
    )
