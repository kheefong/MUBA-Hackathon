"""
calibration.py — Step 6: per-model confidence calibration and dynamic
trust weights, based on historical (stated_confidence, actual_correctness)
pairs rather than models' self-reported confidence.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import (
    ALL_MODELS,
    CALIBRATION_MIN_SAMPLES_FOR_ISOTONIC,
    CALIBRATION_WINDOW,
    ModelConfig,
)
from database import get_calibration_history


@dataclass
class ModelCalibration:
    model: ModelConfig
    rolling_accuracy: float
    calibrated_confidence: float  # for the current call, 0-100
    trust_weight: float  # normalized across models, sums to 1.0


def _rolling_accuracy(history: list[tuple[float, int]]) -> float:
    """(correct + 1) / (total + 2) — Laplace-smoothed rolling accuracy."""
    if not history:
        return 0.5  # neutral prior with no data
    correct = sum(c for _, c in history)
    total = len(history)
    return (correct + 1) / (total + 2)


def _isotonic_calibrate(raw_confidence: float, history: list[tuple[float, int]]) -> float:
    """Map raw stated confidence (0-100) to calibrated confidence (0-100)
    using isotonic regression fit on historical (confidence, correctness)
    pairs. Falls back to a simple accuracy-scaled heuristic when there
    isn't enough history yet, per spec.
    """
    rolling_acc = _rolling_accuracy(history)
    if len(history) < CALIBRATION_MIN_SAMPLES_FOR_ISOTONIC:
        return max(0.0, min(100.0, raw_confidence * (rolling_acc + 0.1)))

    from sklearn.isotonic import IsotonicRegression

    x = np.array([h[0] for h in history], dtype=float)
    y = np.array([h[1] for h in history], dtype=float)
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(x, y)
    calibrated_fraction = float(iso.predict([raw_confidence])[0])
    return max(0.0, min(100.0, calibrated_fraction * 100))


async def calibrate_model(model: ModelConfig, raw_confidence: float) -> ModelCalibration:
    history = await get_calibration_history(model.name, CALIBRATION_WINDOW)
    rolling_acc = _rolling_accuracy(history)
    calibrated = _isotonic_calibrate(raw_confidence, history)
    # trust_weight is filled in by calibrate_all (needs cross-model normalization)
    return ModelCalibration(model=model, rolling_accuracy=rolling_acc, calibrated_confidence=calibrated, trust_weight=0.0)


async def calibrate_all(raw_confidences: dict[str, float]) -> dict[str, ModelCalibration]:
    """raw_confidences: {model_name: raw_confidence_0_to_100}

    Returns {model_name: ModelCalibration} with trust_weight normalized
    across all configured models. If no history exists for any model,
    trust weights default to equal (1/3 each), per spec.
    """
    calibrations: dict[str, ModelCalibration] = {}
    for model in ALL_MODELS:
        raw_conf = raw_confidences.get(model.name, 0.0)
        calibrations[model.name] = await calibrate_model(model, raw_conf)

    total_acc = sum(c.rolling_accuracy for c in calibrations.values())
    if total_acc <= 0:
        equal = 1.0 / len(ALL_MODELS)
        for c in calibrations.values():
            c.trust_weight = equal
    else:
        for c in calibrations.values():
            c.trust_weight = c.rolling_accuracy / total_acc

    return calibrations
