from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sklearn.metrics import log_loss


EPS = 1e-8


def normalize_probs(prob: np.ndarray) -> np.ndarray:
    arr = np.asarray(prob, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"Expected probability matrix with shape (n, 3), got {arr.shape}")
    arr = np.clip(arr, EPS, 1.0)
    denom = arr.sum(axis=1, keepdims=True)
    if np.any(~np.isfinite(denom)) or np.any(denom <= 0):
        raise ValueError("Probability rows must have a finite positive sum")
    return arr / denom


def market_probs(frame) -> np.ndarray:
    required = ["x1x2_avg_prob_home", "x1x2_avg_prob_draw", "x1x2_avg_prob_away"]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise KeyError(f"Missing market probability columns: {missing}")
    return normalize_probs(frame[required].to_numpy(dtype=float))


def probs_to_logits(prob: np.ndarray) -> np.ndarray:
    p = normalize_probs(prob)
    return np.log(np.clip(p, EPS, 1.0))


def softmax(logits: np.ndarray) -> np.ndarray:
    x = np.asarray(logits, dtype=float)
    x = x - np.max(x, axis=1, keepdims=True)
    exp = np.exp(x)
    return normalize_probs(exp)


def multiclass_brier(y: np.ndarray, prob: np.ndarray) -> float:
    y = np.asarray(y, dtype=int)
    p = normalize_probs(prob)
    one_hot = np.zeros_like(p)
    one_hot[np.arange(len(y)), y] = 1.0
    return float(np.mean(np.sum((p - one_hot) ** 2, axis=1)))


def multiclass_ece(y: np.ndarray, prob: np.ndarray, bins: int = 15) -> float:
    y = np.asarray(y, dtype=int)
    p = normalize_probs(prob)
    confidence = p.max(axis=1)
    prediction = p.argmax(axis=1)
    correct = (prediction == y).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (confidence >= low) & (confidence < high if high < 1.0 else confidence <= high)
        if mask.any():
            result += float(mask.mean()) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return float(result)


def probability_metrics(y: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=int)
    p = normalize_probs(prob)
    return {
        "log_loss": float(log_loss(y, p, labels=[0, 1, 2])),
        "brier": multiclass_brier(y, p),
        "ece": multiclass_ece(y, p),
        "accuracy": float((p.argmax(axis=1) == y).mean()),
    }


@dataclass(frozen=True)
class BlendTemperature:
    alpha: float
    temperature: float
    log_loss: float
    brier: float
    ece: float


def apply_blend_temperature(
    model_prob: np.ndarray,
    market_prob: np.ndarray,
    alpha: float,
    temperature: float,
) -> np.ndarray:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    model_logits = probs_to_logits(model_prob)
    market_logits = probs_to_logits(market_prob)
    blended_logits = ((1.0 - alpha) * market_logits + alpha * model_logits) / float(temperature)
    return softmax(blended_logits)


def fit_blend_temperature(
    y: np.ndarray,
    model_prob: np.ndarray,
    market_prob: np.ndarray,
    alphas: Iterable[float],
    temperatures: Iterable[float],
) -> BlendTemperature:
    y = np.asarray(y, dtype=int)
    candidates: list[BlendTemperature] = []
    for alpha in alphas:
        for temperature in temperatures:
            prob = apply_blend_temperature(model_prob, market_prob, float(alpha), float(temperature))
            metrics = probability_metrics(y, prob)
            candidates.append(
                BlendTemperature(
                    alpha=float(alpha),
                    temperature=float(temperature),
                    log_loss=metrics["log_loss"],
                    brier=metrics["brier"],
                    ece=metrics["ece"],
                )
            )
    if not candidates:
        raise ValueError("No calibration candidates supplied")
    # Prefer the lower log loss, then lower Brier, then stronger shrinkage to market,
    # then a temperature closer to 1.0. This guards against needless complexity.
    return min(
        candidates,
        key=lambda c: (
            c.log_loss,
            c.brier,
            c.alpha,
            abs(c.temperature - 1.0),
        ),
    )
