#!/usr/bin/env python3
"""
Evaluation metrics for AMBIFC soft-label veracity prediction.

Label order:
    0 = refuting
    1 = neutral
    2 = supporting
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


EPS = 1e-12
LABEL_NAMES = ["refuting", "neutral", "supporting"]


def to_numpy(x):
    return np.asarray(x, dtype=np.float64)


def normalize_probs(probs: np.ndarray) -> np.ndarray:
    probs = to_numpy(probs)
    probs = np.clip(probs, EPS, 1.0)

    if probs.ndim == 1:
        return probs / probs.sum()

    return probs / probs.sum(axis=1, keepdims=True)


def hard_labels_from_probs(probs: np.ndarray) -> np.ndarray:
    probs = to_numpy(probs)
    return probs.argmax(axis=1)


def accuracy_from_probs(gold_probs, pred_probs) -> float:
    gold_hard = hard_labels_from_probs(gold_probs)
    pred_hard = hard_labels_from_probs(pred_probs)
    return float(accuracy_score(gold_hard, pred_hard))


def macro_f1_from_probs(gold_probs, pred_probs) -> float:
    gold_hard = hard_labels_from_probs(gold_probs)
    pred_hard = hard_labels_from_probs(pred_probs)
    return float(f1_score(gold_hard, pred_hard, average="macro", labels=[0, 1, 2]))


def entropy(probs: np.ndarray) -> np.ndarray:
    probs = normalize_probs(probs)
    return -np.sum(probs * np.log(probs + EPS), axis=1)


def entce(gold_probs, pred_probs) -> float:
    """
    Human Entropy Calibration Error.

    Lower is better.
    Measures how close model uncertainty is to human soft-label uncertainty.
    """
    gold_probs = normalize_probs(gold_probs)
    pred_probs = normalize_probs(pred_probs)

    gold_entropy = entropy(gold_probs)
    pred_entropy = entropy(pred_probs)

    return float(np.mean(np.abs(gold_entropy - pred_entropy)))


def distcs(gold_probs, pred_probs) -> float:
    """
    Human Distribution Calibration Score.

    Higher is better.
    DistCS = 1 - total variation distance.
    """
    gold_probs = normalize_probs(gold_probs)
    pred_probs = normalize_probs(pred_probs)

    tvd = 0.5 * np.sum(np.abs(gold_probs - pred_probs), axis=1)
    return float(np.mean(1.0 - tvd))


def kl_divergence(gold_probs, pred_probs) -> float:
    """
    KL(gold || predicted).

    Lower is better.
    """
    gold_probs = normalize_probs(gold_probs)
    pred_probs = normalize_probs(pred_probs)

    kl = np.sum(gold_probs * np.log((gold_probs + EPS) / (pred_probs + EPS)), axis=1)
    return float(np.mean(kl))


def rankcs(gold_probs, pred_probs, tie_epsilon: float = 1e-8) -> float:
    """
    Human Ranking Calibration Score.

    Higher is better.

    This pairwise implementation checks whether the model preserves
    the human ranking between labels. If the human probabilities are tied,
    that pair is ignored because multiple rankings are valid.
    """
    gold_probs = normalize_probs(gold_probs)
    pred_probs = normalize_probs(pred_probs)

    scores = []

    for gold, pred in zip(gold_probs, pred_probs):
        correct = 0
        total = 0

        for i in range(3):
            for j in range(i + 1, 3):
                gold_diff = gold[i] - gold[j]

                if abs(gold_diff) <= tie_epsilon:
                    continue

                pred_diff = pred[i] - pred[j]

                if gold_diff > 0 and pred_diff > 0:
                    correct += 1
                elif gold_diff < 0 and pred_diff < 0:
                    correct += 1

                total += 1

        if total == 0:
            scores.append(1.0)
        else:
            scores.append(correct / total)

    return float(np.mean(scores))


def multilabel_f1_from_probs(gold_probs, pred_probs, gold_threshold: float = 0.20, pred_threshold: float = 0.50) -> float:
    """
    Multi-label F1 used for ambiguity-aware evaluation.

    Gold labels are labels chosen by at least 20% of annotators.
    Predicted labels are labels with predicted probability >= 0.50.
    """
    gold_probs = normalize_probs(gold_probs)
    pred_probs = normalize_probs(pred_probs)

    gold_multi = (gold_probs >= gold_threshold).astype(int)
    pred_multi = (pred_probs >= pred_threshold).astype(int)

    # If a prediction has no selected label, use argmax as fallback.
    for i in range(len(pred_multi)):
        if pred_multi[i].sum() == 0:
            pred_multi[i, pred_probs[i].argmax()] = 1

    return float(f1_score(gold_multi, pred_multi, average="samples", zero_division=0))


def compute_all_metrics(gold_probs, pred_probs) -> dict:
    gold_probs = normalize_probs(gold_probs)
    pred_probs = normalize_probs(pred_probs)

    return {
        "accuracy": accuracy_from_probs(gold_probs, pred_probs),
        "macro_f1": macro_f1_from_probs(gold_probs, pred_probs),
        "distcs": distcs(gold_probs, pred_probs),
        "entce": entce(gold_probs, pred_probs),
        "rankcs": rankcs(gold_probs, pred_probs),
        "kl": kl_divergence(gold_probs, pred_probs),
        "multilabel_f1": multilabel_f1_from_probs(gold_probs, pred_probs),
    }


def format_metrics(metrics: dict) -> str:
    order = [
        "accuracy",
        "macro_f1",
        "distcs",
        "entce",
        "rankcs",
        "kl",
        "multilabel_f1",
    ]

    lines = []
    for key in order:
        if key in metrics:
            lines.append(f"{key:15s}: {metrics[key]:.4f}")

    return "\n".join(lines)