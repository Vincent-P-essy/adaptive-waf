"""Detection metrics.

Evaluates the engine against a labelled set and reports the numbers a WAF is
actually judged on: precision, recall, F1, the false-positive rate, and an
estimated false-positives-per-day at a given traffic volume — the metric that
decides whether analysts can live with the deployment.
"""

from __future__ import annotations

from typing import Any

from .engine import WAFEngine
from .models import ALLOW, Request


def _blocked(verdict: str) -> bool:
    """A request is 'flagged' if it is not allowed (BLOCK or CHALLENGE)."""
    return verdict != ALLOW


def evaluate(
    engine: WAFEngine,
    requests: list[Request],
    labels: list[int],
    daily_requests: int = 1_000_000,
) -> dict[str, Any]:
    tp = fp = tn = fn = 0
    for request, label in zip(requests, labels, strict=True):
        flagged = _blocked(engine.inspect(request).verdict)
        if flagged and label == 1:
            tp += 1
        elif flagged and label == 0:
            fp += 1
        elif not flagged and label == 0:
            tn += 1
        else:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    fp_rate = fp / (fp + tn) if (fp + tn) else 0.0
    total = tp + fp + tn + fn
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "false_positive_rate": round(fp_rate, 4),
        "accuracy": round((tp + tn) / total, 4) if total else 0.0,
        "estimated_fp_per_day": int(round(fp_rate * daily_requests)),
    }
