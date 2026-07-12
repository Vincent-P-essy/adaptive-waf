"""AdaptiveWAF — the stateful service that closes the loop.

Owns the training corpus, the engine and the feedback store, and exposes the
operations the API and the demo drive: inspect a request, record analyst
feedback, and retrain the anomaly model on the corrected corpus so a reported
false positive stops being flagged.
"""

from __future__ import annotations

from typing import Any

from .config import config
from .engine import WAFEngine
from .feedback import FeedbackStore
from .metrics import evaluate
from .models import Decision, Request


class AdaptiveWAF:
    def __init__(
        self,
        base_corpus: list[Request],
        engine: WAFEngine | None = None,
        feedback_weight: int | None = None,
    ) -> None:
        self._base = list(base_corpus)
        self.engine = engine or WAFEngine()
        self.feedback = FeedbackStore()
        self.feedback_weight = feedback_weight if feedback_weight is not None else config.feedback_weight
        self.retrains = 0
        self.engine.fit(self._training_corpus())

    def _training_corpus(self) -> list[Request]:
        # Base normal traffic plus analyst-confirmed benign requests, oversampled:
        # a confirmed normal is strong evidence that its region of feature space is
        # legitimate, so weighting it makes those requests less isolatable (lower
        # anomaly score) and stops them being flagged after a retrain.
        return self._base + self.feedback.benign_examples() * self.feedback_weight

    def inspect(self, request: Request) -> Decision:
        return self.engine.inspect(request)

    def report_feedback(self, request: Request, label: str, was: str = "") -> dict[str, Any]:
        self.feedback.add(request, label, was)
        return self.feedback.summary()

    def retrain(self) -> dict[str, Any]:
        """Re-fit the anomaly model on the base corpus + benign corrections."""
        self.engine.fit(self._training_corpus())
        self.retrains += 1
        return {
            "retrains": self.retrains,
            "corpus_size": len(self._training_corpus()),
            "feedback": self.feedback.summary(),
        }

    def evaluate(self, requests: list[Request], labels: list[int]) -> dict[str, Any]:
        return evaluate(self.engine, requests, labels)
