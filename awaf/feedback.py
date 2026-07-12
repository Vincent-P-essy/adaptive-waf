"""Feedback store — the raw material of the learning loop.

Analysts label requests the WAF got wrong: a *benign* request that was blocked (a
false positive) or a *malicious* request that was allowed (a false negative).
Benign corrections are folded back into the training corpus so the anomaly model
learns them as normal; malicious corrections are tracked for coverage reporting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .models import Request

BENIGN = "benign"
MALICIOUS = "malicious"


@dataclass
class FeedbackItem:
    request: Request
    label: str  # benign | malicious
    was: str  # the verdict the WAF originally returned


@dataclass
class FeedbackStore:
    items: list[FeedbackItem] = field(default_factory=list)

    def add(self, request: Request, label: str, was: str = "") -> FeedbackItem:
        if label not in (BENIGN, MALICIOUS):
            raise ValueError(f"label must be '{BENIGN}' or '{MALICIOUS}', got {label!r}")
        item = FeedbackItem(request=request, label=label, was=was)
        self.items.append(item)
        return item

    def benign_examples(self) -> list[Request]:
        return [i.request for i in self.items if i.label == BENIGN]

    def malicious_examples(self) -> list[Request]:
        return [i.request for i in self.items if i.label == MALICIOUS]

    def summary(self) -> dict[str, Any]:
        return {
            "total": len(self.items),
            "benign": len(self.benign_examples()),
            "malicious": len(self.malicious_examples()),
        }
