"""Three-layer decision engine.

Layer 1 (rules) runs first and short-circuits on a known-attack signature. Layer 2
(Isolation Forest) scores everything else; scores at/above the block threshold are
blocked outright, scores clearly below are allowed. Only the thin ambiguous band
just under the threshold is escalated to layer 3 (the LLM/heuristic adjudicator).
This ordering keeps the expensive layer off the hot path for the vast majority of
requests.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import rules
from .config import config
from .features import extract, extract_batch
from .llm import Adjudication, build_adjudicator
from .ml.isolation_forest import IsolationForest
from .ml.scaler import StandardScaler
from .models import ALLOW, BLOCK, CHALLENGE, Decision, Request


class WAFEngine:
    def __init__(
        self,
        block_threshold: float | None = None,
        ambiguous_band: float | None = None,
        adjudicator: Any = None,
    ) -> None:
        self.block_threshold = block_threshold if block_threshold is not None else config.block_threshold
        self.ambiguous_band = ambiguous_band if ambiguous_band is not None else config.ambiguous_band
        self.scaler = StandardScaler()
        self.forest = IsolationForest(config.n_trees, config.sample_size, config.random_seed)
        self._adjudicator = adjudicator or build_adjudicator()

    @property
    def adjudicator_name(self) -> str:
        return self._adjudicator.name

    @property
    def fitted(self) -> bool:
        return self.forest.fitted

    # -- training ----------------------------------------------------------
    def fit(self, normal_requests: list[Request]) -> "WAFEngine":
        x = extract_batch(normal_requests)
        xs = self.scaler.fit_transform(x)
        self.forest.fit(xs)
        return self

    def anomaly_score(self, request: Request) -> float:
        vec = self.scaler.transform(extract(request).reshape(1, -1))
        return float(self.forest.score_one(vec[0]))

    # -- decision ----------------------------------------------------------
    def inspect(self, request: Request) -> Decision:
        # Layer 1: signatures.
        rule = rules.match(request)
        if rule is not None:
            return Decision(BLOCK, "rules", f"{rule.name} [{rule.id}]", confidence=0.99)

        # Layer 2: anomaly score.
        if not self.forest.fitted:
            raise RuntimeError("engine is not fitted")
        score = self.anomaly_score(request)

        if score >= self.block_threshold:
            return Decision(BLOCK, "ml", f"anomaly score {score:.3f} ≥ {self.block_threshold}",
                            anomaly_score=score, confidence=round(score, 3))

        if score >= self.block_threshold - self.ambiguous_band:
            # Layer 3: adjudicate the ambiguous band.
            adj: Adjudication = self._adjudicator.adjudicate(request)
            if adj.malicious and adj.confidence >= 0.6:
                verdict = BLOCK
            elif adj.malicious:
                verdict = CHALLENGE
            else:
                verdict = ALLOW
            return Decision(verdict, "llm", adj.reason, anomaly_score=score, confidence=adj.confidence)

        return Decision(ALLOW, "ml", f"anomaly score {score:.3f} below threshold",
                        anomaly_score=score, confidence=round(1 - score, 3))

    def scores(self, requests: list[Request]) -> np.ndarray:
        xs = self.scaler.transform(extract_batch(requests))
        return self.forest.anomaly_score(xs)

    # -- persistence -------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "block_threshold": self.block_threshold,
            "ambiguous_band": self.ambiguous_band,
            "scaler": self.scaler.to_dict(),
            "forest": self.forest.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any], adjudicator: Any = None) -> "WAFEngine":
        engine = cls(d["block_threshold"], d["ambiguous_band"], adjudicator=adjudicator)
        engine.scaler = StandardScaler.from_dict(d["scaler"])
        engine.forest = IsolationForest.from_dict(d["forest"])
        return engine
