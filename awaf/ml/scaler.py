"""Standard scaler (zero-mean, unit-variance) — from scratch on numpy.

The Isolation Forest is scale-sensitive because splits are drawn uniformly across
each feature's range; standardising first keeps a large-magnitude feature (e.g.
body length) from dominating the random splits.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class StandardScaler:
    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> StandardScaler:
        self.mean_ = x.mean(axis=0)
        std = x.std(axis=0)
        std[std == 0] = 1.0  # avoid divide-by-zero on constant features
        self.std_ = std
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("scaler is not fitted")
        return (x - self.mean_) / self.std_

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)

    def to_dict(self) -> dict[str, Any]:
        return {"mean": self.mean_.tolist(), "std": self.std_.tolist()}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> StandardScaler:
        s = cls()
        s.mean_ = np.asarray(d["mean"])
        s.std_ = np.asarray(d["std"])
        return s
