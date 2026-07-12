"""Isolation Forest, implemented from first principles.

Isolation Forest isolates anomalies rather than profiling normal points: an
anomaly is separated from the rest of the data in fewer random splits, so it has a
shorter expected path length in an ensemble of random "isolation trees". This
module implements the algorithm from Liu, Ting & Zhou (2008) directly on numpy —
tree construction, the ``c(n)`` path-length normalisation, and the
``2^(-E[h(x)]/c(n))`` anomaly score — with no scikit-learn dependency.

Score semantics: ~0.5 is average (normal); values approaching 1.0 are anomalous;
values well below 0.5 are very normal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

_EULER = 0.5772156649015329


def _c(n: int) -> float:
    """Average path length of an unsuccessful search in a BST of ``n`` points."""
    if n <= 1:
        return 0.0
    if n == 2:
        return 1.0
    harmonic = math.log(n - 1) + _EULER
    return 2.0 * harmonic - 2.0 * (n - 1) / n


@dataclass
class _Node:
    # Internal node: split_feature/split_value + children. External: size, size>0, children None.
    size: int = 0
    split_feature: int | None = None
    split_value: float | None = None
    left: _Node | None = None
    right: _Node | None = None

    @property
    def is_external(self) -> bool:
        return self.split_feature is None


class IsolationTree:
    def __init__(self, height_limit: int, rng: np.random.RandomState) -> None:
        self._height_limit = height_limit
        self._rng = rng
        self.root: _Node | None = None

    def fit(self, x: np.ndarray) -> IsolationTree:
        self.root = self._grow(x, 0)
        return self

    def _grow(self, x: np.ndarray, height: int) -> _Node:
        n = x.shape[0]
        if height >= self._height_limit or n <= 1:
            return _Node(size=n)

        # Pick a random feature that actually varies; give up after a few tries.
        n_features = x.shape[1]
        for feature in self._rng.permutation(n_features):
            col = x[:, feature]
            lo, hi = col.min(), col.max()
            if lo == hi:
                continue
            split = self._rng.uniform(lo, hi)
            left_mask = col < split
            return _Node(
                size=n,
                split_feature=int(feature),
                split_value=float(split),
                left=self._grow(x[left_mask], height + 1),
                right=self._grow(x[~left_mask], height + 1),
            )
        return _Node(size=n)  # all features constant on this subsample

    def path_length(self, x: np.ndarray) -> float:
        node, height = self.root, 0
        while node is not None and not node.is_external:
            node = node.left if x[node.split_feature] < node.split_value else node.right
            height += 1
        return height + _c(node.size if node else 1)


class IsolationForest:
    def __init__(self, n_trees: int = 120, sample_size: int = 256, seed: int = 42) -> None:
        self.n_trees = n_trees
        self.sample_size = sample_size
        self.seed = seed
        self._trees: list[IsolationTree] = []
        self._c_norm = 1.0

    def fit(self, x: np.ndarray) -> IsolationForest:
        rng = np.random.RandomState(self.seed)
        n = x.shape[0]
        sample = min(self.sample_size, n)
        height_limit = max(1, math.ceil(math.log2(sample)))
        self._c_norm = _c(sample) or 1.0
        self._trees = []
        for _ in range(self.n_trees):
            idx = rng.choice(n, size=sample, replace=False) if n > sample else np.arange(n)
            self._trees.append(IsolationTree(height_limit, rng).fit(x[idx]))
        return self

    def _mean_path(self, x: np.ndarray) -> float:
        return float(np.mean([t.path_length(x) for t in self._trees]))

    def score_one(self, x: np.ndarray) -> float:
        if not self._trees:
            raise RuntimeError("model is not fitted")
        return 2.0 ** (-self._mean_path(x) / self._c_norm)

    def anomaly_score(self, x: np.ndarray) -> np.ndarray:
        return np.asarray([self.score_one(row) for row in x])

    @property
    def fitted(self) -> bool:
        return bool(self._trees)

    # -- persistence -------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        def node(n: _Node | None) -> Any:
            if n is None:
                return None
            if n.is_external:
                return {"s": n.size}
            return {"f": n.split_feature, "v": n.split_value, "l": node(n.left), "r": node(n.right)}

        return {
            "n_trees": self.n_trees,
            "sample_size": self.sample_size,
            "seed": self.seed,
            "c_norm": self._c_norm,
            "trees": [node(t.root) for t in self._trees],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> IsolationForest:
        forest = cls(d["n_trees"], d["sample_size"], d["seed"])
        forest._c_norm = d["c_norm"]

        def node(j: Any) -> _Node | None:
            if j is None:
                return None
            if "s" in j:
                return _Node(size=j["s"])
            return _Node(split_feature=j["f"], split_value=j["v"], left=node(j["l"]), right=node(j["r"]))

        forest._trees = []
        for tj in d["trees"]:
            t = IsolationTree(0, np.random.RandomState(0))
            t.root = node(tj)
            forest._trees.append(t)
        return forest
