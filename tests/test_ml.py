"""Tests for the from-scratch Isolation Forest, scaler and features."""

from __future__ import annotations

import numpy as np
import pytest

from awaf.features import FEATURE_NAMES, extract, extract_batch
from awaf.ml.isolation_forest import IsolationForest, _c
from awaf.ml.scaler import StandardScaler
from awaf.models import Request


def test_c_normalisation():
    assert _c(1) == 0.0
    assert _c(2) == 1.0
    assert _c(256) > _c(2)  # grows with n


def test_forest_separates_outliers():
    rng = np.random.RandomState(0)
    normal = rng.normal(0, 1, size=(300, 5))
    outliers = rng.uniform(7, 10, size=(15, 5))
    x = np.vstack([normal, outliers])
    f = IsolationForest(n_trees=100, sample_size=256, seed=1).fit(x)
    scores = f.anomaly_score(x)
    assert scores[:300].mean() < 0.55
    assert scores[300:].mean() > 0.65


def test_forest_unfitted_raises():
    with pytest.raises(RuntimeError):
        IsolationForest().score_one(np.zeros(4))


def test_forest_persistence_round_trip():
    x = np.random.RandomState(2).normal(size=(200, 4))
    f = IsolationForest(n_trees=50, seed=3).fit(x)
    f2 = IsolationForest.from_dict(f.to_dict())
    assert np.allclose(f.anomaly_score(x[:10]), f2.anomaly_score(x[:10]))


def test_scaler_standardises():
    x = np.array([[1.0, 10.0], [3.0, 30.0], [5.0, 50.0]])
    s = StandardScaler()
    xs = s.fit_transform(x)
    assert np.allclose(xs.mean(axis=0), 0, atol=1e-9)
    assert np.allclose(xs.std(axis=0), 1, atol=1e-9)


def test_scaler_handles_constant_feature():
    x = np.array([[1.0, 7.0], [2.0, 7.0], [3.0, 7.0]])  # 2nd column constant
    xs = StandardScaler().fit_transform(x)
    assert not np.isnan(xs).any()


def test_feature_vector_shape_and_keywords():
    benign = extract(Request("GET", "/products?id=42"))
    attack = extract(Request("GET", "/products?id=1' OR 1=1--"))
    assert benign.shape == (len(FEATURE_NAMES),)
    kw = FEATURE_NAMES.index("keyword_hits")
    assert attack[kw] > benign[kw]  # attack has suspicious tokens


def test_extract_batch_empty():
    assert extract_batch([]).shape == (0, len(FEATURE_NAMES))
