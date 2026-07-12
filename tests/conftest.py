"""Shared fixtures."""

from __future__ import annotations

import pytest

from awaf.ingest import build_dataset
from awaf.service import AdaptiveWAF


@pytest.fixture(scope="session")
def dataset():
    # Smaller corpus keeps the suite fast while preserving the normal/attack boundary.
    return build_dataset(n_train=800, n_eval_normal=150, n_eval_attack=100)


@pytest.fixture(scope="session")
def trained_waf(dataset):
    """A trained WAF for read-only tests (inspect/metrics)."""
    train, _, _ = dataset
    return AdaptiveWAF(train)


@pytest.fixture
def fresh_waf(dataset):
    """A fresh WAF for tests that mutate state (feedback/retrain)."""
    train, _, _ = dataset
    return AdaptiveWAF(train)
