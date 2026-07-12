"""Tests for the feedback store and the adaptive service loop."""

from __future__ import annotations

import pytest

from awaf.feedback import BENIGN, MALICIOUS, FeedbackStore
from awaf.ingest import synth_normal
from awaf.models import ALLOW, Request


def test_feedback_store_rejects_bad_label():
    store = FeedbackStore()
    with pytest.raises(ValueError):
        store.add(Request(), "maybe")


def test_feedback_store_summary():
    store = FeedbackStore()
    store.add(Request(url="/a"), BENIGN)
    store.add(Request(url="/b"), MALICIOUS)
    store.add(Request(url="/c"), BENIGN)
    s = store.summary()
    assert s == {"total": 3, "benign": 2, "malicious": 1}
    assert len(store.benign_examples()) == 2


def test_feedback_loop_eliminates_false_positives(fresh_waf):
    production = synth_normal(400, seed=777)
    fps = [r for r in production if fresh_waf.inspect(r).verdict != ALLOW]
    assert fps, "expected at least one baseline false positive to correct"

    for r in fps:
        fresh_waf.report_feedback(r, BENIGN, was="BLOCK")
    fresh_waf.retrain()

    remaining = [r for r in fps if fresh_waf.inspect(r).verdict != ALLOW]
    assert remaining == [], "feedback loop should stop flagging corrected requests"
    assert fresh_waf.retrains == 1


def test_feedback_retrain_preserves_attack_detection(fresh_waf, dataset):
    _, eval_requests, eval_labels = dataset
    before = fresh_waf.evaluate(eval_requests, eval_labels)
    for r in [r for r in synth_normal(400, seed=777) if fresh_waf.inspect(r).verdict != ALLOW]:
        fresh_waf.report_feedback(r, BENIGN)
    fresh_waf.retrain()
    after = fresh_waf.evaluate(eval_requests, eval_labels)
    assert after["recall"] == before["recall"] == 1.0  # attacks still caught
    assert after["fp"] <= before["fp"]  # false positives did not increase
