"""Tests for the rules layer and the 3-layer engine."""

from __future__ import annotations

from awaf import rules
from awaf.models import ALLOW, BLOCK, Request


def test_each_attack_signature_fires():
    cases = {
        "sqli": Request("GET", "/x?id=1' OR 1=1--"),
        "xss": Request("GET", "/x?q=<script>alert(1)</script>"),
        "path_traversal": Request("GET", "/x?f=../../etc/passwd"),
        "ssrf": Request("GET", "/x?u=http://169.254.169.254/"),
        "cmdi": Request("GET", "/x?h=127.0.0.1;whoami"),
    }
    for attack, req in cases.items():
        rule = rules.match(req)
        assert rule is not None and rule.attack == attack, attack


def test_scanner_user_agent_signature():
    r = Request("GET", "/", {"User-Agent": "sqlmap/1.7"})
    assert rules.match(r).attack == "scanner"


def test_rules_do_not_fire_on_normal():
    assert rules.match(Request("GET", "/products?id=42",
                               {"User-Agent": "Mozilla/5.0 Firefox/125.0"})) is None


def test_engine_blocks_sqli_via_rules(trained_waf):
    d = trained_waf.inspect(Request("GET", "/products?id=1' UNION SELECT * FROM users--"))
    assert d.verdict == BLOCK and d.layer == "rules"


def test_engine_allows_normal(trained_waf):
    d = trained_waf.inspect(Request("GET", "/products?id=42",
                                    {"User-Agent": "Mozilla/5.0 Firefox/125.0"}))
    assert d.verdict == ALLOW


def test_engine_threshold_is_calibrated(trained_waf):
    # Calibration should not leave the default 0.62 in place.
    assert trained_waf.engine.block_threshold != 0.62
    assert 0.5 < trained_waf.engine.block_threshold < 0.9


def test_engine_persistence_round_trip(trained_waf):
    from awaf.engine import WAFEngine

    d = trained_waf.engine.to_dict()
    restored = WAFEngine.from_dict(d)
    r = Request("GET", "/products?id=42", {"User-Agent": "Mozilla/5.0 Firefox/125.0"})
    assert restored.anomaly_score(r) == trained_waf.engine.anomaly_score(r)
