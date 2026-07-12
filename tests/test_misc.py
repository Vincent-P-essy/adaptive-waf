"""Tests for LLM adjudicator, metrics, ingest and ModSecurity export."""

from __future__ import annotations

from awaf.ingest import build_dataset, parse_nginx_json, synth_anomalous, synth_normal
from awaf.llm import AnthropicAdjudicator, HeuristicAdjudicator
from awaf.models import Request
from awaf.modsec_export import export_modsecurity
from awaf.rules import RULES


def test_heuristic_adjudicator_flags_attack():
    adj = HeuristicAdjudicator()
    mal = adj.adjudicate(Request("GET", "/x?q=' OR 1=1-- UNION SELECT",
                                 {"User-Agent": "sqlmap"}))
    ben = adj.adjudicate(Request("GET", "/products?id=42",
                                 {"User-Agent": "Mozilla/5.0"}))
    assert mal.malicious is True
    assert ben.malicious is False


def test_anthropic_adjudicator_parses_and_falls_back(monkeypatch):
    class _Block:
        type = "text"
        text = "VERDICT: MALICIOUS | CONFIDENCE: 0.9 | REASON: injection tokens"

    class _Resp:
        content = [_Block()]

    class _Messages:
        def create(self, **_):
            return _Resp()

    adj = AnthropicAdjudicator.__new__(AnthropicAdjudicator)
    adj._client = type("C", (), {"messages": _Messages()})()
    adj._model = "claude-opus-4-8"
    adj._fallback = HeuristicAdjudicator()
    out = adj.adjudicate(Request("GET", "/x?q=test"))
    assert out.malicious is True and out.confidence == 0.9


def test_nginx_json_parsing():
    line = '{"request_method":"POST","request_uri":"/login","http_user_agent":"curl/8","request_body":"a=1"}'
    r = parse_nginx_json(line)
    assert r.method == "POST" and r.path == "/login" and r.body == "a=1"
    assert r.user_agent == "curl/8"


def test_dataset_shapes_and_labels():
    train, requests, labels = build_dataset(n_train=100, n_eval_normal=30, n_eval_attack=20)
    assert len(train) == 100
    assert len(requests) == len(labels) == 50
    assert sum(labels) == 20  # attacks labelled 1


def test_synth_generators_are_deterministic():
    assert [r.url for r in synth_normal(10, seed=5)] == [r.url for r in synth_normal(10, seed=5)]
    assert [r.url for r in synth_anomalous(10, seed=5)] == [r.url for r in synth_anomalous(10, seed=5)]


def test_modsecurity_export_has_a_rule_per_signature():
    text = export_modsecurity()
    assert text.count("@rx") == len(RULES)  # one operator per exported rule
    assert "id:9000001" in text
    assert "phase:2,deny" in text
