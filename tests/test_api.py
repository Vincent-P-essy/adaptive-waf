"""API endpoint tests."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from awaf.api import create_app


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(create_app())


def test_healthz(client):
    d = client.get("/healthz").json()
    assert d["status"] == "ok" and d["fitted"] is True


def test_inspect_blocks_sqli(client):
    d = client.post("/inspect", json={
        "method": "GET", "url": "/products?id=1' OR 1=1--",
        "headers": {"User-Agent": "sqlmap/1.7"}, "body": "",
    }).json()
    assert d["decision"] == "BLOCK"
    assert d["latency_ms"] < 5.0  # signature layer is well within budget


def test_inspect_allows_normal(client):
    d = client.post("/inspect", json={
        "method": "GET", "url": "/products?id=42",
        "headers": {"User-Agent": "Mozilla/5.0 Firefox/125.0"}, "body": "",
    }).json()
    assert d["decision"] == "ALLOW"


def test_metrics_endpoint(client):
    m = client.get("/metrics").json()
    assert m["recall"] == 1.0
    assert 0.0 <= m["precision"] <= 1.0


def test_rules_export_endpoint(client):
    text = client.get("/rules/export").text
    assert "SecRule" in text


def test_feedback_and_retrain_endpoints(client):
    fb = client.post("/feedback", json={
        "request": {"method": "GET", "url": "/products?id=7", "headers": {}, "body": ""},
        "label": "benign",
    }).json()
    assert fb["benign"] >= 1
    rt = client.post("/retrain").json()
    assert rt["retrains"] >= 1


def test_index_served(client):
    assert client.get("/").status_code == 200
