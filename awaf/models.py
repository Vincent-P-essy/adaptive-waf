"""Core request/decision models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

ALLOW = "ALLOW"
CHALLENGE = "CHALLENGE"
BLOCK = "BLOCK"


@dataclass
class Request:
    method: str = "GET"
    url: str = "/"
    headers: dict[str, str] = field(default_factory=dict)
    body: str = ""

    # -- parsed views ------------------------------------------------------
    @property
    def split(self):
        return urlsplit(self.url)

    @property
    def path(self) -> str:
        return self.split.path or "/"

    @property
    def query_string(self) -> str:
        return self.split.query

    @property
    def params(self) -> dict[str, list[str]]:
        return parse_qs(self.query_string, keep_blank_values=True)

    @property
    def user_agent(self) -> str:
        for k, v in self.headers.items():
            if k.lower() == "user-agent":
                return v
        return ""

    @property
    def decoded_payload(self) -> str:
        """URL-decoded concatenation of the query and body — what attacks hide in."""
        return unquote(f"{self.query_string} {self.body}")

    def to_dict(self) -> dict[str, Any]:
        return {"method": self.method, "url": self.url, "headers": self.headers, "body": self.body}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Request:
        return cls(
            method=d.get("method", "GET"),
            url=d.get("url", "/"),
            headers=d.get("headers", {}) or {},
            body=d.get("body", "") or "",
        )


@dataclass
class Decision:
    verdict: str  # ALLOW | CHALLENGE | BLOCK
    layer: str  # rules | ml | llm
    reason: str
    anomaly_score: float | None = None
    confidence: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.verdict,
            "layer": self.layer,
            "reason": self.reason,
            "anomaly_score": self.anomaly_score,
            "confidence": self.confidence,
        }
