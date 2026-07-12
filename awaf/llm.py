"""LLM adjudicator — layer 3.

Only the narrow ambiguous band from the ML layer reaches here — requests that are
neither clearly normal nor clearly anomalous. Claude adjudicates when a key is
set; otherwise a feature-based heuristic decides, so the layer always returns a
verdict offline. Keeping this layer to a thin band is what preserves latency: the
expensive path runs on a small fraction of traffic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .config import config
from .features import _SUSPICIOUS_TOKENS
from .models import Request

_SYSTEM = """\
You are a web application firewall adjudicator. Decide whether a single HTTP
request is malicious (an attack such as SQL injection, XSS, path traversal, SSRF,
or command injection) or benign. Reply with exactly one line:
VERDICT: MALICIOUS|BENIGN | CONFIDENCE: 0.0-1.0 | REASON: <short reason>
"""


@dataclass
class Adjudication:
    malicious: bool
    confidence: float
    reason: str


class HeuristicAdjudicator:
    name = "heuristic"

    def adjudicate(self, request: Request) -> Adjudication:
        payload = request.decoded_payload.lower()
        hits = sum(payload.count(tok) for tok in _SUSPICIOUS_TOKENS)
        special = sum(1 for ch in payload if not ch.isalnum() and not ch.isspace())
        ratio = special / max(len(payload), 1)
        scanner = bool(re.search(r"sqlmap|nikto|nmap|masscan", request.user_agent, re.I))
        score = min(1.0, 0.12 * hits + 1.4 * ratio + (0.4 if scanner else 0.0))
        malicious = score >= 0.5
        return Adjudication(
            malicious=malicious,
            confidence=round(abs(score - 0.5) * 2, 2),
            reason=f"heuristic score {score:.2f} ({hits} keyword hits, {ratio:.2f} special-char ratio)",
        )


class AnthropicAdjudicator:
    name = "anthropic"

    def __init__(self) -> None:
        import anthropic

        self._client = anthropic.Anthropic(api_key=config.anthropic_api_key)
        self._model = config.model
        self._fallback = HeuristicAdjudicator()

    def adjudicate(self, request: Request) -> Adjudication:
        summary = (
            f"method={request.method} path={request.path} "
            f"query={request.query_string!r} body={request.body[:200]!r} "
            f"user_agent={request.user_agent!r}"
        )
        try:
            resp = self._client.messages.create(
                model=self._model,
                max_tokens=200,
                system=_SYSTEM,
                messages=[{"role": "user", "content": summary}],
            )
            text = " ".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
            return self._parse(text) or self._fallback.adjudicate(request)
        except Exception:  # pragma: no cover - network/SDK failure falls back
            return self._fallback.adjudicate(request)

    @staticmethod
    def _parse(text: str) -> Adjudication | None:
        m = re.search(r"VERDICT:\s*(MALICIOUS|BENIGN)", text, re.I)
        if not m:
            return None
        conf = re.search(r"CONFIDENCE:\s*([01](?:\.\d+)?)", text, re.I)
        reason = re.search(r"REASON:\s*(.+)", text, re.I)
        return Adjudication(
            malicious=m.group(1).upper() == "MALICIOUS",
            confidence=float(conf.group(1)) if conf else 0.5,
            reason=(reason.group(1).strip() if reason else "LLM adjudication"),
        )


def build_adjudicator() -> HeuristicAdjudicator | AnthropicAdjudicator:
    if config.llm_enabled:
        try:
            return AnthropicAdjudicator()
        except Exception:  # pragma: no cover
            return HeuristicAdjudicator()
    return HeuristicAdjudicator()
