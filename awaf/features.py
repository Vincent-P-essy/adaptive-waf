"""Request feature extraction.

Turns a :class:`Request` into a fixed 11-dimensional numeric vector. The features
are chosen to separate normal traffic from injection/enumeration/scanner activity
without depending on any single signature — that is what the anomaly model keys
on. Extraction is pure and allocation-light so it stays within the latency budget.
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np

from .models import Request

FEATURE_NAMES = [
    "path_depth",
    "path_length",
    "param_count",
    "query_length",
    "body_length",
    "payload_entropy",
    "special_char_ratio",
    "digit_ratio",
    "keyword_hits",
    "ua_length",
    "method_code",
]

# Tokens that recur in SQLi / XSS / traversal / command-injection payloads.
_SUSPICIOUS_TOKENS = [
    "select", "union", "insert", "update", "delete", "drop", "or ", "and ",
    "script", "onerror", "onload", "alert(", "javascript:", "../", "..\\",
    "/etc/passwd", "cmd=", "exec", "system(", "eval(", "0x", "char(", "concat(",
    "information_schema", "sleep(", "benchmark(", "<", ">", "'", "\"", ";",
]
_METHODS = {"GET": 0.0, "POST": 1.0, "HEAD": 2.0, "PUT": 3.0}


def _entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def extract(request: Request) -> np.ndarray:
    payload = request.decoded_payload
    payload_lower = payload.lower()
    n = max(len(payload), 1)

    special = sum(1 for ch in payload if not ch.isalnum() and not ch.isspace())
    digits = sum(1 for ch in payload if ch.isdigit())
    keyword_hits = sum(payload_lower.count(tok) for tok in _SUSPICIOUS_TOKENS)

    vec = [
        float(request.path.count("/")),
        float(len(request.path)),
        float(len(request.params)),
        float(len(request.query_string)),
        float(len(request.body)),
        _entropy(payload),
        special / n,
        digits / n,
        float(keyword_hits),
        float(len(request.user_agent)),
        _METHODS.get(request.method.upper(), 4.0),
    ]
    return np.asarray(vec, dtype=np.float64)


def extract_batch(requests: list[Request]) -> np.ndarray:
    if not requests:
        return np.empty((0, len(FEATURE_NAMES)))
    return np.vstack([extract(r) for r in requests])
