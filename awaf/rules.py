"""Signature (rule) layer — layer 1.

Fast, high-precision detection of *known* attack shapes. A hit here is a
confident BLOCK and short-circuits the pipeline before the ML/LLM layers run. The
same rule objects are exported to ModSecurity ``SecRule`` directives, so the
signature layer is portable to an existing CRS deployment.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .models import Request

# target: which part of the request the pattern is matched against.
TARGET_PAYLOAD = "payload"  # decoded query + body
TARGET_UA = "user_agent"
TARGET_PATH = "path"


@dataclass
class Rule:
    id: str
    name: str
    attack: str
    pattern: str
    target: str = TARGET_PAYLOAD

    def __post_init__(self) -> None:
        self._re = re.compile(self.pattern, re.IGNORECASE)

    def _subject(self, request: Request) -> str:
        if self.target == TARGET_UA:
            return request.user_agent
        if self.target == TARGET_PATH:
            return request.path
        return request.decoded_payload

    def matches(self, request: Request) -> bool:
        return self._re.search(self._subject(request)) is not None


RULES: list[Rule] = [
    Rule("SQLI-1", "SQL injection — boolean/union", "sqli",
         r"(\bunion\b.+\bselect\b|\bor\b\s+['\"]?\d+['\"]?\s*=\s*['\"]?\d+|'\s*or\s*'1'\s*=\s*'1)"),
    Rule("SQLI-2", "SQL injection — comment/stacked/time", "sqli",
         r"(--\s|#$|;\s*drop\b|\bsleep\s*\(|\bbenchmark\s*\(|information_schema)"),
    Rule("XSS-1", "Cross-site scripting", "xss",
         r"(<script\b|onerror\s*=|onload\s*=|javascript:|<img[^>]+src\s*=\s*x)"),
    Rule("LFI-1", "Path traversal / LFI", "path_traversal",
         r"(\.\./|\.\.%2f|\.\.\\|/etc/passwd|/etc/shadow|boot\.ini|win\.ini)"),
    Rule("SSRF-1", "Server-side request forgery", "ssrf",
         r"(169\.254\.169\.254|file://|gopher://|127\.0\.0\.1|localhost:\d+|metadata\.google)"),
    Rule("CMDI-1", "OS command injection", "cmdi",
         r"(;\s*(cat|ls|id|whoami|uname)\b|\|\s*(whoami|id|nc)\b|\$\([^)]+\)|`[^`]+`)"),
    Rule("SCAN-1", "Automated scanner user-agent", "scanner",
         r"(sqlmap|nikto|nmap|masscan|acunetix|dirbuster|wpscan)", target=TARGET_UA),
]


def match(request: Request) -> Rule | None:
    """Return the first rule that fires, or None."""
    for rule in RULES:
        if rule.matches(request):
            return rule
    return None
