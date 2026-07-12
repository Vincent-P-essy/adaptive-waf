"""ModSecurity export.

Translates the signature layer into ModSecurity ``SecRule`` directives so the
high-precision rules are portable to an existing ModSecurity / OWASP CRS
deployment. (The ML layer stays in this service — anomaly-model decisions are not
expressible as static SecRules, which is exactly why the ML layer exists.)
"""

from __future__ import annotations

from .rules import RULES, TARGET_PATH, TARGET_UA, Rule

_TARGET_VAR = {
    "payload": "ARGS|REQUEST_BODY|QUERY_STRING",
    TARGET_UA: "REQUEST_HEADERS:User-Agent",
    TARGET_PATH: "REQUEST_URI",
}
_BASE_ID = 9000001


def _rule_to_secrule(rule: Rule, rule_id: int) -> str:
    variable = _TARGET_VAR.get(rule.target, "ARGS|REQUEST_BODY")
    # Escape double quotes in the regex for the SecRule operator string.
    pattern = rule.pattern.replace('"', '\\"')
    return (
        f'SecRule {variable} "@rx {pattern}" \\\n'
        f'    "id:{rule_id},phase:2,deny,status:403,t:none,t:urlDecodeUni,t:lowercase,\\\n'
        f"    msg:'{rule.name}',tag:'adaptive-waf',tag:'attack-{rule.attack}',severity:'CRITICAL'\""
    )


def export_modsecurity(rules: list[Rule] | None = None) -> str:
    rules = rules or RULES
    header = (
        "# Adaptive WAF — exported ModSecurity rules\n"
        "# Signature layer only; the ML anomaly layer runs in the Adaptive WAF service.\n"
        "SecRuleEngine On\n"
    )
    body = "\n\n".join(_rule_to_secrule(r, _BASE_ID + i) for i, r in enumerate(rules))
    return header + "\n" + body + "\n"
