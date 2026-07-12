"""Traffic ingestion and datasets.

Three entry points:
- :func:`parse_nginx_json` turns a structured nginx access-log line into a Request.
- :func:`build_dataset` returns a train corpus (normal traffic only — the model is
  unsupervised) plus a labelled evaluation set for metrics.
- :func:`load_csic` reads the CSIC 2010 dataset if a CSV path is configured.

The synthetic generators are deterministic (seeded) so training and CI are
reproducible, and are realistic enough that the anomaly model has a genuine
normal/attack boundary to learn.
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from urllib.parse import quote

from .config import config
from .models import Request

_BROWSER_UAS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_4 like Mac OS X) AppleWebKit/605.1.15 Mobile/15E148",
]
_SCANNER_UAS = ["sqlmap/1.7#stable", "Nikto/2.5.0", "Nmap Scripting Engine", "curl/8.4.0", ""]
_TERMS = ["laptop", "phone case", "running shoes", "coffee maker", "desk lamp", "usb+cable", "keyboard"]

_SQLI = [
    "1' OR '1'='1", "1' UNION SELECT username,password FROM users--",
    "'; DROP TABLE orders--", "1 AND SLEEP(5)", "admin'--", "1' OR 1=1#",
]
_XSS = ["<script>alert(1)</script>", "<img src=x onerror=alert(1)>", "javascript:alert(document.cookie)"]
_TRAVERSAL = ["../../../../etc/passwd", "..%2f..%2f..%2fetc/passwd", "....//....//etc/shadow"]
_SSRF = ["http://169.254.169.254/latest/meta-data/", "http://127.0.0.1:6379/", "file:///etc/passwd"]
_CMDI = ["127.0.0.1;cat /etc/passwd", "8.8.8.8|whoami", "$(id)", "`uname -a`"]


def parse_nginx_json(line: str) -> Request:
    """Parse one JSON-formatted nginx access-log line into a Request.

    Expects a log_format emitting JSON with request_method, request_uri,
    http_user_agent and (optionally) request_body.
    """
    rec = json.loads(line)
    headers = {}
    if rec.get("http_user_agent"):
        headers["User-Agent"] = rec["http_user_agent"]
    return Request(
        method=rec.get("request_method", "GET"),
        url=rec.get("request_uri", rec.get("uri", "/")),
        headers=headers,
        body=rec.get("request_body", "") or "",
    )


def _normal(rng: random.Random) -> Request:
    choice = rng.random()
    ua = {"User-Agent": rng.choice(_BROWSER_UAS)}
    if choice < 0.30:
        return Request("GET", f"/products?id={rng.randint(1, 5000)}", ua)
    if choice < 0.50:
        return Request("GET", f"/search?q={quote(rng.choice(_TERMS))}", ua)
    if choice < 0.65:
        return Request("GET", f"/api/v1/orders/{rng.randint(1000, 9999)}", ua)
    if choice < 0.80:
        return Request("GET", rng.choice(["/", "/index.html", "/cart", "/static/app.js"]), ua)
    if choice < 0.92:
        return Request(
            "POST", "/login", ua,
            body=f"username=user{rng.randint(1, 999)}&password=pass{rng.randint(1000, 9999)}",
        )
    return Request("GET", f"/images/{rng.randint(1, 300)}.png", ua)


def _anomalous(rng: random.Random) -> Request:
    kind = rng.choice(["sqli", "xss", "traversal", "ssrf", "cmdi"])
    scanner_ua = {"User-Agent": rng.choice(_SCANNER_UAS)}
    if kind == "sqli":
        return Request("GET", f"/products?id={quote(rng.choice(_SQLI))}", scanner_ua)
    if kind == "xss":
        return Request("GET", f"/search?q={quote(rng.choice(_XSS))}", scanner_ua)
    if kind == "traversal":
        return Request("GET", f"/download?file={quote(rng.choice(_TRAVERSAL))}", scanner_ua)
    if kind == "ssrf":
        return Request("GET", f"/fetch?url={quote(rng.choice(_SSRF))}", scanner_ua)
    return Request("GET", f"/ping?host={quote(rng.choice(_CMDI))}", scanner_ua)


def synth_normal(n: int, seed: int = 0) -> list[Request]:
    rng = random.Random(seed)
    return [_normal(rng) for _ in range(n)]


def synth_anomalous(n: int, seed: int = 1) -> list[Request]:
    rng = random.Random(seed)
    return [_anomalous(rng) for _ in range(n)]


def build_dataset(
    n_train: int = 1500, n_eval_normal: int = 300, n_eval_attack: int = 150
) -> tuple[list[Request], list[Request], list[int]]:
    """Return (train_normal, eval_requests, eval_labels).

    The model trains on normal traffic only. The evaluation set mixes held-out
    normal traffic with attacks; labels are 1 for attack, 0 for normal.
    """
    train = synth_normal(n_train, seed=config.random_seed)
    eval_normal = synth_normal(n_eval_normal, seed=config.random_seed + 100)
    eval_attack = synth_anomalous(n_eval_attack, seed=config.random_seed + 200)
    requests = eval_normal + eval_attack
    labels = [0] * len(eval_normal) + [1] * len(eval_attack)
    # Shuffle deterministically so ordering can't leak into evaluation.
    rng = random.Random(config.random_seed + 300)
    combined = list(zip(requests, labels, strict=True))
    rng.shuffle(combined)
    requests, labels = zip(*combined, strict=True)
    return train, list(requests), list(labels)


def load_csic(path: str | None = None) -> tuple[list[Request], list[int]]:
    """Load the CSIC 2010 dataset from a CSV (columns: method,url,body,label).

    ``label`` is ``anom``/``1`` for anomalous, anything else for normal.
    """
    csv_path = Path(path or config.csic_path or "")
    if not csv_path.exists():
        raise FileNotFoundError(f"CSIC dataset not found at {csv_path}")
    requests: list[Request] = []
    labels: list[int] = []
    with csv_path.open(encoding="utf-8", errors="replace") as fh:
        for row in csv.DictReader(fh):
            requests.append(
                Request(
                    method=row.get("method", "GET"),
                    url=row.get("url", "/"),
                    headers={"User-Agent": row.get("user_agent", "")},
                    body=row.get("body", "") or "",
                )
            )
            label = str(row.get("label", "")).strip().lower()
            labels.append(1 if label in {"anom", "anomalous", "1", "attack"} else 0)
    return requests, labels
