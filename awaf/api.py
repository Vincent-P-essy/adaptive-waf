"""FastAPI service.

Exposes the WAF: inspect a request, submit analyst feedback, trigger a retrain,
read evaluation metrics, and export ModSecurity rules. A trained AdaptiveWAF is
built at startup from the bootstrap corpus, and the labelled evaluation set is
kept so ``/metrics`` reflects live model quality after retrains.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .models import Request as WafRequest
from .modsec_export import export_modsecurity
from .train import build_service

WEB_DIR = Path(__file__).resolve().parents[1] / "web"


class RequestModel(BaseModel):
    method: str = "GET"
    url: str = "/"
    headers: dict[str, str] = {}
    body: str = ""


class FeedbackModel(BaseModel):
    request: RequestModel
    label: str  # "benign" | "malicious"
    was: str = ""


def create_app() -> FastAPI:
    app = FastAPI(title="Adaptive WAF", version="0.1.0")
    waf, eval_requests, eval_labels = build_service()
    state: dict[str, Any] = {"waf": waf, "eval": (eval_requests, eval_labels)}

    @app.get("/healthz")
    def healthz() -> dict:
        w = state["waf"]
        return {
            "status": "ok",
            "fitted": w.engine.fitted,
            "adjudicator": w.engine.adjudicator_name,
            "corpus_size": len(w._base),
            "retrains": w.retrains,
            "feedback": w.feedback.summary(),
        }

    @app.post("/inspect")
    def inspect(req: RequestModel) -> dict:
        r = WafRequest.from_dict(req.model_dump())
        start = time.perf_counter()
        decision = state["waf"].inspect(r)
        latency_ms = (time.perf_counter() - start) * 1000
        return {**decision.to_dict(), "latency_ms": round(latency_ms, 3)}

    @app.post("/feedback")
    def feedback(fb: FeedbackModel) -> dict:
        r = WafRequest.from_dict(fb.request.model_dump())
        return state["waf"].report_feedback(r, fb.label, fb.was)

    @app.post("/retrain")
    def retrain() -> dict:
        return state["waf"].retrain()

    @app.get("/metrics")
    def metrics() -> dict:
        reqs, labels = state["eval"]
        return state["waf"].evaluate(reqs, labels)

    @app.get("/rules/export", response_class=PlainTextResponse)
    def rules_export() -> str:
        return export_modsecurity()

    if WEB_DIR.exists():
        app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")

    return app


app = create_app()
