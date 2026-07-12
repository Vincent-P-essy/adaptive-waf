"""Bootstrap trainer.

Builds the corpus, fits the engine and prints evaluation metrics against the
labelled set. Run it to see the WAF's out-of-the-box precision/recall.
"""

from __future__ import annotations

from .config import config
from .ingest import build_dataset, load_csic
from .service import AdaptiveWAF


def build_service() -> tuple[AdaptiveWAF, list, list]:
    """Build a trained service plus a labelled evaluation set."""
    if config.csic_path:
        requests, labels = load_csic()
        normal = [r for r, y in zip(requests, labels, strict=True) if y == 0]
        train = normal[: int(0.7 * len(normal))]
        eval_requests, eval_labels = requests, labels
    else:
        train, eval_requests, eval_labels = build_dataset()
    return AdaptiveWAF(train), eval_requests, eval_labels


def main() -> int:
    waf, eval_requests, eval_labels = build_service()
    m = waf.evaluate(eval_requests, eval_labels)
    print(f"Trained on {len(waf._base)} normal requests; adjudicator: {waf.engine.adjudicator_name}")
    print(f"Evaluation set: {len(eval_requests)} requests "
          f"({sum(eval_labels)} attacks, {len(eval_labels) - sum(eval_labels)} normal)\n")
    print(f"  precision            {m['precision']}")
    print(f"  recall               {m['recall']}")
    print(f"  f1                   {m['f1']}")
    print(f"  false-positive rate  {m['false_positive_rate']}")
    print(f"  est. FP / day (1M)   {m['estimated_fp_per_day']}")
    print(f"  confusion            tp={m['tp']} fp={m['fp']} tn={m['tn']} fn={m['fn']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
