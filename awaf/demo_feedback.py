"""Demonstrate the adaptive feedback loop end to end.

1. Train the WAF and measure false positives on a batch of production-like benign
   traffic.
2. An analyst reports the flagged benign requests as false positives.
3. Retrain on the corrected corpus.
4. Re-measure: the same traffic now passes — the model learned from its mistakes.
"""

from __future__ import annotations

from .feedback import BENIGN
from .ingest import build_dataset, synth_normal
from .models import ALLOW
from .service import AdaptiveWAF


def main() -> int:
    train, _, _ = build_dataset()
    waf = AdaptiveWAF(train)

    # A fresh batch of legitimate production traffic.
    production = synth_normal(400, seed=777)

    def false_positives() -> list:
        return [r for r in production if waf.inspect(r).verdict != ALLOW]

    before = false_positives()
    print(f"Baseline: {len(before)}/{len(production)} benign requests wrongly flagged "
          f"(FP rate {len(before)/len(production):.3%})")

    if before:
        sample = before[0]
        print(f"\nExample false positive:\n  {sample.method} {sample.url}")
        print(f"  verdict before feedback: {waf.inspect(sample).verdict}")

    # Analyst labels the flagged requests as benign and the model retrains.
    for r in before:
        waf.report_feedback(r, BENIGN, was="BLOCK")
    result = waf.retrain()
    print(f"\nReported {result['feedback']['benign']} false positives · "
          f"retrained on {result['corpus_size']} requests")

    after = false_positives()
    print(f"\nAfter retrain: {len(after)}/{len(production)} benign requests flagged "
          f"(FP rate {len(after)/len(production):.3%})")

    if before:
        print(f"  verdict after feedback:  {waf.inspect(before[0]).verdict}")

    reduction = len(before) - len(after)
    print(f"\nFalse positives eliminated by the feedback loop: {reduction} "
          f"({reduction / max(len(before), 1):.0%} of them)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
