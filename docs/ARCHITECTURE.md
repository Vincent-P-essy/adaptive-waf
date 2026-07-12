# Architecture & Design Notes

## 1. Why three layers

No single technique is the right tool for every request:

- **Signatures** are exact and instant but only catch *known* shapes. They are
  the right answer for an unambiguous SQLi payload — block it in microseconds,
  no model needed.
- **An anomaly model** catches *novel* deviations from normal traffic without a
  signature, but at the cost of false positives and no explanation.
- **An LLM** can reason about an ambiguous request in context, but it is too slow
  and expensive to run on every request.

The engine orders them by cost and certainty: rules first (short-circuit on a
hit), the Isolation Forest second (block clear anomalies, allow clear normals),
and the LLM only for the thin ambiguous band just below the block threshold. In
practice the LLM runs on a small fraction of traffic, which is what keeps the
added latency within budget while still getting a reasoned verdict where it
matters.

## 2. The Isolation Forest, from scratch

The anomaly layer is implemented from first principles rather than imported,
because implementing it is the part worth understanding. Isolation Forest exploits
a simple asymmetry: an anomaly is *easy to isolate*. If you repeatedly split the
data on a random feature at a random value, an outlier gets cut off from the rest
in only a few splits, while a normal point — buried in a dense cluster — needs
many. The **expected path length** to isolate a point, averaged over many random
trees, is therefore short for anomalies and long for normals.

The implementation follows Liu, Ting & Zhou (2008):

- Each tree is grown on a random subsample of size ψ (256) to height
  `ceil(log2 ψ)`; sampling keeps trees decorrelated and bounds depth.
- Unbuilt subtrees are corrected with `c(n)`, the average path length of an
  unsuccessful search in a binary tree of `n` points, so a point that lands in a
  leaf of 30 unsplit points isn't treated as fully isolated.
- The anomaly score is `2^(-E[h(x)] / c(ψ))`, normalising the mean path length to
  `[0, 1]`: ~0.5 is average, →1 is anomalous.

Features are standardised first (from-scratch `StandardScaler`) because splits are
drawn uniformly over each feature's range — without scaling, a large-magnitude
feature like body length would dominate every split.

## 3. Threshold calibration

A fixed anomaly-score cutoff is arbitrary and will either over- or under-block
depending on the traffic. Instead the block threshold is **calibrated on fit** to
a target false-positive rate: it's set to the `(1 − target_fpr)` quantile of the
scores on the training (normal) traffic. That turns a hard-to-reason-about score
threshold into an operational knob — "I can tolerate ~1% of normal traffic being
challenged" — and adapts automatically to the feature distribution.

## 4. The feedback loop — the point of the project

Most ML-security demos predict and stop. This one closes the loop:

1. An analyst marks a request the WAF got wrong — a benign request that was
   blocked (false positive) or an attack that was allowed (false negative).
2. Confirmed-benign requests are folded back into the training corpus,
   **oversampled** by a weight (default 40). Oversampling is what makes the
   correction take: a confirmed-normal point, repeated, makes its region of
   feature space denser, so it becomes *harder to isolate* — its anomaly score
   drops below the threshold and it stops being flagged.
3. The threshold is **pinned** after the initial calibration, so a retrain adapts
   the *scores* to feedback rather than silently moving the decision boundary.
   This makes corrections deterministic and measurable.

End to end, feeding back the false positives from a batch of production traffic
takes precision from ~0.96 to 1.0 while recall stays at 1.0 — the model learns the
operator's traffic without losing attack detection. That is continuous learning
from operator feedback, the pattern most valued in production.

## 5. ModSecurity export

The signature layer exports to ModSecurity `SecRule` directives, so an existing
CRS deployment can adopt the high-precision rules directly. The ML layer stays in
this service by design — an anomaly-model decision boundary is not expressible as
a static SecRule, which is exactly the gap the ML layer fills.

## 6. Datasets and evaluation

The model trains on **normal traffic only** (it is unsupervised); the labelled
evaluation set is used purely to measure precision/recall/F1/FP-rate. Synthetic
generators produce a realistic normal/attack corpus deterministically so the whole
pipeline runs offline and reproducibly, and `load_csic()` reads the CSIC 2010
dataset when a CSV is provided for evaluation against the standard benchmark.
