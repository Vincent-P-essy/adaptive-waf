"""Environment-driven configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    anthropic_api_key: str | None = os.getenv("ANTHROPIC_API_KEY") or None
    model: str = os.getenv("AWAF_MODEL", "claude-opus-4-8")
    # If AWAF_BLOCK_THRESHOLD is unset the threshold is *calibrated* on fit to the
    # target false-positive rate on training traffic (recommended). Set it to pin
    # a fixed anomaly-score cutoff instead.
    block_threshold: float | None = (
        float(os.environ["AWAF_BLOCK_THRESHOLD"]) if os.getenv("AWAF_BLOCK_THRESHOLD") else None
    )
    target_fpr: float = float(os.getenv("AWAF_TARGET_FPR", "0.01"))
    ambiguous_band: float = float(os.getenv("AWAF_AMBIGUOUS_BAND", "0.08"))
    csic_path: str | None = os.getenv("CSIC_PATH") or None
    n_trees: int = int(os.getenv("AWAF_N_TREES", "120"))
    sample_size: int = int(os.getenv("AWAF_SAMPLE_SIZE", "256"))
    random_seed: int = int(os.getenv("AWAF_SEED", "42"))
    # How strongly an analyst-confirmed benign request influences retraining.
    # Confirmed normals are high-value signal, so they are oversampled.
    feedback_weight: int = int(os.getenv("AWAF_FEEDBACK_WEIGHT", "40"))

    @property
    def llm_enabled(self) -> bool:
        return self.anthropic_api_key is not None


config = Config()
