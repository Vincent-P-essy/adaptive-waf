"""Environment-driven configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    anthropic_api_key: str | None = os.getenv("ANTHROPIC_API_KEY") or None
    model: str = os.getenv("AWAF_MODEL", "claude-opus-4-8")
    block_threshold: float = float(os.getenv("AWAF_BLOCK_THRESHOLD", "0.62"))
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
