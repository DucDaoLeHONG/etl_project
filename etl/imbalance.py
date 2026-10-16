"""Class-imbalance helpers (rubric R1). The strategy is documented in README.md ("Class imbalance").

Rules this module enforces by design:
  * split FIRST (etl/splits.py), then balance - resampling before the split leaks copies into test;
  * only ever touch the TRAIN rows; validation/test keep their natural class distribution.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def class_weights(y_train: pd.Series) -> dict:
    """'Balanced' weights n / (k * n_c); pass to class_weight= / sample_weight= of your classifier."""
    counts = y_train.value_counts()
    return {cls: len(y_train) / (len(counts) * n) for cls, n in counts.items()}


def random_resample(train: pd.DataFrame, label: str, strategy: str = "over", seed: int = 42) -> pd.DataFrame:
    """Random over-sampling (copies minority rows up to the majority size) or under-sampling (down to the minority size)."""
    rng = np.random.default_rng(seed)
    sizes = train[label].value_counts()
    target = sizes.max() if strategy == "over" else sizes.min()
    parts = []
    for cls, n in sizes.items():
        rows = train[train[label] == cls]
        parts.append(rows.iloc[rng.choice(n, size=target, replace=(target > n))])
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)
