"""Seeded stratified train/val/test assignment (numpy only).

Run on the CLEANED tables (duplicates and conflicting labels already removed), so identical feature
vectors can never land on both sides of a split. Classes with fewer than 3 rows stay in train.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def stratified_split(y: pd.Series, fractions: dict[str, float] | None = None, seed: int | None = None) -> pd.Series:
    fractions = fractions or config.SPLIT_FRACTIONS
    rng = np.random.default_rng(config.SPLIT_SEED if seed is None else seed)
    y = y.reset_index(drop=True)
    out = np.full(len(y), "train", dtype=object)
    for _, idx in y.groupby(y).indices.items():
        idx = rng.permutation(idx)
        n = len(idx)
        if n < 3:
            continue
        n_test = max(1, round(n * fractions["test"]))
        n_val = max(1, round(n * fractions["val"]))
        out[idx[:n_test]] = "test"
        out[idx[n_test:n_test + n_val]] = "val"
    return pd.Series(out, name="split")
