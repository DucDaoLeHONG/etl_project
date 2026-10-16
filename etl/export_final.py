"""Copy the model-ready tables to data/final/ as PLAIN CSV with a stratified `split` column.

Only tables listed in config.FINAL_TABLES are exported: staging tables, hash/metadata tables and the
scratch catalog stay in data/processed and are not part of the submission.
"""
from __future__ import annotations

import logging

import pandas as pd

from . import config
from .common import read_table
from .splits import stratified_split

log = logging.getLogger("etl")


def export_final() -> pd.DataFrame:
    config.FINAL_DIR.mkdir(parents=True, exist_ok=True)
    for old in config.FINAL_DIR.glob("*"):
        old.unlink()
    rows = []
    for name, strat_col in config.FINAL_TABLES.items():
        df = read_table(f"{name}_model")
        df = df.copy().assign(split=stratified_split(df[strat_col]).values)
        out = config.FINAL_DIR / f"{name}.csv"
        df.to_csv(out, index=False)
        log.info("final  %-22s %6d rows x %3d cols  stratified on %s", out.name, *df.shape, strat_col)
        ct = df.groupby(["split", strat_col]).size().rename("rows").reset_index().rename(columns={strat_col: "class"})
        ct.insert(0, "dataset", name)
        ct.insert(1, "stratified_on", strat_col)
        rows.append(ct)
    summary = pd.concat(rows, ignore_index=True)
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(config.REPORT_DIR / "class_balance_by_split.csv", index=False)
    return summary
