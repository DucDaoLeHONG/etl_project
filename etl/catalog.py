"""SQLite catalog: one row per sample per dataset, so datasets can be cross-referenced."""
from __future__ import annotations

import sqlite3

import pandas as pd

from . import config

COLUMNS = ["dataset", "sample_key", "sha256", "md5", "label", "family"]


def register(dataset: str, rows: pd.DataFrame) -> int:
    """Replace all catalog rows for `dataset` (idempotent re-runs). Returns row count."""
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = rows.reindex(columns=COLUMNS[1:]).assign(dataset=dataset)[COLUMNS]
    with sqlite3.connect(config.CATALOG_DB) as con:
        con.execute(
            "CREATE TABLE IF NOT EXISTS sample_catalog ("
            "dataset TEXT, sample_key TEXT, sha256 TEXT, md5 TEXT, label INTEGER, family TEXT)"
        )
        con.execute("CREATE INDEX IF NOT EXISTS ix_cat_sha ON sample_catalog(sha256)")
        con.execute("DELETE FROM sample_catalog WHERE dataset = ?", (dataset,))
        out.to_sql("sample_catalog", con, if_exists="append", index=False)
    return len(out)


def overlap_report() -> pd.DataFrame:
    """SHA-256 values that appear in more than one dataset (potential train/test leakage)."""
    # The figshare_* tables are views of ONE corpus, so they count as one source.
    q = (
        "SELECT sha256, COUNT(DISTINCT src) AS n_sources, GROUP_CONCAT(DISTINCT dataset) AS datasets "
        "FROM (SELECT sha256, dataset, CASE WHEN dataset LIKE 'figshare_%' THEN 'figshare' ELSE dataset END AS src "
        "FROM sample_catalog WHERE sha256 IS NOT NULL) "
        "GROUP BY sha256 HAVING n_sources > 1"
    )
    with sqlite3.connect(config.CATALOG_DB) as con:
        return pd.read_sql_query(q, con)
