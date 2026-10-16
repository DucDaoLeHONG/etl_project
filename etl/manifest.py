"""Rubric R7 helper: checksum every raw file listed in data_sources.csv and flag missing provenance.

Checksums prove a backup/teammate copy is byte-identical. Licences cannot be inferred from the data, so
`licence` / `source_url` / `backup_location` in data_sources.csv must be filled in by hand; this check
lists what is still missing.
"""
from __future__ import annotations

import hashlib
import logging

import pandas as pd

from . import config

log = logging.getLogger("etl")


def _sha256(path, chunk=1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def check_sources() -> pd.DataFrame:
    src = pd.read_csv(config.SOURCES_CSV, dtype=str).fillna("")
    rows = []
    for r in src.itertuples():
        candidates = [r.raw_file] + (["PE-Malware-Dataset1-main.zip"] if r.dataset == "pe_malware_dataset1" else [])
        path = next((config.RAW_DIR / c for c in candidates if (config.RAW_DIR / c).exists()), None)
        missing = [f for f in ("source_url", "licence", "backup_location") if not getattr(r, f).strip()]
        rows.append({**r._asdict(), "present": path is not None, "bytes": path.stat().st_size if path else None,
                     "sha256": _sha256(path) if path else "", "missing_fields": ",".join(missing)})
    out = pd.DataFrame(rows).drop(columns="Index")
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(config.REPORT_DIR / "data_manifest.csv", index=False)
    n_missing = int((out["missing_fields"] != "").sum())
    log.log(logging.WARNING if n_missing else logging.INFO,
            "R7 sources: %d/%d files present, %d rows still missing licence/url/backup details (see reports/data_manifest.csv)",
            int(out["present"].sum()), len(out), n_missing)
    return out
