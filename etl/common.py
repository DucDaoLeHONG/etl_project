"""Shared ETL plumbing: zip reading, writers, quality report, pipeline base class."""
from __future__ import annotations

import json
import logging
import re
import time
import zipfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd

from . import config

log = logging.getLogger("etl")


# ----------------------------------------------------------------- extract
def read_csv_from_zip(zip_path: Path, member: str, **kwargs) -> pd.DataFrame:
    """Read one CSV straight out of a zip (no extraction, no path traversal risk)."""
    with zipfile.ZipFile(zip_path) as zf:
        if member not in zf.namelist():
            raise FileNotFoundError(f"{member!r} not in {zip_path.name}: {zf.namelist()}")
        with zf.open(member) as fh:
            return pd.read_csv(fh, low_memory=False, **kwargs)


def read_csv_plain(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - put the file in data/raw/")
    return pd.read_csv(path, low_memory=False, **kwargs)


def read_table(name: str) -> pd.DataFrame:
    """Read back a table written by write_table() (used by pipelines that build on earlier outputs)."""
    for ext in ("parquet", "csv.gz"):
        path = config.PROCESSED_DIR / f"{name}.{ext}"
        if path.exists():
            return pd.read_parquet(path) if ext == "parquet" else pd.read_csv(path, low_memory=False)
    raise FileNotFoundError(f"processed table {name!r} not found - run its pipeline first")


# --------------------------------------------------------------- transform
def snake_case(name: str) -> str:
    """'DIGITAL SIG. ' -> 'digital_sig', 'SHA-256' -> 'sha_256', 'e_lfanew' unchanged."""
    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name.strip())  # camelCase split
    s = re.sub(r"[^0-9a-zA-Z]+", "_", s).strip("_").lower()
    return s


def downcast(df: pd.DataFrame) -> pd.DataFrame:
    """Shrink numeric dtypes (int64 -> uint8/int32...) without changing values."""
    for col in df.select_dtypes(include=["integer"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="unsigned" if df[col].min() >= 0 else "integer")
    for col in df.select_dtypes(include=["floating"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    return df


# -------------------------------------------------------------------- load
def _parquet_available() -> bool:
    for mod in ("pyarrow", "fastparquet"):
        try:
            __import__(mod)
            return True
        except ImportError:
            continue
    return False


def output_extension() -> str:
    fmt = config.OUTPUT_FORMAT
    if fmt == "auto":
        fmt = "parquet" if _parquet_available() else "csv.gz"
    return fmt


def write_table(df: pd.DataFrame, name: str) -> Path:
    """Write df to data/processed/<name>.<ext>; returns the path."""
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    ext = output_extension()
    path = config.PROCESSED_DIR / f"{name}.{ext}"
    if ext == "parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False, compression="gzip")
    log.info("wrote %s  (%d rows x %d cols)", path.name, *df.shape)
    return path


# ----------------------------------------------------------------- quality
class QualityReport:
    """Collects counts and checks during a run; saved as reports/<pipeline>.json."""

    def __init__(self, pipeline: str):
        self.pipeline = pipeline
        self.data: dict[str, Any] = {"pipeline": pipeline, "checks": {}, "metrics": {}}

    def metric(self, key: str, value: Any) -> None:
        self.data["metrics"][key] = value.item() if hasattr(value, "item") else value

    def check(self, name: str, passed: bool, detail: str = "") -> None:
        self.data["checks"][name] = {"passed": bool(passed), "detail": detail}
        (log.info if passed else log.warning)("check %-34s %s %s", name, "OK " if passed else "FAIL", detail)

    def save(self) -> Path:
        config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
        path = config.REPORT_DIR / f"{self.pipeline}.json"
        path.write_text(json.dumps(self.data, indent=2, default=str))
        return path

    @property
    def ok(self) -> bool:
        return all(c["passed"] for c in self.data["checks"].values())


# ---------------------------------------------------------------- pipeline
class Pipeline(ABC):
    """extract -> transform -> validate -> load. Subclasses implement the four steps."""

    name: str = "pipeline"

    def __init__(self) -> None:
        self.report = QualityReport(self.name)

    @abstractmethod
    def extract(self) -> Any: ...
    @abstractmethod
    def transform(self, raw: Any) -> Any: ...
    @abstractmethod
    def validate(self, data: Any) -> None: ...
    @abstractmethod
    def load(self, data: Any) -> dict[str, Path]: ...

    def run(self) -> QualityReport:
        t0 = time.time()
        log.info("=== %s: start", self.name)
        raw = self.extract()
        data = self.transform(raw)
        self.validate(data)
        outputs = self.load(data)
        self.report.data["outputs"] = {k: str(v) for k, v in outputs.items()}
        self.report.metric("seconds", round(time.time() - t0, 2))
        self.report.save()
        log.info("=== %s: done in %.1fs (checks %s)", self.name, time.time() - t0, "all passed" if self.report.ok else "HAD FAILURES")
        return self.report
