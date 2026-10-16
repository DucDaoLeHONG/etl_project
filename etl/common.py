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
    for ext in ("csv", "csv.gz", "parquet"):
        path = config.PROCESSED_DIR / f"{name}.{ext}"
        if path.exists():
            return pd.read_parquet(path) if ext == "parquet" else pd.read_csv(path, low_memory=False)
    raise FileNotFoundError(f"processed table {name!r} not found - run its pipeline first")


def find_member(zip_path: Path, basename: str) -> str:
    """Zip member whose file name is `basename` (archives often add a top-level folder)."""
    with zipfile.ZipFile(zip_path) as zf:
        hits = [n for n in zf.namelist() if n.rsplit("/", 1)[-1] == basename]
    if not hits:
        raise FileNotFoundError(f"{basename!r} not found in {zip_path.name}")
    return hits[0]


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


def drop_duplicates_and_conflicts(
    df: pd.DataFrame, feature_cols: list[str], label_cols: list[str], report: "QualityReport", tag: str = ""
) -> pd.DataFrame:
    """Remove (1) every row whose feature vector appears with more than one label, then
    (2) repeated identical feature vectors (first kept). Row counts are logged at each step.

    Identical rows in train and test inflate scores, and one vector with two labels is unlearnable,
    so this must happen BEFORE splitting. Vectors are compared through a 64-bit row hash.
    """
    pre = f"{tag}: " if tag else ""
    report.step(f"{pre}entering de-duplication", len(df), len(feature_cols))
    key = pd.util.hash_pandas_object(df[feature_cols], index=False)
    label = df[label_cols[0]].astype(str)
    for c in label_cols[1:]:
        label = label + "|" + df[c].astype(str)
    conflict = label.groupby(key).transform("nunique") > 1
    report.metric(f"{pre}conflicting_label_groups", int(key[conflict].nunique()))
    report.metric(f"{pre}conflicting_label_rows", int(conflict.sum()))
    if len(label_cols) > 1:  # how many of those conflict even on the coarse label alone
        coarse = df[label_cols[0]].astype(str).groupby(key).transform("nunique") > 1
        report.metric(f"{pre}conflicting_rows_on_{label_cols[0]}_alone", int(coarse.sum()))
    if config.REMOVE_CONFLICTING_LABELS:
        df, key = df[~conflict], key[~conflict]
    report.step(f"{pre}after conflicting-label removal", len(df))
    dup = key.duplicated(keep="first")
    report.metric(f"{pre}duplicate_feature_rows", int(dup.sum()))
    if config.DROP_FEATURE_DUPLICATES:
        df = df[~dup]
    report.step(f"{pre}after duplicate-feature removal", len(df))
    return df.reset_index(drop=True)


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
    if fmt == "parquet" and not _parquet_available():
        raise RuntimeError("OUTPUT_FORMAT='parquet' but pyarrow/fastparquet is not installed")
    return fmt


def write_table(df: pd.DataFrame, name: str) -> Path:
    """Write df to data/processed/<name>.<ext>; returns the path."""
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    ext = output_extension()
    path = config.PROCESSED_DIR / f"{name}.{ext}"
    if ext == "parquet":
        df.to_parquet(path, index=False)
    elif ext == "csv.gz":
        df.to_csv(path, index=False, compression="gzip")
    else:
        df.to_csv(path, index=False)
    log.info("wrote %s  (%d rows x %d cols)", path.name, *df.shape)
    return path


# ----------------------------------------------------------------- quality
class QualityReport:
    """Collects counts and checks during a run; saved as reports/<pipeline>.json."""

    def __init__(self, pipeline: str):
        self.pipeline = pipeline
        self.data: dict[str, Any] = {"pipeline": pipeline, "checks": {}, "metrics": {}}

    def step(self, name: str, rows: int, cols: int | None = None) -> None:
        """Log the row count after a processing step (kept in order, saved in the report)."""
        steps = self.data.setdefault("row_counts", [])
        prev = steps[-1]["rows"] if steps else None
        steps.append({"step": name, "rows": int(rows), **({"cols": int(cols)} if cols is not None else {})})
        delta = "" if prev is None else f"  ({rows - prev:+d})"
        log.info("rows  %-52s %8d%s", name, rows, delta)

    def class_balance(self, labels: pd.Series, name: str = "class_balance") -> None:
        vc = labels.value_counts().sort_index()
        self.data["metrics"][name] = {
            "counts": {str(k): int(v) for k, v in vc.items()},
            "minority_share": round(float(vc.min() / vc.sum()), 4),
            "imbalance_ratio_max_over_min": round(float(vc.max() / max(vc.min(), 1)), 2),
        }

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
