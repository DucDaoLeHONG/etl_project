"""Figshare corpus, view 1/3 - DLLs_Imported.csv (archive__7_): 629 binary "imports this DLL" columns + Type 0-6.

`label` = 0 when Type == config.FIGSHARE_BENIGN_TYPE (0) else 1. See config.py for where that mapping comes from.
Writes `figshare_dll_staging` (de-duplicated by SHA-256 only; consumed by figshare_multiview) and
`figshare_dll_model` (identical DLL vectors and conflicting-label vectors removed).
"""
from __future__ import annotations

import pandas as pd

from .. import catalog, config
from ..common import Pipeline, drop_duplicates_and_conflicts, read_csv_from_zip, write_table


def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    return df.copy().assign(
        label=(df["type_code"] != config.FIGSHARE_BENIGN_TYPE).astype("uint8"),
        type_name=df["type_code"].map(config.FIGSHARE_TYPE_NAMES),
    )


class FigshareDllPipeline(Pipeline):
    name = "figshare_dll"

    def extract(self) -> pd.DataFrame:
        df = read_csv_from_zip(config.FIGSHARE_DLL_ARCHIVE, "DLLs_Imported.csv")
        self.report.step("raw", len(df), df.shape[1])
        return df

    def transform(self, raw: pd.DataFrame):
        df = raw.rename(columns={"SHA256": "sha256", "Type": "type_code"})
        dll_cols = [c.strip().lower() for c in df.columns[2:]]
        df.columns = ["sha256", "type_code"] + dll_cols
        dups = pd.Index(dll_cols)[pd.Index(dll_cols).duplicated()].unique().tolist()
        self.report.metric("dll_columns_merged_after_lowercase", dups)
        if dups:  # merge columns that collapse after lowercasing with logical OR
            df = pd.concat([df[["sha256", "type_code"]], df[dll_cols].T.groupby(level=0).max().T], axis=1)
            dll_cols = list(df.columns[2:])

        df["sha256"] = df["sha256"].str.strip().str.lower()
        conflict = df[df.duplicated("sha256", keep=False)].groupby("sha256")["type_code"].nunique().gt(1).sum()
        self.report.metric("duplicate_sha256_with_conflicting_type", int(conflict))
        df = df.drop_duplicates("sha256", keep="first").reset_index(drop=True)
        self.report.step("after duplicate-SHA-256 removal", len(df))

        bits = df[dll_cols].clip(0, 1).astype("uint8")
        staging = pd.concat([df[["sha256"]], df["type_code"].astype("uint8"),
                             bits.sum(axis=1).astype("uint16").rename("n_dlls"), bits], axis=1)
        staging = add_labels(staging)
        self.dll_cols = dll_cols
        self.staging = staging

        model = drop_duplicates_and_conflicts(staging, dll_cols, ["label", "type_code"], self.report)
        constants = [c for c in dll_cols if model[c].nunique() == 1]
        self.report.metric("constant_columns_dropped", len(constants))
        if config.DROP_CONSTANT_COLUMNS:
            model = model.drop(columns=constants)
        self.report.step("final model table", len(model), model.shape[1])
        return {"staging": staging, "model": model}

    def validate(self, data) -> None:
        s, m, r = data["staging"], data["model"], self.report
        r.check("unique_sha256", s["sha256"].is_unique)
        r.check("sha256_format", s["sha256"].str.fullmatch(config.SHA256_RE).all())
        r.check("no_nulls", not s.drop(columns="type_name").isna().any().any())
        r.check("features_binary", int(s[self.dll_cols].max().max()) <= 1)
        r.check("type_code_range_0_6", s["type_code"].between(0, 6).all())
        r.check("label_consistent_with_type", ((s["type_code"] == 0) == (s["label"] == 0)).all())
        r.check("benign_type_count_1877", int((s["type_code"] == 0).sum()) == 1877, "matches the documented benign count")
        feats = [c for c in m.columns if c.endswith(".dll") or c in self.dll_cols]
        r.check("no_duplicate_feature_vectors", not m.duplicated(feats).any())
        r.check("no_conflicting_labels", not m.groupby(feats, dropna=False)["type_code"].nunique().gt(1).any())
        r.metric("rows_with_zero_imports", int((s["n_dlls"] == 0).sum()))
        r.class_balance(m["label"], "class_balance_binary")
        r.class_balance(m["type_code"], "class_balance_type")

    def load(self, data):
        paths = {k: write_table(v.drop(columns="type_name"), f"figshare_dll_{k}") for k, v in data.items()}
        s = data["staging"]
        self.report.metric("catalog_rows", catalog.register(
            "figshare_dll", s[["sha256", "type_name"]].assign(sample_key=s["sha256"], family=s["type_name"])))
        return paths
