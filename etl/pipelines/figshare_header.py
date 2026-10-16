"""Figshare corpus, view 2/3 - PE_Header.csv: optional-header fields for 29,807 samples (SHA256 + Type 0-6 + binary label)."""
from __future__ import annotations

import pandas as pd

from .. import catalog, config
from ..common import Pipeline, downcast, drop_duplicates_and_conflicts, read_csv_plain, snake_case, write_table
from ..pe_features import add_header_features
from .figshare_dll import add_labels


class FigshareHeaderPipeline(Pipeline):
    name = "figshare_header"

    def extract(self) -> pd.DataFrame:
        df = read_csv_plain(config.FIGSHARE_HEADER_CSV)
        self.report.step("raw", len(df), df.shape[1])
        return df

    def transform(self, raw: pd.DataFrame):
        df = raw.rename(columns=snake_case).rename(columns={"type": "type_code"})
        df["sha256"] = df["sha256"].str.strip().str.lower()
        df = df.drop_duplicates("sha256").reset_index(drop=True)
        self.report.step("after duplicate-SHA-256 removal", len(df))
        df = add_labels(add_header_features(df))
        staging = downcast(df.drop(columns="type_name"))
        feats = [c for c in staging.columns if c not in ("sha256", "type_code", "label")]
        self.staging, self.meta = staging, df[["sha256", "type_code", "type_name"]]

        model = drop_duplicates_and_conflicts(staging, feats, ["label", "type_code"], self.report)
        constants = [c for c in feats if model[c].nunique() == 1]
        self.report.metric("constant_columns", constants)
        if config.DROP_CONSTANT_COLUMNS:
            model = model.drop(columns=constants)
        self.report.step("final model table", len(model), model.shape[1])
        return {"staging": staging, "model": model}

    def validate(self, data) -> None:
        s, m, r = data["staging"], data["model"], self.report
        r.check("unique_sha256", s["sha256"].is_unique)
        r.check("sha256_format", s["sha256"].str.fullmatch(config.SHA256_RE).all())
        r.check("no_nulls", not s.isna().any().any())
        r.check("e_magic_is_MZ", (s["e_magic"] == 0x5A4D).all())
        r.check("optional_magic_valid", s["magic"].isin([267, 523]).all(), "0x10b PE32 / 0x20b PE32+")
        r.check("type_code_range_0_6", s["type_code"].between(0, 6).all())
        r.check("label_consistent_with_type", ((s["type_code"] == 0) == (s["label"] == 0)).all())
        r.check("number_of_sections_positive", (s["number_of_sections"] >= 1).all())
        bad = int((s["timestamp_valid"] == 0).sum())
        r.check("compile_timestamps_plausible", bad / len(s) < 0.15, f"{bad} rows ({bad/len(s):.1%}) zero/future")
        feats = [c for c in m.columns if c not in ("sha256", "type_code", "label")]
        r.check("no_duplicate_feature_vectors", not m.duplicated(feats).any())
        r.check("no_conflicting_labels", not m.groupby(feats, dropna=False)["type_code"].nunique().gt(1).any())
        r.class_balance(m["label"], "class_balance_binary")
        r.class_balance(m["type_code"], "class_balance_type")

    def load(self, data):
        paths = {k: write_table(v, f"figshare_header_{k}") for k, v in data.items()}
        m = self.meta
        self.report.metric("catalog_rows", catalog.register(
            "figshare_header", m.assign(sample_key=m["sha256"], family=m["type_name"])))
        return paths
