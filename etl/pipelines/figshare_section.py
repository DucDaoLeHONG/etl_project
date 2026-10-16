"""Figshare corpus, view 3/3 - PE_Section.csv: 9 fields x 10 well-known sections (29,760 samples, Type 0-6 + binary label).

Absent sections are all zeros in the source; presence/exec/write flags and totals are derived in pe_features.py.
"""
from __future__ import annotations

import pandas as pd

from .. import catalog, config
from ..common import Pipeline, downcast, drop_duplicates_and_conflicts, read_csv_plain, snake_case, write_table
from ..pe_features import SECTIONS, add_section_features
from .figshare_dll import add_labels


class FigshareSectionPipeline(Pipeline):
    name = "figshare_section"

    def extract(self) -> pd.DataFrame:
        df = read_csv_plain(config.FIGSHARE_SECTION_CSV)
        self.report.step("raw", len(df), df.shape[1])
        return df

    def transform(self, raw: pd.DataFrame):
        df = raw.rename(columns=snake_case).rename(columns={"type": "type_code"})
        df["sha256"] = df["sha256"].str.strip().str.lower()
        df = df.drop_duplicates("sha256").reset_index(drop=True)
        self.report.step("after duplicate-SHA-256 removal", len(df))
        df = add_labels(add_section_features(df))
        staging = downcast(df.drop(columns="type_name"))
        feats = [c for c in staging.columns if c not in ("sha256", "type_code", "label")]
        self.staging, self.meta = staging, df[["sha256", "type_code", "type_name"]]

        model = drop_duplicates_and_conflicts(staging, feats, ["label", "type_code"], self.report)
        constants = [c for c in feats if model[c].nunique(dropna=False) == 1]
        self.report.metric("constant_columns", constants)
        if config.DROP_CONSTANT_COLUMNS:
            model = model.drop(columns=constants)
        self.report.step("final model table", len(model), model.shape[1])
        return {"staging": staging, "model": model}

    def validate(self, data) -> None:
        s, m, r = data["staging"], data["model"], self.report
        r.check("unique_sha256", s["sha256"].is_unique)
        r.check("sha256_format", s["sha256"].str.fullmatch(config.SHA256_RE).all())
        r.check("no_negative_values", (s.drop(columns="sha256").select_dtypes("number").min() >= 0).all())
        r.check("no_nulls_except_ratio", not s.drop(columns="raw_to_virtual_ratio").isna().any().any())
        r.check("type_code_range_0_6", s["type_code"].between(0, 6).all())
        r.check("label_consistent_with_type", ((s["type_code"] == 0) == (s["label"] == 0)).all())
        r.check("text_section_common", s["text_present"].mean() > 0.8, f"{s['text_present'].mean():.1%} have .text")
        feats = [c for c in m.columns if c not in ("sha256", "type_code", "label")]
        r.check("no_duplicate_feature_vectors", not m.duplicated(feats).any())
        r.check("no_conflicting_labels", not m.groupby(feats, dropna=False)["type_code"].nunique().gt(1).any())
        r.metric("samples_with_wx_section", int((s["n_wx_sections"] > 0).sum()))
        r.metric("section_prevalence", {x: round(float(s[f"{x}_present"].mean()), 4) for x in SECTIONS})
        r.class_balance(m["label"], "class_balance_binary")
        r.class_balance(m["type_code"], "class_balance_type")

    def load(self, data):
        paths = {k: write_table(v, f"figshare_section_{k}") for k, v in data.items()}
        m = self.meta
        self.report.metric("catalog_rows", catalog.register(
            "figshare_section", m.assign(sample_key=m["sha256"], family=m["type_name"])))
        return paths
