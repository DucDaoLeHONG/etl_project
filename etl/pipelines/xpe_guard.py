"""XPE-Guard (archive__6_): 9,600 PE files, 55 static features, benign + 3 malware families.

Quirks handled: duplicate rows, constant columns, `label` (0/1) vs `Family_Label` (text) must agree,
four columns named is_*/has_* that hold raw PE bit values (8192, 64, 256...) instead of 0/1.
"""
from __future__ import annotations

import pandas as pd

from .. import catalog, config
from ..common import Pipeline, downcast, drop_duplicates_and_conflicts, read_csv_from_zip, snake_case, write_table


class XpeGuardPipeline(Pipeline):
    name = "xpe_guard"

    def extract(self) -> pd.DataFrame:
        df = read_csv_from_zip(config.XPE_ARCHIVE, "XPE_Guard_Final_Dataset.csv")
        self.report.step("raw", len(df), df.shape[1])
        return df

    def transform(self, raw: pd.DataFrame) -> dict[str, pd.DataFrame]:
        df = raw.rename(columns=snake_case).rename(columns={"family_label": "family"})
        df["family"] = df["family"].str.strip().str.lower()
        base = [c for c in df.columns if c not in ("label", "family")]

        # Columns named is_*/has_* that are not 0/1: keep the raw value, add an unambiguous "<col>_any" flag.
        self.bitfield_cols = [c for c in base if c.startswith(("has_", "is_")) and not set(df[c].unique()) <= {0, 1}]
        self.report.metric("non_binary_flag_columns", self.bitfield_cols)
        for c in self.bitfield_cols:
            df[f"{c}_any"] = (df[c] != 0).astype("uint8")
        feats = base + [f"{c}_any" for c in self.bitfield_cols]
        df["family_id"] = df["family"].astype("category").cat.codes.astype("uint8")
        df = downcast(df)

        model = drop_duplicates_and_conflicts(df, feats, ["label", "family"], self.report)
        constants = [c for c in feats if model[c].nunique(dropna=False) == 1]
        self.report.metric("constant_columns", constants)
        if config.DROP_CONSTANT_COLUMNS:
            model = model.drop(columns=constants)
        model.insert(0, "sample_id", [f"xpe_{i:05d}" for i in range(len(model))])  # source has no key
        self.report.step("final model table", len(model), model.shape[1])
        return {"model": model}

    def validate(self, data) -> None:
        m, r = data["model"], self.report
        feats = [c for c in m.columns if c not in ("sample_id", "label", "family", "family_id")]
        r.check("no_nulls", not m.isna().any().any())
        r.check("no_negative_values", (m[feats].min() >= 0).all())
        r.check("label_values", set(m["label"].unique()) <= {0, 1})
        r.check("label_matches_family", ((m["family"] == "benign") == (m["label"] == 0)).all())
        flags = [c for c in feats if c.startswith(("has_", "is_")) and c not in self.bitfield_cols]
        r.check("flag_columns_binary", all(set(m[c].unique()) <= {0, 1} for c in flags), f"{len(flags)} flag columns")
        r.check("min_le_max_entropy", (m["min_entropy"] <= m["max_entropy"] + 1e-6).all() if "min_entropy" in m else True)
        r.check("no_duplicate_feature_vectors", not m.duplicated(feats).any())
        r.check("no_conflicting_labels", not m.groupby(feats, dropna=False)["family"].nunique().gt(1).any())
        r.class_balance(m["label"], "class_balance_binary")
        r.class_balance(m["family"], "class_balance_family")

    def load(self, data):
        m = data["model"]
        paths = {"model": write_table(m, "xpe_guard_model")}
        self.report.metric("catalog_rows", catalog.register("xpe_guard", m.assign(sample_key=m["sample_id"])))
        return paths
