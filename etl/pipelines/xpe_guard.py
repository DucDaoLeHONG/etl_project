"""archive__6_ - XPE_Guard_Final_Dataset.csv: 9,600 PE files, 55 static features, benign vs 3 malware families.

Source quirks handled here: 192 exact duplicate rows, 8 constant (zero-information) columns,
`label` (0/1) and `Family_Label` (text) that must agree.
"""
from __future__ import annotations

import pandas as pd

from .. import catalog, config
from ..common import Pipeline, downcast, read_csv_from_zip, snake_case, write_table

MEMBER = "XPE_Guard_Final_Dataset.csv"


class XpeGuardPipeline(Pipeline):
    name = "xpe_guard"

    def extract(self) -> pd.DataFrame:
        df = read_csv_from_zip(config.XPE_ARCHIVE, MEMBER)
        self.report.metric("rows_raw", len(df))
        return df

    def transform(self, raw: pd.DataFrame) -> dict[str, pd.DataFrame]:
        df = raw.rename(columns=snake_case).rename(columns={"family_label": "family"})
        df["family"] = df["family"].str.strip().str.lower()

        feature_cols = [c for c in df.columns if c not in ("label", "family")]  # before derived columns are added
        # Same features but different label would poison training: count before dropping.
        grp = df[df.duplicated(feature_cols, keep=False)].groupby(feature_cols)["label"].nunique()
        self.report.metric("duplicate_feature_rows_with_conflicting_labels", int((grp > 1).sum()))
        self.report.metric("duplicate_rows_removed", int(df.duplicated().sum()))
        df = df.drop_duplicates().reset_index(drop=True)

        # Columns named is_*/has_* that are NOT 0/1 in the source (they carry raw PE bit values such as
        # 8192 or 64). Keep the raw value and add an unambiguous "<col>_any" boolean (value != 0).
        self.bitfield_cols = [c for c in df.columns if c.startswith(("has_", "is_")) and not set(df[c].unique()) <= {0, 1}]
        self.report.metric("non_binary_flag_columns", self.bitfield_cols)
        for c in self.bitfield_cols:
            df[f"{c}_any"] = (df[c] != 0).astype("uint8")

        df["sample_id"] = [f"xpe_{i:05d}" for i in range(len(df))]  # source has no key
        df["family_id"] = df["family"].astype("category").cat.codes.astype("uint8")
        df = downcast(df)

        constants = [c for c in df.columns if c not in ("label", "family", "family_id", "sample_id") and df[c].nunique(dropna=False) == 1]
        self.report.metric("constant_columns", constants)
        model = df.drop(columns=constants) if config.DROP_CONSTANT_COLUMNS else df
        return {"clean": df, "model": model.drop(columns=["sample_id"])}

    def validate(self, data: dict[str, pd.DataFrame]) -> None:
        df, r = data["clean"], self.report
        feats = df.drop(columns=["sample_id", "family", "family_id", "label"])
        r.check("no_nulls", not df.isna().any().any())
        r.check("no_negative_values", (feats.min() >= 0).all())
        r.check("no_infinite_values", not feats.select_dtypes("floating").isin([float("inf")]).any().any())
        r.check("label_values", set(df["label"].unique()) <= {0, 1})
        r.check("label_matches_family", ((df["family"] == "benign") == (df["label"] == 0)).all())
        flag_cols = [c for c in feats.columns if c.startswith(("has_", "is_")) and c not in self.bitfield_cols]
        r.check("flag_columns_binary", all(set(df[c].unique()) <= {0, 1} for c in flag_cols), f"{len(flag_cols)} flag columns")
        r.check("bitfield_columns_integer", all(pd.api.types.is_integer_dtype(df[c]) for c in self.bitfield_cols), str(self.bitfield_cols))
        r.check("min_le_avg_le_max_entropy", ((df["min_entropy"] <= df["avg_entropy"] + 1e-6) & (df["avg_entropy"] <= df["max_entropy"] + 1e-6)).all())
        r.check("min_le_max_section_size", (df["min_section_size"] <= df["max_section_size"]).all())
        r.metric("rows_clean", len(df))
        r.metric("family_counts", df["family"].value_counts().to_dict())

    def load(self, data):
        paths = {k: write_table(v, f"xpe_guard_{k}") for k, v in data.items()}
        c = data["clean"]
        n = catalog.register("xpe_guard", c.assign(sample_key=c["sample_id"]))
        self.report.metric("catalog_rows", n)
        return paths
