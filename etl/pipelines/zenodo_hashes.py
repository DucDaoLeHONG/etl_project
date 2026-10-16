"""Zenodo dataset (15038698.zip): 1,000 malicious + 1,000 benign samples with hashes and import lists.

Quirks handled: spaced/dashed column names, whitespace in labels, duplicate SHA-256 rows,
LIBRARIES / FUNCTIONS stored as comma-separated strings (sometimes empty).
Model table = numeric features + one-hot of the TOP_N most common imported DLLs.
"""
from __future__ import annotations

import pandas as pd

from .. import catalog, config
from ..common import Pipeline, downcast, drop_duplicates_and_conflicts, read_csv_from_zip, snake_case, write_table

FILES = {"1000_malware_sample_malicious.csv": "malicious", "1000_whitelist_sample_benign.csv": "benign"}


def split_list(value) -> list[str]:
    """'a.dll, B.dll,,a.dll' -> ['a.dll', 'b.dll'] (lowercased, trimmed, de-duplicated, order kept)."""
    if pd.isna(value):
        return []
    seen, out = set(), []
    for tok in str(value).split(","):
        tok = tok.strip().lower()
        if tok and tok not in seen:
            seen.add(tok)
            out.append(tok)
    return out


class ZenodoHashesPipeline(Pipeline):
    name = "zenodo_hashes"

    def extract(self) -> pd.DataFrame:
        frames = []
        for member, kind in FILES.items():
            df = read_csv_from_zip(config.ZENODO_ARCHIVE, member, dtype={"MD5": str, "SHA-1": str, "SHA-256": str})
            df["source_file"] = member
            self.report.step(f"raw {kind} file", len(df))
            frames.append(df)
        out = pd.concat(frames, ignore_index=True)
        self.report.step("raw files combined", len(out))
        return out

    def transform(self, raw: pd.DataFrame) -> dict[str, pd.DataFrame]:
        df = raw.rename(columns=snake_case).rename(
            columns={"sha_1": "sha1", "sha_256": "sha256", "digital_sig": "is_signed", "classification": "label_name"})
        for col in ("md5", "sha1", "sha256"):
            df[col] = df[col].str.strip().str.lower()
        df["label_name"] = df["label_name"].str.strip().str.lower()
        df["label"] = (df["label_name"] == "malicious").astype("uint8")

        dup = df[df.duplicated("sha256", keep=False)].groupby("sha256")["label"].nunique().gt(1).sum()
        self.report.metric("duplicate_sha256_with_conflicting_labels", int(dup))
        df = df.drop_duplicates("sha256", keep="first").reset_index(drop=True)
        self.report.step("after duplicate-SHA-256 removal", len(df))

        df["libraries_list"] = df["libraries"].map(split_list)
        df["functions_list"] = df["functions"].map(split_list)
        df["has_import_info"] = (df["libraries_list"].str.len() > 0).astype("uint8")
        df["n_libraries"] = df["libraries_list"].str.len()
        df["n_functions"] = df["functions_list"].str.len()

        imports = df[["sha256", "libraries_list"]].explode("libraries_list").dropna().reset_index(drop=True)
        top = imports["libraries_list"].value_counts().head(config.TOP_N_LIBRARIES).index
        onehot = pd.crosstab(imports["sha256"], imports["libraries_list"]).reindex(columns=top, fill_value=0).clip(upper=1)
        onehot.columns = [f"lib_{c}" for c in onehot.columns]
        base = ["size", "is_signed", "has_import_info", "n_libraries", "n_functions"]
        feats = df[["sha256", "label"] + base].merge(onehot, how="left", left_on="sha256", right_index=True)
        lib_cols = [c for c in feats.columns if c.startswith("lib_")]
        feats[lib_cols] = feats[lib_cols].fillna(0)
        feats = downcast(feats)
        self.meta = df[["sha256", "md5", "label", "label_name"]]
        self.report.step("feature table built", len(feats), feats.shape[1])

        model = drop_duplicates_and_conflicts(feats, base + lib_cols, ["label"], self.report)
        constants = [c for c in base + lib_cols if model[c].nunique() == 1]
        self.report.metric("constant_columns", constants)
        if config.DROP_CONSTANT_COLUMNS:
            model = model.drop(columns=constants)
        self.report.step("final model table", len(model), model.shape[1])
        return {"model": model}

    def validate(self, data) -> None:
        m, r = data["model"], self.report
        r.check("unique_sha256", m["sha256"].is_unique)
        r.check("sha256_format", m["sha256"].str.fullmatch(config.SHA256_RE).all())
        r.check("label_values", set(m["label"].unique()) <= {0, 1})
        r.check("both_classes_present", m["label"].nunique() == 2)
        r.check("size_positive", (m["size"] > 0).all())
        r.check("no_nulls", not m.isna().any().any())
        feats = [c for c in m.columns if c not in ("sha256", "label")]
        r.check("no_duplicate_feature_vectors", not m.duplicated(feats).any())
        r.check("no_conflicting_labels", not m.groupby(feats, dropna=False)["label"].nunique().gt(1).any())
        r.class_balance(m["label"])

    def load(self, data):
        paths = {"model": write_table(data["model"], "zenodo_hashes_model")}
        r = catalog.register("zenodo_hashes", self.meta.assign(sample_key=self.meta["sha256"], family=self.meta["label_name"]))
        self.report.metric("catalog_rows", r)
        return paths
