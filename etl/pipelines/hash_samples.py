"""15038698.zip - 1,000 malicious + 1,000 benign samples described by hashes and import lists.

Source quirks handled here:
  * column names with spaces/trailing blanks ('DIGITAL SIG. ') and dashes ('SHA-256')
  * label text with stray whitespace ('   benign')
  * 29 duplicate SHA-256 rows in the benign file
  * LIBRARIES / FUNCTIONS are comma-separated strings, sometimes empty (NaN)
"""
from __future__ import annotations

import pandas as pd

from .. import catalog, config
from ..common import Pipeline, count_conflicting_duplicate_keys, downcast, read_csv_from_zip, snake_case, write_table

FILES = {
    "1000_malware_sample_malicious.csv": "malicious",
    "1000_whitelist_sample_benign.csv": "benign",
}


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


class HashSamplesPipeline(Pipeline):
    name = "hash_samples"

    # ------------------------------------------------------------ extract
    def extract(self) -> pd.DataFrame:
        frames = []
        for member, expected in FILES.items():
            df = read_csv_from_zip(config.HASH_ARCHIVE, member, dtype={"MD5": str, "SHA-1": str, "SHA-256": str})
            df["source_file"] = member
            self.report.metric(f"rows_raw[{expected}]", len(df))
            frames.append(df)
        return pd.concat(frames, ignore_index=True)

    # ---------------------------------------------------------- transform
    def transform(self, raw: pd.DataFrame) -> dict[str, pd.DataFrame]:
        df = raw.rename(columns=snake_case).rename(
            columns={"sha_1": "sha1", "sha_256": "sha256", "digital_sig": "is_signed", "classification": "label_name"}
        )
        for col in ("md5", "sha1", "sha256"):
            df[col] = df[col].str.strip().str.lower()
        df["label_name"] = df["label_name"].str.strip().str.lower()
        df["label"] = (df["label_name"] == "malicious").astype("uint8")

        # De-duplicate on SHA-256; record any duplicate whose label disagrees.
        dup_mask = df.duplicated("sha256", keep=False)
        conflicts = df[dup_mask].groupby("sha256")["label"].nunique().gt(1).sum()
        self.report.metric("duplicate_rows_removed", int(df.duplicated("sha256").sum()))
        self.report.metric("duplicate_hashes_with_conflicting_labels", int(conflicts))
        self.report.metric("duplicate_hashes_with_conflicting_records", count_conflicting_duplicate_keys(df, "sha256"))
        df = df.drop_duplicates("sha256", keep="first").reset_index(drop=True)

        df["libraries_list"] = df["libraries"].map(split_list)
        df["functions_list"] = df["functions"].map(split_list)
        df["has_import_info"] = (df["libraries_list"].str.len() > 0).astype("uint8")
        df["n_libraries"] = df["libraries_list"].str.len()
        df["n_functions"] = df["functions_list"].str.len()
        df["size"] = df["size"].astype("int64")

        samples = df[["md5", "sha1", "sha256", "size", "is_signed", "label", "label_name", "source_file",
                      "has_import_info", "n_libraries", "n_functions"]].copy()
        # Lists stored as '|'-joined strings so CSV and Parquet outputs look the same.
        samples["libraries"] = df["libraries_list"].str.join("|")
        samples["functions"] = df["functions_list"].str.join("|")

        imports_long = (
            df[["sha256", "libraries_list"]].explode("libraries_list").dropna()
            .rename(columns={"libraries_list": "library"}).reset_index(drop=True)
        )

        # Model-ready table: numeric features + one-hot for the N most common libraries.
        top = imports_long["library"].value_counts().head(config.TOP_N_LIBRARIES).index
        onehot = (
            pd.crosstab(imports_long["sha256"], imports_long["library"]).reindex(columns=top, fill_value=0).clip(upper=1)
        )
        onehot.columns = [f"lib_{c}" for c in onehot.columns]
        features = samples[["sha256", "label", "size", "is_signed", "has_import_info", "n_libraries", "n_functions"]].merge(
            onehot, how="left", left_on="sha256", right_index=True
        )
        lib_cols = [c for c in features.columns if c.startswith("lib_")]
        features[lib_cols] = features[lib_cols].fillna(0)
        features = downcast(features)
        return {"samples": samples, "imports_long": imports_long, "features": features}

    # ----------------------------------------------------------- validate
    def validate(self, data: dict[str, pd.DataFrame]) -> None:
        s, r = data["samples"], self.report
        r.check("unique_sha256", s["sha256"].is_unique)
        r.check(
            "duplicate_hashes_have_consistent_records",
            r.data["metrics"]["duplicate_hashes_with_conflicting_records"] == 0,
        )
        r.check("sha256_format", s["sha256"].str.fullmatch(config.SHA256_RE.strip("^$")).all())
        r.check("sha1_format", s["sha1"].str.fullmatch(config.SHA1_RE.strip("^$")).all())
        r.check("md5_format", s["md5"].str.fullmatch(config.MD5_RE.strip("^$")).all())
        r.check("label_values", set(s["label"].unique()) <= {0, 1})
        r.check("size_positive", (s["size"] > 0).all())
        r.check("is_signed_binary", set(s["is_signed"].unique()) <= {0, 1})
        r.check("both_classes_present", s["label"].nunique() == 2)
        r.metric("rows_clean", len(s))
        r.metric("class_counts", s["label_name"].value_counts().to_dict())
        r.metric("rows_without_import_info", int((s["has_import_info"] == 0).sum()))

    # --------------------------------------------------------------- load
    def load(self, data):
        paths = {k: write_table(v, f"hash_samples_{k}") for k, v in data.items()}
        s = data["samples"]
        n = catalog.register("hash_samples", s.assign(sample_key=s["sha256"], family=s["label_name"]))
        self.report.metric("catalog_rows", n)
        return paths
