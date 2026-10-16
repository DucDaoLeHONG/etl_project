"""archive__8_ - PE-header features: dataset_malwares.csv (19,611 labelled) + dataset_test.csv (17 unlabelled).

Source quirks handled here: CamelCase/mixed column names, VirusShare_<md5> naming for malware,
impossible compile timestamps (0 or in the future), 8 constant columns, many identical feature vectors,
and a test file that lacks the `Malware` label column.
"""
from __future__ import annotations

import pandas as pd

from .. import catalog, config
from ..common import Pipeline, downcast, read_csv_from_zip, snake_case, write_table

PE_EPOCH_MIN = pd.Timestamp("1992-01-01")  # first PE-capable Windows toolchains
VIRUSSHARE_RE = r"^VirusShare_([0-9a-f]{32})$"


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    new = [snake_case(c) for c in df.columns]
    if len(set(new)) != len(new):
        raise ValueError(f"column name collision after snake_case: {sorted({c for c in new if new.count(c) > 1})}")
    df = df.copy()
    df.columns = new
    df = df.rename(columns={"malware": "label"})
    df["name"] = df["name"].str.strip()
    # Name is either VirusShare_<md5>, a bare sha256, or a file name - pull out whatever hashes it carries.
    df["md5"] = df["name"].str.extract(VIRUSSHARE_RE, expand=False)
    df["sha256"] = df["name"].str.lower().where(df["name"].str.fullmatch(r"[0-9a-fA-F]{64}"))
    ts = pd.to_datetime(df["time_date_stamp"], unit="s", errors="coerce")
    now = pd.Timestamp.now()
    valid = ts.between(PE_EPOCH_MIN, now)
    df["compile_time"] = ts.where(valid)
    df["timestamp_valid"] = valid.astype("uint8")
    return df


class PeHeadersPipeline(Pipeline):
    name = "pe_headers"

    def extract(self) -> dict[str, pd.DataFrame]:
        raw = {
            "train": read_csv_from_zip(config.PEHDR_ARCHIVE, "dataset_malwares.csv"),
            "holdout": read_csv_from_zip(config.PEHDR_ARCHIVE, "dataset_test.csv"),
        }
        for k, v in raw.items():
            self.report.metric(f"rows_raw[{k}]", len(v))
        return raw

    def transform(self, raw: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        train, hold = _clean(raw["train"]), _clean(raw["holdout"])

        # Schema alignment: holdout must have exactly the train features (no label).
        feats = [c for c in train.columns if c not in ("name", "label", "md5", "sha256", "compile_time", "timestamp_valid")]
        self.report.metric("feature_columns", len(feats))
        self.missing_in_holdout = sorted(set(feats) - set(hold.columns))
        self.extra_in_holdout = sorted(set(hold.columns) - set(train.columns))

        self.report.metric("duplicate_names_removed", int(train.duplicated("name").sum()))
        train = train.drop_duplicates("name").reset_index(drop=True)

        # Flag (and optionally drop) identical feature vectors, and measure label conflicts among them.
        train["is_feature_duplicate"] = train.duplicated(feats, keep="first").astype("uint8")
        g = train[train.duplicated(feats, keep=False)].groupby(feats)["label"].nunique()
        self.report.metric("feature_duplicate_rows", int(train["is_feature_duplicate"].sum()))
        self.report.metric("feature_duplicate_groups_with_conflicting_labels", int((g > 1).sum()))
        if config.DROP_FEATURE_DUPLICATES:
            train = train[train["is_feature_duplicate"] == 0].reset_index(drop=True)

        constants = [c for c in feats if train[c].nunique(dropna=False) == 1]
        self.report.metric("constant_columns", constants)
        model_feats = [c for c in feats if c not in constants] if config.DROP_CONSTANT_COLUMNS else feats

        keep_meta = ["name", "md5", "sha256", "compile_time", "timestamp_valid"]
        out_train = downcast(train[keep_meta + model_feats + ["label", "is_feature_duplicate"]].copy())
        out_hold = downcast(hold.reindex(columns=keep_meta + model_feats).copy())
        out_train["label"] = out_train["label"].astype("uint8")
        return {"train": out_train, "holdout": out_hold, "_feats": model_feats}

    def validate(self, data) -> None:
        t, h, r = data["train"], data["holdout"], self.report
        feats = data["_feats"]
        r.check("unique_name", t["name"].is_unique)
        r.check("label_values", set(t["label"].unique()) <= {0, 1})
        r.check("holdout_has_all_train_features", not self.missing_in_holdout, str(self.missing_in_holdout))
        r.check("holdout_only_extra_is_none", not [c for c in self.extra_in_holdout if c not in ("name", "md5", "sha256", "compile_time", "timestamp_valid")], str(self.extra_in_holdout))
        r.check("no_nulls_in_features", not t[feats].isna().any().any() and not h[feats].isna().any().any())
        r.check("number_of_sections_nonneg", (t["number_of_sections"] >= 0).all() if "number_of_sections" in t else True)
        r.check("virusshare_names_are_malware", (t.loc[t["name"].str.startswith("VirusShare_"), "label"] == 1).all())
        bad_ts = int((t["timestamp_valid"] == 0).sum())
        r.check("compile_timestamps_plausible", bad_ts / len(t) < 0.10, f"{bad_ts} rows ({bad_ts/len(t):.1%}) have 0/future timestamps -> compile_time NaT")
        r.metric("rows_clean[train]", len(t))
        r.metric("rows_clean[holdout]", len(h))
        r.metric("class_counts", t["label"].value_counts().to_dict())

    def load(self, data):
        paths = {k: write_table(v, f"pe_headers_{k}") for k, v in data.items() if not k.startswith("_")}
        t = data["train"]
        n = catalog.register("pe_headers", t.assign(sample_key=t["name"], family=t["label"].map({1: "malware", 0: "benign"})))
        self.report.metric("catalog_rows", n)
        return paths
