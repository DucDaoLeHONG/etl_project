"""Kaggle PE-header dataset (archive__8_): dataset_malwares.csv (19,611 labelled) + dataset_test.csv (17 unlabelled).

Quirks handled: CamelCase/mixed column names, VirusShare_<md5> naming, impossible compile timestamps,
constant columns, thousands of identical feature vectors (some with CONFLICTING labels), and a test
file without the `Malware` column.

NOTE: "train"/"holdout" are just the two source files. The 17-row holdout is unlabelled, so it is NOT
part of the model table; make your own stratified split from the model table.
"""
from __future__ import annotations

import pandas as pd

from .. import catalog, config
from ..common import Pipeline, downcast, drop_duplicates_and_conflicts, read_csv_from_zip, snake_case, write_table
from ..pe_features import PE_EPOCH_MIN

VIRUSSHARE_RE = r"^VirusShare_([0-9a-f]{32})$"


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    new = [snake_case(c) for c in df.columns]
    if len(set(new)) != len(new):
        raise ValueError(f"column name collision after snake_case: {sorted({c for c in new if new.count(c) > 1})}")
    df = df.copy()
    df.columns = new
    df = df.rename(columns={"malware": "label"})
    df["name"] = df["name"].str.strip()
    df["md5"] = df["name"].str.extract(VIRUSSHARE_RE, expand=False)
    df["sha256"] = df["name"].str.lower().where(df["name"].str.fullmatch(r"[0-9a-fA-F]{64}"))
    ts = pd.to_datetime(df["time_date_stamp"], unit="s", errors="coerce")
    df["timestamp_valid"] = ts.between(PE_EPOCH_MIN, pd.Timestamp.now()).astype("uint8")
    return df


class KaggleHeaderPipeline(Pipeline):
    name = "kaggle_header"

    def extract(self) -> dict[str, pd.DataFrame]:
        raw = {"train": read_csv_from_zip(config.KAGGLE_ARCHIVE, "dataset_malwares.csv"),
               "holdout": read_csv_from_zip(config.KAGGLE_ARCHIVE, "dataset_test.csv")}
        self.report.step("raw dataset_malwares.csv", len(raw["train"]), raw["train"].shape[1])
        self.report.step("raw dataset_test.csv (unlabelled, not modelled)", len(raw["holdout"]))
        self.report.step("labelled rows entering pipeline", len(raw["train"]))
        return raw

    def transform(self, raw):
        train, hold = _clean(raw["train"]), _clean(raw["holdout"])
        meta = ["name", "md5", "sha256", "timestamp_valid"]
        feats = [c for c in train.columns if c not in meta + ["label"]]
        self.report.metric("feature_columns", len(feats))
        self.missing_in_holdout = sorted(set(feats) - set(hold.columns))
        self.extra_in_holdout = sorted(set(hold.columns) - set(train.columns))

        train = train.drop_duplicates("name").reset_index(drop=True)
        self.report.step("after duplicate-name removal", len(train))
        train["label"] = train["label"].astype("uint8")
        train = downcast(train)

        model = drop_duplicates_and_conflicts(train, feats, ["label"], self.report)
        constants = [c for c in feats if model[c].nunique(dropna=False) == 1]
        self.report.metric("constant_columns", constants)
        keep = [c for c in feats if c not in constants] if config.DROP_CONSTANT_COLUMNS else feats
        self.feats = keep
        self.meta = model[["name", "md5", "sha256", "label"]]
        model = model[["name", "timestamp_valid"] + keep + ["label"]]
        self.report.step("final model table", len(model), model.shape[1])
        holdout = downcast(hold.reindex(columns=["name"] + keep).copy())
        return {"model": model, "holdout_unlabelled": holdout}

    def validate(self, data) -> None:
        m, h, r = data["model"], data["holdout_unlabelled"], self.report
        r.check("unique_name", m["name"].is_unique)
        r.check("label_values", set(m["label"].unique()) <= {0, 1})
        r.check("both_classes_present", m["label"].nunique() == 2)
        r.check("holdout_has_all_train_features", not self.missing_in_holdout, str(self.missing_in_holdout))
        r.check("no_nulls_in_features", not m[self.feats].isna().any().any())
        r.check("virusshare_names_are_malware", (m.loc[m["name"].str.startswith("VirusShare_"), "label"] == 1).all())
        r.check("no_duplicate_feature_vectors", not m.duplicated(self.feats).any())
        r.check("no_conflicting_labels", not m.groupby(self.feats, dropna=False)["label"].nunique().gt(1).any())
        bad = int((m["timestamp_valid"] == 0).sum())
        r.check("compile_timestamps_plausible", bad / len(m) < 0.15, f"{bad} rows ({bad/len(m):.1%}) zero/future")
        r.class_balance(m["label"])

    def load(self, data):
        paths = {k: write_table(v, f"kaggle_header_{k}") for k, v in data.items()}
        m = self.meta
        self.report.metric("catalog_rows", catalog.register(
            "kaggle_header", m.assign(sample_key=m["name"], family=m["label"].map({1: "malware", 0: "benign"}))))
        return paths
