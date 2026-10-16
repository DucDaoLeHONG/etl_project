"""PE_Header.csv - DOS/COFF/optional-header fields for 29,807 samples (SHA256 + Type 0-6).

Quirks handled: SHA256 case, CamelCase names, ImageBase beyond uint32 (64-bit images), 0/future compile
timestamps (flagged, not dropped), 2 constant columns (e_magic, LoaderFlags).
"""
from __future__ import annotations

import pandas as pd

from .. import catalog, config
from ..common import Pipeline, count_conflicting_duplicate_keys, downcast, read_csv_plain, snake_case, write_table

MACHINES = {332: "i386", 34404: "amd64", 512: "ia64", 452: "arm", 43620: "arm64"}
PE_EPOCH_MIN = pd.Timestamp("1992-01-01")


class PeHeaderPipeline(Pipeline):
    name = "pe_header"

    def extract(self) -> pd.DataFrame:
        df = read_csv_plain(config.PE_HEADER_CSV)
        self.report.metric("rows_raw", len(df))
        return df

    def transform(self, raw: pd.DataFrame) -> dict[str, pd.DataFrame]:
        df = raw.rename(columns=snake_case).rename(columns={"type": "type_code"})
        df["sha256"] = df["sha256"].str.strip().str.lower()
        self.report.metric("duplicate_hashes_with_conflicting_records", count_conflicting_duplicate_keys(df, "sha256"))
        self.report.metric("duplicate_rows_removed", int(df.duplicated("sha256").sum()))
        df = df.drop_duplicates("sha256").reset_index(drop=True)

        ts = pd.to_datetime(df["time_date_stamp"], unit="s", errors="coerce")
        valid = ts.between(PE_EPOCH_MIN, pd.Timestamp.now())
        df["compile_time"] = ts.where(valid)
        df["timestamp_valid"] = valid.astype("uint8")
        df["is_64bit"] = (df["magic"] == 523).astype("uint8")           # PE32+ optional header
        df["machine_name"] = df["machine"].map(MACHINES).fillna("other")
        df = downcast(df)

        feats = [c for c in df.columns[2:] if c not in ("compile_time", "machine_name")]
        constants = [c for c in feats if df[c].nunique() == 1]
        self.report.metric("constant_columns", constants)
        drop = constants if config.DROP_CONSTANT_COLUMNS else []
        model = df.drop(columns=drop + ["compile_time", "machine_name"])
        return {"clean": df, "model": model}

    def validate(self, data) -> None:
        df, r = data["clean"], self.report
        r.check("unique_sha256", df["sha256"].is_unique)
        r.check(
            "duplicate_hashes_have_consistent_records",
            r.data["metrics"]["duplicate_hashes_with_conflicting_records"] == 0,
        )
        r.check("sha256_format", df["sha256"].str.fullmatch(config.SHA256_RE.strip("^$")).all())
        r.check("no_nulls", not df.drop(columns="compile_time").isna().any().any())
        r.check("e_magic_is_MZ", (df["e_magic"] == 0x5A4D).all())
        r.check("optional_magic_valid", df["magic"].isin([267, 523]).all(), "0x10b PE32 / 0x20b PE32+")
        r.check("type_code_range_0_6", df["type_code"].between(0, 6).all())
        r.check("number_of_sections_positive", (df["number_of_sections"] >= 1).all())
        r.check("pe32plus_matches_amd64", (df["is_64bit"] == (df["machine"] == 34404)).all())
        bad = int((df["timestamp_valid"] == 0).sum())
        r.check("compile_timestamps_plausible", bad / len(df) < 0.15, f"{bad} rows ({bad/len(df):.1%}) zero/future -> NaT")
        r.metric("rows_clean", len(df))
        r.metric("type_counts", df["type_code"].value_counts().sort_index().to_dict())

    def load(self, data):
        paths = {k: write_table(v, f"pe_header_{k}") for k, v in data.items()}
        c = data["clean"]
        r = catalog.register("pe_header", c.assign(sample_key=c["sha256"], family=c["type_code"].astype(str)))
        self.report.metric("catalog_rows", r)
        return paths
