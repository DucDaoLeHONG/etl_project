"""archive__7_ - DLLs_Imported.csv: 29,498 samples x 629 binary "imports this DLL" columns + Type (0-6).

Feature vectors repeat heavily by nature (many files import the same DLL set), so only rows with the
same SHA-256 are de-duplicated; identical vectors under different hashes are legitimate samples.
"""
from __future__ import annotations

import pandas as pd

from .. import catalog, config
from ..common import Pipeline, read_csv_from_zip, write_table

MEMBER = "DLLs_Imported.csv"


class DllImportsPipeline(Pipeline):
    name = "dll_imports"

    def extract(self) -> pd.DataFrame:
        df = read_csv_from_zip(config.DLL_ARCHIVE, MEMBER)
        self.report.metric("rows_raw", len(df))
        self.report.metric("columns_raw", df.shape[1])
        return df

    def transform(self, raw: pd.DataFrame) -> dict[str, pd.DataFrame]:
        df = raw.rename(columns={"SHA256": "sha256", "Type": "type_code"})
        dll_cols = [c for c in df.columns if c not in ("sha256", "type_code")]
        df.columns = ["sha256", "type_code"] + [c.strip().lower() for c in dll_cols]
        dll_cols = list(df.columns[2:])

        # Column names that collapse after lowercasing are merged with logical OR.
        dup_names = pd.Index(dll_cols)[pd.Index(dll_cols).duplicated()].unique().tolist()
        self.report.metric("dll_columns_merged_after_lowercase", dup_names)
        if dup_names:
            df = pd.concat([df[["sha256", "type_code"]], df[dll_cols].T.groupby(level=0).max().T], axis=1)
            dll_cols = list(df.columns[2:])

        df["sha256"] = df["sha256"].str.strip().str.lower()
        dups = df[df.duplicated("sha256", keep=False)].groupby("sha256")["type_code"].nunique()
        self.report.metric("duplicate_hashes_with_conflicting_type", int((dups > 1).sum()))
        self.report.metric("duplicate_rows_removed", int(df.duplicated("sha256").sum()))
        df = df.drop_duplicates("sha256", keep="first").reset_index(drop=True)

        bits = df[dll_cols].clip(0, 1).astype("uint8")
        type_code = df["type_code"].astype("uint8")
        type_name = type_code.map(config.DLL_TYPE_NAMES) if config.DLL_TYPE_NAMES else pd.Series(pd.NA, index=df.index)
        df = pd.concat(
            [df[["sha256"]], type_code.rename("type_code"), type_name.rename("type_name"),
             bits.sum(axis=1).astype("uint16").rename("n_dlls"), bits],
            axis=1,
        )
        self.dll_cols = dll_cols

        vocab = (
            df[dll_cols].sum().rename("n_samples").rename_axis("dll").reset_index()
            .sort_values("n_samples", ascending=False, ignore_index=True)
        )
        vocab["prevalence"] = (vocab["n_samples"] / len(df)).round(5)
        out_cols = ["sha256", "type_code", "n_dlls"] + dll_cols
        return {"features": df[out_cols], "dll_vocabulary": vocab, "_meta": df[["sha256", "type_code", "type_name"]]}

    def validate(self, data: dict[str, pd.DataFrame]) -> None:
        df, r = data["features"], self.report
        r.check("unique_sha256", df["sha256"].is_unique)
        r.check("sha256_format", df["sha256"].str.fullmatch(config.SHA256_RE.strip("^$")).all())
        r.check("no_nulls", not df.isna().any().any())
        r.check("features_binary", int(df[self.dll_cols].max().max()) <= 1)
        r.check("type_code_range_0_6", df["type_code"].between(0, 6).all())
        r.check("unique_dll_columns", len(set(self.dll_cols)) == len(self.dll_cols))
        empty = int((df["n_dlls"] == 0).sum())
        r.check("no_rows_with_zero_imports", empty == 0, f"{empty} rows import no DLL (kept, flagged)")
        never = [c for c in self.dll_cols if df[c].sum() == 0]
        r.check("no_all_zero_dll_columns", not never, f"{len(never)} columns")
        r.metric("rows_clean", len(df))
        r.metric("dll_columns", len(self.dll_cols))
        r.metric("type_counts", df["type_code"].value_counts().sort_index().to_dict())

    def load(self, data):
        paths = {k: write_table(v, f"dll_imports_{k}") for k, v in data.items() if not k.startswith("_")}
        m = data["_meta"]
        n = catalog.register("dll_imports", m.assign(sample_key=m["sha256"], family=m["type_code"].astype(str)))
        self.report.metric("catalog_rows", n)
        return paths
