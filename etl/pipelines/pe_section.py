"""PE_Section.csv - 9 fields x 10 well-known sections (.text .data .rdata .bss .idata .edata .rsrc .reloc .tls .pdata).

A section that is absent from a binary is all zeros, so "present" is derived (virtual size or
characteristics non-zero). Characteristics bit 0x20000000 = executable, 0x80000000 = writable.
Outputs a wide table (one row per sample) and a long table (one row per sample x present section).
"""
from __future__ import annotations

import pandas as pd

from .. import catalog, config
from ..common import Pipeline, downcast, read_csv_plain, snake_case, write_table

SCN_EXEC, SCN_WRITE = 0x20000000, 0x80000000
SECTIONS = ["text", "data", "rdata", "bss", "idata", "edata", "rsrc", "reloc", "tls", "pdata"]
FIELDS = ["misc_virtual_size", "virtual_address", "size_of_raw_data", "pointer_to_raw_data",
          "pointer_to_relocations", "pointer_to_linenumbers", "number_of_relocations",
          "number_of_linenumbers", "characteristics"]


class PeSectionPipeline(Pipeline):
    name = "pe_section"

    def extract(self) -> pd.DataFrame:
        df = read_csv_plain(config.PE_SECTION_CSV)
        self.report.metric("rows_raw", len(df))
        return df

    def transform(self, raw: pd.DataFrame) -> dict[str, pd.DataFrame]:
        df = raw.rename(columns=snake_case).rename(columns={"type": "type_code"})
        df["sha256"] = df["sha256"].str.strip().str.lower()
        self.report.metric("duplicate_rows_removed", int(df.duplicated("sha256").sum()))
        df = df.drop_duplicates("sha256").reset_index(drop=True)
        missing = [f"{s}_{f}" for s in SECTIONS for f in FIELDS if f"{s}_{f}" not in df.columns]
        if missing:
            raise ValueError(f"unexpected section schema, missing: {missing[:5]}")

        derived, long_parts = {}, []
        for s in SECTIONS:
            ch, vs = df[f"{s}_characteristics"], df[f"{s}_misc_virtual_size"]
            present = ((ch != 0) | (vs != 0)).astype("uint8")
            derived[f"{s}_present"] = present
            derived[f"{s}_exec"] = ((ch & SCN_EXEC) != 0).astype("uint8")
            derived[f"{s}_write"] = ((ch & SCN_WRITE) != 0).astype("uint8")
            part = df.loc[present == 1, ["sha256", "type_code"] + [f"{s}_{f}" for f in FIELDS]].copy()
            part.columns = ["sha256", "type_code"] + FIELDS
            part.insert(2, "section", s)
            part["is_exec"] = ((part["characteristics"] & SCN_EXEC) != 0).astype("uint8")
            part["is_write"] = ((part["characteristics"] & SCN_WRITE) != 0).astype("uint8")
            long_parts.append(part)

        d = pd.DataFrame(derived)
        totals = pd.DataFrame({
            "n_sections_present": d[[f"{s}_present" for s in SECTIONS]].sum(axis=1),
            "n_exec_sections": d[[f"{s}_exec" for s in SECTIONS]].sum(axis=1),
            "n_write_sections": d[[f"{s}_write" for s in SECTIONS]].sum(axis=1),
            "n_wx_sections": sum(d[f"{s}_exec"] & d[f"{s}_write"] for s in SECTIONS),   # write+execute = suspicious
            "total_virtual_size": sum(df[f"{s}_misc_virtual_size"].astype("int64") for s in SECTIONS),
            "total_raw_size": sum(df[f"{s}_size_of_raw_data"].astype("int64") for s in SECTIONS),
        })
        totals["raw_to_virtual_ratio"] = (totals["total_raw_size"] / totals["total_virtual_size"].where(totals["total_virtual_size"] > 0)).round(4)
        wide = downcast(pd.concat([df, d, totals], axis=1))

        constants = [c for c in df.columns[2:] if df[c].nunique() == 1]
        self.report.metric("constant_columns", constants)
        features = wide.drop(columns=constants) if config.DROP_CONSTANT_COLUMNS else wide
        long = downcast(pd.concat(long_parts, ignore_index=True))
        self.report.metric("rows_without_any_section_data", int((totals["n_sections_present"] == 0).sum()))
        return {"features": features, "sections_long": long, "_meta": wide[["sha256", "type_code"]]}

    def validate(self, data) -> None:
        w, lg, r = data["features"], data["sections_long"], self.report
        r.check("unique_sha256", w["sha256"].is_unique)
        r.check("sha256_format", w["sha256"].str.fullmatch(config.SHA256_RE.strip("^$")).all())
        num = w.drop(columns=["sha256"]).select_dtypes("number")
        r.check("no_negative_values", (num.min() >= 0).all())
        r.check("no_nulls_except_ratio", not w.drop(columns="raw_to_virtual_ratio").isna().any().any())
        r.check("type_code_range_0_6", w["type_code"].between(0, 6).all())
        r.check("long_matches_wide_counts", len(lg) == int(w["n_sections_present"].sum()))
        r.check("long_hashes_in_wide", lg["sha256"].isin(w["sha256"]).all())
        r.check("text_section_common", w["text_present"].mean() > 0.8, f"{w['text_present'].mean():.1%} have .text")
        r.metric("rows_clean", len(w))
        r.metric("long_rows", len(lg))
        r.metric("samples_with_wx_section", int((w["n_wx_sections"] > 0).sum()))
        r.metric("section_prevalence", {s: round(float(w[f'{s}_present'].mean()), 4) for s in SECTIONS})

    def load(self, data):
        paths = {k: write_table(v, f"pe_section_{k}") for k, v in data.items() if not k.startswith("_")}
        m = data["_meta"]
        n = catalog.register("pe_section", m.assign(sample_key=m["sha256"], family=m["type_code"].astype(str)))
        self.report.metric("catalog_rows", n)
        return paths
