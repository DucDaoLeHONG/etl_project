"""Joins the three views of the same corpus into one model-ready table keyed on SHA-256:

    pe_header (optional-header fields)  +  pe_section (section tables)  +  dll_imports (imported DLL flags)

Runs AFTER those three pipelines (it reads their processed outputs). Keeps only samples present in all
three views (inner join) and writes a coverage table so you can see exactly what each view is missing.
"""
from __future__ import annotations

import pandas as pd

from ..common import Pipeline, downcast, read_table, write_table


class PeMultiviewPipeline(Pipeline):
    name = "pe_multiview"

    def extract(self) -> dict[str, pd.DataFrame]:
        return {
            "header": read_table("pe_header_model"),
            "section": read_table("pe_section_features"),
            "dll": read_table("dll_imports_features"),
        }

    def transform(self, raw: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        h, s, d = (raw[k].copy() for k in ("header", "section", "dll"))

        # Type must agree across views for every shared hash before we trust the join.
        types = {k: v.set_index("sha256")["type_code"] for k, v in zip(("header", "section", "dll"), (h, s, d))}
        disagree = 0
        for a, b in (("header", "section"), ("header", "dll"), ("section", "dll")):
            both = types[a].index.intersection(types[b].index)
            disagree += int((types[a].loc[both] != types[b].loc[both]).sum())
        self.report.metric("type_disagreements_across_views", disagree)

        # Avoid silent column clashes: each view's feature columns must be disjoint.
        feats = {k: set(v.columns) - {"sha256", "type_code"} for k, v in zip(("header", "section", "dll"), (h, s, d))}
        clash = (feats["header"] & feats["section"]) | (feats["header"] & feats["dll"]) | (feats["section"] & feats["dll"])
        self.report.metric("clashing_columns", sorted(clash))
        if clash:
            raise ValueError(f"feature name clash between views: {sorted(clash)[:10]}")

        sets = {k: set(v["sha256"]) for k, v in zip(("header", "section", "dll"), (h, s, d))}
        universe = sorted(set.union(*sets.values()))
        coverage = pd.DataFrame({"sha256": universe})
        for k, v in sets.items():
            coverage[f"in_{k}"] = coverage["sha256"].isin(v).astype("uint8")
        coverage["n_views"] = coverage[["in_header", "in_section", "in_dll"]].sum(axis=1)
        self.report.metric("universe_size", len(universe))
        self.report.metric("coverage_counts", coverage["n_views"].value_counts().sort_index().to_dict())

        joined = (
            h.merge(s.drop(columns="type_code"), on="sha256", how="inner")
            .merge(d.drop(columns="type_code"), on="sha256", how="inner")
        )
        joined = downcast(joined)
        return {"features": joined, "coverage": coverage, "_n": (len(h), len(s), len(d))}

    def validate(self, data) -> None:
        j, cov, r = data["features"], data["coverage"], self.report
        r.check("unique_sha256", j["sha256"].is_unique)
        r.check("no_type_disagreement", r.data["metrics"]["type_disagreements_across_views"] == 0)
        r.check("no_clashing_columns", not r.data["metrics"]["clashing_columns"])
        r.check("joined_equals_full_coverage", len(j) == int((cov["n_views"] == 3).sum()))
        r.check("type_code_range_0_6", j["type_code"].between(0, 6).all())
        nulls = j.drop(columns=["raw_to_virtual_ratio"], errors="ignore").isna().sum().sum()
        r.check("no_nulls_after_join", nulls == 0, f"{nulls} nulls")
        lost = len(cov) - len(j)
        r.check("join_loss_under_5pct", lost / len(cov) < 0.05, f"{lost} of {len(cov)} samples dropped ({lost/len(cov):.1%}) - see coverage table")
        r.metric("rows_clean", len(j))
        r.metric("n_columns", j.shape[1])
        r.metric("type_counts", j["type_code"].value_counts().sort_index().to_dict())

    def load(self, data):
        return {k: write_table(v, f"pe_multiview_{k}") for k, v in data.items() if not k.startswith("_")}
