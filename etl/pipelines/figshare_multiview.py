"""Figshare corpus, joined: header + section + DLL views on SHA-256 -> one model-ready table.

Runs AFTER figshare_header / figshare_section / figshare_dll (reads their *_staging tables, which are
only de-duplicated by SHA-256). Inner join; identical joined vectors and conflicting-label vectors are
then removed on the combined feature set. A coverage table (reports/) shows which view each hash lacks.
"""
from __future__ import annotations

import pandas as pd

from .. import config
from ..common import Pipeline, downcast, drop_duplicates_and_conflicts, read_table, write_table

KEYS = ["sha256", "type_code", "label"]


class FigshareMultiviewPipeline(Pipeline):
    name = "figshare_multiview"

    def extract(self) -> dict[str, pd.DataFrame]:
        raw = {k: read_table(f"figshare_{k}_staging") for k in ("header", "section", "dll")}
        for k, v in raw.items():
            self.report.step(f"{k} view (staging)", len(v), v.shape[1])
        return raw

    def transform(self, raw):
        views = {k: v.drop(columns=[c for c in ("type_name",) if c in v.columns]) for k, v in raw.items()}
        # all three views must agree on type for every hash they share
        dis = 0
        names = list(views)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                ta, tb = views[a].set_index("sha256")["type_code"], views[b].set_index("sha256")["type_code"]
                both = ta.index.intersection(tb.index)
                dis += int((ta.loc[both] != tb.loc[both]).sum())
        self.report.metric("type_disagreements_across_views", dis)

        feats = {k: set(v.columns) - set(KEYS) for k, v in views.items()}
        clash = (feats["header"] & feats["section"]) | (feats["header"] & feats["dll"]) | (feats["section"] & feats["dll"])
        self.report.metric("clashing_columns", sorted(clash))
        if clash:
            raise ValueError(f"feature name clash between views: {sorted(clash)[:10]}")

        sets = {k: set(v["sha256"]) for k, v in views.items()}
        cov = pd.DataFrame({"sha256": sorted(set.union(*sets.values()))})
        for k, v in sets.items():
            cov[f"in_{k}"] = cov["sha256"].isin(v).astype("uint8")
        cov["n_views"] = cov[[f"in_{k}" for k in views]].sum(axis=1)
        self.coverage = cov
        self.report.metric("coverage_counts", {str(k): int(v) for k, v in cov["n_views"].value_counts().sort_index().items()})

        j = views["header"].merge(views["section"].drop(columns=["type_code", "label"]), on="sha256", how="inner")
        self.report.step("after header+section inner join", len(j), j.shape[1])
        j = j.merge(views["dll"].drop(columns=["type_code", "label"]), on="sha256", how="inner")
        self.report.step("after +dll inner join", len(j), j.shape[1])
        j = downcast(j)

        fcols = [c for c in j.columns if c not in KEYS]
        model = drop_duplicates_and_conflicts(j, fcols, ["label", "type_code"], self.report)
        constants = [c for c in fcols if model[c].nunique(dropna=False) == 1]
        self.report.metric("constant_columns_dropped", len(constants))
        if config.DROP_CONSTANT_COLUMNS:
            model = model.drop(columns=constants)
        self.report.step("final model table", len(model), model.shape[1])
        return {"model": model}

    def validate(self, data) -> None:
        m, cov, r = data["model"], self.coverage, self.report
        r.check("unique_sha256", m["sha256"].is_unique)
        r.check("no_type_disagreement", r.data["metrics"]["type_disagreements_across_views"] == 0)
        r.check("no_clashing_columns", not r.data["metrics"]["clashing_columns"])
        r.check("type_code_range_0_6", m["type_code"].between(0, 6).all())
        r.check("label_consistent_with_type", ((m["type_code"] == 0) == (m["label"] == 0)).all())
        nulls = int(m.drop(columns=["raw_to_virtual_ratio"], errors="ignore").isna().sum().sum())
        r.check("no_nulls_after_join", nulls == 0, f"{nulls} nulls")
        lost = len(cov) - int((cov["n_views"] == 3).sum())
        r.check("join_loss_under_5pct", lost / len(cov) < 0.05, f"{lost} of {len(cov)} hashes not in all 3 views")
        feats = [c for c in m.columns if c not in KEYS]
        r.check("no_duplicate_feature_vectors", not m.duplicated(feats).any())
        r.check("no_conflicting_labels", not m.groupby(feats, dropna=False)["type_code"].nunique().gt(1).any())
        r.class_balance(m["label"], "class_balance_binary")
        r.class_balance(m["type_code"], "class_balance_type")

    def load(self, data):
        config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
        self.coverage.to_csv(config.REPORT_DIR / "figshare_multiview_coverage.csv", index=False)
        return {"model": write_table(data["model"], "figshare_multiview_model")}
