"""Run with:  python -m unittest discover -s tests -v   (no extra dependencies; needs the raw files in data/raw)"""
import unittest

import pandas as pd

from etl import config
from etl.common import QualityReport, drop_duplicates_and_conflicts, downcast, snake_case
from etl.export_final import export_final
from etl.imbalance import class_weights, random_resample
from etl.manifest import check_sources
from etl.pipelines import ALL
from etl.pipelines.zenodo_hashes import split_list
from etl.splits import stratified_split


class UnitHelpers(unittest.TestCase):
    def test_snake_case(self):
        self.assertEqual(snake_case("DIGITAL SIG. "), "digital_sig")
        self.assertEqual(snake_case("SHA-256"), "sha_256")
        self.assertEqual(snake_case("SizeOfOptionalHeader"), "size_of_optional_header")
        self.assertEqual(snake_case("e_lfanew"), "e_lfanew")

    def test_split_list(self):
        self.assertEqual(split_list("A.dll, b.DLL,,a.dll"), ["a.dll", "b.dll"])
        self.assertEqual(split_list(float("nan")), [])

    def test_downcast_preserves_values(self):
        df = pd.DataFrame({"a": [0, 1, 255], "b": [-5, 0, 70000]})
        out = downcast(df.copy())
        self.assertTrue((out.values == df.values).all())
        self.assertEqual(str(out["a"].dtype), "uint8")

    def test_dedupe_and_conflict_removal(self):
        df = pd.DataFrame({"x": [1, 1, 2, 2, 2, 3], "y": [0, 0, 5, 5, 5, 9], "label": [0, 0, 1, 0, 1, 1]})
        rep = QualityReport("toy")
        out = drop_duplicates_and_conflicts(df, ["x", "y"], ["label"], rep)
        # (1,0) duplicated -> keep 1; (2,5) has labels {1,0} -> removed entirely; (3,9) kept
        self.assertEqual(sorted(out["x"]), [1, 3])
        self.assertEqual([s["rows"] for s in rep.data["row_counts"]], [6, 3, 2])
        self.assertEqual(rep.data["metrics"]["conflicting_label_groups"], 1)

    def test_stratified_split_proportions_and_determinism(self):
        y = pd.Series([0] * 900 + [1] * 100)
        s1, s2 = stratified_split(y), stratified_split(y)
        self.assertTrue(s1.equals(s2))
        for cls in (0, 1):
            share = (s1[y == cls] == "test").mean()
            self.assertAlmostEqual(share, 0.15, delta=0.02)
        self.assertEqual(set(s1), {"train", "val", "test"})

    def test_imbalance_helpers(self):
        train = pd.DataFrame({"f": range(110), "label": [0] * 100 + [1] * 10})
        w = class_weights(train["label"])
        self.assertGreater(w[1], w[0])
        over = random_resample(train, "label", "over")
        under = random_resample(train, "label", "under")
        self.assertEqual(over["label"].value_counts().to_dict(), {0: 100, 1: 100})
        self.assertEqual(under["label"].value_counts().to_dict(), {0: 10, 1: 10})


class EndToEnd(unittest.TestCase):
    """Runs every pipeline on the real raw files, exports data/final, and checks the guarantees."""

    @classmethod
    def setUpClass(cls):
        cls.reports = {name: c().run() for name, c in ALL.items()}
        cls.summary = export_final()

    def test_all_checks_pass(self):
        for name, rep in self.reports.items():
            failed = [k for k, v in rep.data["checks"].items() if not v["passed"]]
            self.assertEqual(failed, [], f"{name}: {failed}")

    def test_final_row_counts(self):
        expected = {"zenodo_hashes": 1909, "xpe_guard": 9380, "kaggle_header": 16623,
                    "figshare_dll": 907, "figshare_header": 23354, "figshare_section": 20744,
                    "figshare_multiview": 23426, "pe_malware_dataset1": 14100}
        for name, n in expected.items():
            self.assertEqual(self.reports[name].data["row_counts"][-1]["rows"], n, name)

    def test_row_counts_logged_at_every_step(self):
        for name, rep in self.reports.items():
            steps = [s["step"] for s in rep.data["row_counts"]]
            self.assertGreaterEqual(len(steps), 4, name)
            self.assertTrue(any("conflicting-label" in s for s in steps), name)
            self.assertTrue(any("duplicate-feature" in s for s in steps), name)
            self.assertEqual(steps[-1].split(": ")[-1], "final model table")

    def test_known_kaggle_conflicts(self):
        m = self.reports["kaggle_header"].data["metrics"]
        self.assertEqual(m["conflicting_label_groups"], 4)

    def test_pemd1_join_keeps_family(self):
        r = self.reports["pe_malware_dataset1"].data
        self.assertEqual(r["metrics"]["hashes_only_in_header"], 5)
        df = pd.read_csv(config.FINAL_DIR / "pe_malware_dataset1.csv", usecols=["sha256", "family"])
        self.assertEqual(df["family"].nunique(), 5)

    def test_figshare_binary_label(self):
        df = pd.read_csv(config.FINAL_DIR / "figshare_multiview.csv", usecols=["type_code", "label"])
        self.assertTrue(((df["type_code"] == config.FIGSHARE_BENIGN_TYPE) == (df["label"] == 0)).all())

    def test_final_folder_is_plain_csv_and_only_model_tables(self):
        files = sorted(p.name for p in config.FINAL_DIR.iterdir())
        self.assertEqual(files, sorted(f"{n}.csv" for n in config.FINAL_TABLES))

    def test_no_duplicate_or_conflicting_vectors_in_final(self):
        ids = {"sha256", "name", "sample_id", "md5", "split", "label", "family", "family_id", "type_code"}
        for name in config.FINAL_TABLES:
            df = pd.read_csv(config.FINAL_DIR / f"{name}.csv", low_memory=False)
            feats = [c for c in df.columns if c not in ids]
            self.assertFalse(df.duplicated(feats).any(), f"{name}: duplicate feature vectors (would leak across splits)")

    def test_splits_cover_all_classes(self):
        s = self.summary
        for name, g in s.groupby("dataset"):
            self.assertEqual(set(g["split"]), {"train", "val", "test"}, name)

    def test_manifest_flags_missing_licences(self):
        out = check_sources()
        self.assertTrue(out["present"].all(), "raw file missing from data/raw")
        self.assertTrue(out["sha256"].str.len().eq(64).all())


if __name__ == "__main__":
    unittest.main()
