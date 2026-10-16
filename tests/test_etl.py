"""Run with:  python -m unittest discover -s tests -v   (no extra dependencies)"""
import sqlite3
import unittest

import pandas as pd

from etl import config
from etl.common import downcast, snake_case
from etl.pipelines import ALL
from etl.pipelines.hash_samples import split_list


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


class EndToEnd(unittest.TestCase):
    """Runs every pipeline on the real archives and asserts all quality checks pass."""

    @classmethod
    def setUpClass(cls):
        cls.reports = {name: cls_().run() for name, cls_ in ALL.items()}

    def test_all_checks_pass(self):
        for name, rep in self.reports.items():
            failed = [k for k, v in rep.data["checks"].items() if not v["passed"]]
            self.assertEqual(failed, [], f"{name}: {failed}")

    def test_expected_row_counts(self):
        m = {n: r.data["metrics"] for n, r in self.reports.items()}
        self.assertEqual(m["hash_samples"]["rows_clean"], 1971)
        self.assertEqual(m["xpe_guard"]["rows_clean"], 9408)
        self.assertEqual(m["dll_imports"]["rows_clean"], 29495)
        self.assertEqual(m["pe_headers"]["rows_clean[train]"], 19611)
        self.assertEqual(m["pe_header"]["rows_clean"], 29807)
        self.assertEqual(m["pe_section"]["rows_clean"], 29760)
        self.assertEqual(m["pe_multiview"]["rows_clean"], 29492)

    def test_views_agree_on_type(self):
        self.assertEqual(self.reports["pe_multiview"].data["metrics"]["type_disagreements_across_views"], 0)

    def test_rerun_is_idempotent(self):
        before = self._catalog_count()
        ALL["xpe_guard"]().run()
        self.assertEqual(before, self._catalog_count())

    @staticmethod
    def _catalog_count():
        with sqlite3.connect(config.CATALOG_DB) as con:
            return con.execute("SELECT COUNT(*) FROM sample_catalog").fetchone()[0]


if __name__ == "__main__":
    unittest.main()
