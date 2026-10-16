"""CLI:  python -m etl.run_etl [pipeline ...]   (no args = run everything, export data/final, check R7 sources)"""
from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from . import catalog, config
from .export_final import export_final
from .manifest import check_sources
from .pipelines import ALL


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("pipelines", nargs="*", help=f"any of: {', '.join(ALL)}")
    p.add_argument("-q", "--quiet", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    log = logging.getLogger("etl")

    names = args.pipelines or list(ALL)
    unknown = [n for n in names if n not in ALL]
    if unknown:
        p.error(f"unknown pipeline(s): {unknown}; choose from {list(ALL)}")

    failed, steps = [], []
    for n in names:
        try:
            rep = ALL[n]().run()
            steps += [{"pipeline": n, **s} for s in rep.data.get("row_counts", [])]
            if not rep.ok:
                failed.append(n)
        except Exception:
            log.exception("pipeline %s crashed", n)
            failed.append(n)

    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    if steps:
        pd.DataFrame(steps).to_csv(config.REPORT_DIR / "row_counts.csv", index=False)
    if config.CATALOG_DB.exists():
        ov = catalog.overlap_report()
        ov.to_csv(config.REPORT_DIR / "cross_dataset_overlap.csv", index=False)
        log.info("cross-source sha256 overlap: %d hashes (reports/cross_dataset_overlap.csv)", len(ov))
    if not args.pipelines:  # full run only
        try:
            export_final()
        except Exception:
            log.exception("final export failed")
            failed.append("export_final")
        check_sources()
    if failed:
        log.error("FAILED: %s", failed)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
