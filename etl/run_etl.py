"""CLI:  python -m etl.run_etl [pipeline ...]   (no args = run all)"""
from __future__ import annotations

import argparse
import logging
import sys

from . import catalog, config
from .pipelines import ALL


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("pipelines", nargs="*", help=f"any of: {', '.join(ALL)}")
    p.add_argument("-q", "--quiet", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")

    names = args.pipelines or list(ALL)
    unknown = [n for n in names if n not in ALL]
    if unknown:
        p.error(f"unknown pipeline(s): {unknown}; choose from {list(ALL)}")

    failed = []
    for n in names:
        try:
            if not ALL[n]().run().ok:
                failed.append(n)
        except Exception:
            logging.getLogger("etl").exception("pipeline %s crashed", n)
            failed.append(n)

    if config.CATALOG_DB.exists():
        ov = catalog.overlap_report()
        ov.to_csv(config.REPORT_DIR / "cross_dataset_overlap.csv", index=False)
        logging.getLogger("etl").info("cross-dataset sha256 overlap: %d hashes", len(ov))
    if failed:
        logging.getLogger("etl").error("FAILED: %s", failed)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
