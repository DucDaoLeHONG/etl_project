# Malware-dataset ETL pipelines

Four independent pipelines (extract → transform → validate → load), one per archive, plus a shared
SQLite sample catalog for cross-dataset checks. Raw zips are read in place; nothing is extracted.

| Pipeline | Source zip | What it produces (`data/processed/`) |
|---|---|---|
| `hash_samples` | `15038698.zip` (1k malicious + 1k benign, hashes + import lists) | `samples`, `imports_long` (sha256↔library), `features` (numeric + top-200 library one-hots) |
| `xpe_guard` | `archive__6_.zip` (9.6k PE files, 55 static features, 3 families) | `clean` (all columns), `model` (constant columns dropped) |
| `dll_imports` | `archive__7_.zip` (29k samples × 629 DLL flags, Type 0-6) | `features`, `dll_vocabulary` |
| `pe_headers` | `archive__8_.zip` (19.6k labelled PE headers + 17 unlabelled) | `train`, `holdout` |
| `pe_header` | `PE_Header.csv` (29.8k samples, optional-header fields, Type 0-6) | `clean`, `model` |
| `pe_section` | `PE_Section.csv` (29.8k samples × 10 sections, Type 0-6) | `features` (+derived present/exec/write flags, totals), `sections_long` |
| `pe_multiview` | built from `dll_imports` + `pe_header` + `pe_section` | `features` (one row per sample, ~800 columns), `coverage` |

> Naming: `pe_headers` (plural, archive 8, different corpus with `Malware` 0/1 label) and `pe_header` (singular, `PE_Header.csv`, Type 0-6 corpus) are different datasets.

Also written: `catalog.sqlite` (`sample_catalog` table), `reports/<pipeline>.json` (counts + pass/fail checks),
`reports/cross_dataset_overlap.csv`.

## Run
```bash
pip install -r requirements.txt
python -m etl.run_etl                  # all pipelines
python -m etl.run_etl pe_section       # just one
python -m etl.run_etl xpe_guard        # just one
python -m unittest discover -s tests   # unit + end-to-end tests
```
Run these commands from the project root. Do not run a file in `etl/pipelines/` directly;
those modules use package-relative imports.
Exit code is non-zero if any quality check fails. Re-runs are idempotent (outputs overwritten, catalog rows replaced).
Output is Parquet when `pyarrow` is installed, otherwise `.csv.gz` (`config.OUTPUT_FORMAT`).

## Cleaning decisions (all counted in the reports)
* **hash_samples** – trimmed labels (`'   benign'`), lower-cased hashes, 29 duplicate SHA-256s removed (no label conflicts), 53 rows have no import info (kept, flagged `has_import_info=0`).
* **xpe_guard** – 192 exact duplicates removed; 8 constant columns dropped from `model`; `is_dll`, `is_executable`, `has_aslr`, `has_dep` are **not** 0/1 in the source (raw bit values like 8192/64/256), so raw values are kept and `<col>_any` booleans added.
* **dll_imports** – only identical SHA-256 rows removed (3). Identical DLL vectors under different hashes are legitimate. `Type` meaning is undocumented, so it stays `type_code`; set `config.DLL_TYPE_NAMES` if you know the mapping.
* **pe_headers** – snake_case columns; 732 impossible compile timestamps (0 / future) → `compile_time` null + `timestamp_valid=0`; 8 constant columns dropped; 2,984 identical feature vectors flagged `is_feature_duplicate` (set `DROP_FEATURE_DUPLICATES=True` to remove). 4 such groups carry **conflicting labels**, so consider removing them before training.

* **pe_header / pe_section / dll_imports** are three views of one corpus: same SHA-256 keys and `Type` agrees on 100% of shared hashes. `pe_multiview` inner-joins them (needs those three to have run first; the CLI runs them in that order). Absent sections are all-zero in the source, so `<section>_present` / `_exec` / `_write` flags and `n_wx_sections` (write+execute) are derived from the characteristics bits.
* **pe_header** – ~10% of compile timestamps are 0 or in the future (kept, flagged `timestamp_valid=0`); `ImageBase` exceeds 32 bits for PE32+ files, so it is stored as uint64.

## Things to know before modelling
* The `archive__8_`, `archive__6_`, `15038698` datasets and the `Type 0-6` corpus have different feature spaces and are **not** merged into one training table; the catalog links them by SHA-256 only (2 hashes overlap between `hash_samples` and `dll_imports`, with the same class).
* `pe_headers` holdout rows are unlabelled; no leakage check is possible without hashes for most of them.
