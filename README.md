# Malware-dataset ETL pipelines

Eight pipelines (extract → transform → validate → load) turn five public malware datasets into
**model-ready, plain-CSV tables** in `data/final/`, each with duplicate feature vectors and conflicting-label
rows removed and a stratified `split` column (train/val/test = 70/15/15, seed 42).

## 1. Set up (Windows / macOS / Linux)

```bash
pip install -r requirements.txt          # pandas + numpy only
# put the raw downloads in data/raw/ with EXACTLY these file names (rename if your download differs):
#   15038698.zip                                 Zenodo
#   archive__6_.zip                              XPE-Guard
#   archive__8_.zip                              Kaggle  (dataset_malwares.csv, dataset_test.csv)
#   archive__7_.zip, PE_Header.csv, PE_Section.csv   Figshare
#   PE-Malware-Dataset1-main.zip                 PE-Malware-Dataset1 (GitHub zip; "PE-Malware-Dataset1-main__1_.zip" also works)
python -m etl.run_etl                    # all pipelines + data/final export + licence check (run from the project root)
python -m etl.run_etl xpe_guard          # a single pipeline
python -m unittest discover -s tests     # 16 tests (needs the raw files)
```
All paths use `pathlib`, so there are no OS-specific separators. Raw files are read in place (nothing is unzipped).
`figshare_multiview` needs the three `figshare_*` view pipelines to run first; the CLI runs them in that order.

## 2. Pipelines and results

| Pipeline | Dataset (raw file) | Raw rows | Final rows | Label columns in final CSV |
|---|---|---:|---:|---|
| `zenodo_hashes` | Zenodo (`15038698.zip`) | 2,000 | 1,909 | `label` (0 benign / 1 malicious) |
| `xpe_guard` | XPE-Guard (`archive__6_.zip`) | 9,600 | 9,380 | `label`, `family` (benign, ransomware, spyware, trojan) |
| `kaggle_header` | Kaggle (`archive__8_.zip`) | 19,611 labelled (+17 unlabelled) | 16,623 | `label` (0 benign / 1 malware) |
| `figshare_dll` / `_header` / `_section` | Figshare (`archive__7_.zip`, `PE_Header.csv`, `PE_Section.csv`) | 29,498 / 29,807 / 29,760 | 907 / 23,354 / 20,744 | `type_code` (0-6), `label` |
| `figshare_multiview` | the three Figshare views joined on `sha256` | - | **23,426** | `type_code`, `label` |
| `pe_malware_dataset1` | PE-Malware-Dataset1 (`Header.csv` ⋈ `Sections.csv`) | 18,551 / 18,546 | 14,100 | `family` (5 families), `label` = 1 for all |

`data/final/` contains one CSV per dataset: `zenodo_hashes`, `xpe_guard`, `kaggle_header`, `figshare_multiview`,
`pe_malware_dataset1` (edit `FINAL_TABLES` in `etl/config.py` to export others, e.g. a single Figshare view). Everything
else (staging tables, hash/metadata, the scratch `catalog.sqlite`) stays in `data/processed/` and is not meant to be submitted.

Row counts after **every** step are in `reports/<pipeline>.json` (`row_counts`) and, combined, `reports/row_counts.csv`.
Quality checks (pass/fail + detail) are in the same JSON; the CLI exits non-zero if any fails.

## 3. Cleaning policy (identical for all datasets, `etl/config.py`)

1. Normalise names/IDs (snake_case columns, lower-case hashes, trimmed labels); remove repeated SHA-256/name keys.
2. **Conflicting labels:** every row of a feature vector that appears with more than one label is removed (`REMOVE_CONFLICTING_LABELS`).
3. **Duplicate feature vectors:** the first row of each identical vector is kept (`DROP_FEATURE_DUPLICATES`).
4. Constant columns are dropped from model tables; unusable values are flagged instead of deleted (e.g. `timestamp_valid`).
5. Only then is the stratified `split` assigned, so identical rows can never land in both train and test.

Conflicts are judged on the finest label (`type_code` / `family`), so the multi-class tasks are safe; the reports also give
the number of rows that conflict on the binary label alone (`conflicting_rows_on_label_alone`).

| Dataset | Conflicting groups / rows removed | Duplicate vectors removed |
|---|---:|---:|
| zenodo_hashes | 0 / 0 | 62 |
| xpe_guard | 12 / 60 | 160 |
| kaggle_header | **4** / 40 | 2,948 |
| figshare_dll | 107 / 22,820 | 5,768 |
| figshare_header | 61 / 395 | 6,058 |
| figshare_section | 49 / 986 | 8,030 |
| figshare_multiview | 23 / 216 | 5,850 |
| pe_malware_dataset1 | 455 / 4,281 | 165 |

Things this exposes - read before modelling:
* **`figshare_dll` alone is not usable** (907 rows left): the same DLL set occurs under different Types, and 15,843 rows conflict even between benign and malware. Use `figshare_multiview` (or header/section) instead.
* **`pe_malware_dataset1` loses 23% of its rows** to conflicting family labels (identical header+section vectors labelled with different VirusTotal families). Family labels are noisy for these features.
* **Leakage between datasets:** 5,083 of its hashes also occur in the Figshare corpus (`reports/cross_dataset_overlap.csv`), and a few hashes occur in the Zenodo set. Do not train on one and test on the other without removing these.
* `pe_malware_dataset1` has no benign samples, so it supports family classification or serves as extra malware beside benign data from elsewhere.
* The Kaggle "holdout" (`dataset_test.csv`, 17 unlabelled rows) is not a split and is not in `data/final/`; use the `split` column.

## 4. Figshare labels (`type_code` → `label`)

`label = 0` if `Type == 0` (benign) else `1`. Mapping used (`config.FIGSHARE_TYPE_NAMES`): 0 benign, 1 RedLineStealer, 2 Downloader,
3 RAT, 4 BankingTrojan, 5 SnakeKeyLogger, 6 Spyware. **Source:** the description of the Kaggle mirror of this corpus
("Windows Malwares", seen on baselight.app) - *not yet confirmed against the Figshare record itself*. The 1,877 Type-0 rows
match the benign count you quoted, and a test asserts it. Confirm on Figshare and tick it in `data_sources.csv`.

## 5. Class imbalance (rubric R1)

Measured in `reports/<pipeline>.json` (`class_balance*`) and per split in `reports/class_balance_by_split.csv`.

| Final table | Class counts | Imbalance (max/min) |
|---|---|---:|
| zenodo_hashes | 949 benign / 960 malicious | 1.0 |
| xpe_guard | 4,678 benign / 4,702 malware; families 1,562-1,576 vs benign 4,678 | 1.0 binary, 3.0 family |
| kaggle_header | 4,962 benign / 11,661 malware | 2.4 |
| figshare_multiview | **1,876 benign / 21,550 malware** (8%); Types 0-6: 1,359-4,813 | **11.5** binary, 3.5 type |
| pe_malware_dataset1 | families 1,454-5,084 | 3.5 |

Strategy (implemented in `etl/splits.py` and `etl/imbalance.py`, applied by the modelling code, not by the ETL):
1. Stratify on the finest label, **split first**, balance second - resampling before the split puts copies of the same row in train and test.
2. Balance **train only**: `class_weights(y_train)` (preferred, no data change) or `random_resample(train, label, "over"|"under")`; validation/test keep the natural distribution.
3. Report macro-F1, per-class recall, PR-AUC and MCC alongside accuracy (accuracy hides a 92 %-malware prior), and compare against the unweighted baseline.
4. Only `figshare_multiview` (binary) is strongly imbalanced; the other tables are near balanced or moderately skewed.

## 6. Data sources & licences (rubric R7)

`data_sources.csv` is the register (source, URL, **licence**, licence_verified, backup location). `python -m etl.run_etl`
checksums every raw file into `reports/data_manifest.csv` (SHA-256 + size, so a backup or a teammate's copy can be verified
byte for byte) and warns about empty fields. **All seven rows still lack a licence and backup location** - they must be filled
from the source pages; they cannot be read from the data. The PE-Malware-Dataset1 zip contains no LICENSE file.

## 7. Layout
```
etl/          pipelines/, config.py, common.py (step logging, dedupe/conflict removal), pe_features.py,
              splits.py, imbalance.py, export_final.py, manifest.py, catalog.py, run_etl.py
tests/        unit tests + end-to-end test on the real data
data/raw/     your downloads (not shipped)      data/processed/  staging, rebuilt each run (not shipped)
data/final/   model-ready CSVs (submit these)   reports/         row counts, checks, class balance, overlap, manifest
data_sources.csv   provenance + licence register
```
