"""Central configuration: paths, file names, and dataset-specific constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"          # you put the downloaded files here (not shipped)
PROCESSED_DIR = ROOT / "data" / "processed"  # staging, rebuilt on every run (not shipped)
FINAL_DIR = ROOT / "data" / "final"      # model-ready plain CSVs (this is what you submit)
REPORT_DIR = ROOT / "reports"
CATALOG_DB = PROCESSED_DIR / "catalog.sqlite"   # scratch cross-dataset index, not shipped
SOURCES_CSV = ROOT / "data_sources.csv"  # licence / provenance register (R7), edited by hand


def raw_path(*candidates: str) -> Path:
    """First candidate file name that exists in data/raw (falls back to the first name for error messages)."""
    for name in candidates:
        if (RAW_DIR / name).exists():
            return RAW_DIR / name
    return RAW_DIR / candidates[0]


# ---- raw inputs (file names exactly as downloaded; teammates must use the same names) ----------
ZENODO_ARCHIVE = raw_path("15038698.zip")                 # hashes + import lists
XPE_ARCHIVE = raw_path("archive__6_.zip")                 # XPE-Guard
KAGGLE_ARCHIVE = raw_path("archive__8_.zip")              # dataset_malwares.csv / dataset_test.csv
FIGSHARE_DLL_ARCHIVE = raw_path("archive__7_.zip")        # DLLs_Imported.csv
FIGSHARE_HEADER_CSV = raw_path("PE_Header.csv")
FIGSHARE_SECTION_CSV = raw_path("PE_Section.csv")
PEMD1_ARCHIVE = raw_path("PE-Malware-Dataset1-main.zip", "PE-Malware-Dataset1-main__1_.zip")

# ---- output format -----------------------------------------------------------------------------
OUTPUT_FORMAT = "csv"   # "csv" (default, what the submission needs) | "csv.gz" | "parquet" (needs pyarrow)

# ---- cleaning policy (applied to every model-ready table) ---------------------------------------
DROP_FEATURE_DUPLICATES = True      # keep the first of each identical feature vector
REMOVE_CONFLICTING_LABELS = True    # drop ALL rows of a feature vector that carries more than one label
DROP_CONSTANT_COLUMNS = True        # zero-variance columns are dropped from model tables

# ---- zenodo_hashes -----------------------------------------------------------------------------
TOP_N_LIBRARIES = 200

# ---- Figshare corpus (PE_Header / PE_Section / DLLs_Imported) ------------------------------------
# Type 0 = benign, everything else = malware. SOURCE: the description of the Kaggle mirror of this
# corpus ("Windows Malwares", via baselight.app) lists 0=Benign, 1=RedLineStealer, 2=Downloader, 3=RAT,
# 4=BankingTrojan, 5=SnakeKeyLogger, 6=Spyware. Type 0 has 1,877 rows. NOT yet confirmed against the
# Figshare record itself - verify there and tick it off in data_sources.csv.
FIGSHARE_BENIGN_TYPE = 0
FIGSHARE_TYPE_NAMES = {0: "benign", 1: "redlinestealer", 2: "downloader", 3: "rat",
                       4: "bankingtrojan", 5: "snakekeylogger", 6: "spyware"}

# ---- final export / splitting -------------------------------------------------------------------
# table -> column used to stratify the train/val/test split (and to report class balance)
FINAL_TABLES = {
    "zenodo_hashes": "label",
    "xpe_guard": "family",
    "kaggle_header": "label",
    "figshare_multiview": "type_code",
    "pe_malware_dataset1": "family",
}
SPLIT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}
SPLIT_SEED = 42

SHA256_RE = r"[0-9a-f]{64}"
SHA1_RE = r"[0-9a-f]{40}"
MD5_RE = r"[0-9a-f]{32}"
