"""Central configuration: paths, file names, and dataset-specific constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports"
CATALOG_DB = PROCESSED_DIR / "catalog.sqlite"

# Source archives (read directly from the zip; nothing is extracted to disk).
HASH_ARCHIVE = RAW_DIR / "15038698.zip"
XPE_ARCHIVE = RAW_DIR / "archive__6_.zip"
DLL_ARCHIVE = RAW_DIR / "archive__7_.zip"
PEHDR_ARCHIVE = RAW_DIR / "archive__8_.zip"
# Plain CSVs (same sample universe / Type scheme as the DLL archive).
PE_HEADER_CSV = RAW_DIR / "PE_Header.csv"
PE_SECTION_CSV = RAW_DIR / "PE_Section.csv"

# "auto" -> parquet if pyarrow/fastparquet is installed, else gzip-compressed CSV.
OUTPUT_FORMAT = "auto"  # "auto" | "parquet" | "csv.gz"

# 15038698.zip: how many most-common imported DLLs become one-hot columns.
TOP_N_LIBRARIES = 200

# archive__7_: the source does not document what Type 0-6 mean, so we keep the
# numeric code. Fill this in if you know the mapping, e.g. {0: "benign", ...}.
DLL_TYPE_NAMES: dict[int, str] = {}

# archive__8_: drop zero-variance columns from the model-ready table.
DROP_CONSTANT_COLUMNS = True
# archive__8_: identical feature vectors are common (same packer/template).
# They are flagged always; set True to also remove them from the output.
DROP_FEATURE_DUPLICATES = False

SHA256_RE = r"^[0-9a-f]{64}$"
SHA1_RE = r"^[0-9a-f]{40}$"
MD5_RE = r"^[0-9a-f]{32}$"
