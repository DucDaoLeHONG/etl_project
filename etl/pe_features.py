"""Feature derivation shared by every dataset that carries PE header / section-table fields."""
from __future__ import annotations

import pandas as pd

SCN_EXEC, SCN_WRITE = 0x20000000, 0x80000000   # section Characteristics bits
SECTIONS = ["text", "data", "rdata", "bss", "idata", "edata", "rsrc", "reloc", "tls", "pdata"]
PE_EPOCH_MIN = pd.Timestamp("1992-01-01")


def add_header_features(df: pd.DataFrame) -> pd.DataFrame:
    """Needs snake_case columns `time_date_stamp` and `magic`. Adds timestamp_valid and is_64bit."""
    ts = pd.to_datetime(df["time_date_stamp"], unit="s", errors="coerce")
    return df.assign(
        timestamp_valid=ts.between(PE_EPOCH_MIN, pd.Timestamp.now()).astype("uint8"),
        is_64bit=(df["magic"] == 523).astype("uint8"),   # PE32+ optional header
    )


def add_section_features(df: pd.DataFrame) -> pd.DataFrame:
    """Needs snake_case `<section>_<field>` columns. A missing section is all zeros in the source, so
    presence is derived. Adds <s>_present/_exec/_write plus per-sample totals; returns a new frame."""
    df = df.copy()  # defragment wide frames before adding columns
    derived = {}
    for s in SECTIONS:
        ch, vs = df[f"{s}_characteristics"], df[f"{s}_misc_virtual_size"]
        derived[f"{s}_present"] = ((ch != 0) | (vs != 0)).astype("uint8")
        derived[f"{s}_exec"] = ((ch & SCN_EXEC) != 0).astype("uint8")
        derived[f"{s}_write"] = ((ch & SCN_WRITE) != 0).astype("uint8")
    d = pd.DataFrame(derived, index=df.index)
    tot_v = sum(df[f"{s}_misc_virtual_size"].astype("int64") for s in SECTIONS)
    tot_r = sum(df[f"{s}_size_of_raw_data"].astype("int64") for s in SECTIONS)
    totals = pd.DataFrame({
        "n_sections_present": d[[f"{s}_present" for s in SECTIONS]].sum(axis=1),
        "n_exec_sections": d[[f"{s}_exec" for s in SECTIONS]].sum(axis=1),
        "n_write_sections": d[[f"{s}_write" for s in SECTIONS]].sum(axis=1),
        "n_wx_sections": sum(d[f"{s}_exec"] & d[f"{s}_write"] for s in SECTIONS),  # write+execute: suspicious
        "total_virtual_size": tot_v,
        "total_raw_size": tot_r,
        "raw_to_virtual_ratio": (tot_r / tot_v.where(tot_v > 0)).round(4),
    }, index=df.index)
    return pd.concat([df, d, totals], axis=1)
