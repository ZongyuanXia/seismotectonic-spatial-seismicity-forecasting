"""Filter WSM2025 records and extract the ten first-stage study regions."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    FILTERED_WSM_PATH,
    REGION_RECORD_COUNTS_PATH,
    REGIONAL_RECORDS_DIR,
    REGIONS,
    WSM_RAW_PATH,
    YEAR_END,
    YEAR_START,
    ensure_output_directories,
    region_keys,
)


REQUIRED_COLUMNS = (
    "ISC_ID", "LAT", "LON", "AZI", "DEPTH", "QUALITY", "REGIME", "DATE", "TIME", "TYPE", "EQ_MAG", "MAG_TYPE",
)


def read_csv(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "utf-8", "latin1"):
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False)
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"Could not decode CSV: {path}")


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={column: re.sub(r"\s+", "", str(column)).upper() for column in df.columns})
    aliases = {
        "LONGITUDE": "LON", "LONG": "LON", "LATITUDE": "LAT", "AZIMUTH": "AZI",
        "MAGTYPE": "MAG_TYPE", "MAG": "EQ_MAG", "QUALITYFLAG": "QUALITY", "METHOD": "TYPE",
        "DATETIMEUTC": "DATE", "DATA": "DATE", "ISCID": "ISC_ID", "ISC": "ISC_ID",
    }
    for source, target in aliases.items():
        if source in df.columns and target not in df.columns:
            df = df.rename(columns={source: target})
    return df


def parse_date(series: pd.Series) -> pd.Series:
    values = series.astype(str).str.strip()
    parsed = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns, UTC]")
    yyyymmdd = values.str.match(r"^\d{8}$", na=False)
    parsed.loc[yyyymmdd] = pd.to_datetime(values.loc[yyyymmdd], format="%Y%m%d", errors="coerce", utc=True)
    normalized = values.str.replace(r"[-.]", "/", regex=True)
    ymd = normalized.str.match(r"^\d{4}/\d{1,2}/\d{1,2}$", na=False)
    parsed.loc[ymd] = pd.to_datetime(normalized.loc[ymd], format="%Y/%m/%d", errors="coerce", utc=True)
    dmy = normalized.str.match(r"^\d{1,2}/\d{1,2}/\d{2,4}$", na=False)
    parsed.loc[dmy] = pd.to_datetime(normalized.loc[dmy], dayfirst=True, errors="coerce", utc=True)
    return parsed


def nonempty(series: pd.Series) -> pd.Series:
    return series.notna() & series.astype(str).str.strip().ne("")


def clean_wsm_records(raw: pd.DataFrame) -> pd.DataFrame:
    df = standardize_columns(raw)
    missing = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"WSM2025 is missing required column(s): {', '.join(missing)}")

    df = df.loc[:, list(REQUIRED_COLUMNS)].copy()
    df["DATE"] = parse_date(df["DATE"])
    df["QUALITY"] = df["QUALITY"].astype(str).str.upper().str.strip().str.extract(r"([A-E])", expand=False)
    df = df.loc[df["QUALITY"].isin(("A", "B", "C"))].copy()
    df = df.loc[np.logical_and.reduce([nonempty(df[column]) for column in REQUIRED_COLUMNS])].copy()

    for column in ("LAT", "LON", "AZI", "DEPTH", "EQ_MAG"):
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.loc[
        df["DATE"].notna()
        & df["DATE"].dt.year.between(YEAR_START, YEAR_END)
        & df["LAT"].between(-90, 90)
        & df["LON"].between(-180, 180)
        & df["AZI"].between(0, 180)
        & (df["DEPTH"] >= 0)
        & (df["EQ_MAG"] > 0)
    ].copy()
    df["DATE"] = df["DATE"].dt.strftime("%Y-%m-%d")
    return df.reset_index(drop=True)


def write_region_subsets(df: pd.DataFrame, selected_regions: list[str]) -> pd.DataFrame:
    counts = []
    for key in selected_regions:
        region = REGIONS[key]
        subset = df.loc[
            df["LAT"].between(region.lat_min, region.lat_max)
            & df["LON"].between(region.lon_min, region.lon_max)
        ].copy()
        subset.to_csv(REGIONAL_RECORDS_DIR / f"{key}.csv", index=False, encoding="utf-8")
        counts.append({"region": key, "region_name": region.name, "n_records": len(subset)})
    return pd.DataFrame(counts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=WSM_RAW_PATH, help="Path to the original WSM2025 CSV.")
    parser.add_argument("--regions", nargs="+", default=["all"], help="Region keys, or all.")
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Place WSM2025 at {args.input} or pass --input PATH.")

    ensure_output_directories()
    filtered = clean_wsm_records(read_csv(args.input))
    filtered.to_csv(FILTERED_WSM_PATH, index=False, encoding="utf-8")
    counts = write_region_subsets(filtered, region_keys(args.regions))
    counts.to_csv(REGION_RECORD_COUNTS_PATH, index=False, encoding="utf-8")
    print(f"Saved {len(filtered)} filtered WSM2025 records.")
    print(counts.to_string(index=False))


if __name__ == "__main__":
    main()
