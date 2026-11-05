# -*- coding: utf-8 -*-
"""
Fig1e_Monthly_Counts.py

Figure 1(e):
Monthly record counts of USGS Mw >= 4.0 earthquakes in the Japan study region
and globally filtered WSM2025 stress-indicator records.

Figure logic
------------
- Top axis: monthly USGS Mw >= 4.0 earthquake counts in the Japan study region,
  plotted from 1964-01 through the forecasting cutoff month 2021-05.
- Bottom axis: monthly counts of globally filtered WSM2025 stress-indicator records
  from 1964-01 through 2024-12.
- 1973-01 marks the start of the formal experiment period, but USGS observations
  before 1973 are retained and plotted for catalog-coverage context.
- The USGS 2021-05 cutoff is shown as the final x-axis tick, not as a dashed line.
- Two figure variants are produced: (1) both panels on linear count axes;
  (2) USGS on a logarithmic y-axis while WSM remains on a linear count axis.

Style is aligned with Fig. 1(a), Fig. 1(b), and Fig. 1(c):
- Times New Roman
- panel label size = 14
- thin gray axes frame
- red for USGS, orange for WSM
- no rolling mean or legend; dataset names are placed above the axes
- y-axis label is Counts; key dates are shown as axis ticks

Outputs
-------
  - Fig1e_Monthly_Counts.png
  - Fig1e_Monthly_Counts.tif
  - Fig1e_Monthly_Counts.pdf
  - Fig1e_Monthly_Counts_USGS_log.png
  - Fig1e_Monthly_Counts_USGS_log.tif
  - Fig1e_Monthly_Counts_USGS_log.pdf
  - Fig1e_monthly_counts.csv
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms
from matplotlib.ticker import LogLocator, LogFormatterSciNotation


# =============================================================================
# 0. User settings
# =============================================================================

# -----------------------------------------------------------------------------
# Input / output paths
# -----------------------------------------------------------------------------
USGS_DIR = Path(r"D:\World_force\1_Project_EP\Phase1\Datas\usgs_csv")
WSM_CSV = Path(r"D:\World_force\2_Project_stress\xzy_202607_final\data\WSM_Database_2025.csv")
OUT_DIR = Path(r"D:\World_force\2_Project_stress\xzy_202607_final\Figures")
FIG_STEM = "Fig1e_Monthly_Counts"
FIG_STEM_USGS_LOG = "Fig1e_Monthly_Counts_USGS_log"

# Cache policy:
# - Fig. 1(a) USGS cache is intentionally disabled here because it may only
#   contain the formal 1973-01 to 2021-05 experiment period. Fig. 1(e) must
#   also show pre-1973 USGS observations.
# - The global WSM cache from Fig. 1(c1) can still be reused.
FIG1A_USGS_CACHE = OUT_DIR / "cache" / "fig1a_usgs_japan_filtered_events.csv"
FIG1C1_WSM_CACHE = OUT_DIR / "cache" / "fig1c1_wsm_global_filtered_records.csv"
USE_FIG1A_USGS_CACHE = False
USE_FIG1C1_WSM_CACHE = True

# -----------------------------------------------------------------------------
# Output switches
# -----------------------------------------------------------------------------
SAVE_PNG = True
SAVE_TIFF = True
SAVE_PDF = True
SAVE_COUNTS_CSV = True
DPI = 600
TIFF_DPI = 600

# -----------------------------------------------------------------------------
# Japan study region (used for USGS only)
# -----------------------------------------------------------------------------
LAT_MIN = 30.05
LAT_MAX = 45.95
LON_MIN = 128.10
LON_MAX = 146.80

# Use the same effective-bound logic as Fig. 1(a) / dataset pipeline for USGS.
USE_GRID_EFFECTIVE_BOUNDS_FOR_USGS = True
GLOBAL_H = 3840
GLOBAL_W = 3840

# WSM2025 is counted globally after the common WSM filtering rules below.
# No regional spatial filter is applied to the WSM time series in Fig. 1(e).

# -----------------------------------------------------------------------------
# Time settings
# -----------------------------------------------------------------------------
# Full plotting range
PLOT_START = "1964-01-01"
PLOT_END = "2024-12-31"

# USGS plotting period. Pre-1973 records are included for catalog-coverage
# context, while 1973-01 remains the formal experiment start.
USGS_PLOT_START_YM = "1964_01"
USGS_PLOT_END_YM = "2021_05"   # month-inclusive

# USGS formal experiment period (reference only; does not truncate plotting).
USGS_TIME_START_YM = "1973_01"
USGS_TIME_END_YM = "2021_05"

# Vertical reference lines
FORMAL_START_DATE = pd.Timestamp("1973-01-01")
FORECAST_CUTOFF_DATE = pd.Timestamp("2021-05-01")

# WSM full coverage in this paper
WSM_START_DATE = pd.Timestamp("1964-01-01", tz="UTC")
WSM_END_DATE = pd.Timestamp("2024-12-31 23:59:59", tz="UTC")

# -----------------------------------------------------------------------------
# USGS magnitude filtering
# -----------------------------------------------------------------------------
START_MAG = 4.0
FILTER_TYPE_EARTHQUAKE = True
DROP_MAGTYPES = {"mint", "unknown", "ma", "mun", "no", "uk", "fa", "lg", "mdl", "mlv"}

# -----------------------------------------------------------------------------
# WSM filtering rules
# -----------------------------------------------------------------------------
USE_ISC_ID_ONLY = True
USE_ABC_QUALITY_ONLY = True
REQUIRE_COMPLETE_CORE_FIELDS = True

# -----------------------------------------------------------------------------
# Figure layout
# -----------------------------------------------------------------------------
FIG_WIDTH_CM = 11.33
FIG_HEIGHT_CM = 8.0

# Axes positions: [left, bottom, width, height]
AX_TOP = [0.085, 0.575, 0.875, 0.335]
AX_BOTTOM = [0.085, 0.135, 0.875, 0.335]

# -----------------------------------------------------------------------------
# Font and axis style
# -----------------------------------------------------------------------------
FONT_FAMILY = "Times New Roman"
FONT_SIZE = 8
AXIS_LABEL_SIZE = 8
TICK_LABEL_SIZE = 7
LEGEND_FONT_SIZE = 7
PANEL_LABEL_SIZE = 14

SPINE_COLOR = "#4D4D4D"
SPINE_WIDTH = 0.70

GRIDLINE_COLOR = "#C8C8C8"
GRIDLINE_WIDTH = 0.35
GRIDLINE_ALPHA = 0.75
GRIDLINE_STYLE = "--"

# -----------------------------------------------------------------------------
# Line styles
# -----------------------------------------------------------------------------
USGS_COLOR = "#B22222"    # aligned with Fig. 1(a)
WSM_COLOR = "#E67E22"     # aligned with Fig. 1(b)/(c1)

# Only draw monthly counts; no smoothed / rolling statistics.
RAW_LINEWIDTH = 0.60
RAW_ALPHA = 0.85

SHOW_ROLLING_MEAN = False
ROLLING_WINDOW_MONTHS = 12
ROLLING_LINEWIDTH = 1.10
ROLLING_ALPHA = 0.95

# -----------------------------------------------------------------------------
# Reference line styles
# -----------------------------------------------------------------------------
REFERENCE_LINE_COLOR = "#C62828"
REFERENCE_LINEWIDTH = 0.70
REFERENCE_LINESTYLE = "--"
SHOW_REFERENCE_LABELS = True
REFERENCE_LABEL_SIZE = 6.5
REFERENCE_LABEL_Y = 0.96

# -----------------------------------------------------------------------------
# Tick style
# -----------------------------------------------------------------------------
TICK_DIRECTION = "in"
X_MAJOR_TICK_LENGTH = 3.0
X_MINOR_TICK_LENGTH = 1.8
Y_MAJOR_TICK_LENGTH = 3.0
Y_MINOR_TICK_LENGTH = 1.8

USGS_MAJOR_TICKS = [
    "1970-01-01", "1980-01-01", "1990-01-01",
    "2000-01-01", "2010-01-01", "2021-05-01",
]
WSM_MAJOR_TICKS = [
    "1964-01-01", "1970-01-01", "1980-01-01",
    "1990-01-01", "2000-01-01", "2010-01-01", "2024-12-01",
]

# -----------------------------------------------------------------------------
# Legend style
# -----------------------------------------------------------------------------
ADD_LEGEND = False
LEGEND_LOC = "upper right"
LEGEND_FRAMEON = True
LEGEND_FACE_COLOR = "white"
LEGEND_FACE_ALPHA = 0.90
LEGEND_EDGE_COLOR = "#B0B0B0"
LEGEND_EDGE_WIDTH = 0.25
LEGEND_BORDER_PAD = 0.45
LEGEND_LABEL_SPACING = 0.30
LEGEND_HANDLE_TEXT_PAD = 0.45
LEGEND_HANDLE_LENGTH = 1.20
LEGEND_BORDER_AXES_PAD = 0.35

# -----------------------------------------------------------------------------
# Dataset labels above axes
# -----------------------------------------------------------------------------
TOP_LABEL_SIZE = 8
TOP_LABEL_Y = 1.035

# -----------------------------------------------------------------------------
# Panel label
# -----------------------------------------------------------------------------
SHOW_PANEL_LABEL = False
PANEL_LABEL = "(e)"


# =============================================================================
# 1. Basic style helpers
# =============================================================================

def cm_to_inch(value_cm: float) -> float:
    return value_cm / 2.54


def apply_matplotlib_style() -> None:
    mpl.rcParams.update({
        "font.family": FONT_FAMILY,
        "font.size": FONT_SIZE,
        "axes.labelsize": AXIS_LABEL_SIZE,
        "xtick.labelsize": TICK_LABEL_SIZE,
        "ytick.labelsize": TICK_LABEL_SIZE,
        "legend.fontsize": LEGEND_FONT_SIZE,
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.unicode_minus": False,
    })


def set_spines(ax) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(SPINE_WIDTH)
        spine.set_edgecolor(SPINE_COLOR)


def add_panel_label(ax) -> None:
    if not SHOW_PANEL_LABEL:
        return

    ax.text(
        0.02, 0.98,
        PANEL_LABEL,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=PANEL_LABEL_SIZE,
        fontweight="bold",
        zorder=20,
    )


def add_axis_top_label(ax, label: str) -> None:
    """Place the dataset label centered above the axes frame."""
    ax.text(
        0.5, TOP_LABEL_Y,
        label,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=TOP_LABEL_SIZE,
        fontweight="normal",
        clip_on=False,
        zorder=20,
    )


# =============================================================================
# 2. Region and time helpers
# =============================================================================

def parse_time_ym(s: str) -> Tuple[int, int]:
    s = str(s).strip().replace("/", "-").replace("_", "-")
    y, m = s.split("-")
    return int(y), int(m)


def month_range(time_start_ym: str, time_end_ym: str) -> list[Tuple[int, int]]:
    y0, m0 = parse_time_ym(time_start_ym)
    y1, m1 = parse_time_ym(time_end_ym)

    months: list[Tuple[int, int]] = []
    y, m = y0, m0
    while (y < y1) or (y == y1 and m <= m1):
        months.append((y, m))
        m += 1
        if m == 13:
            m = 1
            y += 1
    return months


def date_bounds_from_ym(time_start_ym: str, time_end_ym: str) -> Tuple[pd.Timestamp, pd.Timestamp]:
    y0, m0 = parse_time_ym(time_start_ym)
    y1, m1 = parse_time_ym(time_end_ym)
    start = pd.Timestamp(f"{y0:04d}-{m0:02d}-01", tz="UTC")
    end = pd.Timestamp(f"{y1:04d}-{m1:02d}-01", tz="UTC") + pd.offsets.MonthEnd(0)
    return start, end


def region_pixel_box(lat_min: float, lat_max: float, lon_min: float, lon_max: float,
                     h_global: int, w_global: int) -> Tuple[int, int, int, int]:
    x_min = int(np.floor((lon_min + 180.0) / 360.0 * w_global))
    x_max_exclusive = int(np.ceil((lon_max + 180.0) / 360.0 * w_global))
    y_min = int(np.floor((lat_min + 90.0) / 180.0 * h_global))
    y_max_exclusive = int(np.ceil((lat_max + 90.0) / 180.0 * h_global))

    x_min = max(0, min(w_global - 1, x_min))
    y_min = max(0, min(h_global - 1, y_min))
    x_max = max(0, min(w_global - 1, x_max_exclusive - 1))
    y_max = max(0, min(h_global - 1, y_max_exclusive - 1))

    if x_max < x_min or y_max < y_min:
        raise ValueError("Empty crop box from study region bounds.")
    return x_min, x_max, y_min, y_max


def effective_bounds_from_pixel_box(pixel_box: Tuple[int, int, int, int],
                                    h_global: int, w_global: int) -> Tuple[float, float, float, float]:
    x_min, x_max, y_min, y_max = pixel_box
    lon_min_eff = x_min / float(w_global) * 360.0 - 180.0
    lon_max_eff = (x_max + 1) / float(w_global) * 360.0 - 180.0
    lat_min_eff = y_min / float(h_global) * 180.0 - 90.0
    lat_max_eff = (y_max + 1) / float(h_global) * 180.0 - 90.0
    return lat_min_eff, lat_max_eff, lon_min_eff, lon_max_eff


def get_usgs_filter_bounds() -> Tuple[float, float, float, float]:
    if not USE_GRID_EFFECTIVE_BOUNDS_FOR_USGS:
        return LAT_MIN, LAT_MAX, LON_MIN, LON_MAX
    box = region_pixel_box(LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, GLOBAL_H, GLOBAL_W)
    return effective_bounds_from_pixel_box(box, GLOBAL_H, GLOBAL_W)



def month_start_series(dt: pd.Series) -> pd.Series:
    """
    Convert a datetime series to month-start timestamps without timezone.
    """
    return dt.dt.tz_convert(None).dt.to_period("M").dt.to_timestamp()


# =============================================================================
# 3. USGS loading and filtering
# =============================================================================

def find_usgs_month_file(usgs_dir: Path, year: int, month: int) -> Optional[Path]:
    m2 = f"{int(month):02d}"
    y = int(year)

    candidates = [
        usgs_dir / f"{y}_{m2}.csv",
        usgs_dir / f"{y}-{m2}.csv",
        usgs_dir / f"usgs{y}_{m2}.csv",
        usgs_dir / f"usgs_{y}_{m2}.csv",
        usgs_dir / f"USGS{y}_{m2}.csv",
        usgs_dir / f"USGS_{y}_{m2}.csv",
        usgs_dir / f"usgs-{y}-{m2}.csv",
        usgs_dir / f"USGS-{y}-{m2}.csv",
    ]

    for fp in candidates:
        if fp.exists():
            return fp

    rx = re.compile(rf"(?i)^(?:usgs[-_]?)*{y}[-_]?{m2}\.csv$")
    if not usgs_dir.exists():
        return None

    for name in os.listdir(usgs_dir):
        if rx.match(name):
            return usgs_dir / name

    return None


def read_csv_robust(path: Path) -> pd.DataFrame:
    last_err = None
    for enc in ["utf-8-sig", "utf-8", "gbk", "cp936", "latin1"]:
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except Exception as exc:
            last_err = exc
    raise RuntimeError(f"CSV read failed: {path}") from last_err


def energy_from_mag_type(mag: float, mag_type: str) -> float:
    """
    Magnitude-type to energy conversion used in the dataset generation code.
    """
    t = str(mag_type).lower()
    m = float(mag)

    if t in ["mww", "mwc", "mwb", "mwr", "mi", "mwp", "ms", "ms_20", "ml", "mlr", "mlv"]:
        a, b = 1.5, 4.8
    elif t in ["mb", "mb_lg", "mblg", "mlg", "lg", "mfa", "fa"]:
        a, b = 2.4, -1.2
    elif t in ["md", "mdl", "mc"]:
        a, b = 1.404, 4.56
    elif t in ["me"]:
        a, b = 1.5, 4.4
    else:
        a, b = 1.5, 4.8

    return float(np.power(10.0, a * m + b))


def mw_from_energy(energy: np.ndarray) -> np.ndarray:
    return (np.log10(np.clip(energy, 1e-12, None)) - 4.8) / 1.5


def normalize_usgs_columns(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    colmap = {str(c).lower(): c for c in df.columns}
    required = ["time", "latitude", "longitude", "mag"]

    if not all(c in colmap for c in required):
        return None

    use_cols = [colmap[c] for c in required]

    optional = []
    for c in ["magtype", "type", "depth"]:
        if c in colmap:
            optional.append(colmap[c])

    out = df[use_cols + optional].copy()

    rename = {
        colmap["time"]: "time",
        colmap["latitude"]: "latitude",
        colmap["longitude"]: "longitude",
        colmap["mag"]: "mag",
    }

    if "magtype" in colmap:
        rename[colmap["magtype"]] = "magType"
    if "type" in colmap:
        rename[colmap["type"]] = "type"
    if "depth" in colmap:
        rename[colmap["depth"]] = "depth"

    return out.rename(columns=rename)


def load_filtered_usgs_events() -> pd.DataFrame:
    """
    Return filtered USGS events used for monthly count plotting.
    """
    if USE_FIG1A_USGS_CACHE and FIG1A_USGS_CACHE.exists():
        print(f"[CACHE] Loading USGS events from Fig. 1(a) cache:")
        print(f"        {FIG1A_USGS_CACHE}")
        df_cache = pd.read_csv(FIG1A_USGS_CACHE)
        df_cache["time"] = pd.to_datetime(df_cache["time"], errors="coerce", utc=True)
        if "Mw" not in df_cache.columns:
            raise ValueError("Fig. 1(a) cache does not contain 'Mw'.")
        return df_cache

    start, end = date_bounds_from_ym(USGS_PLOT_START_YM, USGS_PLOT_END_YM)
    lat_min_eff, lat_max_eff, lon_min_eff, lon_max_eff = get_usgs_filter_bounds()
    months = month_range(USGS_PLOT_START_YM, USGS_PLOT_END_YM)

    frames: list[pd.DataFrame] = []

    print("[LOAD] Reading monthly USGS CSV files ...")
    for year, month in months:
        fp = find_usgs_month_file(USGS_DIR, year, month)

        if fp is None:
            print(f"  [WARN] Missing monthly file: {year}-{month:02d}")
            continue

        df_raw = read_csv_robust(fp)
        df = normalize_usgs_columns(df_raw)

        if df is None or df.empty:
            continue

        if FILTER_TYPE_EARTHQUAKE and "type" in df.columns:
            df = df[df["type"].astype(str).str.lower().eq("earthquake")].copy()

        if "magType" in df.columns:
            mt = df["magType"].astype(str).str.lower().str.strip()
            df = df[~mt.isin(DROP_MAGTYPES)].copy()
        else:
            df["magType"] = ""

        df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
        df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
        df["mag"] = pd.to_numeric(df["mag"], errors="coerce")
        df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)

        df = df.dropna(subset=["time", "latitude", "longitude", "mag"])
        if df.empty:
            continue

        energy = df.apply(
            lambda r: energy_from_mag_type(float(r["mag"]), str(r.get("magType", ""))),
            axis=1
        ).to_numpy(np.float64)

        df["Mw"] = mw_from_energy(energy)

        df = df[df["Mw"] >= START_MAG].copy()
        df = df[(df["time"] >= start) & (df["time"] <= end)].copy()

        df = df[
            (df["latitude"] >= lat_min_eff) & (df["latitude"] <= lat_max_eff) &
            (df["longitude"] >= lon_min_eff) & (df["longitude"] <= lon_max_eff)
        ].copy()

        if df.empty:
            continue

        frames.append(df[["time", "latitude", "longitude", "Mw"]])

    if not frames:
        raise RuntimeError("No USGS events passed the filters. Check paths and settings.")

    events = pd.concat(frames, ignore_index=True).sort_values("time").reset_index(drop=True)

    print(f"[USGS] final filtered events = {len(events):,}")
    return events


# =============================================================================
# 4. WSM loading and filtering
# =============================================================================

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {c: re.sub(r"\s+", "", str(c)).upper() for c in df.columns}
    df = df.rename(columns=rename)

    mapping = {
        "LONGITUDE": "LON",
        "LONG": "LON",
        "LATITUDE": "LAT",
        "AZIMUTH": "AZI",
        "MAGTYPE": "MAG_TYPE",
        "MAG": "EQ_MAG",
        "QUALITYFLAG": "QUALITY",
        "METHOD": "TYPE",
        "DATETIMEUTC": "DATE",
        "DATA": "DATE",
        "ISCID": "ISC_ID",
        "ISC": "ISC_ID",
    }

    for old, new in mapping.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})

    return df


def parse_date_iso_strict(s: pd.Series) -> pd.Series:
    s0 = s.astype(str).str.strip()
    out = pd.Series(pd.NaT, index=s0.index, dtype="datetime64[ns, UTC]")

    m8 = s0.str.match(r"^\d{8}$", na=False)
    out.loc[m8] = pd.to_datetime(
        s0[m8],
        format="%Y%m%d",
        errors="coerce",
        utc=True,
    )

    s1 = s0.str.replace(r"[-\.]", "/", regex=True)

    ymd = s1.str.match(r"^\d{4}/\d{1,2}/\d{1,2}$", na=False)
    out.loc[ymd & out.isna()] = pd.to_datetime(
        s1[ymd],
        format="%Y/%m/%d",
        errors="coerce",
        utc=True,
    )

    dmy = s1.str.match(r"^\d{1,2}/\d{1,2}/\d{2,4}$", na=False)
    out.loc[dmy & out.isna()] = pd.to_datetime(
        s1[dmy],
        dayfirst=True,
        errors="coerce",
        utc=True,
    )

    bad = s0.str.match(r"^\d{4}$", na=False) | s1.str.match(
        r"^\d{1,2}/\d{4}$|^\d{4}/\d{1,2}$",
        na=False,
    )
    out = out.where(~bad, pd.NaT)

    return out


def clean_quality(q: pd.Series) -> pd.Series:
    return q.astype(str).str.upper().str.strip().str.extract(
        r"\s*([A-E])",
        expand=False,
    )


def has_isc_id(s: pd.Series) -> pd.Series:
    s_str = s.astype(str).str.strip()
    invalid_tokens = r"(?i)nan|null|none|na|^-$|^0$"
    return s.notna() & s_str.ne("") & ~s_str.str.fullmatch(invalid_tokens)


def nonempty_mask(series: pd.Series) -> pd.Series:
    if series.dtype == object:
        return series.astype(str).str.strip().ne("") & series.notna()
    return series.notna()


def load_wsm_global_filtered() -> pd.DataFrame:
    """
    Load WSM records using the same filtering logic as Fig. 1(c1).
    """
    if USE_FIG1C1_WSM_CACHE and FIG1C1_WSM_CACHE.exists():
        print(f"[CACHE] Loading WSM records from Fig. 1(c1) cache:")
        print(f"        {FIG1C1_WSM_CACHE}")
        df_cache = pd.read_csv(FIG1C1_WSM_CACHE)
        df_cache["DATE_DT"] = pd.to_datetime(df_cache["DATE_DT"], errors="coerce", utc=True)
        return df_cache

    print(f"[LOAD] {WSM_CSV}")
    df_raw = read_csv_robust(WSM_CSV)
    df = standardize_columns(df_raw)

    if USE_ISC_ID_ONLY:
        if "ISC_ID" not in df.columns:
            raise ValueError("Column 'ISC_ID' was not found.")
        df = df.loc[has_isc_id(df["ISC_ID"])].copy()

    if "DATE" not in df.columns:
        raise ValueError("Column 'DATE' was not found.")

    df["DATE_DT"] = parse_date_iso_strict(df["DATE"])

    if "QUALITY" not in df.columns:
        df["QUALITY"] = np.nan
    df["QUALITY"] = clean_quality(df["QUALITY"])

    if USE_ABC_QUALITY_ONLY:
        df = df.loc[df["QUALITY"].isin(["A", "B", "C"])].copy()

    required_fields = [
        "ISC_ID", "LAT", "LON", "AZI", "DEPTH", "QUALITY",
        "REGIME", "DATE_DT", "TIME", "EQ_MAG",
    ]

    for c in required_fields:
        if c not in df.columns:
            df[c] = np.nan

    if REQUIRE_COMPLETE_CORE_FIELDS:
        mask_complete = np.ones(len(df), dtype=bool)
        for c in required_fields:
            mask_complete &= nonempty_mask(df[c]).to_numpy()
        df = df.loc[mask_complete].copy()

    to_num = lambda x: pd.to_numeric(x, errors="coerce")
    df["LAT"] = to_num(df["LAT"])
    df["LON"] = to_num(df["LON"])
    df["AZI"] = to_num(df["AZI"])
    df["DEPTH"] = to_num(df["DEPTH"])
    df["EQ_MAG"] = to_num(df["EQ_MAG"])

    valid_mask = (
        df["LAT"].between(-90, 90)
        & df["LON"].between(-180, 180)
        & df["AZI"].between(0, 180)
        & (df["DEPTH"] >= 0)
        & (df["EQ_MAG"] > 0)
        & df["DATE_DT"].notna()
    )
    df = df.loc[valid_mask].copy()

    date_mask = df["DATE_DT"].between(WSM_START_DATE, WSM_END_DATE)
    df = df.loc[date_mask].copy()

    print(f"[WSM] global filtered records = {len(df):,}")
    return df



# =============================================================================
# 5. Monthly counts
# =============================================================================

def make_month_index() -> pd.DatetimeIndex:
    return pd.date_range(
        pd.Timestamp(PLOT_START),
        pd.Timestamp(PLOT_END),
        freq="MS"
    )


def monthly_counts_from_datetime(df: pd.DataFrame, date_col: str, month_index: pd.DatetimeIndex) -> pd.Series:
    tmp = df.copy()
    tmp["MONTH"] = month_start_series(tmp[date_col])

    counts = tmp.groupby("MONTH").size()
    counts = counts.reindex(month_index, fill_value=0).astype(int)

    return counts


def rolling_mean(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, center=True, min_periods=max(1, window // 3)).mean()


def build_monthly_count_table() -> pd.DataFrame:
    month_index = make_month_index()

    usgs = load_filtered_usgs_events()
    wsm_global = load_wsm_global_filtered()

    usgs_counts = monthly_counts_from_datetime(usgs, "time", month_index)
    wsm_counts = monthly_counts_from_datetime(wsm_global, "DATE_DT", month_index)

    out = pd.DataFrame({
        "month": month_index,
        "usgs_mw_ge_4_count": usgs_counts.values,
        "wsm_global_record_count": wsm_counts.values,
    })

    # USGS curve must terminate directly at 2021-05 and should not extend rightward.
    out.loc[out["month"] > FORECAST_CUTOFF_DATE, "usgs_mw_ge_4_count"] = np.nan

    if SHOW_ROLLING_MEAN:
        out[f"usgs_roll{ROLLING_WINDOW_MONTHS}"] = rolling_mean(
            out["usgs_mw_ge_4_count"],
            ROLLING_WINDOW_MONTHS
        )
        out[f"wsm_global_roll{ROLLING_WINDOW_MONTHS}"] = rolling_mean(
            out["wsm_global_record_count"],
            ROLLING_WINDOW_MONTHS
        )

    if SAVE_COUNTS_CSV:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out_csv = OUT_DIR / "Fig1e_monthly_counts.csv"
        out.to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"[SAVE] {out_csv}")

    print("\n========== Monthly count summary ==========")
    print(f"USGS total count in plotted table: {out['usgs_mw_ge_4_count'].sum():,}")
    print(f"WSM global total count in plotted table:  {out['wsm_global_record_count'].sum():,}")

    return out


# =============================================================================
# 6. Plotting
# =============================================================================

def add_reference_lines(
    ax,
    show_start: bool = True,
    show_cutoff: bool = True,
) -> None:
    """
    Add experiment-period reference lines.

    USGS panel:
      - show_start=True
      - show_cutoff=False

    WSM panel:
      - show_start=True
      - show_cutoff=True
    """
    if show_start:
        ax.axvline(
            FORMAL_START_DATE,
            color=REFERENCE_LINE_COLOR,
            linewidth=REFERENCE_LINEWIDTH,
            linestyle=REFERENCE_LINESTYLE,
            zorder=4,
        )

    if show_cutoff:
        ax.axvline(
            FORECAST_CUTOFF_DATE,
            color=REFERENCE_LINE_COLOR,
            linewidth=REFERENCE_LINEWIDTH,
            linestyle=REFERENCE_LINESTYLE,
            zorder=4,
        )

    if SHOW_REFERENCE_LABELS:
        trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
        right_offset = mtransforms.ScaledTranslation(
            4 / 72, 0, ax.figure.dpi_scale_trans
        )
        left_offset = mtransforms.ScaledTranslation(
            -4 / 72, 0, ax.figure.dpi_scale_trans
        )

        if show_start:
            ax.text(
                FORMAL_START_DATE,
                REFERENCE_LABEL_Y,
                "1973-01",
                transform=trans + right_offset,
                ha="left",
                va="top",
                rotation=0,
                fontsize=REFERENCE_LABEL_SIZE,
                color=REFERENCE_LINE_COLOR,
            )

        if show_cutoff:
            ax.text(
                FORECAST_CUTOFF_DATE,
                REFERENCE_LABEL_Y,
                "2021-05",
                transform=trans + left_offset,
                ha="right",
                va="top",
                rotation=0,
                fontsize=REFERENCE_LABEL_SIZE,
                color=REFERENCE_LINE_COLOR,
            )


def format_time_axis(
    ax,
    x_start=PLOT_START,
    x_end=PLOT_END,
    show_xlabel: bool = False,
    show_ticklabels: bool = True,
    major_ticks: Optional[list[str | pd.Timestamp]] = None,
) -> None:
    """Format an independent time axis with explicit YYYY-MM major ticks."""
    x_start_ts = pd.Timestamp(x_start)
    x_end_ts = pd.Timestamp(x_end)
    ax.set_xlim(x_start_ts, x_end_ts)

    if major_ticks is None:
        tick_list = [x_start_ts, x_end_ts]
    else:
        tick_list = [pd.Timestamp(t).replace(day=1) for t in major_ticks]
        tick_list = [t for t in tick_list if x_start_ts <= t <= x_end_ts]
        if not tick_list:
            tick_list = [x_start_ts, x_end_ts]

    tick_list = sorted(set(tick_list))
    tick_labels = [t.strftime("%Y-%m") for t in tick_list]

    ax.xaxis.set_major_locator(mticker.FixedLocator(mdates.date2num(tick_list)))
    ax.xaxis.set_major_formatter(mticker.FixedFormatter(tick_labels))
    ax.xaxis.set_minor_locator(mdates.YearLocator(base=5))

    ax.tick_params(
        axis="x",
        which="major",
        labelsize=TICK_LABEL_SIZE,
        length=X_MAJOR_TICK_LENGTH,
        direction=TICK_DIRECTION,
        labelbottom=show_ticklabels,
        pad=2,
    )
    ax.tick_params(
        axis="x",
        which="minor",
        length=X_MINOR_TICK_LENGTH,
        direction=TICK_DIRECTION,
    )
    ax.tick_params(
        axis="y",
        which="major",
        labelsize=TICK_LABEL_SIZE,
        length=Y_MAJOR_TICK_LENGTH,
        direction=TICK_DIRECTION,
    )
    ax.tick_params(
        axis="y",
        which="minor",
        length=Y_MINOR_TICK_LENGTH,
        direction=TICK_DIRECTION,
    )

    if show_xlabel:
        ax.set_xlabel("Year", fontsize=AXIS_LABEL_SIZE)

    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=4, integer=True))

    ax.grid(
        axis="both",
        color=GRIDLINE_COLOR,
        linewidth=GRIDLINE_WIDTH,
        alpha=GRIDLINE_ALPHA,
        linestyle=GRIDLINE_STYLE,
        zorder=0,
    )

    set_spines(ax)


def add_small_legend(ax, color: str) -> None:
    if not ADD_LEGEND:
        return

    handles = []

    handles.append(
        plt.Line2D(
            [0], [0],
            color=color,
            linewidth=RAW_LINEWIDTH,
            alpha=RAW_ALPHA,
            label="Monthly count",
        )
    )

    if SHOW_ROLLING_MEAN:
        handles.append(
            plt.Line2D(
                [0], [0],
                color=color,
                linewidth=ROLLING_LINEWIDTH,
                alpha=ROLLING_ALPHA,
                label=f"{ROLLING_WINDOW_MONTHS}-month moving average",
            )
        )

    leg = ax.legend(
        handles=handles,
        loc=LEGEND_LOC,
        fontsize=LEGEND_FONT_SIZE,
        frameon=LEGEND_FRAMEON,
        framealpha=LEGEND_FACE_ALPHA,
        facecolor=LEGEND_FACE_COLOR,
        edgecolor=LEGEND_EDGE_COLOR,
        borderpad=LEGEND_BORDER_PAD,
        labelspacing=LEGEND_LABEL_SPACING,
        handletextpad=LEGEND_HANDLE_TEXT_PAD,
        handlelength=LEGEND_HANDLE_LENGTH,
        borderaxespad=LEGEND_BORDER_AXES_PAD,
        fancybox=False,
    )
    leg.get_frame().set_linewidth(LEGEND_EDGE_WIDTH)


def plot_series(ax, x, y, color: str, y_roll=None) -> None:
    ax.plot(
        x, y,
        color=color,
        linewidth=RAW_LINEWIDTH,
        alpha=RAW_ALPHA,
        zorder=2,
    )

    if SHOW_ROLLING_MEAN and y_roll is not None:
        ax.plot(
            x, y_roll,
            color=color,
            linewidth=ROLLING_LINEWIDTH,
            alpha=ROLLING_ALPHA,
            zorder=3,
        )


def _configure_y_axis(ax, y: pd.Series, log_scale: bool) -> None:
    """Configure one panel's y-axis without changing the other panel."""
    if log_scale:
        finite = y.to_numpy(dtype=float)
        finite = finite[np.isfinite(finite) & (finite > 0)]
        ymax = float(np.max(finite)) if finite.size else 10.0
        ax.set_yscale("log")
        ax.set_ylim(0.8, max(1.0, ymax * 1.25))
        ax.yaxis.set_major_locator(LogLocator(base=10.0))
        ax.yaxis.set_major_formatter(LogFormatterSciNotation(base=10.0))
    else:
        ax.set_yscale("linear")
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=4, integer=True))


def draw_fig1e(
    counts: pd.DataFrame,
    usgs_log_y: bool = False,
    wsm_log_y: bool = False,
    basename: str = FIG_STEM,
) -> None:
    apply_matplotlib_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(
        figsize=(cm_to_inch(FIG_WIDTH_CM), cm_to_inch(FIG_HEIGHT_CM)),
        dpi=DPI,
    )

    ax_usgs = fig.add_axes(AX_TOP)
    ax_wsm = fig.add_axes(AX_BOTTOM)

    x = counts["month"]
    usgs_y = counts["usgs_mw_ge_4_count"].copy()
    wsm_y = counts["wsm_global_record_count"].copy()

    # A true logarithmic axis cannot represent zero. Zero-count months are
    # therefore masked only in the panel that uses logarithmic scaling.
    usgs_plot = usgs_y.where(usgs_y > 0, np.nan) if usgs_log_y else usgs_y
    wsm_plot = wsm_y.where(wsm_y > 0, np.nan) if wsm_log_y else wsm_y

    # Top: USGS
    usgs_roll_col = f"usgs_roll{ROLLING_WINDOW_MONTHS}"
    if usgs_roll_col in counts.columns:
        usgs_roll = counts[usgs_roll_col]
        if usgs_log_y:
            usgs_roll = usgs_roll.where(usgs_roll > 0, np.nan)
    else:
        usgs_roll = None

    plot_series(
        ax_usgs,
        x,
        usgs_plot,
        USGS_COLOR,
        usgs_roll,
    )
    ax_usgs.set_ylabel("Counts", fontsize=AXIS_LABEL_SIZE)
    add_axis_top_label(ax_usgs, "USGS catalog")

    # USGS uses an independent x-axis from 1964-01 to 2021-05.
    # 1973-01 marks the formal experiment start. The 2021-05 cutoff is
    # represented by the final x-axis tick rather than a dashed line.
    add_reference_lines(
        ax_usgs,
        show_start=True,
        show_cutoff=False,
    )
    format_time_axis(
        ax_usgs,
        x_start=PLOT_START,
        x_end=FORECAST_CUTOFF_DATE,
        show_xlabel=False,
        show_ticklabels=True,
        major_ticks=USGS_MAJOR_TICKS,
    )
    _configure_y_axis(ax_usgs, usgs_plot, usgs_log_y)
    add_small_legend(ax_usgs, USGS_COLOR)
    add_panel_label(ax_usgs)

    # Bottom: WSM
    wsm_roll_col = f"wsm_global_roll{ROLLING_WINDOW_MONTHS}"
    if wsm_roll_col in counts.columns:
        wsm_roll = counts[wsm_roll_col]
        if wsm_log_y:
            wsm_roll = wsm_roll.where(wsm_roll > 0, np.nan)
    else:
        wsm_roll = None

    plot_series(
        ax_wsm,
        x,
        wsm_plot,
        WSM_COLOR,
        wsm_roll,
    )
    ax_wsm.set_ylabel("Counts", fontsize=AXIS_LABEL_SIZE)
    add_axis_top_label(ax_wsm, "WSM2025 global records")

    # WSM retains the full 1964-2024 global temporal coverage.
    add_reference_lines(
        ax_wsm,
        show_start=False,
        show_cutoff=False,
    )
    format_time_axis(
        ax_wsm,
        x_start=PLOT_START,
        x_end=PLOT_END,
        show_xlabel=True,
        show_ticklabels=True,
        major_ticks=WSM_MAJOR_TICKS,
    )
    _configure_y_axis(ax_wsm, wsm_plot, wsm_log_y)
    add_small_legend(ax_wsm, WSM_COLOR)

    save_figure(fig, basename)
    plt.close(fig)


# =============================================================================
# 7. Save and driver
# =============================================================================

def save_figure(fig, basename: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if SAVE_PDF:
        fp = OUT_DIR / f"{basename}.pdf"
        fig.savefig(fp, bbox_inches="tight", pad_inches=0.02)
        print(f"[SAVE] {fp}")

    if SAVE_TIFF:
        fp = OUT_DIR / f"{basename}.tif"
        try:
            fig.savefig(
                fp,
                dpi=TIFF_DPI,
                bbox_inches="tight",
                pad_inches=0.02,
                pil_kwargs={"compression": "tiff_lzw"},
            )
        except TypeError:
            fig.savefig(
                fp,
                dpi=TIFF_DPI,
                bbox_inches="tight",
                pad_inches=0.02,
            )
        print(f"[SAVE] {fp}")

    if SAVE_PNG:
        fp = OUT_DIR / f"{basename}.png"
        fig.savefig(fp, dpi=DPI, bbox_inches="tight", pad_inches=0.02)
        print(f"[SAVE] {fp}")


def main() -> None:
    print("========== Draw Fig. 1(e): monthly counts ==========")
    print(f"OUT_DIR: {OUT_DIR}")
    print(f"USGS plotted range: {USGS_PLOT_START_YM} to {USGS_PLOT_END_YM}")
    print(f"USGS formal experiment range: {USGS_TIME_START_YM} to {USGS_TIME_END_YM}")
    print(f"WSM x-range: {PLOT_START} to {PLOT_END}")
    print("WSM spatial coverage: global")
    print(f"USE_FIG1A_USGS_CACHE: {USE_FIG1A_USGS_CACHE}")
    print(f"USE_FIG1C1_WSM_CACHE: {USE_FIG1C1_WSM_CACHE}")

    counts = build_monthly_count_table()

    # Variant 1: both panels use ordinary count axes.
    draw_fig1e(
        counts,
        usgs_log_y=False,
        wsm_log_y=False,
        basename=FIG_STEM,
    )

    # Variant 2: USGS uses a logarithmic y-axis, while WSM remains linear.
    draw_fig1e(
        counts,
        usgs_log_y=True,
        wsm_log_y=False,
        basename=FIG_STEM_USGS_LOG,
    )

    print("[DONE] Fig. 1(e) linear-linear and USGS-log/WSM-linear outputs saved.")


if __name__ == "__main__":
    main()