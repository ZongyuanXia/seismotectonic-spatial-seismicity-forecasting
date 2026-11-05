# -*- coding: utf-8 -*-
"""
Fig1d_WSM_Counts.py

Figure 1(d):
Horizontal bar chart of WSM2025 record counts for the Japan main
forecasting region and nine auxiliary RF validation regions.

This script complements Fig1c1_Region_Map_with_Global_WSM_v6.py.

Data logic
----------
1) Prefer reading the global filtered WSM cache produced by Fig. 1(c1):
       cache/fig1c1_wsm_global_filtered_records.csv
2) If the cache does not exist, read the original WSM2025 CSV and apply
   the same filtering logic:
       ISC_ID retained
       QUALITY in {A, B, C}
       core fields complete
       valid LAT/LON/AZI/DEPTH/EQ_MAG/DATE
       DATE from 1964 to 2024
3) Count WSM2025 records in Japan and the nine auxiliary regions.
4) Draw a horizontal bar chart using the same role colors as Fig. 1(c1):
       Japan: main region color
       Auxiliary regions: auxiliary region color

Outputs
-------
  - Fig1d_WSM_Counts.png
  - Fig1d_WSM_Counts.tif
  - Fig1d_WSM_Counts.pdf
  - Fig1d_region_counts.csv
"""

from __future__ import annotations

import re
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# =============================================================================
# 0. User settings
# =============================================================================

# -----------------------------------------------------------------------------
# Input / output paths
# -----------------------------------------------------------------------------
WSM_CSV = Path(r"D:\World_force\2_Project_stress\xzy_202607_final\data\WSM_Database_2025.csv")
OUT_DIR = Path(r"D:\World_force\2_Project_stress\xzy_202607_final\Figures")
FIG_STEM = "Fig1d_WSM_Counts"

# If Fig. 1(c1) has already generated this cache, the script will use it directly.
C1_CACHE_FILE = OUT_DIR / "cache" / "fig1c1_wsm_global_filtered_records.csv"
USE_C1_CACHE_IF_AVAILABLE = True

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
# WSM time range
# -----------------------------------------------------------------------------
WSM_START_DATE = pd.Timestamp("1964-01-01", tz="UTC")
WSM_END_DATE = pd.Timestamp("2024-12-31 23:59:59", tz="UTC")

# -----------------------------------------------------------------------------
# WSM filtering rules
# -----------------------------------------------------------------------------
USE_ISC_ID_ONLY = True
USE_ABC_QUALITY_ONLY = True
REQUIRE_COMPLETE_CORE_FIELDS = True

# -----------------------------------------------------------------------------
# Figure size and axis placement
# -----------------------------------------------------------------------------
# Height is kept the same as Fig. 1(c1), so c1 and d can be combined manually.
BAR_WIDTH_CM = 8.0
BAR_HEIGHT_CM = 9.7

# Axes position in figure coordinates: [left, bottom, width, height]
# Increase left if long region names are clipped.
BAR_AX = [0.36, 0.125, 0.56, 0.790]

# -----------------------------------------------------------------------------
# Font settings; keep aligned with Fig. 1(a), Fig. 1(b), and Fig. 1(c1)
# -----------------------------------------------------------------------------
FONT_FAMILY = "Times New Roman"
FONT_SIZE = 8
AXIS_LABEL_SIZE = 8
TICK_LABEL_SIZE = 7
BAR_VALUE_LABEL_SIZE = 7.5

# -----------------------------------------------------------------------------
# Bar chart style
# -----------------------------------------------------------------------------
MAIN_COLOR = "#B22222"   # Japan, aligned with Fig. 1(a) red
AUX_COLOR = "#3E517A"    # auxiliary regions, aligned with Fig. 1(c1) blue

BAR_ALPHA = 0.88
BAR_EDGE_COLOR = "none"
BAR_HEIGHT = 0.66

BAR_XLABEL = "WSM2025 record count"
SHOW_BAR_VALUES = True
BAR_VALUE_OFFSET_FRAC = 0.015

GRIDLINE_COLOR = "#C8C8C8"
GRIDLINE_WIDTH = 0.35
GRIDLINE_ALPHA = 0.75
GRIDLINE_STYLE = "--"

SPINE_COLOR = "#4D4D4D"
SPINE_WIDTH = 0.70

# -----------------------------------------------------------------------------
# Count settings
# -----------------------------------------------------------------------------
# If True, Japan's count uses the older regional statistics window
# 30-46 deg N, 128-146 deg E. This is useful if the manuscript currently reports
# the Japan WSM count from the earlier regional extraction script.
# The map box in Fig. 1(c1) can still show the formal main study region
# 30.05-45.95 deg N, 128.10-146.80 deg E.
USE_JAPAN_LEGACY_COUNT_BOUNDS = True
JAPAN_COUNT_BOUNDS = {"lat": (30.0, 46.0), "lon": (128.0, 146.0)}

# -----------------------------------------------------------------------------
# Region definitions
# -----------------------------------------------------------------------------
# Turkey is intentionally omitted.
REGIONS = OrderedDict({
    "Japan": {
        "lat": (30.05, 45.95), "lon": (128.10, 146.80),
        "role": "main",
    },
    "Greece": {
        "lat": (33.0, 43.0), "lon": (18.0, 31.0),
        "role": "aux",
    },
    "Indonesia": {
        "lat": (-11.0, 6.0), "lon": (94.0, 123.0),
        "role": "aux",
    },
    "Mexico": {
        "lat": (13.0, 33.0), "lon": (-117.0, -91.0),
        "role": "aux",
    },
    "Chile": {
        "lat": (-35.0, -29.0), "lon": (-75.0, -69.0),
        "role": "aux",
    },
    "Italy": {
        "lat": (36.0, 48.0), "lon": (5.0, 20.0),
        "role": "aux",
    },
    "New Zealand": {
        "lat": (-47.5, -34.5), "lon": (166.5, 178.5),
        "role": "aux",
    },
    "Taiwan": {
        "lat": (21.0, 26.0), "lon": (119.0, 123.0),
        "role": "aux",
    },
    "Iran": {
        "lat": (25.0, 40.0), "lon": (44.0, 64.0),
        "role": "aux",
    },
    "Southern California": {
        "lat": (32.3, 37.0), "lon": (-122.0, -114.0),
        "role": "aux",
    },
})


# =============================================================================
# 1. Style helpers
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


# =============================================================================
# 2. WSM loading and filtering
# =============================================================================

def read_csv_robust(path: Path) -> pd.DataFrame:
    last_err = None
    for enc in ["utf-8-sig", "utf-8", "gbk", "latin1"]:
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except Exception as exc:
            last_err = exc
    raise RuntimeError(f"Failed to read CSV: {path}") from last_err


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

    # Avoid treating incomplete year-only or year-month strings as valid dates.
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
    Load global WSM records with the same filtering logic used in Fig. 1(c1).
    """
    if USE_C1_CACHE_IF_AVAILABLE and C1_CACHE_FILE.exists():
        print("[CACHE] Loading global WSM records from Fig. 1(c1) cache:")
        print(f"        {C1_CACHE_FILE}")
        df = pd.read_csv(C1_CACHE_FILE)
        df["DATE_DT"] = pd.to_datetime(df["DATE_DT"], errors="coerce", utc=True)
        return df

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

    keep_cols = [
        "ISC_ID", "DATE_DT", "LAT", "LON", "AZI", "DEPTH",
        "EQ_MAG", "QUALITY", "REGIME",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].copy()

    print(f"[FILTER] WSM complete A-C ISC records = {len(df):,}")
    return df


def count_regions(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for name, cfg in REGIONS.items():
        if name == "Japan" and USE_JAPAN_LEGACY_COUNT_BOUNDS:
            lat_min, lat_max = JAPAN_COUNT_BOUNDS["lat"]
            lon_min, lon_max = JAPAN_COUNT_BOUNDS["lon"]
        else:
            lat_min, lat_max = cfg["lat"]
            lon_min, lon_max = cfg["lon"]

        mask = (
            df["LAT"].between(lat_min, lat_max)
            & df["LON"].between(lon_min, lon_max)
        )

        rows.append({
            "region": name,
            "role": cfg["role"],
            "lat_min_count": lat_min,
            "lat_max_count": lat_max,
            "lon_min_count": lon_min,
            "lon_max_count": lon_max,
            "wsm_record_count": int(mask.sum()),
        })

    out = pd.DataFrame(rows)
    out = out.sort_values("wsm_record_count", ascending=False).reset_index(drop=True)

    if SAVE_COUNTS_CSV:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out_csv = OUT_DIR / "Fig1d_region_counts.csv"
        out.to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"[SAVE] {out_csv}")

    print("\n========== Region WSM2025 record counts ==========")
    print(out[["region", "role", "wsm_record_count"]].to_string(index=False))

    return out


# =============================================================================
# 3. Plotting
# =============================================================================


def plot_region_bars(ax, counts: pd.DataFrame) -> None:
    # Sort descending, then invert y-axis so the largest appears at the top.
    df = counts.copy().sort_values("wsm_record_count", ascending=False).reset_index(drop=True)

    colors = [MAIN_COLOR if role == "main" else AUX_COLOR for role in df["role"]]
    y = np.arange(len(df))

    ax.barh(
        y,
        df["wsm_record_count"],
        color=colors,
        alpha=BAR_ALPHA,
        edgecolor=BAR_EDGE_COLOR,
        height=BAR_HEIGHT,
        zorder=3,
    )

    ax.set_yticks(y)
    ax.set_yticklabels(df["region"], fontsize=TICK_LABEL_SIZE)
    ax.invert_yaxis()

    ax.tick_params(axis="x", labelsize=TICK_LABEL_SIZE, length=0)
    ax.tick_params(axis="y", length=0)

    ax.set_xlabel(BAR_XLABEL, fontsize=AXIS_LABEL_SIZE)

    ax.grid(
        axis="x",
        color=GRIDLINE_COLOR,
        linewidth=GRIDLINE_WIDTH,
        alpha=GRIDLINE_ALPHA,
        linestyle=GRIDLINE_STYLE,
        zorder=0,
    )

    ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))

    max_val = max(df["wsm_record_count"].max(), 1)
    ax.set_xlim(0, max_val * 1.20)

    if SHOW_BAR_VALUES:
        for yi, val in zip(y, df["wsm_record_count"]):
            ax.text(
                val + max_val * BAR_VALUE_OFFSET_FRAC,
                yi,
                f"{int(val):,}",
                va="center",
                ha="left",
                fontsize=BAR_VALUE_LABEL_SIZE,
                color="#333333",
            )

    set_spines(ax)


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


def draw_d_bar_chart(counts: pd.DataFrame) -> None:
    apply_matplotlib_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(
        figsize=(cm_to_inch(BAR_WIDTH_CM), cm_to_inch(BAR_HEIGHT_CM)),
        dpi=DPI,
    )

    ax = fig.add_axes(BAR_AX)
    plot_region_bars(ax, counts)

    save_figure(fig, FIG_STEM)
    plt.close(fig)


def main() -> None:
    print("========== Draw Fig. 1(d): WSM2025 regional record counts ==========")
    print(f"OUT_DIR: {OUT_DIR}")
    print(f"USE_C1_CACHE_IF_AVAILABLE: {USE_C1_CACHE_IF_AVAILABLE}")
    print(f"USE_JAPAN_LEGACY_COUNT_BOUNDS: {USE_JAPAN_LEGACY_COUNT_BOUNDS}")

    wsm = load_wsm_global_filtered()
    counts = count_regions(wsm)
    draw_d_bar_chart(counts)

    print("[DONE] Fig. 1(d) outputs saved.")


if __name__ == "__main__":
    main()
