# -*- coding: utf-8 -*-
"""
Fig1b_WSM_Japan_unified_v2.py

Figure 1(b): Spatial distribution of WSM2025 stress indicator records
in the Japan study region.

Style rules:
- same extent/style as Fig. 1(a)
- one outer frame only
- no panel title
- panel label at upper-left
- legend at lower-left without frame
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter


# =============================================================================
# 0. User settings
# =============================================================================

# ---- Input/output paths ----
WSM_CSV = Path(r"D:\World_force\2_Project_stress\xzy_202607_final\data\WSM_Database_2025.csv")
OUT_DIR = Path(r"D:\World_force\2_Project_stress\xzy_202607_final\Figures")
FIG_STEM = "Fig1b_WSM_Japan"

# ---- Study region ----
LAT_MIN = 30.05
LAT_MAX = 45.95
LON_MIN = 128.10
LON_MAX = 146.80

# Use the same anchored-crop effective bounds as Fig. 1(a), if needed.
USE_GRID_EFFECTIVE_BOUNDS_FOR_FILTER = True
GLOBAL_H = 3840
GLOBAL_W = 3840

# ---- Temporal split ----
CUTOFF_DATE = pd.Timestamp("2021-05-31", tz="UTC")

# ---- Figure size and output ----
FIG_WIDTH_CM = 8.5
FIG_HEIGHT_CM = 8.2
DPI = 600
TIFF_DPI = 600
SAVE_PNG = True
SAVE_TIFF = True
SAVE_PDF = True

# ---- Font ----
FONT_FAMILY = "Times New Roman"
TICK_SIZE = 7
LEGEND_SIZE = 7
PANEL_LABEL_SIZE = 14  # fixed for all Fig. 1 subpanels

# ---- Map style ----
PROJECTION = ccrs.PlateCarree()
COASTLINE_RESOLUTION = "10m"
LAND_COLOR = "white"
OCEAN_COLOR = "#DDEEF8"
COASTLINE_COLOR = "#B0B0B0"
COASTLINE_WIDTH = 0.45
SPINE_COLOR = "#4D4D4D"
SPINE_WIDTH = 0.70
GRIDLINE_COLOR = "#C8C8C8"
GRIDLINE_WIDTH = 0.35
GRIDLINE_ALPHA = 0.75
GRIDLINE_STYLE = "--"
MAP_PADDING_DEG = 0.35

# ---- Marker style ----
# Use the same size as the Mw 4.0-5.0 points in Fig. 1(a)
# Both temporal groups use the same WSM scatter color in Fig. 1(b).
WSM_MARKER_SIZE = 4.0
WSM_BEFORE_COLOR = "#F5B46F"
WSM_AFTER_COLOR = "#F5B46F"
WSM_ALPHA_BEFORE = 0.45
WSM_ALPHA_AFTER = 0.45

# ---- Tick settings ----
XTICKS = [130, 132.5, 135, 137.5, 140, 142.5, 145]
YTICKS = [30, 32.5, 35, 37.5, 40, 42.5, 45]

# ---- Annotation / legend ----
PANEL_LABEL = "(b)"
ADD_PANEL_LABEL = True
ADD_LEGEND = False    #左下角图例
LEGEND_LOC = "lower left"
LEGEND_FRAMEON = True
LEGEND_FACE_COLOR = "white"
LEGEND_FACE_ALPHA = 0.90
LEGEND_EDGE_COLOR = "#B0B0B0"
LEGEND_EDGE_WIDTH = 0.25
LEGEND_BORDER_PAD = 0.55
LEGEND_LABEL_SPACING = 0.38
LEGEND_HANDLE_TEXT_PAD = 0.55
LEGEND_HANDLE_LENGTH = 1.35
LEGEND_BORDER_AXES_PAD = 0.45
LEGEND_LABEL_BEFORE = r"WSM records before 2021-05"
LEGEND_LABEL_AFTER = r"WSM records after 2021-05"

# ---- Cache ----
USE_CACHE = True
CACHE_DIR = OUT_DIR / "cache"
CACHE_FILE = CACHE_DIR / "fig1b_wsm_japan_filtered_records.csv"
FORCE_REBUILD_CACHE = False


# =============================================================================
# 1. Global matplotlib style
# =============================================================================

def apply_matplotlib_style() -> None:
    mpl.rcParams.update({
        "font.family": FONT_FAMILY,
        "font.size": TICK_SIZE,
        "xtick.labelsize": TICK_SIZE,
        "ytick.labelsize": TICK_SIZE,
        "legend.fontsize": LEGEND_SIZE,
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.unicode_minus": False,
    })


def cm_to_inch(value_cm: float) -> float:
    return value_cm / 2.54


# =============================================================================
# 2. Region and WSM filtering utilities
# =============================================================================

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
    return x_min, x_max, y_min, y_max


def effective_bounds_from_pixel_box(pixel_box: Tuple[int, int, int, int],
                                    h_global: int, w_global: int) -> Tuple[float, float, float, float]:
    x_min, x_max, y_min, y_max = pixel_box
    lon_min_eff = x_min / float(w_global) * 360.0 - 180.0
    lon_max_eff = (x_max + 1) / float(w_global) * 360.0 - 180.0
    lat_min_eff = y_min / float(h_global) * 180.0 - 90.0
    lat_max_eff = (y_max + 1) / float(h_global) * 180.0 - 90.0
    return lat_min_eff, lat_max_eff, lon_min_eff, lon_max_eff


def get_filter_bounds() -> Tuple[float, float, float, float]:
    if not USE_GRID_EFFECTIVE_BOUNDS_FOR_FILTER:
        return LAT_MIN, LAT_MAX, LON_MIN, LON_MAX
    box = region_pixel_box(LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, GLOBAL_H, GLOBAL_W)
    return effective_bounds_from_pixel_box(box, GLOBAL_H, GLOBAL_W)


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
        "LONGITUDE": "LON", "LONG": "LON", "LATITUDE": "LAT", "AZIMUTH": "AZI",
        "MAGTYPE": "MAG_TYPE", "MAG": "EQ_MAG", "QUALITYFLAG": "QUALITY",
        "METHOD": "TYPE", "DATETIMEUTC": "DATE", "DATA": "DATE",
        "ISCID": "ISC_ID", "ISC": "ISC_ID",
    }
    for old, new in mapping.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})
    return df


def parse_date_iso_strict(s: pd.Series) -> pd.Series:
    s0 = s.astype(str).str.strip()
    out = pd.Series(pd.NaT, index=s0.index, dtype="datetime64[ns, UTC]")

    m8 = s0.str.match(r"^\d{8}$", na=False)
    out.loc[m8] = pd.to_datetime(s0[m8], format="%Y%m%d", errors="coerce", utc=True)

    s1 = s0.str.replace(r"[-\.]", "/", regex=True)
    ymd = s1.str.match(r"^\d{4}/\d{1,2}/\d{1,2}$", na=False)
    out.loc[ymd & out.isna()] = pd.to_datetime(
        s1[ymd], format="%Y/%m/%d", errors="coerce", utc=True,
    )

    dmy = s1.str.match(r"^\d{1,2}/\d{1,2}/\d{2,4}$", na=False)
    out.loc[dmy & out.isna()] = pd.to_datetime(
        s1[dmy], dayfirst=True, errors="coerce", utc=True,
    )

    bad = s0.str.match(r"^\d{4}$", na=False) | s1.str.match(
        r"^\d{1,2}/\d{4}$|^\d{4}/\d{1,2}$", na=False,
    )
    out = out.where(~bad, pd.NaT)
    return out


def clean_quality(q: pd.Series) -> pd.Series:
    return q.astype(str).str.upper().str.strip().str.extract(r"\s*([A-E])", expand=False)


def has_isc_id(s: pd.Series) -> pd.Series:
    s_str = s.astype(str).str.strip()
    invalid_tokens = r"(?i)nan|null|none|na|^-$|^0$"
    return s.notna() & s_str.ne("") & ~s_str.str.fullmatch(invalid_tokens)


def load_wsm_filtered() -> pd.DataFrame:
    if USE_CACHE and CACHE_FILE.exists() and not FORCE_REBUILD_CACHE:
        print(f"[CACHE] Loading filtered WSM records: {CACHE_FILE}")
        df_cache = pd.read_csv(CACHE_FILE)
        df_cache["DATE_DT"] = pd.to_datetime(df_cache["DATE_DT"], errors="coerce", utc=True)
        return df_cache

    print(f"[LOAD] {WSM_CSV}")
    df_raw = read_csv_robust(WSM_CSV)
    df = standardize_columns(df_raw)

    if "ISC_ID" not in df.columns:
        raise ValueError("Column 'ISC_ID' was not found.")
    df = df.loc[has_isc_id(df["ISC_ID"])].copy()

    if "DATE" not in df.columns:
        raise ValueError("Column 'DATE' was not found.")
    df["DATE_DT"] = parse_date_iso_strict(df["DATE"])
    df["QUALITY"] = clean_quality(df.get("QUALITY", pd.Series(index=df.index, dtype=object)))

    required_fields = [
        "ISC_ID", "LAT", "LON", "AZI", "DEPTH", "QUALITY",
        "REGIME", "DATE_DT", "TIME", "EQ_MAG",
    ]

    df = df.loc[df["QUALITY"].isin(["A", "B", "C"])].copy()
    for c in required_fields:
        if c not in df.columns:
            df[c] = np.nan

    def _nonempty(series: pd.Series) -> pd.Series:
        if series.dtype == object:
            return series.astype(str).str.strip().ne("") & series.notna()
        return series.notna()

    mask_complete = np.ones(len(df), dtype=bool)
    for c in required_fields:
        mask_complete &= _nonempty(df[c]).to_numpy()
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

    lat_min_eff, lat_max_eff, lon_min_eff, lon_max_eff = get_filter_bounds()
    region_mask = (
        df["LAT"].between(lat_min_eff, lat_max_eff)
        & df["LON"].between(lon_min_eff, lon_max_eff)
    )
    df = df.loc[region_mask].copy()

    df["TIME_GROUP"] = np.where(df["DATE_DT"] <= CUTOFF_DATE, "before", "after")

    print("\n========== Fig. 1(b) WSM filter summary ==========")
    print(f"records_final: {len(df)}")
    print(f"records <= 2021-05: {(df['TIME_GROUP'] == 'before').sum()}")
    print(f"records > 2021-05: {(df['TIME_GROUP'] == 'after').sum()}")

    if USE_CACHE:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(CACHE_FILE, index=False, encoding="utf-8-sig")
        print(f"[CACHE] Saved filtered WSM records: {CACHE_FILE}")

    return df


# =============================================================================
# 3. Plotting
# =============================================================================

def add_base_map(ax) -> None:
    ax.set_facecolor(OCEAN_COLOR)
    ocean = cfeature.NaturalEarthFeature(
        "physical", "ocean", COASTLINE_RESOLUTION,
        edgecolor="none", facecolor=OCEAN_COLOR,
    )
    land = cfeature.NaturalEarthFeature(
        "physical", "land", COASTLINE_RESOLUTION,
        edgecolor="none", facecolor=LAND_COLOR,
    )
    ax.add_feature(ocean, zorder=0)
    ax.add_feature(land, zorder=1)
    ax.coastlines(
        resolution=COASTLINE_RESOLUTION,
        color=COASTLINE_COLOR,
        linewidth=COASTLINE_WIDTH,
        zorder=2,
    )


def style_geo_axes(ax) -> None:
    ax.set_extent(
        [LON_MIN - MAP_PADDING_DEG, LON_MAX + MAP_PADDING_DEG,
         LAT_MIN - MAP_PADDING_DEG, LAT_MAX + MAP_PADDING_DEG],
        crs=ccrs.PlateCarree(),
    )
    add_base_map(ax)

    ax.set_xticks(XTICKS, crs=ccrs.PlateCarree())
    ax.set_yticks(YTICKS, crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter(number_format="g", degree_symbol="°"))
    ax.yaxis.set_major_formatter(LatitudeFormatter(number_format="g", degree_symbol="°"))
    ax.tick_params(axis="both", which="major", labelsize=TICK_SIZE, length=0)

    ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=False,
        linewidth=GRIDLINE_WIDTH,
        color=GRIDLINE_COLOR,
        alpha=GRIDLINE_ALPHA,
        linestyle=GRIDLINE_STYLE,
        xlocs=XTICKS,
        ylocs=YTICKS,
    )

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(SPINE_WIDTH)
        spine.set_edgecolor(SPINE_COLOR)


def plot_records(ax, df: pd.DataFrame) -> None:
    df_after = df.loc[df["TIME_GROUP"] == "after"].copy()
    df_before = df.loc[df["TIME_GROUP"] == "before"].copy()

    # Plot later records first, earlier records on top.
    ax.scatter(
        df_after["LON"], df_after["LAT"],
        s=WSM_MARKER_SIZE,
        c=WSM_AFTER_COLOR,
        alpha=WSM_ALPHA_AFTER,
        edgecolors="none",
        linewidths=0.0,
        transform=ccrs.PlateCarree(),
        zorder=3,
    )
    ax.scatter(
        df_before["LON"], df_before["LAT"],
        s=WSM_MARKER_SIZE,
        c=WSM_BEFORE_COLOR,
        alpha=WSM_ALPHA_BEFORE,
        edgecolors="none",
        linewidths=0.0,
        transform=ccrs.PlateCarree(),
        zorder=4,
    )


def add_wsm_legend(ax) -> None:
    handles = [
        Line2D(
            [0], [0], marker="o", linestyle="None",
            markersize=np.sqrt(WSM_MARKER_SIZE),
            markerfacecolor=WSM_BEFORE_COLOR,
            markeredgecolor="none",
            alpha=WSM_ALPHA_BEFORE,
            label=LEGEND_LABEL_BEFORE,
        ),
        Line2D(
            [0], [0], marker="o", linestyle="None",
            markersize=np.sqrt(WSM_MARKER_SIZE),
            markerfacecolor=WSM_AFTER_COLOR,
            markeredgecolor="none",
            alpha=WSM_ALPHA_AFTER,
            label=LEGEND_LABEL_AFTER,
        ),
    ]
    leg = ax.legend(
        handles=handles,
        loc=LEGEND_LOC,
        frameon=LEGEND_FRAMEON,
        fontsize=LEGEND_SIZE,
        facecolor=LEGEND_FACE_COLOR,
        edgecolor=LEGEND_EDGE_COLOR,
        framealpha=LEGEND_FACE_ALPHA,
        handletextpad=LEGEND_HANDLE_TEXT_PAD,
        labelspacing=LEGEND_LABEL_SPACING,
        borderpad=LEGEND_BORDER_PAD,        handlelength=LEGEND_HANDLE_LENGTH,
        borderaxespad=LEGEND_BORDER_AXES_PAD,

        fancybox=False,
    )
    leg.get_frame().set_linewidth(LEGEND_EDGE_WIDTH)

def save_figure(fig, out_dir: Path, basename: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if SAVE_PNG:
        fp = out_dir / f"{basename}.png"
        fig.savefig(fp, dpi=DPI, bbox_inches="tight")
        print(f"[SAVE] {fp}")
    if SAVE_TIFF:
        fp = out_dir / f"{basename}.tif"
        try:
            fig.savefig(fp, dpi=TIFF_DPI, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
        except TypeError:
            fig.savefig(fp, dpi=TIFF_DPI, bbox_inches="tight")
        print(f"[SAVE] {fp}")
    if SAVE_PDF:
        fp = out_dir / f"{basename}.pdf"
        fig.savefig(fp, bbox_inches="tight")
        print(f"[SAVE] {fp}")


def draw_fig1b(df: pd.DataFrame) -> None:
    apply_matplotlib_style()

    fig = plt.figure(figsize=(cm_to_inch(FIG_WIDTH_CM), cm_to_inch(FIG_HEIGHT_CM)))
    ax = fig.add_subplot(1, 1, 1, projection=PROJECTION)

    style_geo_axes(ax)
    plot_records(ax, df)

    if ADD_PANEL_LABEL:
        ax.text(
            0.02, 0.98, PANEL_LABEL,
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=PANEL_LABEL_SIZE,
            fontweight="bold",
            zorder=10,
        )

    if ADD_LEGEND:
        add_wsm_legend(ax)

    # No title and no count text.
    plt.tight_layout(pad=0.4)
    save_figure(fig, OUT_DIR, FIG_STEM)
    plt.close(fig)


def main() -> None:
    print("========== Draw Fig. 1(b): WSM Japan spatial distribution ==========")
    df = load_wsm_filtered()
    draw_fig1b(df)
    print("[DONE]")


if __name__ == "__main__":
    main()
