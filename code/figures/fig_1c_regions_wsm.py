# -*- coding: utf-8 -*-
"""
Fig1c_Region_Map_with_Global_WSM_v6.py

Purpose
-------
Draw Fig. 1(c): global overview of WSM2025 stress indicator records
and selected study / representative regions.

Figure logic
------------
- Fig. 1(c): global overview
    1) Global WSM records, 1964–2024.
    2) Japan as the main experimental region.
    3) Other selected regions as representative regions.
    4) Map centered at 120°E so that Japan / western Pacific is near the center.

- Fig. 1(b): Japan zoom-in map
    It can remain as the regional enlarged view of WSM records in Japan.

Main modifications in this version
----------------------------------
1) Chile label is placed to the left side of the Chile region box.
2) New Zealand label is placed directly above the New Zealand region box.
3) Region-label font size is increased for readability after figure reduction.
4) Figure-related visual parameters are explicitly annotated in the settings section.
5) The PlateCarree aspect ratio is preserved.
   Do not use ax.set_aspect("auto").

Outputs
-------
  - Fig1c_Region_Map.png
  - Fig1c_Region_Map.tif
  - Fig1c_Region_Map.pdf
  - cache/fig1c_wsm_global_filtered_records.csv
"""

from __future__ import annotations

import re
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Patch

import cartopy.crs as ccrs
import cartopy.feature as cfeature


# =============================================================================
# 0. User settings
# =============================================================================

# -----------------------------------------------------------------------------
# Input / output paths
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RF_ASSOCIATION_DIR = PROJECT_ROOT / "code" / "rf_association"
if str(RF_ASSOCIATION_DIR) not in sys.path:
    sys.path.insert(0, str(RF_ASSOCIATION_DIR))

from config import FILTERED_WSM_PATH, REGIONS as RF_REGIONS

WSM_CSV = FILTERED_WSM_PATH
OUT_DIR = PROJECT_ROOT / "results" / "figures"
FIG_STEM = "Fig1c_Region_Map"

# -----------------------------------------------------------------------------
# Output switches
# -----------------------------------------------------------------------------
SAVE_PNG = True
SAVE_TIFF = True
SAVE_PDF = True
DPI = 600
TIFF_DPI = 600

# -----------------------------------------------------------------------------
# WSM time range for global overview
# -----------------------------------------------------------------------------
# This global overview uses all valid WSM records from 1964 to 2024.
# The color is borrowed from the "before 2021-05" style in Fig. 1(b),
# but the selected data are NOT restricted to before 2021-05.
WSM_START_DATE = pd.Timestamp("1964-01-01", tz="UTC")
WSM_END_DATE = pd.Timestamp("2024-12-31 23:59:59", tz="UTC")

# -----------------------------------------------------------------------------
# WSM filtering rules
# -----------------------------------------------------------------------------
# These rules keep the global WSM point selection consistent with the cleaning
# logic used for Fig. 1(b): ISC_ID retained, A-C quality retained, core fields complete.
USE_ISC_ID_ONLY = True
USE_ABC_QUALITY_ONLY = True
REQUIRE_COMPLETE_CORE_FIELDS = True

# -----------------------------------------------------------------------------
# Cache
# -----------------------------------------------------------------------------
USE_CACHE = False
CACHE_DIR = OUT_DIR / "cache"
CACHE_FILE = CACHE_DIR / "fig1c_wsm_global_filtered_records.csv"
FORCE_REBUILD_CACHE = False

# =============================================================================
# 0.1 Figure visual parameters
# =============================================================================
# All parameters below directly affect figure appearance.
# Modify these when adjusting layout, readability, label visibility, or visual density.

# -----------------------------------------------------------------------------
# Figure size
# -----------------------------------------------------------------------------
# Full global PlateCarree map has theoretical width / height ratio = 360 / 180 = 2.0.
# The figure is set to 17.0 cm × 9.7 cm, leaving enough vertical space for labels,
# legend, and polar latitude ticks.
MAP_WIDTH_CM = 17.0
MAP_HEIGHT_CM = 9.7

# Axes position in figure coordinates: [left, bottom, width, height].
# Increase bottom if x-axis tick labels overlap the figure boundary.
# Increase height if the map looks vertically compressed.
MAP_AX = [0.055, 0.125, 0.900, 0.790]

# -----------------------------------------------------------------------------
# Font sizes
# -----------------------------------------------------------------------------
FONT_FAMILY = "Times New Roman"
FONT_SIZE = 8
AXIS_LABEL_SIZE = 8
TICK_LABEL_SIZE = 7

# Region-label font size.
# Increased from the earlier 5.4 to 6.8 so labels remain readable after reduction.
REGION_LABEL_SIZE = 6.8

LEGEND_FONT_SIZE = 7
PANEL_LABEL_SIZE = 14

# -----------------------------------------------------------------------------
# Map projection and central longitude
# -----------------------------------------------------------------------------
# DATA_CRS is the true geographic coordinate system for WSM points, boxes, labels,
# and gridlines.
# PROJECTION is the displayed map projection.
# CENTER_LONGITUDE = 120 puts 120°E near the visual center of the map,
# so Japan / western Pacific is centered.
CENTER_LONGITUDE = 120.0
DATA_CRS = ccrs.PlateCarree()
PROJECTION = ccrs.PlateCarree(central_longitude=CENTER_LONGITUDE)

# -----------------------------------------------------------------------------
# Base-map style
# -----------------------------------------------------------------------------
COASTLINE_RESOLUTION = "110m"  # use "50m" if more detailed coastlines are needed

LAND_COLOR = "white"
OCEAN_COLOR = "#DDEEF8"
COASTLINE_COLOR = "#B0B0B0"
COASTLINE_WIDTH = 0.45

GRIDLINE_COLOR = "#C8C8C8"
GRIDLINE_WIDTH = 0.35
GRIDLINE_ALPHA = 0.75
GRIDLINE_STYLE = "--"

SPINE_COLOR = "#4D4D4D"
SPINE_WIDTH = 0.70

# -----------------------------------------------------------------------------
# Tick settings
# -----------------------------------------------------------------------------
# Projection x coordinates are relative to central_longitude.
# With central_longitude=120:
#   x = 0    corresponds to 120°E
#   x = -120 corresponds to 0°
#   x = 60   corresponds to 180°
XTICK_PROJ_POS = [-180, -120, -60, 0, 60, 120, 180]
YTICKS = [-90, -45, 0, 45, 90]

# Geographic gridline longitudes.
GRIDLINE_XLOCS = [-180, -120, -60, 0, 60, 120, 180]
GRIDLINE_YLOCS = YTICKS

# -----------------------------------------------------------------------------
# WSM marker style
# -----------------------------------------------------------------------------
# Same as the "before 2021-05" WSM point style in Fig. 1(b).
# If global points look too dense, reduce WSM_MARKER_SIZE or WSM_ALPHA.
WSM_MARKER_SIZE = 4.0
WSM_COLOR = "#F5B46F"
WSM_ALPHA = 0.45
WSM_RASTERIZED = True  # rasterize dense scatter points in vector outputs

# -----------------------------------------------------------------------------
# Region box style
# -----------------------------------------------------------------------------
MAIN_COLOR = "#B22222"
AUX_COLOR = "#3E517A"

MAIN_LINEWIDTH = 1.05
AUX_LINEWIDTH = 0.90
MAIN_LINESTYLE = "-"
AUX_LINESTYLE = "--"

# -----------------------------------------------------------------------------
# Region label style
# -----------------------------------------------------------------------------
SHOW_REGION_LABELS = True

# Label background box.
# Kept to improve readability above WSM scatter points.
REGION_LABEL_BBOX = True
REGION_LABEL_BOX_FACE = "white"
REGION_LABEL_BOX_ALPHA = 0.72
REGION_LABEL_BOX_EDGE = "none"
REGION_LABEL_PAD = 0.65  # slightly increased because label font size is larger

# Label-position offsets.
# These two parameters are only used for label placement readability:
# - Chile label is moved to the LEFT of its box.
# - New Zealand label is moved directly ABOVE its box.
CHILE_LABEL_LEFT_OFFSET_DEG = 5.5
NEW_ZEALAND_LABEL_TOP_OFFSET_DEG = 3.0

# -----------------------------------------------------------------------------
# Legend style
# -----------------------------------------------------------------------------
ADD_MAP_LEGEND = True
LEGEND_LOC = "lower left"
LEGEND_FRAMEON = True
LEGEND_FACE_COLOR = "white"
LEGEND_FACE_ALPHA = 0.90
LEGEND_EDGE_COLOR = "#B0B0B0"
LEGEND_EDGE_WIDTH = 0.25
LEGEND_BORDER_PAD = 0.55
LEGEND_LABEL_SPACING = 0.35
LEGEND_HANDLE_TEXT_PAD = 0.50
LEGEND_HANDLE_LENGTH = 1.25
LEGEND_BORDER_AXES_PAD = 0.35

LEGEND_LABEL_WSM = "WSM records (1964–2024)"
LEGEND_LABEL_MAIN = "Forecasting study region"
LEGEND_LABEL_AUX = "Other study regions"

# -----------------------------------------------------------------------------
# Panel label
# -----------------------------------------------------------------------------
PANEL_LABEL = "(c)"


# =============================================================================
# 1. Region definitions
# =============================================================================

def box_center(lat_range: tuple[float, float], lon_range: tuple[float, float]) -> tuple[float, float]:
    """Return the geographic center of a lat/lon box as (lon_center, lat_center)."""
    lat_min, lat_max = lat_range
    lon_min, lon_max = lon_range
    return (0.5 * (lon_min + lon_max), 0.5 * (lat_min + lat_max))


def shared_region_bounds(key: str) -> tuple[tuple[float, float], tuple[float, float]]:
    region = RF_REGIONS[key]
    return (region.lat_min, region.lat_max), (region.lon_min, region.lon_max)


JAPAN_LAT, JAPAN_LON = shared_region_bounds("japan")
GREECE_LAT, GREECE_LON = shared_region_bounds("greece")
INDONESIA_LAT, INDONESIA_LON = shared_region_bounds("indonesia")
MEXICO_LAT, MEXICO_LON = shared_region_bounds("mexico")
CHILE_LAT, CHILE_LON = shared_region_bounds("central_chile")
ITALY_LAT, ITALY_LON = shared_region_bounds("italy")
NEW_ZEALAND_LAT, NEW_ZEALAND_LON = shared_region_bounds("new_zealand")
TAIWAN_LAT, TAIWAN_LON = shared_region_bounds("taiwan")
IRAN_LAT, IRAN_LON = shared_region_bounds("iran")
SOUTHERN_CALIFORNIA_LAT, SOUTHERN_CALIFORNIA_LON = shared_region_bounds(
    "southern_california"
)


REGIONS = OrderedDict({
    "Japan": {
        "lat": JAPAN_LAT,
        "lon": JAPAN_LON,
        "label": (151.0, 43.0),
        "role": "main",
    },
})

REGIONS.update(OrderedDict({
    "Greece": {
        "lat": GREECE_LAT,
        "lon": GREECE_LON,
        "label": (30.0, 46.0),
        "role": "aux",
    },
    "Indonesia": {
        "lat": INDONESIA_LAT,
        "lon": INDONESIA_LON,
        "label": (110.0, 10.5),
        "role": "aux",
    },
    "Mexico": {
        "lat": MEXICO_LAT,
        "lon": MEXICO_LON,
        "label": (-88.0, 33.5),
        "role": "aux",
    },
    "Chile": {
        "lat": CHILE_LAT,
        "lon": CHILE_LON,
        # Chile label: placed to the LEFT of the Chile box.
        # lon = left boundary - CHILE_LABEL_LEFT_OFFSET_DEG,
        # lat = vertical center of Chile box.
        "label": (
            CHILE_LON[0] - CHILE_LABEL_LEFT_OFFSET_DEG,
            0.5 * (CHILE_LAT[0] + CHILE_LAT[1]),
        ),
        "role": "aux",
    },
    "Italy": {
        "lat": ITALY_LAT,
        "lon": ITALY_LON,
        "label": (7.0, 51.0),
        "role": "aux",
    },
    "New Zealand": {
        "lat": NEW_ZEALAND_LAT,
        "lon": NEW_ZEALAND_LON,
        # New Zealand label: placed directly ABOVE the New Zealand box.
        # lon = horizontal center of box,
        # lat = top boundary + NEW_ZEALAND_LABEL_TOP_OFFSET_DEG.
        "label": (
            0.5 * (NEW_ZEALAND_LON[0] + NEW_ZEALAND_LON[1]),
            NEW_ZEALAND_LAT[1] + NEW_ZEALAND_LABEL_TOP_OFFSET_DEG,
        ),
        "role": "aux",
    },
    "Taiwan": {
        "lat": TAIWAN_LAT,
        "lon": TAIWAN_LON,
        "label": (129.0, 25.5),
        "role": "aux",
    },
    "Iran": {
        "lat": IRAN_LAT,
        "lon": IRAN_LON,
        "label": (57.0, 44.0),
        "role": "aux",
    },
    "Southern California": {
        "lat": SOUTHERN_CALIFORNIA_LAT,
        "lon": SOUTHERN_CALIFORNIA_LON,
        "label": (-110.0, 42.0),
        "role": "aux",
    },
}))


# =============================================================================
# 2. Basic style functions
# =============================================================================

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


def cm_to_inch(value_cm: float) -> float:
    return value_cm / 2.54


def normalize_lon(lon: float) -> float:
    """
    Normalize longitude to [-180, 180], but keep 180 instead of -180
    for cleaner labels.
    """
    lon_norm = ((float(lon) + 180.0) % 360.0) - 180.0
    if abs(lon_norm + 180.0) < 1e-8:
        return 180.0
    return lon_norm


def projection_x_to_lon(x: float) -> float:
    """
    Convert projected PlateCarree x coordinate back to geographic longitude.

    For PlateCarree(central_longitude=120):
        x = 0    -> 120°E
        x = -120 -> 0°
        x = 60   -> 180°
    """
    return normalize_lon(CENTER_LONGITUDE + float(x))


def format_lon(lon: float) -> str:
    lon = normalize_lon(float(lon))

    if abs(lon) < 1e-8:
        return "0°"
    if abs(abs(lon) - 180.0) < 1e-8:
        return "180°"

    hemi = "E" if lon > 0 else "W"
    return f"{abs(int(round(lon)))}°{hemi}"


def format_lat(lat: float) -> str:
    lat = float(lat)

    if abs(lat) < 1e-8:
        return "0°"
    if abs(abs(lat) - 90.0) < 1e-8:
        hemi = "N" if lat > 0 else "S"
        return f"90°{hemi}"

    hemi = "N" if lat > 0 else "S"
    return f"{abs(int(round(lat)))}°{hemi}"


def set_spines(ax) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(SPINE_WIDTH)
        spine.set_edgecolor(SPINE_COLOR)


def set_geo_ticks(ax) -> None:
    """
    Set ticks manually to avoid wrong longitude labels under
    PlateCarree(central_longitude=120).

    The x-axis tick positions are projection coordinates.
    The labels are converted back to geographic longitudes.
    """
    ax.set_xticks(XTICK_PROJ_POS)
    ax.set_xticklabels(
        [format_lon(projection_x_to_lon(x)) for x in XTICK_PROJ_POS],
        fontsize=TICK_LABEL_SIZE,
    )

    ax.set_yticks(YTICKS)
    ax.set_yticklabels(
        [format_lat(y) for y in YTICKS],
        fontsize=TICK_LABEL_SIZE,
    )

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=TICK_LABEL_SIZE,
        length=0,
        pad=3,
    )


# =============================================================================
# 3. WSM loading and filtering
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
    Load and filter global WSM records.

    The cleaning logic follows the Fig. 1(b) WSM filtering convention:
    - keep records with ISC_ID
    - keep A-C quality
    - keep complete core fields
    - keep valid lat/lon/azi/depth/magnitude/date
    - keep dates from 1964 to 2024
    """
    if USE_CACHE and CACHE_FILE.exists() and not FORCE_REBUILD_CACHE:
        print(f"[CACHE] Loading global WSM records: {CACHE_FILE}")
        df_cache = pd.read_csv(CACHE_FILE)
        df_cache["DATE_DT"] = pd.to_datetime(
            df_cache["DATE_DT"],
            errors="coerce",
            utc=True,
        )
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

    keep_cols = [
        "ISC_ID", "DATE_DT", "LAT", "LON", "AZI", "DEPTH",
        "EQ_MAG", "QUALITY", "REGIME",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].copy()

    print("\n========== Fig. 1(c) global WSM filter summary ==========")
    print(f"records_final: {len(df)}")
    print(f"date_min: {df['DATE_DT'].min()}")
    print(f"date_max: {df['DATE_DT'].max()}")
    print(f"lat_range: {df['LAT'].min():.3f} to {df['LAT'].max():.3f}")
    print(f"lon_range: {df['LON'].min():.3f} to {df['LON'].max():.3f}")

    if USE_CACHE:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(CACHE_FILE, index=False, encoding="utf-8-sig")
        print(f"[CACHE] Saved global WSM records: {CACHE_FILE}")

    return df


# =============================================================================
# 4. Map drawing functions
# =============================================================================

def add_base_map(ax) -> None:
    # Full global view with 120°E at the center.
    ax.set_global()

    # Critical:
    # Do not use ax.set_aspect("auto"), otherwise the map will be stretched.
    ax.set_aspect("equal", adjustable="box")

    ax.set_facecolor(OCEAN_COLOR)

    ocean = cfeature.NaturalEarthFeature(
        "physical",
        "ocean",
        COASTLINE_RESOLUTION,
        edgecolor="none",
        facecolor=OCEAN_COLOR,
    )
    land = cfeature.NaturalEarthFeature(
        "physical",
        "land",
        COASTLINE_RESOLUTION,
        edgecolor="none",
        facecolor=LAND_COLOR,
    )

    ax.add_feature(ocean, zorder=0)
    ax.add_feature(land, zorder=1)

    ax.coastlines(
        resolution=COASTLINE_RESOLUTION,
        color=COASTLINE_COLOR,
        linewidth=COASTLINE_WIDTH,
        zorder=2,
    )

    ax.gridlines(
        crs=DATA_CRS,
        draw_labels=False,
        linewidth=GRIDLINE_WIDTH,
        color=GRIDLINE_COLOR,
        alpha=GRIDLINE_ALPHA,
        linestyle=GRIDLINE_STYLE,
        xlocs=GRIDLINE_XLOCS,
        ylocs=GRIDLINE_YLOCS,
        zorder=3,
    )

    set_geo_ticks(ax)
    set_spines(ax)


def plot_wsm_records(ax, df: pd.DataFrame) -> None:
    ax.scatter(
        df["LON"],
        df["LAT"],
        s=WSM_MARKER_SIZE,
        c=WSM_COLOR,
        alpha=WSM_ALPHA,
        edgecolors="none",
        linewidths=0.0,
        transform=DATA_CRS,
        zorder=4,
        rasterized=WSM_RASTERIZED,
    )


def add_region_boxes(ax) -> None:
    for name, cfg in REGIONS.items():
        lat_min, lat_max = cfg["lat"]
        lon_min, lon_max = cfg["lon"]
        role = cfg["role"]

        if role == "main":
            color = MAIN_COLOR
            lw = MAIN_LINEWIDTH
            ls = MAIN_LINESTYLE
            zorder = 7
        else:
            color = AUX_COLOR
            lw = AUX_LINEWIDTH
            ls = AUX_LINESTYLE
            zorder = 6

        rect = Rectangle(
            (lon_min, lat_min),
            lon_max - lon_min,
            lat_max - lat_min,
            linewidth=lw,
            edgecolor=color,
            facecolor="none",
            linestyle=ls,
            transform=DATA_CRS,
            zorder=zorder,
        )
        ax.add_patch(rect)

        if SHOW_REGION_LABELS:
            label_lon, label_lat = cfg["label"]

            bbox = None
            if REGION_LABEL_BBOX:
                bbox = dict(
                    facecolor=REGION_LABEL_BOX_FACE,
                    edgecolor=REGION_LABEL_BOX_EDGE,
                    alpha=REGION_LABEL_BOX_ALPHA,
                    pad=REGION_LABEL_PAD,
                )

            ax.text(
                label_lon,
                label_lat,
                name,
                transform=DATA_CRS,
                ha="center",
                va="center",
                fontsize=REGION_LABEL_SIZE,
                color=color,
                bbox=bbox,
                zorder=8,
            )


def add_map_legend(ax) -> None:
    if not ADD_MAP_LEGEND:
        return

    handles = [
        Line2D(
            [0], [0],
            marker="o",
            linestyle="None",
            markersize=np.sqrt(WSM_MARKER_SIZE),
            markerfacecolor=WSM_COLOR,
            markeredgecolor="none",
            alpha=WSM_ALPHA,
            label=LEGEND_LABEL_WSM,
        ),
        Patch(
            facecolor="none",
            edgecolor=MAIN_COLOR,
            linewidth=MAIN_LINEWIDTH,
            linestyle=MAIN_LINESTYLE,
            label=LEGEND_LABEL_MAIN,
        ),
        Patch(
            facecolor="none",
            edgecolor=AUX_COLOR,
            linewidth=AUX_LINEWIDTH,
            linestyle=AUX_LINESTYLE,
            label=LEGEND_LABEL_AUX,
        ),
    ]

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


def add_panel_label(ax, label: str) -> None:
    ax.text(
        0.02,
        0.98,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=PANEL_LABEL_SIZE,
        fontweight="bold",
        zorder=20,
    )


def plot_region_map(ax, df_wsm: pd.DataFrame) -> None:
    add_base_map(ax)
    plot_wsm_records(ax, df_wsm)
    add_region_boxes(ax)
    add_map_legend(ax)
    add_panel_label(ax, PANEL_LABEL)


# =============================================================================
# 5. Save and driver
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


def draw_c_region_map(df_wsm: pd.DataFrame) -> None:
    apply_matplotlib_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(
        figsize=(cm_to_inch(MAP_WIDTH_CM), cm_to_inch(MAP_HEIGHT_CM)),
        dpi=DPI,
    )

    ax = fig.add_axes(MAP_AX, projection=PROJECTION)
    plot_region_map(ax, df_wsm)

    save_figure(fig, FIG_STEM)
    plt.close(fig)


def main() -> None:
    print("========== Draw Fig. 1(c): global WSM and region overview ==========")
    print(f"OUT_DIR: {OUT_DIR}")
    print(f"Projection center longitude: {CENTER_LONGITUDE}°E")
    print(f"WSM time range: {WSM_START_DATE} to {WSM_END_DATE}")

    df_wsm = load_wsm_global_filtered()
    draw_c_region_map(df_wsm)

    print("[DONE] Fig. 1(c) outputs saved.")


if __name__ == "__main__":
    main()
