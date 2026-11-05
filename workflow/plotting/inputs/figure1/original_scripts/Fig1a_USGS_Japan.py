# -*- coding: utf-8 -*-
"""
Fig1a_USGS_Japan.py

Figure 1(a): Spatial distribution of USGS Mw >= 4.0 earthquakes
in the Japan study region.

Style rules:
- one outer frame only
- no panel title
- panel label at upper-left
- legend at lower-left without frame
- no n-count text
- mathtext labels with italic M_w
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
from matplotlib.lines import Line2D

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter


# =============================================================================
# 0. User settings
# =============================================================================

# ---- Input/output paths ----
USGS_DIR = Path(r"D:\World_force\1_Project_EP\Phase1\Datas\usgs_csv")
OUT_DIR = Path(r"D:\World_force\2_Project_stress\xzy_202607_final\Figures")
FIG_STEM = "Fig1a_USGS_Japan"

# ---- Study region ----
LAT_MIN = 30.05
LAT_MAX = 45.95
LON_MIN = 128.10
LON_MAX = 146.80

# Use the same anchored-crop effective bounds as the dataset pipeline.
USE_GRID_EFFECTIVE_BOUNDS_FOR_FILTER = True
GLOBAL_H = 3840
GLOBAL_W = 3840

# ---- Time / magnitude ----
TIME_START_YM = "1973_01"
TIME_END_YM = "2021_05"  # month-inclusive, same as dataset generation
USE_STRICT_END_DATE = False
STRICT_END_DATE = "2021-05-01"
START_MAG = 4.0

# ---- USGS filtering ----
FILTER_TYPE_EARTHQUAKE = True
DROP_MAGTYPES = {"mint", "unknown", "ma", "mun", "no", "uk", "fa", "lg", "mdl", "mlv"}

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
EVENT_COLOR = "#B22222"
EVENT_ALPHA = 0.45
EVENT_EDGE_COLOR = "none"
EVENT_LINEWIDTH = 0.0
MAG_SIZE_BINS = [
    (4.0, 5.0, 4.0,  r"$4.0 \leq M_{\mathrm{w}} < 5.0$"),
    (5.0, 6.0, 12.0, r"$5.0 \leq M_{\mathrm{w}} < 6.0$"),
    (6.0, 99.0, 28.0, r"$M_{\mathrm{w}} \geq 6.0$"),
]

# ---- Tick settings ----
XTICKS = [130, 132.5, 135, 137.5, 140, 142.5, 145]
YTICKS = [30, 32.5, 35, 37.5, 40, 42.5, 45]

# ---- Annotation / legend ----
PANEL_LABEL = "(a)"
ADD_PANEL_LABEL = True
ADD_LEGEND = True
ADD_COUNT_TEXT = False  # keep False: remove n-count text
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
# ---- Cache ----
USE_CACHE = True
CACHE_DIR = OUT_DIR / "cache"
CACHE_FILE = CACHE_DIR / "fig1a_usgs_japan_filtered_events.csv"
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
# 2. Time, grid, and USGS filtering utilities
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


def date_bounds() -> Tuple[pd.Timestamp, pd.Timestamp]:
    y0, m0 = parse_time_ym(TIME_START_YM)
    y1, m1 = parse_time_ym(TIME_END_YM)
    start = pd.Timestamp(f"{y0:04d}-{m0:02d}-01", tz="UTC")
    if USE_STRICT_END_DATE:
        end = pd.Timestamp(STRICT_END_DATE, tz="UTC")
    else:
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
    last_err: Optional[Exception] = None
    for enc in ["utf-8-sig", "utf-8", "gbk", "cp936", "latin1"]:
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except Exception as exc:
            last_err = exc
    raise RuntimeError(f"CSV read failed: {path}") from last_err


def energy_from_mag_type(mag: float, mag_type: str) -> float:
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
    if USE_CACHE and CACHE_FILE.exists() and not FORCE_REBUILD_CACHE:
        print(f"[CACHE] Loading filtered events: {CACHE_FILE}")
        df_cache = pd.read_csv(CACHE_FILE)
        df_cache["time"] = pd.to_datetime(df_cache["time"], errors="coerce", utc=True)
        return df_cache

    start, end = date_bounds()
    lat_min_eff, lat_max_eff, lon_min_eff, lon_max_eff = get_filter_bounds()
    months = month_range(TIME_START_YM, TIME_END_YM)

    frames: list[pd.DataFrame] = []
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
        if "depth" in df.columns:
            df["depth"] = pd.to_numeric(df["depth"], errors="coerce")
        else:
            df["depth"] = np.nan
        df = df.dropna(subset=["time", "latitude", "longitude", "mag"])
        if df.empty:
            continue

        energy = df.apply(
            lambda r: energy_from_mag_type(float(r["mag"]), str(r.get("magType", ""))),
            axis=1,
        ).to_numpy(np.float64)
        df["Mw"] = mw_from_energy(energy)
        df = df[df["Mw"] >= START_MAG].copy()
        if df.empty:
            continue

        df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)
        df = df.dropna(subset=["time"])
        df = df[(df["time"] >= start) & (df["time"] <= end)].copy()
        if df.empty:
            continue

        df = df[
            (df["latitude"] >= lat_min_eff) & (df["latitude"] <= lat_max_eff) &
            (df["longitude"] >= lon_min_eff) & (df["longitude"] <= lon_max_eff)
        ].copy()
        if df.empty:
            continue

        frames.append(df[["time", "latitude", "longitude", "depth", "mag", "magType", "Mw"]])

    if not frames:
        raise RuntimeError("No USGS events passed the filters. Check paths and settings.")

    events = pd.concat(frames, ignore_index=True)
    events = events.sort_values("time").reset_index(drop=True)

    print("\n========== Fig. 1(a) USGS filter summary ==========")
    print(f"events_final: {len(events)}")
    print(f"Mw_min/Mw_max: {events['Mw'].min():.3f} / {events['Mw'].max():.3f}")

    if USE_CACHE:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        events.to_csv(CACHE_FILE, index=False, encoding="utf-8-sig")
        print(f"[CACHE] Saved filtered events: {CACHE_FILE}")

    return events


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


def plot_events(ax, events: pd.DataFrame) -> None:
    for lo, hi, size, _ in MAG_SIZE_BINS:
        mask = (events["Mw"] >= lo) & (events["Mw"] < hi)
        sub = events.loc[mask]
        if sub.empty:
            continue
        ax.scatter(
            sub["longitude"], sub["latitude"],
            s=size,
            c=EVENT_COLOR,
            alpha=EVENT_ALPHA,
            edgecolors=EVENT_EDGE_COLOR,
            linewidths=EVENT_LINEWIDTH,
            transform=ccrs.PlateCarree(),
            zorder=4,
        )


def add_size_legend(ax) -> None:
    handles = []
    for _, _, size, label in MAG_SIZE_BINS:
        handles.append(
            Line2D(
                [0], [0],
                marker="o",
                linestyle="None",
                markersize=np.sqrt(size),
                markerfacecolor=EVENT_COLOR,
                markeredgecolor="none",
                alpha=EVENT_ALPHA,
                label=label,
            )
        )
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


def draw_fig1a(events: pd.DataFrame) -> None:
    apply_matplotlib_style()

    fig = plt.figure(figsize=(cm_to_inch(FIG_WIDTH_CM), cm_to_inch(FIG_HEIGHT_CM)))
    ax = fig.add_subplot(1, 1, 1, projection=PROJECTION)

    style_geo_axes(ax)
    plot_events(ax, events)

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
        add_size_legend(ax)

    # No title and no n-count text.
    plt.tight_layout(pad=0.4)
    save_figure(fig, OUT_DIR, FIG_STEM)
    plt.close(fig)


def main() -> None:
    print("========== Draw Fig. 1(a): USGS Japan spatial distribution ==========")
    events = load_filtered_usgs_events()
    draw_fig1a(events)
    print("[DONE]")


if __name__ == "__main__":
    main()
