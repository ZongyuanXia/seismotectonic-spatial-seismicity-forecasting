#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 5 plotting script

功能
----
1. 读取代表性测试样本（默认 sample_0400）的预测频次网格和真实目标网格
2. 输出每个模型的单图：
   - 带颜色棒版本
   - 不带颜色棒版本
3. 输出五模型拼接后的合成图（仅一个共享颜色棒）

说明
----
- 输入文件夹采用实际路径：
    ./test_jp/4_1_prediction_frequency_export
- 图中模型名称统一采用 M1–M5
- 底图风格参照 Figure 1(a) 的日本地区绘图风格
- 0 频次区域透明化，避免覆盖蓝色海洋和白色陆地底图
- 不叠加 M1 高频次轮廓线
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


# =============================================================================
# 0. User settings
# =============================================================================

# ---- Input / output paths ----
REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_ROOT = REPO_ROOT / "results" / "convlstm" / "figure5_source"
SAMPLE_GID = 400
OUTPUT_DIR = REPO_ROOT / "results" / "figures" / f"figure5_sample_{SAMPLE_GID:04d}"

# ---- Exported folder -> manuscript model name ----
GROUPS = [
    {"folder": "C0",     "model": "M1", "input_label": "MAG",                     "panel": "(a)"},
    {"folder": "C012",   "model": "M2", "input_label": "MAG + Depth",             "panel": "(b)"},
    {"folder": "C034",   "model": "M3", "input_label": r"MAG + $S_{\mathrm{Hmax}}$ orientation", "panel": "(c)"},
    {"folder": "C056",   "model": "M4", "input_label": "MAG + Stress regime",     "panel": "(d)"},
    {"folder": "C01256", "model": "M5", "input_label": "MAG + Depth + Stress regime", "panel": "(e)"},
]

# ---- Output format ----
SAVE_PNG = True
SAVE_PDF = True
SAVE_SVG = False
DPI = 600

# ---- Figure size ----
# 单图：始终预留 colorbar 位置，保证有/无颜色棒版本高度一致
SINGLE_FIGSIZE = (6.6, 5.4)
COMBINED_FIGSIZE = (12.8, 7.8)

# ---- Font ----
FONT_FAMILY = "Times New Roman"
BASE_FONT_SIZE = 10
TICK_SIZE = 9
TITLE_SIZE = 12
CBAR_LABEL_SIZE = 10
CBAR_TICK_SIZE = 9

# ---- Combined figure layout ----
COMBINED_LEFT = 0.055
COMBINED_RIGHT = 0.985
COMBINED_BOTTOM = 0.10
COMBINED_TOP = 0.94
COMBINED_WSPACE = 0.07
COMBINED_HSPACE = 0.13
SHARED_CBAR_WIDTH = 0.014
SHARED_CBAR_HEIGHT_FRAC = 0.72
SHARED_CBAR_PAD = 0.025

# ---- Study region: Japan main study region ----
LAT_MIN = 30.05
LAT_MAX = 45.95
LON_MIN = 128.10
LON_MAX = 146.80
MAP_PADDING_DEG = 0.35

EXTENT = {
    "lon_min": LON_MIN - MAP_PADDING_DEG,
    "lon_max": LON_MAX + MAP_PADDING_DEG,
    "lat_min": LAT_MIN - MAP_PADDING_DEG,
    "lat_max": LAT_MAX + MAP_PADDING_DEG,
}

# ---- Ticks ----
XTICKS = [130, 132.5, 135, 137.5, 140, 142.5, 145]
YTICKS = [30, 32.5, 35, 37.5, 40, 42.5, 45]

# ---- Map style: refer to Fig. 1(a) ----
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

# ---- Prediction-frequency colormap ----
CMAP = LinearSegmentedColormap.from_list(
    "transparent_to_softred",
    ["#ffffff", "#f7d6d6", "#efaaaa", "#de6f6f", "#c93f3f"]
)

# 让 masked value 透明，不覆盖底图
CMAP = CMAP.copy()
CMAP.set_bad((1, 1, 1, 0))

# True: 0 频次区域透明，露出蓝色海洋和白色陆地
MASK_ZERO_FREQUENCY = True

# ---- Ground-truth point style ----
GT_POINT_SIZE = 14
GT_POINT_COLOR = "black"

# ---- Cartopy ----
USE_CARTOPY_IF_AVAILABLE = True

HAS_CARTOPY = False
if USE_CARTOPY_IF_AVAILABLE:
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
        HAS_CARTOPY = True
    except Exception:
        HAS_CARTOPY = False


# =============================================================================
# 1. Global style
# =============================================================================

def apply_matplotlib_style() -> None:
    mpl.rcParams.update({
        "font.family": FONT_FAMILY,
        "font.size": BASE_FONT_SIZE,
        "axes.labelsize": BASE_FONT_SIZE,
        "xtick.labelsize": TICK_SIZE,
        "ytick.labelsize": TICK_SIZE,
        "legend.fontsize": BASE_FONT_SIZE,
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.unicode_minus": False,
    })


# =============================================================================
# 2. Data loading
# =============================================================================

def load_grid(csv_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load grid CSV:
        index = lat
        columns = lon

    Return:
        lat_vals, lon_vals, values
    """
    df = pd.read_csv(csv_path, index_col=0)
    lat_vals = df.index.astype(float).to_numpy()
    lon_vals = df.columns.astype(float).to_numpy()
    values = df.to_numpy(dtype=float)
    return lat_vals, lon_vals, values


def get_group_paths(input_root: str, sample_gid: int, folder: str) -> Dict[str, str]:
    sample_dir = Path(input_root) / f"sample_{sample_gid:04d}" / folder

    freq_csv = sample_dir / f"{folder}_pred_frequency_grid.csv"
    true_csv = sample_dir / f"{folder}_true_mask_grid.csv"

    if not freq_csv.exists():
        raise FileNotFoundError(f"Prediction-frequency file not found: {freq_csv}")
    if not true_csv.exists():
        raise FileNotFoundError(f"True-mask file not found: {true_csv}")

    return {
        "sample_dir": str(sample_dir),
        "freq_csv": str(freq_csv),
        "true_csv": str(true_csv),
    }


def compute_edges(vals: np.ndarray) -> np.ndarray:
    """
    Convert grid centers to grid edges.
    """
    vals = np.asarray(vals, dtype=float)

    if len(vals) < 2:
        step = 1.0
        return np.array([vals[0] - step / 2.0, vals[0] + step / 2.0], dtype=float)

    diffs = np.diff(vals)
    first = vals[0] - diffs[0] / 2.0
    last = vals[-1] + diffs[-1] / 2.0
    mids = (vals[:-1] + vals[1:]) / 2.0
    return np.concatenate([[first], mids, [last]])


def prepare_frequency(freq: np.ndarray) -> np.ndarray:
    """
    Mask zero-frequency cells so that the base map remains visible.
    """
    freq = np.asarray(freq, dtype=float)

    if MASK_ZERO_FREQUENCY:
        return np.ma.masked_where(freq <= 0.0, freq)

    return freq


# =============================================================================
# 3. Axes styling
# =============================================================================

def add_base_map(ax) -> None:
    """
    Add ocean, land, and coastline.
    """
    ax.set_facecolor(OCEAN_COLOR)

    if HAS_CARTOPY and hasattr(ax, "projection"):
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
            zorder=6,
        )


def style_plain_axes(ax, show_left: bool = True, show_bottom: bool = True) -> None:
    """
    Fallback axes style when Cartopy is unavailable.
    """
    ax.set_facecolor(OCEAN_COLOR)
    ax.set_xlim(EXTENT["lon_min"], EXTENT["lon_max"])
    ax.set_ylim(EXTENT["lat_min"], EXTENT["lat_max"])

    ax.set_xticks(XTICKS)
    ax.set_yticks(YTICKS)

    ax.tick_params(
        axis="both",
        which="both",
        direction="in",
        bottom=True,
        left=True,
        top=False,
        right=False,
        labelbottom=show_bottom,
        labelleft=show_left,
        labeltop=False,
        labelright=False,
        length=3.0,
        width=0.7,
    )

    ax.grid(
        True,
        color=GRIDLINE_COLOR,
        linewidth=GRIDLINE_WIDTH,
        alpha=GRIDLINE_ALPHA,
        linestyle=GRIDLINE_STYLE,
        zorder=1,
    )

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(SPINE_COLOR)
        spine.set_linewidth(SPINE_WIDTH)


def style_cartopy_axes(ax, show_left: bool = True, show_bottom: bool = True) -> None:
    """
    Cartopy axes style following the Japan map style.
    """
    ax.set_extent(
        [
            EXTENT["lon_min"],
            EXTENT["lon_max"],
            EXTENT["lat_min"],
            EXTENT["lat_max"],
        ],
        crs=ccrs.PlateCarree(),
    )

    add_base_map(ax)

    ax.set_xticks(XTICKS, crs=ccrs.PlateCarree())
    ax.set_yticks(YTICKS, crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(
        LongitudeFormatter(number_format="g", degree_symbol="°")
    )
    ax.yaxis.set_major_formatter(
        LatitudeFormatter(number_format="g", degree_symbol="°")
    )

    ax.tick_params(
        axis="both",
        which="both",
        direction="in",
        bottom=True,
        left=True,
        top=False,
        right=False,
        labelbottom=show_bottom,
        labelleft=show_left,
        labeltop=False,
        labelright=False,
        length=3.0,
        width=0.7,
    )

    ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=False,
        linewidth=GRIDLINE_WIDTH,
        color=GRIDLINE_COLOR,
        alpha=GRIDLINE_ALPHA,
        linestyle=GRIDLINE_STYLE,
        xlocs=XTICKS,
        ylocs=YTICKS,
        zorder=2,
    )

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(SPINE_COLOR)
        spine.set_linewidth(SPINE_WIDTH)


def add_panel_title(ax, panel: str, model: str, input_label: str) -> None:
    ax.set_title(
        f"{panel} {model}: {input_label}",
        fontsize=TITLE_SIZE,
        pad=4,
    )


# =============================================================================
# 4. Map drawing
# =============================================================================

def draw_one_map(
    ax,
    lon_vals: np.ndarray,
    lat_vals: np.ndarray,
    freq: np.ndarray,
    true_mask: np.ndarray,
    panel: str,
    model: str,
    input_label: str,
    show_left: bool = True,
    show_bottom: bool = True,
):
    """
    Draw one prediction-frequency map.
    """
    lon_edges = compute_edges(lon_vals)
    lat_edges = compute_edges(lat_vals)
    freq_plot = prepare_frequency(freq)

    if HAS_CARTOPY and hasattr(ax, "projection"):
        style_cartopy_axes(
            ax,
            show_left=show_left,
            show_bottom=show_bottom,
        )

        mesh = ax.pcolormesh(
            lon_edges,
            lat_edges,
            freq_plot,
            cmap=CMAP,
            vmin=0.0,
            vmax=1.0,
            shading="auto",
            transform=ccrs.PlateCarree(),
            zorder=3,
        )

        yy, xx = np.where(true_mask > 0)
        gt_lons = lon_vals[xx]
        gt_lats = lat_vals[yy]

        ax.scatter(
            gt_lons,
            gt_lats,
            s=GT_POINT_SIZE,
            c=GT_POINT_COLOR,
            linewidths=0,
            transform=ccrs.PlateCarree(),
            zorder=7,
        )

        # Redraw coastline above heatmap and points.
        ax.coastlines(
            resolution=COASTLINE_RESOLUTION,
            color=COASTLINE_COLOR,
            linewidth=COASTLINE_WIDTH,
            zorder=8,
        )

    else:
        style_plain_axes(
            ax,
            show_left=show_left,
            show_bottom=show_bottom,
        )

        mesh = ax.pcolormesh(
            lon_edges,
            lat_edges,
            freq_plot,
            cmap=CMAP,
            vmin=0.0,
            vmax=1.0,
            shading="auto",
            zorder=3,
        )

        yy, xx = np.where(true_mask > 0)
        gt_lons = lon_vals[xx]
        gt_lats = lat_vals[yy]

        ax.scatter(
            gt_lons,
            gt_lats,
            s=GT_POINT_SIZE,
            c=GT_POINT_COLOR,
            linewidths=0,
            zorder=7,
        )

    add_panel_title(ax, panel=panel, model=model, input_label=input_label)
    return mesh


# =============================================================================
# 5. Single map export
# =============================================================================

def save_single_map(item: Dict, with_colorbar: bool = True) -> None:
    """
    Save single map.

    To keep map height identical between with-colorbar and no-colorbar versions,
    always reserve the colorbar axes.
    """
    if HAS_CARTOPY:
        fig = plt.figure(figsize=SINGLE_FIGSIZE)
        gs = fig.add_gridspec(
            nrows=1,
            ncols=2,
            width_ratios=[1.0, 0.045],
            wspace=0.06,
        )
        ax = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
        cax = fig.add_subplot(gs[0, 1])
    else:
        fig = plt.figure(figsize=SINGLE_FIGSIZE)
        gs = fig.add_gridspec(
            nrows=1,
            ncols=2,
            width_ratios=[1.0, 0.045],
            wspace=0.06,
        )
        ax = fig.add_subplot(gs[0, 0])
        cax = fig.add_subplot(gs[0, 1])

    mesh = draw_one_map(
        ax=ax,
        lon_vals=item["lon_vals"],
        lat_vals=item["lat_vals"],
        freq=item["freq"],
        true_mask=item["true_mask"],
        panel=item["panel"],
        model=item["model"],
        input_label=item["input_label"],
        show_left=True,
        show_bottom=True,
    )

    if with_colorbar:
        cbar = fig.colorbar(mesh, cax=cax)
        cbar.set_label("Prediction frequency", fontsize=CBAR_LABEL_SIZE)
        cbar.set_ticks([0.0, 0.25, 0.50, 0.75, 1.0])
        cbar.ax.tick_params(
            labelsize=CBAR_TICK_SIZE,
            direction="in",
        )
    else:
        cax.set_visible(False)

    fig.tight_layout()

    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    suffix = "with_cbar" if with_colorbar else "no_cbar"
    stem = f"Figure5_{item['model']}_{suffix}"

    if SAVE_PNG:
        fig.savefig(out_dir / f"{stem}.png", dpi=DPI, bbox_inches="tight")
    if SAVE_PDF:
        fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    if SAVE_SVG:
        fig.savefig(out_dir / f"{stem}.svg", bbox_inches="tight")

    plt.close(fig)


# =============================================================================
# 6. Combined figure export
# =============================================================================

def save_combined_figure(group_data: List[Dict]) -> None:
    """
    Save combined Figure 5 with five subplots and one shared colorbar.

    Design requirements:
    - Panels (a)–(e) keep identical map sizes.
    - Panel (e) stays in the middle column.
    - The unused lower-right slot is removed so no black area appears.
    - One shared colorbar is placed to the right of panel (e).
    """
    nrows, ncols = 2, 3

    if HAS_CARTOPY:
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=COMBINED_FIGSIZE,
            subplot_kw={"projection": ccrs.PlateCarree()},
        )
    else:
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=COMBINED_FIGSIZE,
        )

    fig.patch.set_facecolor("white")
    axes = np.asarray(axes).reshape(nrows, ncols)

    # Keep a regular 2x3 layout so all map panels have the same size.
    # Use only the first five slots: (a)(b)(c) on top, (d)(e) on bottom-left/bottom-middle.
    used_positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]
    last_mesh = None
    used_axes = []

    for item, (row, col) in zip(group_data, used_positions):
        ax = axes[row, col]
        used_axes.append(ax)

        show_left = (col == 0)
        show_bottom = (row == 1)

        last_mesh = draw_one_map(
            ax=ax,
            lon_vals=item["lon_vals"],
            lat_vals=item["lat_vals"],
            freq=item["freq"],
            true_mask=item["true_mask"],
            panel=item["panel"],
            model=item["model"],
            input_label=item["input_label"],
            show_left=show_left,
            show_bottom=show_bottom,
        )

    # Remove the unused lower-right axes completely to avoid the black area.
    unused_ax = axes[1, 2]
    unused_ax.remove()

    fig.subplots_adjust(
        left=COMBINED_LEFT,
        right=COMBINED_RIGHT,
        bottom=COMBINED_BOTTOM,
        top=COMBINED_TOP,
        wspace=COMBINED_WSPACE,
        hspace=COMBINED_HSPACE,
    )

    # Shared colorbar positioned to the right of panel (e).
    if last_mesh is not None:
        ax_e = axes[1, 1]
        bbox = ax_e.get_position()

        cbar_height = bbox.height * SHARED_CBAR_HEIGHT_FRAC
        cbar_x0 = bbox.x1 + SHARED_CBAR_PAD
        cbar_y0 = bbox.y0 + (bbox.height - cbar_height) / 2.0

        cax = fig.add_axes([
            cbar_x0,
            cbar_y0,
            SHARED_CBAR_WIDTH,
            cbar_height,
        ])
        cbar = fig.colorbar(last_mesh, cax=cax, orientation="vertical")
        cbar.set_label("Prediction frequency", fontsize=CBAR_LABEL_SIZE)
        cbar.set_ticks([0.0, 0.25, 0.50, 0.75, 1.0])
        cbar.ax.tick_params(labelsize=CBAR_TICK_SIZE, direction="in")

    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = f"Figure5_combined_sample_{SAMPLE_GID:04d}"

    if SAVE_PNG:
        fig.savefig(out_dir / f"{stem}.png", dpi=DPI, bbox_inches="tight", facecolor="white")
    if SAVE_PDF:
        fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    if SAVE_SVG:
        fig.savefig(out_dir / f"{stem}.svg", bbox_inches="tight", facecolor="white")

    plt.close(fig)


# =============================================================================
# 7. Main
# =============================================================================

def main() -> None:
    apply_matplotlib_style()

    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    group_data: List[Dict] = []

    for group in GROUPS:
        folder = group["folder"]
        paths = get_group_paths(INPUT_ROOT, SAMPLE_GID, folder)

        lat_vals, lon_vals, freq = load_grid(paths["freq_csv"])
        lat_vals_t, lon_vals_t, true_mask = load_grid(paths["true_csv"])

        if not np.allclose(lat_vals, lat_vals_t):
            raise ValueError(f"{folder}: latitude coordinates are inconsistent.")
        if not np.allclose(lon_vals, lon_vals_t):
            raise ValueError(f"{folder}: longitude coordinates are inconsistent.")

        group_data.append({
            "folder": folder,
            "model": group["model"],
            "input_label": group["input_label"],
            "panel": group["panel"],
            "lon_vals": lon_vals,
            "lat_vals": lat_vals,
            "freq": freq,
            "true_mask": true_mask,
        })

    # Single maps: with colorbar and without colorbar.
    for item in group_data:
        save_single_map(item, with_colorbar=True)
        save_single_map(item, with_colorbar=False)

    # Combined figure: one shared colorbar.
    save_combined_figure(group_data)

    print(f"[DONE] Output directory: {out_dir}")
    for p in sorted(out_dir.iterdir()):
        print(p.name)


if __name__ == "__main__":
    main()
