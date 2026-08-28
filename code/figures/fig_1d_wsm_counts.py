# -*- coding: utf-8 -*-
"""
Fig1d_WSM_Counts.py

Figure 1(d):
Horizontal bar chart of WSM2025 record counts for the Japan main
forecasting region and nine auxiliary RF validation regions.

This script complements Fig1c1_Region_Map_with_Global_WSM_v6.py.

Data logic
----------
1) Read the regional WSM2025 record-count table produced by
   ``01_prepare_wsm_data.py``:
       data/derived/region_record_counts.csv
2) Draw a horizontal bar chart using the same role colors as Fig. 1(c1):
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

import sys
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
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RF_ASSOCIATION_DIR = PROJECT_ROOT / "code" / "rf_association"
if str(RF_ASSOCIATION_DIR) not in sys.path:
    sys.path.insert(0, str(RF_ASSOCIATION_DIR))

from config import REGION_RECORD_COUNTS_PATH

REGION_COUNTS_CSV = REGION_RECORD_COUNTS_PATH
OUT_DIR = PROJECT_ROOT / "results" / "figures"
FIG_STEM = "Fig1d_WSM_Counts"

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
# 2. Regional count-table loading
# =============================================================================

def load_region_counts() -> pd.DataFrame:
    if not REGION_COUNTS_CSV.exists():
        raise FileNotFoundError(
            f"Regional count table not found: {REGION_COUNTS_CSV}. "
            "Run 01_prepare_wsm_data.py first."
        )

    source = pd.read_csv(REGION_COUNTS_CSV)
    required = {"region", "region_name", "n_records"}
    missing = required.difference(source.columns)
    if missing:
        raise ValueError(f"Missing columns in {REGION_COUNTS_CSV}: {sorted(missing)}")

    counts = source.rename(
        columns={"region_name": "display_region", "n_records": "wsm_record_count"}
    )[["region", "display_region", "wsm_record_count"]].copy()
    counts["wsm_record_count"] = pd.to_numeric(
        counts["wsm_record_count"], errors="raise"
    ).astype(int)
    counts["role"] = counts["region"].map({"japan": "main"}).fillna("aux")
    counts = counts.rename(columns={"display_region": "region_name"})

    if SAVE_COUNTS_CSV:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out_csv = OUT_DIR / "Fig1d_region_counts.csv"
        counts.to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"[SAVE] {out_csv}")

    print("\n========== Region WSM2025 record counts ==========")
    print(counts[["region_name", "role", "wsm_record_count"]].to_string(index=False))
    return counts


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
    ax.set_yticklabels(df["region_name"], fontsize=TICK_LABEL_SIZE)
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
    print(f"REGION_COUNTS_CSV: {REGION_COUNTS_CSV}")
    print(f"OUT_DIR: {OUT_DIR}")

    counts = load_region_counts()
    draw_d_bar_chart(counts)

    print("[DONE] Fig. 1(d) outputs saved.")


if __name__ == "__main__":
    main()
