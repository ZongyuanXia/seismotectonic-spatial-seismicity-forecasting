# -*- coding: utf-8 -*-
"""
Plot Figure 6 and Figure 7 for the robustness analysis:

Figure 6
--------
Repeated-training CSI ranking curve.

Figure 7
--------
(a) Prediction-threshold sensitivity heatmap of Delta CSI.
(b) Spatial-expansion-scale sensitivity heatmap of Delta CSI.

Figure 7 uses:
- a shared color scale from 0 to 8.0 percentage points;
- one shared colorbar for panels (a) and (b);
- full rectangular outlines for the formal evaluation settings
  (prediction threshold = 100 and spatial expansion scale d = 3);
- panel labels (a) and (b);
- a shared colorbar with tick labels kept and shown without plus signs.

Input CSV files:
- 5_1_official_CSI_sorted_for_plot.csv
- 5_1_threshold_sensitivity_delta_CSI.csv
- 5_1_d_sensitivity_delta_CSI.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Rectangle


# ============================================================
# 0. Paths
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "results" / "convlstm" / "robustness"
OUT_DIR = REPO_ROOT / "results" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_SORTED = DATA_DIR / "5_1_official_CSI_sorted_for_plot.csv"
CSV_THRESHOLD = DATA_DIR / "5_1_threshold_sensitivity_delta_CSI.csv"
CSV_D = DATA_DIR / "5_1_d_sensitivity_delta_CSI.csv"


# ============================================================
# 1. Global style
# ============================================================

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 18,
    "axes.labelsize": 20,
    "axes.titlesize": 20,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18,
    "axes.linewidth": 1.2,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 6,
    "ytick.major.size": 6,
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.unicode_minus": False,
    "mathtext.fontset": "custom",
    "mathtext.rm": "Times New Roman",
    "mathtext.it": "Times New Roman:italic",
    "mathtext.bf": "Times New Roman:bold",
})


# ============================================================
# 2. Model names and plotting order
# ============================================================

MODEL_ORDER_ALL = ["C0", "C012", "C034", "C056", "C01256"]
MODEL_ORDER_EXT = ["C012", "C034", "C056", "C01256"]

MODEL_LABELS_SIMPLE = {
    "C0": "M1",
    "C012": "M2",
    "C034": "M3",
    "C056": "M4",
    "C01256": "M5",
}

LINE_STYLES = {
    "C0":     dict(marker="o", linestyle="--", linewidth=2.2, markersize=7),
    "C012":   dict(marker="s", linestyle="-",  linewidth=2.4, markersize=7),
    "C034":   dict(marker="^", linestyle="-.", linewidth=2.4, markersize=7),
    "C056":   dict(marker="D", linestyle="-",  linewidth=2.4, markersize=7),
    "C01256": dict(marker="v", linestyle=":",  linewidth=2.6, markersize=7),
}


# ============================================================
# 3. Figure 7 shared heatmap settings
# ============================================================

# Use one fixed color scale for both Figure 7 panels.
FIG7_VMIN = 0.0
FIG7_VMAX = 8.0
FIG7_CMAP = "Blues"
FIG7_CBAR_LABEL = r"$\Delta$CSI (percentage points)"

# Inset the formal-setting outline slightly from the cell boundary so all
# four sides remain visible, including when the formal column is the first one.
FORMAL_BOX_EXPAND = 0.035
FORMAL_BOX_LINEWIDTH = 2.2
FORMAL_BOX_COLOR = "#4D4D4D"

# Panel-label placement in axes coordinates.
PANEL_LABEL_X = -0.01
PANEL_LABEL_Y = 1.035
PANEL_LABEL_SIZE = 20

# Shared colorbar placement relative to panel (b).
FIG7_CBAR_GAP = 0.012
FIG7_CBAR_WIDTH = 0.015


# ============================================================
# 4. Helper functions
# ============================================================

def as_percent(values):
    """
    Convert decimal values such as 0.0715 to percentage points 7.15.
    If values already look like percentages, keep unchanged.
    """
    values = np.asarray(values, dtype=float)
    if np.nanmax(np.abs(values)) <= 1.5:
        return values * 100.0
    return values


def format_box_with_left_bottom_ticks(ax):
    """
    Keep full outer box, including top and right spines.
    Keep tick marks only on left and bottom.
    """
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)

    ax.tick_params(
        axis="both",
        which="both",
        direction="in",
        top=False,
        right=False,
        bottom=True,
        left=True,
        length=6,
        width=1.2
    )


def format_heatmap_axis(ax):
    """
    Heatmap axis:
    - keep full outer box;
    - keep labels on left and bottom;
    - remove short tick marks.
    """
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)

    ax.tick_params(
        axis="both",
        which="both",
        top=False,
        right=False,
        bottom=True,
        left=True,
        length=0,
        width=1.2
    )


def save_figure(fig, stem):
    """Save each figure as PNG, TIFF, and PDF."""
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=600, bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{stem}.tif", dpi=600, bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{stem}.pdf", bbox_inches="tight")


def add_heatmap_cell_grid(ax, data_shape):
    """Add white cell boundaries without visible minor tick marks."""
    nrows, ncols = data_shape
    ax.set_xticks(np.arange(ncols + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(nrows + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.4)
    ax.tick_params(which="minor", bottom=False, left=False, length=0)


def annotate_heatmap(ax, data):
    """Write signed Delta CSI values in each heatmap cell."""
    data = np.asarray(data, dtype=float)
    midpoint = (FIG7_VMIN + FIG7_VMAX) / 2.0

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            text_color = "white" if data[i, j] > midpoint else "black"
            ax.text(
                j,
                i,
                f"{data[i, j]:+.2f}",
                ha="center",
                va="center",
                fontsize=18,
                color=text_color,
                zorder=6
            )


def add_formal_column_outline(ax, formal_col, nrows):
    """
    Outline the entire formal-setting column with a complete rectangle.

    The rectangle is expanded slightly outward so the full border reads clearly,
    while using a dark-gray outline instead of pure black.
    """
    if formal_col is None:
        return

    expand = FORMAL_BOX_EXPAND
    rect = Rectangle(
        (formal_col - 0.5 - expand, -0.5 - expand),
        1.0 + 2.0 * expand,
        float(nrows) + 2.0 * expand,
        fill=False,
        edgecolor=FORMAL_BOX_COLOR,
        linewidth=FORMAL_BOX_LINEWIDTH,
        zorder=7,
        clip_on=False
    )
    ax.add_patch(rect)


def draw_heatmap_panel(
    ax,
    data,
    row_labels,
    col_labels,
    xlabel,
    formal_col,
    panel_label
):
    """
    Draw one Figure 7 heatmap panel using the shared color scale.
    """
    data = np.asarray(data, dtype=float)

    im = ax.imshow(
        data,
        aspect="auto",
        cmap=FIG7_CMAP,
        vmin=FIG7_VMIN,
        vmax=FIG7_VMAX
    )

    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Model")

    add_heatmap_cell_grid(ax, data.shape)
    annotate_heatmap(ax, data)
    add_formal_column_outline(ax, formal_col, data.shape[0])
    format_heatmap_axis(ax)

    ax.text(
        PANEL_LABEL_X,
        PANEL_LABEL_Y,
        panel_label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=PANEL_LABEL_SIZE,
        clip_on=False
    )

    return im


def load_threshold_sensitivity_data():
    """Load and prepare Figure 7(a) data."""
    df = pd.read_csv(CSV_THRESHOLD)

    if "pred_threshold" not in df.columns:
        raise ValueError(f"Missing column 'pred_threshold' in {CSV_THRESHOLD.name}")

    for col in MODEL_ORDER_EXT:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in {CSV_THRESHOLD.name}")

    # Horizontal axis from 100 to 50, preserving the experimental progression.
    df = df.sort_values("pred_threshold", ascending=False).reset_index(drop=True)

    data = as_percent(df[MODEL_ORDER_EXT].T.values)
    row_labels = [MODEL_LABELS_SIMPLE[m] for m in MODEL_ORDER_EXT]
    col_labels = [
        f"{int(v)}" if float(v).is_integer() else f"{v:g}"
        for v in df["pred_threshold"].values
    ]

    # Formal prediction threshold = 100.
    pred_values = list(df["pred_threshold"].astype(float))
    formal_col = pred_values.index(100.0) if 100.0 in pred_values else None

    return data, row_labels, col_labels, formal_col


def load_d_sensitivity_data():
    """Load and prepare Figure 7(b) data."""
    df = pd.read_csv(CSV_D)

    if "expand_d" not in df.columns:
        raise ValueError(f"Missing column 'expand_d' in {CSV_D.name}")

    for col in MODEL_ORDER_EXT:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in {CSV_D.name}")

    df = df.sort_values("expand_d", ascending=True).reset_index(drop=True)

    data = as_percent(df[MODEL_ORDER_EXT].T.values)
    row_labels = [MODEL_LABELS_SIMPLE[m] for m in MODEL_ORDER_EXT]
    col_labels = [
        f"{int(v)}" if float(v).is_integer() else f"{v:g}"
        for v in df["expand_d"].values
    ]

    # Formal spatial expansion scale d = 3.
    d_values = list(df["expand_d"].astype(float))
    formal_col = d_values.index(3.0) if 3.0 in d_values else None

    return data, row_labels, col_labels, formal_col


# ============================================================
# 5. Figure 6: repeated-training CSI ranking curve
#    Plotting settings are intentionally unchanged.
# ============================================================

def plot_figure6_csi_sorted_curve():
    df = pd.read_csv(CSV_SORTED)

    required_cols = {"experiment", "rank", "CSI"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {CSV_SORTED.name}: {missing}")

    df["CSI_percent"] = as_percent(df["CSI"].values)

    fig, ax = plt.subplots(figsize=(9.5, 6.8))

    for exp in MODEL_ORDER_ALL:
        sub = df[df["experiment"] == exp].copy()
        if sub.empty:
            continue

        sub = sub.sort_values("rank")
        ax.plot(
            sub["rank"],
            sub["CSI_percent"],
            label=MODEL_LABELS_SIMPLE[exp],
            **LINE_STYLES[exp]
        )

    ax.set_xlabel("Rank")
    ax.set_ylabel("CSI (%)")

    ax.set_xlim(0.7, 9.3)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    ymin = np.floor(df["CSI_percent"].min() - 1.0)
    ymax = np.ceil(df["CSI_percent"].max() + 1.0)
    ax.set_ylim(ymin, ymax)

    # Full outer box, ticks only on left and bottom
    format_box_with_left_bottom_ticks(ax)

    # Legend inside lower-left corner
    ax.legend(
        loc="lower left",
        frameon=False,
        handlelength=2.5,
        borderaxespad=0.6,
        labelspacing=0.4
    )

    fig.tight_layout()
    save_figure(fig, "Fig_6_repeated_training_CSI_rank")
    plt.close(fig)


# ============================================================
# 6. Figure 7: sensitivity of Delta CSI to evaluation settings
#    (a) Prediction threshold
#    (b) Spatial expansion scale d
# ============================================================

def plot_figure7_sensitivity_heatmaps():
    threshold_data, threshold_rows, threshold_cols, threshold_formal_col = (
        load_threshold_sensitivity_data()
    )
    d_data, d_rows, d_cols, d_formal_col = load_d_sensitivity_data()

    # Width ratios are proportional to the number of columns so the heatmap
    # cells have similar visual widths in panels (a) and (b).
    fig = plt.figure(figsize=(17.0, 5.8))
    gs = fig.add_gridspec(
        nrows=1,
        ncols=2,
        width_ratios=[len(threshold_cols), len(d_cols)],
        wspace=0.30
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    im_a = draw_heatmap_panel(
        ax=ax_a,
        data=threshold_data,
        row_labels=threshold_rows,
        col_labels=threshold_cols,
        xlabel="Prediction threshold",
        formal_col=threshold_formal_col,
        panel_label="(a)"
    )

    im_b = draw_heatmap_panel(
        ax=ax_b,
        data=d_data,
        row_labels=d_rows,
        col_labels=d_cols,
        xlabel="Spatial expansion scale, d",
        formal_col=d_formal_col,
        panel_label="(b)"
    )

    # Shared colorbar for both panels. Place it close to panel (b) without
    # changing the larger spacing already set between panels (a) and (b).
    fig.subplots_adjust(right=0.93)
    bbox_b = ax_b.get_position()
    cax = fig.add_axes([
        bbox_b.x1 + FIG7_CBAR_GAP,
        bbox_b.y0,
        FIG7_CBAR_WIDTH,
        bbox_b.height,
    ])

    cbar = fig.colorbar(im_b, cax=cax)
    cbar.set_label(FIG7_CBAR_LABEL, fontsize=20)
    cbar.set_ticks([0, 2, 4, 6, 8])
    cbar.set_ticklabels(["0", "2", "4", "6", "8"])
    cbar.ax.tick_params(length=5, width=1.0)
    cbar.outline.set_linewidth(1.2)

    save_figure(fig, "Fig_7_sensitivity_delta_CSI")
    plt.close(fig)


# ============================================================
# 7. Run
# ============================================================

if __name__ == "__main__":
    plot_figure6_csi_sorted_curve()
    plot_figure7_sensitivity_heatmaps()

    print("Done. Figure 6 and Figure 7 saved to:", OUT_DIR.resolve())
