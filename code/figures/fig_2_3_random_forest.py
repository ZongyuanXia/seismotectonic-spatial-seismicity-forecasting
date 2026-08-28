# -*- coding: utf-8 -*-
"""
plot_rf_fig2_fig3.py

用途
----
读取各地区、各震级阈值下随机森林全特征实验结果，并绘制：

Figure 2:
    多地区随机森林震级分组表现
    (a) Threshold: Mw = 4
    (b) Threshold: Mw = 5
    (c) Threshold: Mw = 6

Figure 3:
    应力场构造特征重要性
    (a) Threshold: Mw = 4
    (b) Threshold: Mw = 5
    (c) Threshold: Mw = 6
    (d) All thresholds

读取逻辑
--------
读取 ``05_summarize_rf_results.py`` 生成的冻结汇总表：

    results/random_forest/summaries/rf_accuracy_by_region_threshold.csv
    results/random_forest/summaries/rf_importance_by_region_threshold.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms


# =============================================================================
# 0. User settings
# =============================================================================

# ---- Input / output paths ----
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ACCURACY_SUMMARY_CSV = (
    PROJECT_ROOT / "results" / "random_forest" / "summaries"
    / "rf_accuracy_by_region_threshold.csv"
)
IMPORTANCE_SUMMARY_CSV = (
    PROJECT_ROOT / "results" / "random_forest" / "summaries"
    / "rf_importance_by_region_threshold.csv"
)
OUT_DIR = PROJECT_ROOT / "results" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- Figure settings ----
DPI = 600
SAVE_PNG = True
SAVE_PDF = True
SAVE_TIFF = False

FONT_FAMILY = "Times New Roman"
BASE_FONT_SIZE = 8
TICK_SIZE = 7
LABEL_SIZE = 8
PANEL_LABEL_SIZE = 10
LEGEND_SIZE = 7

# ---- Figure size ----
FIG2_WIDTH_CM = 18.0
FIG2_HEIGHT_CM = 9.5

FIG3_WIDTH_CM = 18.0
FIG3_HEIGHT_CM = 11.5

# ---- Figure 2 sorting ----
# "within_threshold": 每个子图内部按 Accuracy 从低到高排序
# "global_mean": 三个子图保持同一地区顺序，便于横向比较
FIG2_SORT_MODE = "within_threshold"

# ---- Axis settings ----
FIG2_XLIM = (0.4, 0.9)
FIG2_XTICKS = np.arange(0.4, 0.91, 0.1)
RANDOM_BASELINE = 0.5

FIG3_YLIM = None  # None means automatic common ylim across four panels

# ---- Figure 3 text settings ----
FIG3_YLABEL = "Mean feature importance"
FIG3_PANEL_TITLES = {
    4: rf"Threshold: $M_w = 4.0$",
    5: rf"Threshold: $M_w = 5.0$",
    6: rf"Threshold: $M_w = 6.0$",
    "all": "Mean across thresholds",
}

# ---- Colors ----
COLOR_BLUE = "#3E517A"
COLOR_BLUE_ERR = "#7F90B0"
COLOR_RED = "#B22222"
COLOR_RED_ERR = "#D98C8C"
COLOR_BASELINE = "#5F5F5F"
COLOR_GRID = "#D0D0D0"
COLOR_SPINE = "#4D4D4D"
COLOR_GROUP_TEXT = "#4D4D4D"


# =============================================================================
# 1. Feature display settings
# =============================================================================

FEATURE_ORDER = [
    "MEAN_LOCAL_DEPTH",
    "MEDIAN_LOCAL_DEPTH",
    "STD_LOCAL_DEPTH",
    "REGIME_ENTROPY",
    "REGIME_GINI",
    "MAX_SPREAD_100KM_DEG",
    "CONSISTENCY_R_MEAN",
    "STD_LOCAL",
    "MAX_ROTATION_DEG",
]

FEATURE_LABELS = {
    "MEAN_LOCAL_DEPTH": r"$\bar{h}$",
    "MEDIAN_LOCAL_DEPTH": r"$\tilde{h}$",
    "STD_LOCAL_DEPTH": r"$\sigma_h$",
    "REGIME_ENTROPY": r"$H$",
    "REGIME_GINI": r"$D$",
    "MAX_SPREAD_100KM_DEG": r"$\Delta\theta_{\max}$",
    "CONSISTENCY_R_MEAN": r"$R$",
    "STD_LOCAL": r"$s_0$",
    "MAX_ROTATION_DEG": r"$\Delta\theta_{mi}$",
}

# start/end 是 FEATURE_ORDER 中的闭区间索引
FEATURE_GROUPS = [
    ("Depth", 0, 2),
    ("Stress regime", 3, 4),
    ("$S_{Hmax}$ orientation", 5, 8),
]

FINAL_SELECTED_FEATURES = {
    "MEAN_LOCAL_DEPTH",
    "STD_LOCAL_DEPTH",
    "REGIME_ENTROPY",
    "REGIME_GINI",
    "MAX_SPREAD_100KM_DEG",
    "STD_LOCAL",
}


# =============================================================================
# 3. Matplotlib style
# =============================================================================

def cm_to_inch(value_cm: float) -> float:
    return value_cm / 2.54


def apply_style() -> None:
    mpl.rcParams.update({
        "font.family": FONT_FAMILY,
        "font.size": BASE_FONT_SIZE,
        "axes.labelsize": LABEL_SIZE,
        "xtick.labelsize": TICK_SIZE,
        "ytick.labelsize": TICK_SIZE,
        "legend.fontsize": LEGEND_SIZE,
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.unicode_minus": False,
    })


def style_axes_common(ax: plt.Axes) -> None:
    """统一坐标轴样式：仅保留下方和左方刻度及刻度值。"""
    ax.tick_params(
        axis="both",
        which="both",
        direction="in",
        bottom=True,
        left=True,
        top=False,
        right=False,
        labelbottom=True,
        labelleft=True,
        labeltop=False,
        labelright=False,
        length=3.0,
        width=0.7,
    )

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.7)
        spine.set_color(COLOR_SPINE)


def save_figure(fig: plt.Figure, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if SAVE_PNG:
        fp = OUT_DIR / f"{stem}.png"
        fig.savefig(fp, dpi=DPI, bbox_inches="tight")
        print(f"[SAVE] {fp}")

    if SAVE_PDF:
        fp = OUT_DIR / f"{stem}.pdf"
        fig.savefig(fp, bbox_inches="tight")
        print(f"[SAVE] {fp}")

    if SAVE_TIFF:
        fp = OUT_DIR / f"{stem}.tif"
        fig.savefig(fp, dpi=DPI, bbox_inches="tight")
        print(f"[SAVE] {fp}")


# =============================================================================
# 4. Reading utilities
# =============================================================================

def read_csv_robust(path: Path) -> pd.DataFrame:
    for enc in ["utf-8-sig", "utf-8", "gbk", "cp936", "latin1"]:
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except Exception:
            continue
    raise RuntimeError(f"CSV read failed: {path}")


def collect_results() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not ACCURACY_SUMMARY_CSV.exists():
        raise FileNotFoundError(
            f"Accuracy summary not found: {ACCURACY_SUMMARY_CSV}. "
            "Run 05_summarize_rf_results.py first."
        )
    if not IMPORTANCE_SUMMARY_CSV.exists():
        raise FileNotFoundError(
            f"Feature-importance summary not found: {IMPORTANCE_SUMMARY_CSV}. "
            "Run 05_summarize_rf_results.py first."
        )

    accuracy_source = read_csv_robust(ACCURACY_SUMMARY_CSV)
    importance_source = read_csv_robust(IMPORTANCE_SUMMARY_CSV)

    accuracy_required = {
        "region", "region_name", "threshold_mw", "accuracy_mean", "accuracy_std",
    }
    importance_required = {
        "region", "region_name", "threshold_mw", "feature",
        "importance_mean", "importance_std",
    }
    missing_accuracy = accuracy_required.difference(accuracy_source.columns)
    missing_importance = importance_required.difference(importance_source.columns)
    if missing_accuracy:
        raise ValueError(
            f"Missing columns in {ACCURACY_SUMMARY_CSV}: {sorted(missing_accuracy)}"
        )
    if missing_importance:
        raise ValueError(
            f"Missing columns in {IMPORTANCE_SUMMARY_CSV}: {sorted(missing_importance)}"
        )

    df_acc = accuracy_source.rename(
        columns={
            "region": "region_code",
            "region_name": "region",
            "threshold_mw": "threshold",
        }
    )[["region_code", "region", "threshold", "accuracy_mean", "accuracy_std"]].copy()
    df_fi_raw = importance_source.rename(
        columns={
            "region": "region_code",
            "region_name": "region",
            "threshold_mw": "threshold",
        }
    )[
        [
            "region_code", "region", "threshold", "feature",
            "importance_mean", "importance_std",
        ]
    ].copy()

    for frame, numeric_columns in [
        (df_acc, ["threshold", "accuracy_mean", "accuracy_std"]),
        (df_fi_raw, ["threshold", "importance_mean", "importance_std"]),
    ]:
        for column in numeric_columns:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
    df_acc["threshold"] = df_acc["threshold"].astype(int)
    df_fi_raw["threshold"] = df_fi_raw["threshold"].astype(int)
    df_fi_raw = df_fi_raw[df_fi_raw["feature"].isin(FEATURE_ORDER)].copy()

    df_acc.to_csv(OUT_DIR / "rf_accuracy_summary_for_fig2.csv", index=False, encoding="utf-8-sig")
    df_fi_raw.to_csv(OUT_DIR / "rf_feature_importance_raw_for_fig3.csv", index=False, encoding="utf-8-sig")

    return df_acc, df_fi_raw


# =============================================================================
# 5. Figure 2
# =============================================================================

def get_global_region_order(df_acc: pd.DataFrame) -> list[str]:
    tmp = (
        df_acc.groupby("region", as_index=False)["accuracy_mean"]
        .mean()
        .sort_values("accuracy_mean", ascending=True)
    )
    return tmp["region"].tolist()


def plot_figure2(df_acc: pd.DataFrame) -> None:
    apply_style()

    thresholds = [4, 5, 6]
    panel_labels = ["(a)", "(b)", "(c)"]

    fig, axes = plt.subplots(
        1, 3,
        figsize=(cm_to_inch(FIG2_WIDTH_CM), cm_to_inch(FIG2_HEIGHT_CM)),
        sharex=True,
    )

    if FIG2_SORT_MODE == "global_mean":
        global_order = get_global_region_order(df_acc)
    else:
        global_order = None

    for ax, threshold, panel in zip(axes, thresholds, panel_labels):
        sub = df_acc[df_acc["threshold"] == threshold].copy()

        if sub.empty:
            ax.text(
                0.5, 0.5, f"No data for threshold {threshold}",
                ha="center", va="center", transform=ax.transAxes
            )
            ax.set_axis_off()
            continue

        if FIG2_SORT_MODE == "global_mean":
            sub["region_order"] = sub["region"].map({r: i for i, r in enumerate(global_order)})
            sub = sub.sort_values("region_order", ascending=True)
        else:
            sub = sub.sort_values("accuracy_mean", ascending=True)

        y = np.arange(len(sub))

        # Plot each region separately so Japan can be highlighted in red.
        for idx, (_, row) in enumerate(sub.iterrows()):
            is_japan = str(row["region"]).lower() == "japan"
            point_color = COLOR_RED if is_japan else COLOR_BLUE
            err_color = COLOR_RED_ERR if is_japan else COLOR_BLUE_ERR

            ax.errorbar(
                row["accuracy_mean"],
                idx,
                xerr=row["accuracy_std"],
                fmt="o",
                markersize=4.2,
                capsize=2.5,
                elinewidth=0.9,
                markeredgewidth=0.7,
                color=point_color,
                ecolor=err_color,
                zorder=4 if is_japan else 3,
            )

        # Keep the baseline line, remove the text annotation.
        ax.axvline(
            RANDOM_BASELINE,
            color=COLOR_BASELINE,
            linestyle="--",
            linewidth=0.9,
            zorder=2,
        )

        ax.set_yticks(y)
        ax.set_yticklabels(sub["region"].tolist())
        ax.set_xlim(*FIG2_XLIM)
        ax.set_xticks(FIG2_XTICKS)
        ax.set_xticklabels([f"{v:.1f}" for v in FIG2_XTICKS])
        ax.set_xlabel("Accuracy")

        ax.set_title(
            rf"{panel} Threshold: $M_w$={threshold:.1f}",
            fontsize=LABEL_SIZE,
            pad=4,
        )

        ax.grid(axis="x", color=COLOR_GRID, linestyle="--", linewidth=0.45, alpha=0.8)
        ax.grid(axis="y", visible=False)

        style_axes_common(ax)

    fig.tight_layout(w_pad=1.2)
    save_figure(fig, "Figure2_RF_multiregion_accuracy")
    plt.close(fig)


# =============================================================================
# 6. Figure 3
# =============================================================================

def summarize_feature_importance_for_threshold(
    df_fi_raw: pd.DataFrame,
    threshold: int | str,
) -> pd.DataFrame:
    """
    threshold:
        4 / 5 / 6 / "all"

    For threshold-specific panels:
        each region contributes one importance_mean per feature.

    For all-threshold panel:
        each region-threshold experiment contributes one importance_mean per feature.
    """
    if str(threshold).lower() == "all":
        sub = df_fi_raw.copy()
    else:
        sub = df_fi_raw[df_fi_raw["threshold"] == int(threshold)].copy()

    rows = []
    for feature in FEATURE_ORDER:
        vals = sub.loc[sub["feature"] == feature, "importance_mean"]
        vals = pd.to_numeric(vals, errors="coerce").dropna()

        if len(vals) == 0:
            mean = np.nan
            std = np.nan
            n = 0
        elif len(vals) == 1:
            mean = float(vals.mean())
            std = 0.0
            n = 1
        else:
            mean = float(vals.mean())
            std = float(vals.std(ddof=1))
            n = int(len(vals))

        rows.append({
            "feature": feature,
            "feature_label": FEATURE_LABELS[feature],
            "importance_mean": mean,
            "importance_std": std,
            "n": n,
            "threshold": threshold,
        })

    return pd.DataFrame(rows)


def get_final_selected_mask(summary: pd.DataFrame) -> np.ndarray:
    """Mark the six characteristics retained as ConvLSTM inputs."""
    return summary["feature"].isin(FINAL_SELECTED_FEATURES).to_numpy(dtype=bool)


def add_feature_group_guides(ax: plt.Axes) -> None:
    """
    Add separators and category labels for Depth / Mechanism / Direction.

    Group labels are positioned using blended transform:
    x uses data coordinates, y uses axes coordinates.
    This keeps Depth / Mechanism / Direction at the same visual height in all panels.
    """
    ax.axvline(2.5, color="#BDBDBD", linestyle="--", linewidth=0.6, zorder=1)
    ax.axvline(4.5, color="#BDBDBD", linestyle="--", linewidth=0.6, zorder=1)

    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)

    for name, start, end in FEATURE_GROUPS:
        x_mid = (start + end) / 2.0
        ax.text(
            x_mid,
            0.96,
            name,
            transform=trans,
            ha="center",
            va="top",
            fontsize=6.5,
            color=COLOR_GROUP_TEXT,
            zorder=5,
        )


def plot_feature_panel(
    ax: plt.Axes,
    summary: pd.DataFrame,
    title: str,
    panel: str,
    highlight_final_selected: bool = False,
) -> None:
    x = np.arange(len(FEATURE_ORDER))
    y = summary["importance_mean"].to_numpy()
    yerr = summary["importance_std"].to_numpy()

    highlight_mask = (
        get_final_selected_mask(summary)
        if highlight_final_selected
        else np.zeros(len(summary), dtype=bool)
    )

    for i in range(len(summary)):
        is_highlight = bool(highlight_mask[i])
        point_color = COLOR_RED if is_highlight else COLOR_BLUE
        err_color = COLOR_RED_ERR if is_highlight else COLOR_BLUE_ERR

        ax.errorbar(
            x[i],
            y[i],
            yerr=yerr[i],
            fmt="o",
            markersize=4.0,
            capsize=2.4,
            elinewidth=0.9,
            color=point_color,
            ecolor=err_color,
            zorder=4 if is_highlight else 3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(summary["feature_label"].tolist(), rotation=0)
    ax.set_ylabel(FIG3_YLABEL)
    ax.set_title(f"{panel} {title}", fontsize=LABEL_SIZE, pad=4)

    ax.grid(axis="y", color=COLOR_GRID, linestyle="--", linewidth=0.45, alpha=0.8)
    ax.grid(axis="x", visible=False)

    add_feature_group_guides(ax)
    style_axes_common(ax)


def compute_common_fig3_ylim(summaries: list[pd.DataFrame]) -> tuple[float, float]:
    if FIG3_YLIM is not None:
        return FIG3_YLIM

    max_vals = []
    for summary in summaries:
        y = summary["importance_mean"].to_numpy(dtype=float)
        yerr = summary["importance_std"].to_numpy(dtype=float)
        max_vals.append(np.nanmax(y + np.nan_to_num(yerr, nan=0.0)))

    max_y = np.nanmax(max_vals)
    if not np.isfinite(max_y) or max_y <= 0:
        max_y = 0.1

    return 0.0, max_y * 1.28 + 1e-4


def plot_figure3(df_fi_raw: pd.DataFrame) -> None:
    apply_style()

    panels = [
        (4, FIG3_PANEL_TITLES[4], "(a)", False),
        (5, FIG3_PANEL_TITLES[5], "(b)", False),
        (6, FIG3_PANEL_TITLES[6], "(c)", False),
        ("all", FIG3_PANEL_TITLES["all"], "(d)", True),
    ]

    summaries = [
        summarize_feature_importance_for_threshold(df_fi_raw, threshold)
        for threshold, _, _, _ in panels
    ]

    common_ylim = compute_common_fig3_ylim(summaries)

    fig, axes = plt.subplots(
        2, 2,
        figsize=(cm_to_inch(FIG3_WIDTH_CM), cm_to_inch(FIG3_HEIGHT_CM)),
        sharey=True,
    )
    axes = axes.ravel()

    all_summaries = []

    for ax, summary, (threshold, title, panel, highlight) in zip(axes, summaries, panels):
        ax.set_ylim(*common_ylim)
        plot_feature_panel(
            ax=ax,
            summary=summary,
            title=title,
            panel=panel,
            highlight_final_selected=highlight,
        )
        all_summaries.append(summary)

    fig.tight_layout(w_pad=1.2, h_pad=1.4)
    save_figure(fig, "Figure3_RF_feature_importance")
    plt.close(fig)

    pd.concat(all_summaries, ignore_index=True).to_csv(
        OUT_DIR / "rf_feature_importance_summary_for_fig3.csv",
        index=False,
        encoding="utf-8-sig",
    )


# =============================================================================
# 7. Main
# =============================================================================

def main() -> None:
    print("========== Collect RF results ==========")
    print(f"[ACCURACY_SUMMARY_CSV] {ACCURACY_SUMMARY_CSV}")
    print(f"[IMPORTANCE_SUMMARY_CSV] {IMPORTANCE_SUMMARY_CSV}")
    print(f"[OUT_DIR]  {OUT_DIR}")
    print(f"[DPI] {DPI}")

    df_acc, df_fi_raw = collect_results()

    print("\n========== Collected accuracy results ==========")
    print(df_acc[["region_code", "region", "threshold", "accuracy_mean", "accuracy_std"]])

    print("\n========== Draw Figure 2 ==========")
    plot_figure2(df_acc)

    print("\n========== Draw Figure 3 ==========")
    plot_figure3(df_fi_raw)

    print("\n[DONE]")


if __name__ == "__main__":
    main()
