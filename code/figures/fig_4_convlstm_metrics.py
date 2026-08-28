# -*- coding: utf-8 -*-
"""
Figure 4:
融合方案——横向排序综合指标图

图形设计
--------
- 横轴：Metric value (%)
- 纵轴：模型方案，按 CSI 均值从高到低排序，高值在上、低值在下
- CSI：由白到蓝色的渐变横向柱状图 + 横向误差棒
- Precision：黑色圆点 + 折线 + 横向误差棒
- Recall：红色方点 + 折线 + 横向误差棒
- 具体模型标签：放在 CSI 柱状图内部
- 右侧标注：相对于 M1 的增益（pp）
- 不显示图名

输出
----
Figure4_fused_ranked.png
Figure4_fused_ranked.pdf
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D


# =============================================================================
# 1. 路径与基础设置
# =============================================================================

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_CSV = REPO_ROOT / "results" / "convlstm" / "summary_metrics.csv"
OUT_DIR = REPO_ROOT / "results" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DPI = 600
FONT_FAMILY = "Times New Roman"
FONT_SIZE = 18

# 颜色
COLOR_CSI = "#4E79A7"          # CSI 蓝色
COLOR_CSI_ERR = "#2F5D8A"      # CSI 误差棒
COLOR_PREC = "#B22222"         # Precision 红色
COLOR_RECALL = "#6E6E6E"       # Recall 灰色，与 reference 一致
COLOR_GRID = "#D0D0D0"
COLOR_SPINE = "#4D4D4D"
COLOR_TEXT = "#2F2F2F"
COLOR_M1_BG = "#F3F3F3"        # M1 行淡灰底色

# CSI 渐变色图：白色 -> 蓝色
CSI_CMAP = LinearSegmentedColormap.from_list(
    "csi_grad",
    ["#FFFFFF", COLOR_CSI]
)



# =============================================================================
# 2. Table 2 source data
# =============================================================================

models = np.array(["M1", "M2", "M3", "M4", "M5"])

display_labels = np.array([
    "MAG",
    "MAG + Depth",
    r"MAG + $S_{\mathrm{Hmax}}$ orientation",
    "MAG + Stress regime",
    "MAG + Depth + Stress regime"
])


if not DATA_CSV.exists():
    raise FileNotFoundError(f"Missing Figure 4 source table: {DATA_CSV}")
source = pd.read_csv(DATA_CSV).set_index("model").loc[models]
precision_mean = source["precision_mean"].to_numpy(dtype=float) * 100.0
precision_std = source["precision_std"].to_numpy(dtype=float) * 100.0
recall_mean = source["recall_mean"].to_numpy(dtype=float) * 100.0
recall_std = source["recall_std"].to_numpy(dtype=float) * 100.0
csi_mean = source["CSI_mean"].to_numpy(dtype=float) * 100.0
csi_std = source["CSI_std"].to_numpy(dtype=float) * 100.0
delta_csi = source["delta_CSI"].to_numpy(dtype=float) * 100.0
gain_labels = np.array(["baseline" if model == "M1" else f"{gain:+.2f} pp" for model, gain in zip(models, delta_csi)])


# =============================================================================
# 3. 绘图风格
# =============================================================================

mpl.rcParams.update({
    "font.family": FONT_FAMILY,
    "font.size": FONT_SIZE,
    "axes.labelsize": FONT_SIZE,
    "axes.titlesize": FONT_SIZE,
    "xtick.labelsize": FONT_SIZE,
    "ytick.labelsize": FONT_SIZE,
    "legend.fontsize": FONT_SIZE,
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.unicode_minus": False,
})


def style_axes(ax):
    """统一坐标轴样式：仅保留左方和下方刻度值，刻度朝内。"""
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
        length=5.0,
        width=1.0,
    )

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_color(COLOR_SPINE)


def save_fig(fig, stem):
    png_path = OUT_DIR / f"{stem}.png"
    pdf_path = OUT_DIR / f"{stem}.pdf"

    fig.savefig(png_path, dpi=DPI, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")

    print(f"[SAVE] {png_path}")
    print(f"[SAVE] {pdf_path}")



def draw_gradient_barh(ax, left, y_center, width, height, cmap, zorder=2):
    """
    绘制单个横向渐变柱。
    """
    grad = np.linspace(0, 1, 256).reshape(1, -1)

    ax.imshow(
        grad,
        extent=[
            left,
            left + width,
            y_center - height / 2,
            y_center + height / 2,
        ],
        origin="lower",
        aspect="auto",
        cmap=cmap,
        interpolation="bicubic",
        zorder=zorder,
    )



def draw_xerr(ax, x, y, xerr, color, lw=1.4, cap=0.105, zorder=6):
    """
    统一绘制横向误差棒：
    - 水平误差线
    - 左右端竖向 cap
    这样 CSI、Precision、Recall 的误差棒样式完全一致。
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xerr = np.asarray(xerr, dtype=float)

    for xi, yi, ei in zip(x, y, xerr):
        if not np.isfinite(xi) or not np.isfinite(yi) or not np.isfinite(ei):
            continue

        x0 = xi - ei
        x1 = xi + ei

        ax.hlines(
            yi,
            x0,
            x1,
            color=color,
            linewidth=lw,
            zorder=zorder,
        )

        ax.vlines(
            [x0, x1],
            yi - cap,
            yi + cap,
            color=color,
            linewidth=lw,
            zorder=zorder,
        )


# =============================================================================
# 4. 绘图函数
# =============================================================================

def plot_figure4_fused():
    # -------------------------------------------------------------------------
    # 按 CSI 均值从高到低排序
    # 目标显示顺序：M4 → M2 → M5 → M3 → M1
    # -------------------------------------------------------------------------
    order = np.argsort(-csi_mean)

    models_s = models[order]
    display_labels_s = display_labels[order]

    precision_mean_s = precision_mean[order]
    precision_std_s = precision_std[order]

    recall_mean_s = recall_mean[order]
    recall_std_s = recall_std[order]

    csi_mean_s = csi_mean[order]
    csi_std_s = csi_std[order]

    gain_labels_s = gain_labels[order]

    # -------------------------------------------------------------------------
    # y 位置设置
    # 不使用 invert_yaxis，直接用递减 y 坐标保证第一个模型在最上方
    # -------------------------------------------------------------------------
    n = len(models_s)
    y = np.arange(n - 1, -1, -1)   # 4, 3, 2, 1, 0
    offset = 0.16
    bar_height = 0.48

    # 适配 18 号字体的图幅
    fig, ax = plt.subplots(figsize=(12.5, 7.2))

    # -------------------------------------------------------------------------
    # 高亮基线模型 M1 所在行
    # -------------------------------------------------------------------------
    idx_m1 = np.where(models_s == "M1")[0][0]
    y_m1 = y[idx_m1]
    ax.axhspan(
        y_m1 - 0.42,
        y_m1 + 0.42,
        color=COLOR_M1_BG,
        zorder=0,
    )

    # -------------------------------------------------------------------------
    # CSI 渐变横向柱状图：由白到蓝色，无外框
    # -------------------------------------------------------------------------
    for yi, csi_v in zip(y, csi_mean_s):
        draw_gradient_barh(
            ax=ax,
            left=0,
            y_center=yi,
            width=csi_v,
            height=bar_height,
            cmap=CSI_CMAP,
            zorder=2,
        )

    # CSI 横向误差棒：统一样式
    draw_xerr(
        ax=ax,
        x=csi_mean_s,
        y=y,
        xerr=csi_std_s,
        color=COLOR_CSI_ERR,
        lw=1.4,
        cap=0.105,
        zorder=6,
    )

    # -------------------------------------------------------------------------
    # Precision：黑色圆点 + 折线 + 横向误差棒
    # -------------------------------------------------------------------------
    y_prec = y + offset

    ax.plot(
        precision_mean_s,
        y_prec,
        color=COLOR_PREC,
        linewidth=1.5,
        zorder=4,
    )

    draw_xerr(
        ax=ax,
        x=precision_mean_s,
        y=y_prec,
        xerr=precision_std_s,
        color=COLOR_PREC,
        lw=1.4,
        cap=0.105,
        zorder=6,
    )

    ax.scatter(
        precision_mean_s,
        y_prec,
        s=95,
        marker="o",
        color=COLOR_PREC,
        edgecolors="none",
        zorder=7,
        label="Precision",
    )

    # -------------------------------------------------------------------------
    # Recall：红色方点 + 折线 + 横向误差棒
    # -------------------------------------------------------------------------
    y_rec = y - offset

    ax.plot(
        recall_mean_s,
        y_rec,
        color=COLOR_RECALL,
        linewidth=1.5,
        zorder=4,
    )

    draw_xerr(
        ax=ax,
        x=recall_mean_s,
        y=y_rec,
        xerr=recall_std_s,
        color=COLOR_RECALL,
        lw=1.4,
        cap=0.105,
        zorder=6,
    )

    ax.scatter(
        recall_mean_s,
        y_rec,
        s=90,
        marker="s",
        color=COLOR_RECALL,
        edgecolors="none",
        zorder=7,
        label="Recall",
    )

    # -------------------------------------------------------------------------
    # ΔCSI 标注
    # -------------------------------------------------------------------------
    for i, (csi_v, csi_e, txt, model) in enumerate(
        zip(csi_mean_s, csi_std_s, gain_labels_s, models_s)
    ):
        color = "#6E6E6E" if model == "M1" else COLOR_TEXT

        ax.text(
            csi_v + csi_e + 1.2,
            y[i],
            txt,
            ha="left",
            va="center",
            fontsize=FONT_SIZE,
            color=color,
            zorder=8,
        )

    # -------------------------------------------------------------------------
    # y 轴标签：只显示 M1-M5
    # -------------------------------------------------------------------------
    ax.set_yticks(y)
    ax.set_yticklabels(models_s, color="black")

    # -------------------------------------------------------------------------
    # 输入组合：放在 CSI 柱状图内部，黑色字
    # -------------------------------------------------------------------------
    for i, in_label in enumerate(display_labels_s):
        ax.text(
            1.6,
            y[i],
            in_label,
            ha="left",
            va="center",
            fontsize=FONT_SIZE,
            color="black",
            zorder=8,
        )

    # -------------------------------------------------------------------------
    # 横轴设置
    # -------------------------------------------------------------------------
    ax.set_xlim(0, 95)
    ax.set_xticks(np.arange(0, 96, 10))
    ax.set_xlabel("Metric value (%)")
    ax.set_ylabel("Model")

    # 固定 y 轴范围，确保高 CSI 模型在上、低 CSI 模型在下
    ax.set_ylim(-0.6, n - 0.4)

    # 网格线
    ax.grid(
        axis="x",
        linestyle="--",
        linewidth=0.6,
        color=COLOR_GRID,
        alpha=0.8,
        zorder=0,
    )
    ax.grid(axis="y", visible=False)

    # -------------------------------------------------------------------------
    # 图例：右上角
    # -------------------------------------------------------------------------
    csi_proxy = Line2D([0], [0], color=COLOR_CSI, linewidth=10)
    prec_proxy = Line2D(
        [0], [0],
        marker="o",
        color=COLOR_PREC,
        linestyle="-",
        markersize=8.5,
        linewidth=1.5,
    )
    rec_proxy = Line2D(
        [0], [0],
        marker="s",
        color=COLOR_RECALL,
        linestyle="-",
        markersize=8.0,
        linewidth=1.5,
    )

    ax.legend(
        [csi_proxy, prec_proxy, rec_proxy],
        ["CSI", "Precision", "Recall"],
        frameon=False,
        loc="upper right",
        handlelength=2.0,
        borderpad=0.2,
    )

    # 不显示图名
    # ax.set_title(...)

    style_axes(ax)
    fig.tight_layout()

    save_fig(fig, "Figure4_fused_ranked")
    plt.close(fig)


# =============================================================================
# 5. 主程序
# =============================================================================

if __name__ == "__main__":
    plot_figure4_fused()
    print("[DONE] Figure 4 fused ranked version has been generated.")
