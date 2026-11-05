# -*- coding: utf-8 -*-
"""
04a_rf_full_and_stability.py
---------------------------------
用途：
    针对“单地区 + 单阈值”下的已标注数据集，进行：
    1) 全特征输入的随机森林分类实验
    2) 稳健性实验（通过手动调整参数后重复运行）
    3) 自动保存本次实验详细结果
    4) 自动将“参数设置 + 结果摘要”追加到同一个汇总表中

说明：
    - 本脚本不做单特征或分组特征实验
    - 你每次手动修改参数区后运行一次
    - 每次运行结果会：
        a) 保存到本次实验子目录
        b) 追加到总汇总表 full_feature_stability_summary.csv
"""

from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import train_test_split


# =========================================================
# 1. 参数区：你每次主要手动修改这里
# =========================================================

# ---------- 输入数据 ----------
INPUT_CSV = Path(
    r"D:\World_force\Project_stress\idea_final\02_characteristic\jp\events_metrics_jp_Mw_alltime_Mc0_label5.csv"
)

# ---------- 总输出目录 ----------
OUT_DIR = Path(
    r"D:\World_force\Project_stress\idea_final\02_characteristic\jp\full_and_stability_noact"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- 本次实验名称 ----------
# 建议每次修改，便于总表和子目录中区分
EXPERIMENT_NAME = "rf_baseline_label5_depth8_trees100_leaf5_test05"

# ---------- 实验分组 ----------
# 便于你后续筛选：如 rf_param / test_ratio / balance / shuffle / baseline
EXPERIMENT_GROUP = "baseline"

# ---------- 标签列 ----------
LABEL_COL = "MAG_LABEL"

# ---------- 全特征列 ----------
# 必须与当前 02 / 03a / 03b 的输出保持一致
FEATURE_COLS = [
    "MEAN_LOCAL_DEPTH",
    "MEDIAN_LOCAL_DEPTH",
    "STD_LOCAL_DEPTH",
    "STD_LOCAL",
    "CONSISTENCY_R_MEAN",
    "MAX_SPREAD_100KM_DEG",
    "MAX_ROTATION_DEG",
    "REGIME_ENTROPY",
    "REGIME_GINI",
]

# ---------- 样本与实验设置 ----------
BALANCE = True
SHUFFLE_LABELS = False
TEST_RATIO = 0.5
N_REPEATS = 1000

# ---------- Random Forest 参数 ----------
RF_PARAMS = {
    "n_estimators": 100,
    "max_depth": 8,
    "min_samples_leaf": 5,
    "n_jobs": -1,
    "random_state": None,   # 每次重复实验内部会替换为 i
}

# ---------- 备注 ----------
NOTES = "baseline setting for full-feature experiment"


# =========================================================
# 2. 工具函数
# =========================================================
def try_read_csv(path: Path) -> pd.DataFrame:
    """尝试多种编码读取 CSV。"""
    for enc in ["utf-8-sig", "utf-8", "gbk", "cp936", "latin1"]:
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except Exception:
            continue
    raise RuntimeError(f"CSV 读取失败：{path}")


def sample_balanced(df: pd.DataFrame, label_col: str, random_state: int | None = None) -> pd.DataFrame:
    """
    对正负样本平衡抽样：
    - 各抽取 min(n_pos, n_neg) 个
    - 再整体打乱
    """
    df_pos = df[df[label_col] == 1]
    df_neg = df[df[label_col] == 0]

    n_pos = len(df_pos)
    n_neg = len(df_neg)
    n = min(n_pos, n_neg)

    if n == 0:
        raise ValueError("正类或负类样本数为 0，无法进行平衡抽样。")

    df_balanced = pd.concat([
        df_pos.sample(n=n, replace=False, random_state=random_state),
        df_neg.sample(n=n, replace=False, random_state=random_state),
    ], axis=0)

    df_balanced = df_balanced.sample(
        frac=1.0, replace=False, random_state=random_state
    ).reset_index(drop=True)

    return df_balanced


def safe_roc_auc(y_true, y_score) -> float:
    """安全计算 ROC-AUC。"""
    try:
        return roc_auc_score(y_true, y_score)
    except Exception:
        return np.nan


def safe_pr_auc(y_true, y_score) -> float:
    """安全计算 PR-AUC。"""
    try:
        return average_precision_score(y_true, y_score)
    except Exception:
        return np.nan


def save_json(obj: dict, out_path: Path) -> None:
    """保存 json 配置文件。"""
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# =========================================================
# 3. 核心实验函数
# =========================================================
def run_repeated_rf_experiment(
    df_clean: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    balance: bool,
    shuffle_labels: bool,
    test_ratio: float,
    n_repeats: int,
    rf_params: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    在给定设置下重复运行 RF 分类实验。

    返回：
        df_scores        : 每次重复实验的指标
        df_importances   : 每次重复实验的特征重要性
    """
    df_exp_base = df_clean.copy()

    # 乱序标签（只在整个实验开始前做一次）
    if shuffle_labels:
        rng = np.random.RandomState(42)
        df_exp_base[label_col] = rng.permutation(df_exp_base[label_col].values)

    score_records = []
    importance_records = []

    for i in range(n_repeats):
        # 1) 平衡抽样 or 不平衡
        if balance:
            df_exp = sample_balanced(df_exp_base, label_col=label_col, random_state=i)
        else:
            df_exp = df_exp_base.copy()

        # 2) 构造 X / y
        X = df_exp[feature_cols].values
        y = df_exp[label_col].values

        # 3) 划分训练集 / 测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_ratio,
            stratify=y,
            random_state=i
        )

        # 4) 模型训练
        params = rf_params.copy()
        params["random_state"] = i

        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        # 5) 预测
        y_pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_proba = y_pred.astype(float)

        # 6) 指标
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = safe_roc_auc(y_test, y_proba)
        pr_auc = safe_pr_auc(y_test, y_proba)

        score_records.append({
            "repeat": i,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1": f1,
            "ROC_AUC": roc_auc,
            "PR_AUC": pr_auc,
        })

        importance_records.append(model.feature_importances_)

        if (i + 1) % 20 == 0 or i == n_repeats - 1:
            print(f"    - 已完成 {i + 1}/{n_repeats} 次")

    df_scores = pd.DataFrame(score_records)
    df_importances = pd.DataFrame(importance_records, columns=feature_cols)

    return df_scores, df_importances


# =========================================================
# 4. 汇总函数
# =========================================================
def summarize_metrics(df_scores: pd.DataFrame) -> pd.DataFrame:
    """计算各指标的 mean/std。"""
    metrics = ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC", "PR_AUC"]
    mean_series = df_scores[metrics].mean()
    std_series = df_scores[metrics].std()

    summary_df = pd.DataFrame({
        "metric": metrics,
        "mean": mean_series.values,
        "std": std_series.values
    })
    return summary_df


def summarize_feature_importance(df_importances: pd.DataFrame) -> pd.DataFrame:
    """计算特征重要性的 mean/std，并按均值降序排列。"""
    fi_mean = df_importances.mean()
    fi_std = df_importances.std()

    fi_df = pd.DataFrame({
        "feature": fi_mean.index,
        "importance_mean": fi_mean.values,
        "importance_std": fi_std.values
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)

    return fi_df


def build_summary_row(
    input_csv: Path,
    experiment_name: str,
    experiment_group: str,
    notes: str,
    n_raw: int,
    n_clean: int,
    n_pos_clean: int,
    n_neg_clean: int,
    balance: bool,
    shuffle_labels: bool,
    test_ratio: float,
    n_repeats: int,
    rf_params: dict,
    metrics_summary: pd.DataFrame,
    fi_summary: pd.DataFrame,
) -> dict:
    """
    组装“本次实验的一行总表记录”。
    """
    metric_map = {}
    for _, row in metrics_summary.iterrows():
        metric_name = str(row["metric"]).lower()
        metric_map[f"{metric_name}_mean"] = float(row["mean"])
        metric_map[f"{metric_name}_std"] = float(row["std"])

    # 取前三特征
    top_features = fi_summary.head(3).copy()

    summary_row = {
        "experiment_name": experiment_name,
        "experiment_group": experiment_group,
        "notes": notes,
        "input_csv": str(input_csv),
        "n_raw": int(n_raw),
        "n_clean": int(n_clean),
        "n_pos_clean": int(n_pos_clean),
        "n_neg_clean": int(n_neg_clean),
        "n_features": len(FEATURE_COLS),
        "feature_set_name": "all_features",
        "feature_cols": "|".join(FEATURE_COLS),

        # 参数设置
        "balance": bool(balance),
        "shuffle_labels": bool(shuffle_labels),
        "test_ratio": float(test_ratio),
        "n_repeats": int(n_repeats),
        "rf_n_estimators": rf_params.get("n_estimators", np.nan),
        "rf_max_depth": rf_params.get("max_depth", np.nan),
        "rf_min_samples_leaf": rf_params.get("min_samples_leaf", np.nan),
        "rf_min_samples_split": rf_params.get("min_samples_split", np.nan),
        "rf_max_features": rf_params.get("max_features", np.nan),
        "rf_bootstrap": rf_params.get("bootstrap", np.nan),

        # top-3 特征
        "top1_feature": np.nan,
        "top1_importance_mean": np.nan,
        "top1_importance_std": np.nan,
        "top2_feature": np.nan,
        "top2_importance_mean": np.nan,
        "top2_importance_std": np.nan,
        "top3_feature": np.nan,
        "top3_importance_mean": np.nan,
        "top3_importance_std": np.nan,
    }

    for rank in range(min(3, len(top_features))):
        summary_row[f"top{rank+1}_feature"] = str(top_features.iloc[rank]["feature"])
        summary_row[f"top{rank+1}_importance_mean"] = float(top_features.iloc[rank]["importance_mean"])
        summary_row[f"top{rank+1}_importance_std"] = float(top_features.iloc[rank]["importance_std"])

    # 指标 mean/std
    summary_row.update(metric_map)

    return summary_row


def append_summary_row(summary_csv: Path, summary_row: dict) -> None:
    """
    将本次实验的一行摘要追加到总表：
    - 若总表不存在，则新建
    - 若存在，则读取后追加
    """
    row_df = pd.DataFrame([summary_row])

    if summary_csv.exists():
        old_df = pd.read_csv(summary_csv, encoding="utf-8-sig")
        new_df = pd.concat([old_df, row_df], axis=0, ignore_index=True)
    else:
        new_df = row_df.copy()

    new_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")


# =========================================================
# 5. 绘图函数
# =========================================================
def plot_metrics_summary(metrics_summary: pd.DataFrame, out_path: Path) -> None:
    """绘制指标 mean ± std 柱状图。"""
    metrics = metrics_summary["metric"].tolist()
    means = metrics_summary["mean"].values
    stds = metrics_summary["std"].values

    plt.figure(figsize=(8, 5))
    x = np.arange(len(metrics))
    plt.bar(x, means, yerr=stds, capsize=4)
    plt.xticks(x, metrics, rotation=30)
    plt.ylabel("Score")
    plt.title("Classification Metrics (mean ± std)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_feature_importance_summary(fi_summary: pd.DataFrame, out_path: Path) -> None:
    """绘制特征重要性 mean ± std 图。"""
    features = fi_summary["feature"].values
    means = fi_summary["importance_mean"].values
    stds = fi_summary["importance_std"].values

    plt.figure(figsize=(10, 6))
    x = np.arange(len(features))
    plt.errorbar(
        x, means, yerr=stds,
        fmt='o', ecolor='lightcoral', elinewidth=1.5,
        capsize=3, markersize=6, color='red', label="Feature importance"
    )
    plt.xticks(x, features, rotation=45, ha="right")
    plt.ylabel("Importance")
    plt.title("Feature Importance (mean ± std)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


# =========================================================
# 6. 主流程
# =========================================================
def main():
    print(f"[*] 读取数据：{INPUT_CSV}")
    df = try_read_csv(INPUT_CSV)

    # -----------------------------
    # 检查必要列
    # -----------------------------
    required_cols = [LABEL_COL] + FEATURE_COLS
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"输入文件缺少必要列：{missing_cols}")

    # -----------------------------
    # 数值化标签与特征
    # -----------------------------
    df[LABEL_COL] = pd.to_numeric(df[LABEL_COL], errors="coerce")
    for col in FEATURE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # -----------------------------
    # 原始样本统计
    # -----------------------------
    n_raw = len(df)

    # -----------------------------
    # 清洗：仅保留标签 + 全特征完整样本
    # -----------------------------
    df_clean = df.dropna(subset=required_cols).copy()
    df_clean = df_clean[df_clean[LABEL_COL].isin([0, 1])].copy()
    df_clean[LABEL_COL] = df_clean[LABEL_COL].astype(int)

    n_clean = len(df_clean)
    n_pos_clean = int((df_clean[LABEL_COL] == 1).sum())
    n_neg_clean = int((df_clean[LABEL_COL] == 0).sum())

    print(f"[*] 原始样本数：{n_raw}")
    print(f"[*] 清洗后样本数：{n_clean} (正类={n_pos_clean}, 负类={n_neg_clean})")

    if n_clean == 0:
        raise ValueError("清洗后数据为空，请检查标签列或特征列缺失情况。")
    if n_pos_clean == 0 or n_neg_clean == 0:
        raise ValueError("清洗后正类或负类样本数为 0，无法进行分类训练。")

    # -----------------------------
    # 当前实验输出目录
    # -----------------------------
    exp_dir = OUT_DIR / EXPERIMENT_NAME
    exp_dir.mkdir(parents=True, exist_ok=True)

    # 保存当前实验配置
    config_dict = {
        "INPUT_CSV": str(INPUT_CSV),
        "OUT_DIR": str(OUT_DIR),
        "EXPERIMENT_NAME": EXPERIMENT_NAME,
        "EXPERIMENT_GROUP": EXPERIMENT_GROUP,
        "LABEL_COL": LABEL_COL,
        "FEATURE_COLS": FEATURE_COLS,
        "BALANCE": BALANCE,
        "SHUFFLE_LABELS": SHUFFLE_LABELS,
        "TEST_RATIO": TEST_RATIO,
        "N_REPEATS": N_REPEATS,
        "RF_PARAMS": RF_PARAMS,
        "NOTES": NOTES,
    }
    save_json(config_dict, exp_dir / "experiment_config.json")

    # -----------------------------
    # 运行重复实验
    # -----------------------------
    print(f"[*] 开始实验：{EXPERIMENT_NAME}")
    print(f"[*] 设置：BALANCE={BALANCE}, SHUFFLE_LABELS={SHUFFLE_LABELS}, TEST_RATIO={TEST_RATIO}")
    print(f"[*] RF_PARAMS={RF_PARAMS}")
    print(f"[*] N_REPEATS={N_REPEATS}")

    df_scores, df_importances = run_repeated_rf_experiment(
        df_clean=df_clean,
        feature_cols=FEATURE_COLS,
        label_col=LABEL_COL,
        balance=BALANCE,
        shuffle_labels=SHUFFLE_LABELS,
        test_ratio=TEST_RATIO,
        n_repeats=N_REPEATS,
        rf_params=RF_PARAMS,
    )

    # -----------------------------
    # 保存每次重复实验的原始结果
    # -----------------------------
    df_scores.to_csv(exp_dir / "scores_per_repeat.csv", index=False, encoding="utf-8-sig")
    df_importances.to_csv(exp_dir / "feature_importances_per_repeat.csv", index=False, encoding="utf-8-sig")

    # -----------------------------
    # 汇总
    # -----------------------------
    metrics_summary = summarize_metrics(df_scores)
    fi_summary = summarize_feature_importance(df_importances)

    metrics_summary.to_csv(exp_dir / "metrics_summary.csv", index=False, encoding="utf-8-sig")
    fi_summary.to_csv(exp_dir / "feature_importance_summary.csv", index=False, encoding="utf-8-sig")

    # 打印
    print("[*] 指标汇总（mean ± std）：")
    for _, row in metrics_summary.iterrows():
        print(f"    {row['metric']}: {row['mean']:.4f} ({row['std']:.4f})")

    print("[*] Top-3 特征：")
    for i in range(min(3, len(fi_summary))):
        row = fi_summary.iloc[i]
        print(f"    Top{i+1}: {row['feature']} -> {row['importance_mean']:.4f} ({row['importance_std']:.4f})")

    # -----------------------------
    # 画图
    # -----------------------------
    plot_metrics_summary(metrics_summary, exp_dir / "metrics_mean_std.png")
    plot_feature_importance_summary(fi_summary, exp_dir / "feature_importance_mean_std.png")

    # -----------------------------
    # 组装一行总表记录并追加
    # -----------------------------
    summary_row = build_summary_row(
        input_csv=INPUT_CSV,
        experiment_name=EXPERIMENT_NAME,
        experiment_group=EXPERIMENT_GROUP,
        notes=NOTES,
        n_raw=n_raw,
        n_clean=n_clean,
        n_pos_clean=n_pos_clean,
        n_neg_clean=n_neg_clean,
        balance=BALANCE,
        shuffle_labels=SHUFFLE_LABELS,
        test_ratio=TEST_RATIO,
        n_repeats=N_REPEATS,
        rf_params=RF_PARAMS,
        metrics_summary=metrics_summary,
        fi_summary=fi_summary,
    )

    summary_csv = OUT_DIR / "full_feature_stability_summary.csv"
    append_summary_row(summary_csv, summary_row)

    print(f"[*] 本次实验详细结果已保存至：{exp_dir}")
    print(f"[*] 总汇总表已更新：{summary_csv}")


if __name__ == "__main__":
    main()