#!/usr/bin/env python3
"""Replay the archived nine-feature RF procedure on rebuilt WSM event tables.

The archived order is preserved: balance the full event table, stratify a
50/50 train/test split, then fit a random forest.  The same integer is used
for the three random states in each repeat, as in 04a_rf_full_and_stability.py.
This script does not reinterpret the resulting scores as future forecasts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


SOURCE_DIR = Path(
    "/Users/xiazongyuan/1 课题/7 地震震级预测/"
    "xzy-小论文1-文件汇总/代码与结果（归档）/"
    "九项指标重构归档/datasets/rf_wsm"
)
ARCHIVED_SCRIPT = Path(
    "/Users/xiazongyuan/1 课题/7 地震震级预测/"
    "xzy-小论文1-文件汇总/代码与结果（归档）/"
    "第一阶段-RF模型/04a_rf_full_and_stability.py"
)
FEATURES = [
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
SCORES = ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC", "PR_AUC"]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def id_sha256(values: pd.Series) -> str:
    joined = "\n".join(sorted(values.astype(str)))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def validate_table(table: pd.DataFrame, period: str) -> dict:
    required = ["region", "WSM_ID", "ISC_ID", "MW", "all_nine_valid", *FEATURES]
    required += [f"label_ge_{n}" for n in (4, 5, 6)]
    absent = sorted(set(required) - set(table.columns))
    if absent:
        raise ValueError(f"{period}: missing columns {absent}")
    if table[required].isna().any().any():
        raise ValueError(f"{period}: missing values in required columns")
    features = table[FEATURES].to_numpy(dtype=float)
    if not np.isfinite(features).all():
        raise ValueError(f"{period}: non-finite feature values")
    if not np.isfinite(pd.to_numeric(table["MW"]).to_numpy(dtype=float)).all():
        raise ValueError(f"{period}: non-finite converted magnitude")
    if not table["all_nine_valid"].astype(bool).all():
        raise ValueError(f"{period}: a training row has an invalid indicator")
    for n in (4, 5, 6):
        label = pd.to_numeric(table[f"label_ge_{n}"], errors="raise")
        if not label.isin((0, 1)).all():
            raise ValueError(f"{period}: label_ge_{n} is not binary")
        if not (label.to_numpy(dtype=int) == (table["MW"].to_numpy() >= n)).all():
            raise ValueError(f"{period}: label_ge_{n} disagrees with MW")
    if table.duplicated(["region", "WSM_ID"]).any():
        raise ValueError(f"{period}: duplicate WSM_ID within a region")
    return {
        "period": period,
        "rows": len(table),
        "regions": int(table["region"].nunique()),
        "duplicate_isc_id_within_region": int(table.duplicated(["region", "ISC_ID"]).sum()),
        "duplicate_isc_id_overall": int(table.duplicated("ISC_ID").sum()),
    }


def audit_task_counts(table: pd.DataFrame, counts: pd.DataFrame, period: str) -> None:
    subset = counts[counts["period"] == period]
    for _, item in subset.iterrows():
        rows = table[table["region"] == item["region"]]
        label = rows[f"label_ge_{int(item['threshold'])}"].astype(int)
        observed = (len(rows), int(label.sum()), int((label == 0).sum()))
        expected = tuple(int(item[col]) for col in ("usable_events", "positive", "negative"))
        if observed != expected:
            raise ValueError(
                f"{period} {item['region']} Mw>={item['threshold']}: "
                f"table counts {observed}, formal counts {expected}"
            )


def one_repeat(table: pd.DataFrame, label_col: str, seed: int) -> tuple[dict, dict, dict, pd.DataFrame]:
    """Match archived 04a sampling, splitting, and RF parameters."""
    positive = table[table[label_col] == 1]
    negative = table[table[label_col] == 0]
    n = min(len(positive), len(negative))
    balanced = pd.concat(
        [
            positive.sample(n=n, replace=False, random_state=seed),
            negative.sample(n=n, replace=False, random_state=seed),
        ],
        axis=0,
    ).sample(frac=1.0, replace=False, random_state=seed).reset_index(drop=True)
    y = balanced[label_col].to_numpy(dtype=int)
    indices = np.arange(len(balanced))
    train, test = train_test_split(
        indices, test_size=0.5, stratify=y, random_state=seed
    )
    x = balanced[FEATURES].to_numpy(dtype=float)
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=seed,
    )
    model.fit(x[train], y[train])
    pred = model.predict(x[test])
    probability = model.predict_proba(x[test])[:, 1]
    score = {
        "seed": seed,
        "Accuracy": accuracy_score(y[test], pred),
        "Precision": precision_score(y[test], pred, zero_division=0),
        "Recall": recall_score(y[test], pred, zero_division=0),
        "F1": f1_score(y[test], pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y[test], probability),
        "PR_AUC": average_precision_score(y[test], probability),
    }
    importance = {"seed": seed, **dict(zip(FEATURES, model.feature_importances_))}
    splits = {
        "seed": seed,
        "n_selected": len(balanced),
        "n_train": len(train),
        "n_test": len(test),
        "n_positive_train": int(y[train].sum()),
        "n_positive_test": int(y[test].sum()),
        "selected_wsm_ids_sha256": id_sha256(balanced["WSM_ID"]),
        "train_wsm_ids_sha256": id_sha256(balanced.iloc[train]["WSM_ID"]),
        "test_wsm_ids_sha256": id_sha256(balanced.iloc[test]["WSM_ID"]),
    }
    seed_zero_membership = pd.DataFrame()
    if seed == 0:
        seed_zero_membership = balanced[["WSM_ID", "ISC_ID", "MW", label_col]].copy()
        seed_zero_membership["set"] = ""
        seed_zero_membership.loc[train, "set"] = "train"
        seed_zero_membership.loc[test, "set"] = "test"
    return score, importance, splits, seed_zero_membership


def summarize(scores: pd.DataFrame, importance: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = pd.DataFrame({
        "metric": SCORES,
        "mean": scores[SCORES].mean().to_numpy(),
        "std": scores[SCORES].std().to_numpy(),
        "min": scores[SCORES].min().to_numpy(),
        "max": scores[SCORES].max().to_numpy(),
    })
    features = pd.DataFrame({
        "feature": FEATURES,
        "importance_mean": importance[FEATURES].mean().to_numpy(),
        "importance_std": importance[FEATURES].std().to_numpy(),
    }).sort_values("importance_mean", ascending=False)
    return metrics, features


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=SOURCE_DIR)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).parent / "results")
    parser.add_argument("--repeats", type=int, default=1000)
    parser.add_argument("--period", choices=["both", "1978_2008", "1978_2024"], default="both")
    parser.add_argument("--region", action="append", help="Limit to a named region; repeatable")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    count_path = args.source_dir / "rf_formal_task_counts.csv"
    count_table = pd.read_csv(count_path)
    periods = ["1978_2008", "1978_2024"] if args.period == "both" else [args.period]
    metadata = {
        "source_dir": str(args.source_dir),
        "source_hashes_sha256": {count_path.name: file_sha256(count_path)},
        "archived_reference": str(ARCHIVED_SCRIPT),
        "archived_reference_sha256": file_sha256(ARCHIVED_SCRIPT),
        "repeats": args.repeats,
        "features": FEATURES,
        "balance_before_split": True,
        "test_ratio": 0.5,
        "stratify": True,
        "same_seed_for_balance_split_forest": True,
        "forest": {"n_estimators": 100, "max_depth": 8, "min_samples_leaf": 5, "n_jobs": -1},
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
        "table_audits": [],
    }
    task_rows = []
    started = time.monotonic()
    for period in periods:
        path = args.source_dir / f"rf_training_{period}.csv"
        table = pd.read_csv(path)
        metadata["source_hashes_sha256"][path.name] = file_sha256(path)
        metadata["table_audits"].append(validate_table(table, period))
        audit_task_counts(table, count_table, period)
        for _, spec in count_table[count_table["period"] == period].iterrows():
            region = str(spec["region"])
            if args.region and region not in args.region:
                continue
            threshold = int(spec["threshold"])
            label_col = f"label_ge_{threshold}"
            task = table[table["region"] == region].copy()
            positive = int(task[label_col].sum())
            negative = len(task) - positive
            row = {
                "period": period, "region": region, "threshold": threshold,
                "n_total": len(task), "n_positive": positive, "n_negative": negative,
                "min_class_count": min(positive, negative),
            }
            if min(positive, negative) < 2:
                row.update(status="skipped", reason="fewer than two events in one class")
                task_rows.append(row)
                print(f"SKIP {period} {region} Mw>={threshold}: {positive} positive, {negative} negative", flush=True)
                continue
            name = f"{period}__{region}__ge{threshold}"
            directory = args.out_dir / name
            directory.mkdir(parents=True, exist_ok=True)
            scores, importances, splits = [], [], []
            print(f"RUN  {name}: {positive} positive, {negative} negative, {args.repeats} repeat(s)", flush=True)
            for seed in range(args.repeats):
                score, importance, split, membership = one_repeat(task, label_col, seed)
                scores.append(score)
                importances.append(importance)
                splits.append(split)
                if seed == 0:
                    membership.to_csv(directory / "seed_0_membership.csv", index=False)
                if (seed + 1) % 20 == 0 or seed + 1 == args.repeats:
                    pd.DataFrame(scores).to_csv(directory / "scores_per_repeat.csv", index=False)
                    pd.DataFrame(importances).to_csv(directory / "feature_importances_per_repeat.csv", index=False)
                    pd.DataFrame(splits).to_csv(directory / "split_audit.csv", index=False)
                    print(f"  {name}: {seed + 1}/{args.repeats}", flush=True)
            metrics, features = summarize(pd.DataFrame(scores), pd.DataFrame(importances))
            metrics.to_csv(directory / "metrics_summary.csv", index=False)
            features.to_csv(directory / "feature_importance_summary.csv", index=False)
            row.update(status="completed", reason="", n_repeats=args.repeats)
            for _, metric in metrics.iterrows():
                row[f"{metric['metric']}_mean"] = metric["mean"]
                row[f"{metric['metric']}_std"] = metric["std"]
            for rank, (_, feature) in enumerate(features.head(3).iterrows(), start=1):
                row[f"top{rank}_feature"] = feature["feature"]
                row[f"top{rank}_importance_mean"] = feature["importance_mean"]
            task_rows.append(row)
            pd.DataFrame(task_rows).to_csv(args.out_dir / "task_summary.csv", index=False)
    metadata["duration_seconds"] = time.monotonic() - started
    (args.out_dir / "run_config.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(task_rows).to_csv(args.out_dir / "task_summary.csv", index=False)
    print(f"DONE {len(task_rows)} tasks in {metadata['duration_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
