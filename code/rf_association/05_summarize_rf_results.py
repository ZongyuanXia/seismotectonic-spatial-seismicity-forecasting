"""Create manuscript-ready summaries for the 23 formal RF tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from config import FEATURE_COLUMNS, FORMAL_RF_TASKS, RUNS_DIR, SUMMARIES_DIR, ensure_output_directories


def load_run(run_directory: Path) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    metadata_path = run_directory / "run_metadata.json"
    metrics_path = run_directory / "metrics_summary.csv"
    importance_path = run_directory / "feature_importance_summary.csv"
    if not all(path.exists() for path in (metadata_path, metrics_path, importance_path)):
        raise FileNotFoundError(f"Incomplete RF run: {run_directory}")
    return (
        json.loads(metadata_path.read_text(encoding="utf-8")),
        pd.read_csv(metrics_path, encoding="utf-8"),
        pd.read_csv(importance_path, encoding="utf-8"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    ensure_output_directories()
    run_directories = sorted(path.parent for path in RUNS_DIR.glob("*/mw_*/run_metadata.json"))
    if not run_directories:
        raise FileNotFoundError(f"No completed RF runs found in {RUNS_DIR}")

    formal_tasks = {(region, float(threshold)) for region, threshold in FORMAL_RF_TASKS}
    completed_tasks: set[tuple[str, float]] = set()
    accuracy_rows: list[dict] = []
    importance_rows: list[dict] = []
    for run_directory in run_directories:
        metadata_path = run_directory / "run_metadata.json"
        metadata_preview = json.loads(metadata_path.read_text(encoding="utf-8"))
        task = (
            str(metadata_preview["region"]),
            float(metadata_preview["threshold_mw"]),
        )
        if task not in formal_tasks:
            continue
        metadata, metrics, importance = load_run(run_directory)
        completed_tasks.add(task)
        accuracy = metrics.loc[metrics["metric"] == "accuracy"].iloc[0]
        accuracy_rows.append({
            "region": metadata["region"],
            "region_name": metadata["region_name"],
            "threshold_mw": metadata["threshold_mw"],
            "n_events": metadata["n_events"],
            "n_positive": metadata["n_positive"],
            "n_negative": metadata["n_negative"],
            "accuracy_mean": accuracy["mean"],
            "accuracy_std": accuracy["std"],
            "n_repeats": metadata["n_repeats"],
            "top_three_features": "; ".join(importance["feature"].head(3)),
        })
        for _, row in importance.iterrows():
            importance_rows.append({
                "region": metadata["region"],
                "region_name": metadata["region_name"],
                "threshold_mw": metadata["threshold_mw"],
                "feature": row["feature"],
                "importance_mean": row["importance_mean"],
                "importance_std": row["importance_std"],
            })

    missing_tasks = formal_tasks - completed_tasks
    if missing_tasks:
        missing_text = ", ".join(
            f"{region}:Mw{threshold:g}" for region, threshold in sorted(missing_tasks)
        )
        raise FileNotFoundError(f"Missing completed formal RF task(s): {missing_text}")

    accuracy_summary = pd.DataFrame(accuracy_rows).sort_values(by=["threshold_mw", "region"])
    importance_summary = pd.DataFrame(importance_rows).sort_values(
        by=["threshold_mw", "region", "importance_mean"], ascending=[True, True, False]
    )
    accuracy_summary.to_csv(SUMMARIES_DIR / "rf_accuracy_by_region_threshold.csv", index=False, encoding="utf-8")
    importance_summary.to_csv(SUMMARIES_DIR / "rf_importance_by_region_threshold.csv", index=False, encoding="utf-8")

    japan_summary = accuracy_summary.loc[accuracy_summary["region"] == "japan"].copy()
    japan_summary.to_csv(SUMMARIES_DIR / "japan_rf_summary.csv", index=False, encoding="utf-8")

    by_threshold = (
        importance_summary.groupby(["threshold_mw", "feature"], as_index=False)
        .agg(importance_mean=("importance_mean", "mean"), importance_std=("importance_mean", "std"), n_regions=("region", "nunique"))
    )
    by_threshold["feature"] = pd.Categorical(by_threshold["feature"], categories=list(FEATURE_COLUMNS), ordered=True)
    by_threshold = by_threshold.sort_values(by=["threshold_mw", "feature"])
    by_threshold.to_csv(SUMMARIES_DIR / "rf_importance_by_threshold.csv", index=False, encoding="utf-8")

    all_thresholds = (
        importance_summary.groupby("feature", as_index=False)
        .agg(importance_mean=("importance_mean", "mean"), importance_std=("importance_mean", "std"), n_region_threshold_tasks=("region", "size"))
    )
    all_thresholds["feature"] = pd.Categorical(all_thresholds["feature"], categories=list(FEATURE_COLUMNS), ordered=True)
    all_thresholds = all_thresholds.sort_values("feature")
    all_thresholds.to_csv(SUMMARIES_DIR / "rf_importance_all_thresholds.csv", index=False, encoding="utf-8")
    print(f"Saved RF summary tables to {SUMMARIES_DIR}")


if __name__ == "__main__":
    main()
