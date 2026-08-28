"""Run the repeated nine-feature random-forest association experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from config import (
    FEATURE_COLUMNS,
    FORMAL_RF_TASKS,
    LABELS_DIR,
    MAGNITUDE_THRESHOLDS,
    PROJECT_ROOT,
    REGIONS,
    RF_MAX_DEPTH,
    RF_MIN_SAMPLES_LEAF,
    RF_N_ESTIMATORS,
    RF_REPEATS,
    RF_TEST_SIZE,
    RUNS_DIR,
    ensure_output_directories,
)


def label_column(threshold: float) -> str:
    return f"LABEL_MW{int(threshold)}"


def balanced_sample(df: pd.DataFrame, label: str, seed: int) -> pd.DataFrame:
    positive = df.loc[df[label] == 1]
    negative = df.loc[df[label] == 0]
    sample_size = min(len(positive), len(negative))
    if sample_size < 2:
        raise ValueError("At least two positive and two negative samples are required.")
    sampled = pd.concat((
        positive.sample(n=sample_size, random_state=seed),
        negative.sample(n=sample_size, random_state=seed),
    ))
    return sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def clean_input(df: pd.DataFrame, label: str) -> pd.DataFrame:
    required = [label, *FEATURE_COLUMNS]
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"Input is missing required column(s): {', '.join(missing)}")
    clean = df.loc[:, required].copy()
    for column in required:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")
    clean = clean.dropna(subset=required)
    clean = clean.loc[clean[label].isin((0, 1))].copy()
    clean[label] = clean[label].astype(int)
    return clean


def run_experiment(clean: pd.DataFrame, label: str, n_repeats: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    scores: list[dict[str, float]] = []
    importances: list[np.ndarray] = []
    for repeat in range(n_repeats):
        sampled = balanced_sample(clean, label, repeat)
        x_train, x_test, y_train, y_test = train_test_split(
            sampled.loc[:, list(FEATURE_COLUMNS)], sampled[label], test_size=RF_TEST_SIZE,
            stratify=sampled[label], random_state=repeat,
        )
        model = RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS,
            max_depth=RF_MAX_DEPTH,
            min_samples_leaf=RF_MIN_SAMPLES_LEAF,
            n_jobs=-1,
            random_state=repeat,
        )
        model.fit(x_train, y_train)
        scores.append({"repeat": repeat, "accuracy": accuracy_score(y_test, model.predict(x_test))})
        importances.append(model.feature_importances_)
        if (repeat + 1) % 25 == 0 or repeat + 1 == n_repeats:
            print(f"  completed {repeat + 1}/{n_repeats} repeats")
    return pd.DataFrame(scores), pd.DataFrame(importances, columns=FEATURE_COLUMNS)


def write_run(region_key: str, threshold: float, source_path: Path, n_repeats: int, overwrite: bool) -> None:
    label = label_column(threshold)
    source = pd.read_csv(source_path, encoding="utf-8", low_memory=False)
    clean = clean_input(source, label)
    positive_count = int((clean[label] == 1).sum())
    negative_count = int((clean[label] == 0).sum())
    if positive_count == 0 or negative_count == 0:
        raise ValueError(f"{region_key}, Mw >= {threshold:g}: one class is empty after feature filtering.")

    output_dir = RUNS_DIR / region_key / f"mw_{int(threshold)}"
    if output_dir.exists() and not overwrite:
        raise FileExistsError(f"Run directory exists: {output_dir}. Pass --overwrite to replace it.")
    output_dir.mkdir(parents=True, exist_ok=True)
    scores, importances = run_experiment(clean, label, n_repeats)
    metrics = pd.DataFrame({
        "metric": ["accuracy"],
        "mean": [scores["accuracy"].mean()],
        "std": [scores["accuracy"].std(ddof=1)],
    })
    feature_summary = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "importance_mean": importances.mean().reindex(FEATURE_COLUMNS).to_numpy(),
        "importance_std": importances.std(ddof=1).reindex(FEATURE_COLUMNS).to_numpy(),
    }).sort_values("importance_mean", ascending=False, ignore_index=True)
    metadata = {
        "region": region_key,
        "region_name": REGIONS[region_key].name,
        "threshold_mw": threshold,
        "source_file": str(source_path.relative_to(PROJECT_ROOT)),
        "n_events_before_feature_filter": int(len(source)),
        "n_events": int(len(clean)),
        "n_positive": positive_count,
        "n_negative": negative_count,
        "feature_columns": list(FEATURE_COLUMNS),
        "balanced_sampling": True,
        "test_size": RF_TEST_SIZE,
        "n_repeats": n_repeats,
        "random_forest": {
            "n_estimators": RF_N_ESTIMATORS,
            "max_depth": RF_MAX_DEPTH,
            "min_samples_leaf": RF_MIN_SAMPLES_LEAF,
        },
    }
    scores.to_csv(output_dir / "scores_per_repeat.csv", index=False, encoding="utf-8")
    importances.to_csv(output_dir / "feature_importances_per_repeat.csv", index=False, encoding="utf-8")
    metrics.to_csv(output_dir / "metrics_summary.csv", index=False, encoding="utf-8")
    feature_summary.to_csv(output_dir / "feature_importance_summary.csv", index=False, encoding="utf-8")
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-candidates", action="store_true", help="Run all 30 region-threshold candidates instead of the formal task list.")
    parser.add_argument("--n-repeats", type=int, default=RF_REPEATS, help="Number of repeated experiments.")
    parser.add_argument("--overwrite", action="store_true", help="Allow writing into existing run directories.")
    args = parser.parse_args()

    if args.n_repeats < 1:
        raise ValueError("--n-repeats must be positive.")
    ensure_output_directories()
    if args.all_candidates:
        tasks = [(region_key, threshold) for region_key in REGIONS for threshold in MAGNITUDE_THRESHOLDS]
    else:
        tasks = list(FORMAL_RF_TASKS)
    for region_key, threshold in tasks:
        source_path = LABELS_DIR / f"{region_key}_labeled_events.csv"
        if not source_path.exists():
            raise FileNotFoundError(f"Run step 03 first: {source_path}")
        print(f"Running {region_key}, Mw >= {threshold:g}")
        write_run(region_key, threshold, source_path, args.n_repeats, args.overwrite)


if __name__ == "__main__":
    main()
