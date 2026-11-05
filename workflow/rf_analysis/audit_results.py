#!/usr/bin/env python3
"""Check that a completed RF replay has all requested repeats and valid splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()
    root = args.result_dir
    config = json.loads((root / "run_config.json").read_text(encoding="utf-8"))
    tasks = pd.read_csv(root / "task_summary.csv")
    errors: list[str] = []
    audited = []
    for _, task in tasks[tasks.status == "completed"].iterrows():
        name = f"{task.period}__{task.region}__ge{int(task.threshold)}"
        directory = root / name
        scores = pd.read_csv(directory / "scores_per_repeat.csv")
        importance = pd.read_csv(directory / "feature_importances_per_repeat.csv")
        splits = pd.read_csv(directory / "split_audit.csv")
        membership = pd.read_csv(directory / "seed_0_membership.csv")
        n = config["repeats"]
        if not (len(scores) == len(importance) == len(splits) == n):
            errors.append(f"{name}: repeat row count mismatch")
        for frame in (scores, importance, splits):
            if frame.seed.tolist() != list(range(n)):
                errors.append(f"{name}: missing or unordered seed")
        values = scores[["Accuracy", "Precision", "Recall", "F1", "ROC_AUC", "PR_AUC"]].to_numpy()
        if not (np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all()):
            errors.append(f"{name}: invalid score")
        weights = importance.drop(columns="seed").to_numpy()
        totals = weights.sum(axis=1)
        if not (np.isfinite(weights).all() and np.all(np.isclose(totals, 0) | np.isclose(totals, 1))):
            errors.append(f"{name}: invalid feature importance")
        expected = 2 * int(task.min_class_count)
        if not (splits.n_selected == expected).all():
            errors.append(f"{name}: balancing size mismatch")
        if not (splits.n_train + splits.n_test == splits.n_selected).all():
            errors.append(f"{name}: split size mismatch")
        if not (splits.n_positive_train + splits.n_positive_test == int(task.min_class_count)).all():
            errors.append(f"{name}: positive class count changed across the split")
        if not ((splits.n_positive_train >= 1) & (splits.n_positive_test >= 1)).all():
            errors.append(f"{name}: a split lost the positive class")
        if len(membership) != expected or membership.WSM_ID.duplicated().any():
            errors.append(f"{name}: seed 0 membership mismatch")
        if set(membership["set"]) != {"train", "test"}:
            errors.append(f"{name}: seed 0 partition mismatch")
        audited.append(name)
    result = {
        "tasks_total": len(tasks),
        "tasks_completed": len(audited),
        "tasks_skipped": int((tasks.status == "skipped").sum()),
        "repeats_per_completed_task": config["repeats"],
        "errors": errors,
        "passed": not errors,
    }
    (root / "audit_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
