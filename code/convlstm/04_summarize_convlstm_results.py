import argparse

import pandas as pd

from config import EXCLUDED_RUNS, MODEL_CHANNELS, RESULTS_DIR


METRICS = ["precision", "recall", "CSI", "predicted_area_fraction", "target_area_fraction"]


def collect_results(allow_incomplete=False):
    rows = []
    for model_name in MODEL_CHANNELS:
        for run in range(1, 11):
            path = RESULTS_DIR / "evaluation" / model_name / f"run_{run:02d}_metrics.csv"
            if not path.exists():
                if allow_incomplete:
                    continue
                raise FileNotFoundError(f"Missing formal evaluation: {path}")
            row = pd.read_csv(path).iloc[0].to_dict()
            row["model"] = model_name
            row["run"] = run
            row["retained"] = run != EXCLUDED_RUNS[model_name]
            rows.append(row)
    if not rows:
        raise FileNotFoundError("No ConvLSTM evaluation files were found.")
    return pd.DataFrame(rows).sort_values(["model", "run"])


def summarize(frame):
    rows = []
    for model_name, group in frame[frame["retained"]].groupby("model", sort=False):
        row = {"model": model_name, "n_retained": len(group)}
        for metric in METRICS:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_std"] = group[metric].std(ddof=1)
        rows.append(row)
    summary = pd.DataFrame(rows)
    baseline = summary.loc[summary["model"] == "M1", "CSI_mean"]
    if not baseline.empty:
        summary["delta_CSI"] = summary["CSI_mean"] - float(baseline.iloc[0])
    return summary


def main():
    parser = argparse.ArgumentParser(description="Summarize the retained ConvLSTM runs.")
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    runs = collect_results(args.allow_incomplete)
    runs.to_csv(RESULTS_DIR / "run_metrics.csv", index=False)
    summary = summarize(runs)
    summary.to_csv(RESULTS_DIR / "summary_metrics.csv", index=False)
    area_columns = [
        "model",
        "n_retained",
        "predicted_area_fraction_mean",
        "predicted_area_fraction_std",
        "target_area_fraction_mean",
        "target_area_fraction_std",
    ]
    summary[area_columns].to_csv(RESULTS_DIR / "area_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

