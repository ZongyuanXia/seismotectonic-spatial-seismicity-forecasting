import argparse

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from config import CHECKPOINTS_DIR, EXCLUDED_RUNS, MODEL_CHANNELS, RESULTS_DIR, STAGE4_PREFIX, Config
from dataset import MultiDataset
from model import EncoderForecaster
from spatial_tolerance import spacelayer


N_TEST = 806
TARGET_THRESHOLD = 100.0
OFFICIAL_THRESHOLD = 100.0
OFFICIAL_DISTANCE = 3
THRESHOLDS = [50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
DISTANCES = [1, 2, 3, 4, 5]
MODEL_CODES = {"M1": "C0", "M2": "C012", "M3": "C034", "M4": "C056", "M5": "C01256"}
MODEL_LABELS = {
    "M1": "MAG",
    "M2": "MAG+Depth",
    "M3": "MAG+Direction",
    "M4": "MAG+Mechanism",
    "M5": "MAG+Depth+Mechanism",
}


def align_4d(tensor):
    if tensor.ndim == 5 and tensor.shape[2] == 1:
        return tensor[:, :, 0, :, :]
    if tensor.ndim == 4:
        return tensor
    raise ValueError(f"Unexpected tensor shape: {tuple(tensor.shape)}")


def expand_continuous(tensor, distance, device):
    values = spacelayer(tensor.detach().cpu().numpy(), d=distance)
    return torch.from_numpy(values).to(device=device, dtype=torch.float32)


def sample_csi(prediction, target, pred_threshold, target_threshold=TARGET_THRESHOLD):
    pred_positive = prediction * 255.0 >= float(pred_threshold)
    target_positive = target * 255.0 >= float(target_threshold)
    tp = torch.sum(pred_positive & target_positive).item()
    fp = torch.sum(pred_positive & ~target_positive).item()
    fn = torch.sum(~pred_positive & target_positive).item()
    denominator = tp + fp + fn
    return np.nan if denominator == 0 else float(tp / denominator)


def evaluate_checkpoint(model_name, run):
    channels = MODEL_CHANNELS[model_name]
    cfg = Config(selected_channels=channels)
    device = cfg.device
    checkpoint_path = CHECKPOINTS_DIR / model_name / f"run_{run:02d}" / "second_n_conv2d.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = EncoderForecaster(cfg).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    dataset = MultiDataset(STAGE4_PREFIX, N_TEST, "test", channels)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=cfg.num_workers)

    combinations = sorted(
        set([(threshold, OFFICIAL_DISTANCE) for threshold in THRESHOLDS])
        | set([(OFFICIAL_THRESHOLD, distance) for distance in DISTANCES])
    )
    values = {combination: [] for combination in combinations}
    with torch.no_grad():
        for images, target in loader:
            images = images.to(device=device, dtype=torch.float32)
            target = align_4d(target.to(device=device, dtype=torch.float32))
            prediction = align_4d(model(images))
            expanded = {}
            for distance in DISTANCES:
                expanded[distance] = (
                    expand_continuous(prediction, distance, device),
                    expand_continuous(target, distance, device),
                )
            for threshold, distance in combinations:
                pred_eval, target_eval = expanded[distance]
                values[(threshold, distance)].append(
                    sample_csi(pred_eval, target_eval, threshold, TARGET_THRESHOLD)
                )

    rows = []
    for (threshold, distance), sample_values in values.items():
        rows.append(
            {
                "model": model_name,
                "experiment": MODEL_CODES[model_name],
                "label": MODEL_LABELS[model_name],
                "run_idx": run,
                "pred_threshold": threshold,
                "target_threshold": TARGET_THRESHOLD,
                "expand_d": distance,
                "CSI": float(pd.Series(sample_values).mean()),
                "retained": run != EXCLUDED_RUNS[model_name],
            }
        )
    return rows


def write_summaries(frame, output_dir):
    frame.to_csv(output_dir / "5_1_run_level_with_retained_flag.csv", index=False)
    retained = frame[frame["retained"]].copy()
    summary = (
        retained.groupby(
            ["model", "experiment", "label", "pred_threshold", "target_threshold", "expand_d"],
            as_index=False,
        )
        .agg(CSI_mean=("CSI", "mean"), CSI_std=("CSI", "std"), n_runs=("CSI", "count"))
    )
    summary.to_csv(output_dir / "5_1_trimmed_summary_mean_std.csv", index=False)

    official = retained[
        (retained["pred_threshold"] == OFFICIAL_THRESHOLD)
        & (retained["expand_d"] == OFFICIAL_DISTANCE)
    ]
    sorted_rows = []
    for experiment, group in official.groupby("experiment", sort=False):
        for rank, (_, row) in enumerate(group.sort_values("CSI", ascending=False).iterrows(), start=1):
            sorted_rows.append(
                {
                    "experiment": experiment,
                    "label": row["label"],
                    "rank": rank,
                    "run_idx": int(row["run_idx"]),
                    "CSI": row["CSI"],
                }
            )
    pd.DataFrame(sorted_rows).to_csv(
        output_dir / "5_1_official_CSI_sorted_for_plot.csv", index=False
    )

    baseline = summary[summary["experiment"] == "C0"][["pred_threshold", "expand_d", "CSI_mean"]]
    baseline = baseline.rename(columns={"CSI_mean": "CSI_baseline"})
    delta = summary.merge(baseline, on=["pred_threshold", "expand_d"], how="left")
    delta["delta_CSI_vs_baseline"] = delta["CSI_mean"] - delta["CSI_baseline"]

    threshold_delta = delta[
        (delta["expand_d"] == OFFICIAL_DISTANCE) & (delta["experiment"] != "C0")
    ].pivot(index="pred_threshold", columns="experiment", values="delta_CSI_vs_baseline")
    threshold_delta.reset_index().to_csv(
        output_dir / "5_1_threshold_sensitivity_delta_CSI.csv", index=False
    )

    distance_delta = delta[
        (delta["pred_threshold"] == OFFICIAL_THRESHOLD) & (delta["experiment"] != "C0")
    ].pivot(index="expand_d", columns="experiment", values="delta_CSI_vs_baseline")
    distance_delta.reset_index().to_csv(
        output_dir / "5_1_d_sensitivity_delta_CSI.csv", index=False
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate ConvLSTM threshold and distance robustness.")
    parser.add_argument("--models", nargs="+", choices=MODEL_CHANNELS, default=list(MODEL_CHANNELS))
    parser.add_argument("--runs", nargs="+", type=int, default=list(range(1, 11)))
    args = parser.parse_args()

    output_dir = RESULTS_DIR / "robustness"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for model_name in args.models:
        for run in args.runs:
            rows.extend(evaluate_checkpoint(model_name, run))
            print(f"Evaluated {model_name}, run {run}")
    write_summaries(pd.DataFrame(rows), output_dir)


if __name__ == "__main__":
    main()

