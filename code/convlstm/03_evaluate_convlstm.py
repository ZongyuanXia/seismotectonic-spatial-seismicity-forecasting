import argparse

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from config import CHECKPOINTS_DIR, MODEL_CHANNELS, RESULTS_DIR, STAGE4_PREFIX, Config
from dataset import MultiDataset
from metrics import confusion_counts, scores_from_counts
from model import EncoderForecaster
from spatial_tolerance import spacelayer


N_TEST = 806


def align_4d(tensor):
    if tensor.ndim == 5 and tensor.shape[2] == 1:
        return tensor[:, :, 0, :, :]
    if tensor.ndim == 4:
        return tensor
    raise ValueError(f"Expected (B,1,H,W) or (B,1,1,H,W), got {tuple(tensor.shape)}.")


def expand_continuous(tensor, distance, device):
    expanded = spacelayer(tensor.detach().cpu().numpy(), d=distance)
    return torch.from_numpy(expanded).to(device=device, dtype=torch.float32)


def checkpoint_path(model_name, run):
    return CHECKPOINTS_DIR / model_name / f"run_{run:02d}" / "second_n_conv2d.pth"


def evaluate_run(model_name, run, threshold=100.0, distance=3):
    channels = MODEL_CHANNELS[model_name]
    cfg = Config(selected_channels=channels)
    device = cfg.device
    path = checkpoint_path(model_name, run)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=device)
    checkpoint_channels = checkpoint.get("selected_channels")
    if checkpoint_channels is not None and list(checkpoint_channels) != channels:
        raise ValueError(
            f"Checkpoint channels {checkpoint_channels} do not match {model_name}: {channels}."
        )
    model = EncoderForecaster(cfg).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    dataset = MultiDataset(STAGE4_PREFIX, N_TEST, "test", channels)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=cfg.num_workers)
    rows = []
    with torch.no_grad():
        for sample_id, (images, target) in enumerate(loader):
            images = images.to(device=device, dtype=torch.float32)
            raw_target = align_4d(target.to(device=device, dtype=torch.float32))
            raw_prediction = align_4d(model(images))
            expanded_prediction = expand_continuous(raw_prediction, distance, device)
            expanded_target = expand_continuous(raw_target, distance, device)

            p_tp, p_fp, p_fn, _ = confusion_counts(raw_prediction, expanded_target, threshold)
            r_tp, r_fp, r_fn, _ = confusion_counts(expanded_prediction, raw_target, threshold)
            c_tp, c_fp, c_fn, _ = confusion_counts(expanded_prediction, expanded_target, threshold)
            precision, _, _ = scores_from_counts(p_tp, p_fp, p_fn)
            _, recall, _ = scores_from_counts(r_tp, r_fp, r_fn)
            _, _, csi = scores_from_counts(c_tp, c_fp, c_fn)

            pred_area = float(torch.mean((raw_prediction * 255.0 >= threshold).float()).item())
            target_area = float(torch.mean((raw_target * 255.0 >= threshold).float()).item())
            rows.append(
                {
                    "sample": sample_id,
                    "precision": precision,
                    "recall": recall,
                    "CSI": csi,
                    "predicted_area_fraction": pred_area,
                    "target_area_fraction": target_area,
                }
            )

    samples = pd.DataFrame(rows)
    out_dir = RESULTS_DIR / "evaluation" / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    samples.to_csv(out_dir / f"run_{run:02d}_sample_metrics.csv", index=False)
    summary = pd.DataFrame(
        [
            {
                "model": model_name,
                "run": run,
                "seed": checkpoint.get("seed", np.nan),
                "channels": ",".join(map(str, channels)),
                "threshold": threshold,
                "distance": distance,
                "precision": samples["precision"].mean(),
                "recall": samples["recall"].mean(),
                "CSI": samples["CSI"].mean(),
                "predicted_area_fraction": samples["predicted_area_fraction"].mean(),
                "target_area_fraction": samples["target_area_fraction"].mean(),
            }
        ]
    )
    output = out_dir / f"run_{run:02d}_metrics.csv"
    summary.to_csv(output, index=False)
    print(summary.to_string(index=False))
    return output


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ConvLSTM checkpoints.")
    parser.add_argument("--model", choices=MODEL_CHANNELS, required=True)
    parser.add_argument("--run", type=int, choices=range(1, 11))
    parser.add_argument("--all-runs", action="store_true")
    parser.add_argument("--threshold", type=float, default=100.0)
    parser.add_argument("--distance", type=int, default=3)
    args = parser.parse_args()
    if args.all_runs == (args.run is not None):
        parser.error("Specify exactly one of --run or --all-runs.")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    runs = range(1, 11) if arguments.all_runs else [arguments.run]
    for run_number in runs:
        evaluate_run(arguments.model, run_number, arguments.threshold, arguments.distance)
