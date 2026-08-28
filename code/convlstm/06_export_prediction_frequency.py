import argparse

import numpy as np
import pandas as pd
import torch

from config import CHECKPOINTS_DIR, EXCLUDED_RUNS, MODEL_CHANNELS, RESULTS_DIR, STAGE4_PREFIX, Config
from model import EncoderForecaster
from spatial_tolerance import spacelayer


MODEL_CODES = {"M1": "C0", "M2": "C012", "M3": "C034", "M4": "C056", "M5": "C01256"}


def sample_paths(sample_id):
    stem = f"{STAGE4_PREFIX}_half_{sample_id}_n_test"
    return f"{stem}_x.npz", f"{stem}_y.npz"


def load_sample(sample_id):
    x_path, y_path = sample_paths(sample_id)
    with np.load(x_path) as x_data, np.load(y_path) as y_data:
        x = x_data["sequence_array"].astype(np.float32)
        y = y_data["sequence_array"].astype(np.float32)
    if y.ndim == 5 and y.shape[2] == 1:
        y = y[:, :, 0, :, :]
    return x, y


def grid_centers(height, width, cfg):
    latitudes = cfg.lat_min + (np.arange(height) + 0.5) * (cfg.lat_max - cfg.lat_min) / height
    longitudes = cfg.lon_min + (np.arange(width) + 0.5) * (cfg.lon_max - cfg.lon_min) / width
    return latitudes, longitudes


def save_grid(matrix, latitudes, longitudes, path):
    frame = pd.DataFrame(
        matrix,
        index=np.round(latitudes, 6),
        columns=np.round(longitudes, 6),
    )
    frame.index.name = "lat"
    frame.columns.name = "lon"
    frame.to_csv(path, float_format="%.6f")


def export_model(model_name, sample_id, threshold=100.0, distance=3):
    channels = MODEL_CHANNELS[model_name]
    cfg = Config(selected_channels=channels)
    device = cfg.device
    x, y = load_sample(sample_id)
    input_tensor = torch.from_numpy(x[:, :, channels, :, :]).to(device=device, dtype=torch.float32)
    true_mask = (y[0, 0] * 255.0 >= threshold).astype(np.float32)

    masks = []
    retained_runs = [run for run in range(1, 11) if run != EXCLUDED_RUNS[model_name]]
    for run in retained_runs:
        checkpoint_path = CHECKPOINTS_DIR / model_name / f"run_{run:02d}" / "second_n_conv2d.pth"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model = EncoderForecaster(cfg).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        with torch.no_grad():
            prediction = model(input_tensor).detach().cpu().numpy()
        expanded = spacelayer(prediction, d=distance)[0, 0]
        masks.append((expanded * 255.0 >= threshold).astype(np.float32))

    frequency = np.mean(np.stack(masks, axis=0), axis=0)
    code = MODEL_CODES[model_name]
    output_dir = RESULTS_DIR / "figure5_source" / f"sample_{sample_id:04d}" / code
    output_dir.mkdir(parents=True, exist_ok=True)
    latitudes, longitudes = grid_centers(frequency.shape[0], frequency.shape[1], cfg)
    save_grid(frequency, latitudes, longitudes, output_dir / f"{code}_pred_frequency_grid.csv")
    save_grid(true_mask, latitudes, longitudes, output_dir / f"{code}_true_mask_grid.csv")
    pd.DataFrame({"model": [model_name] * 9, "retained_run": retained_runs}).to_csv(
        output_dir / f"{code}_retained_runs.csv", index=False
    )
    print(f"Exported {model_name}: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Export the Figure 5 prediction-frequency grids.")
    parser.add_argument("--sample", type=int, default=400)
    parser.add_argument("--models", nargs="+", choices=MODEL_CHANNELS, default=list(MODEL_CHANNELS))
    parser.add_argument("--threshold", type=float, default=100.0)
    parser.add_argument("--distance", type=int, default=3)
    args = parser.parse_args()
    for model_name in args.models:
        export_model(model_name, args.sample, args.threshold, args.distance)


if __name__ == "__main__":
    main()

