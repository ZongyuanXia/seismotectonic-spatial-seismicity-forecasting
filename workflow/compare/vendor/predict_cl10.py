"""Export per-sample CL10 predictions after model selection is finished."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from cl10_data import CL10Data, HEIGHT, WIDTH, sha256
from model_cl10 import build_model


def seed_metadata(config: dict) -> dict:
    """Preserve the archived single-seed format and identify new dual-seed runs."""
    if config.get("seed_scheme") == "independent_model_initialization_and_loader_order":
        seeds = {name: config.get(name) for name in ("init_seed", "order_seed")}
        if any(type(value) is not int or not 0 <= value < 2**32 for value in seeds.values()):
            raise ValueError("Dual-seed run has missing or invalid random seeds")
        return seeds
    if type(config.get("seed")) is int:
        return {"seed": config["seed"]}
    raise ValueError("Run configuration has no recognized random seed metadata")


def predict(args):
    run = Path(args.run).resolve()
    if not (run / "summary.json").exists():
        raise ValueError("Training run is not complete; best checkpoint is not final")
    config = json.loads((run / "run.json").read_text(encoding="utf-8"))
    root = Path(args.data).resolve()
    for name, filename in (("manifest", "manifest.json"),
                           ("sample_index", "sample_index.csv"),
                           ("usgs_daily", "usgs_daily.npz")):
        if sha256(root / filename) != config["dataset_sha256"][name]:
            raise ValueError(f"Dataset {filename} differs from the training dataset")
    channels = tuple(config["selected_channels"])
    data = CL10Data(root, args.split, channels)
    checkpoint_path = run / "best.pt"
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model(len(channels)).to(device)
    # This checkpoint must be a trusted file produced by the CL10 trainer.
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint["run_config"] != config:
        raise ValueError("Best checkpoint does not match the run configuration")
    model.load_state_dict(checkpoint["model"])
    model.eval()

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    predictions = np.lib.format.open_memmap(
        out / "predictions.npy", mode="w+", dtype=np.float32,
        shape=(len(data), HEIGHT, WIDTH),
    )
    loader = DataLoader(data, batch_size=1, shuffle=False, num_workers=0)
    with (out / "sample_index.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("row", "target_start", "target_end"))
        writer.writeheader()
        with torch.no_grad():
            for i, (x, _) in enumerate(loader):
                prediction = model(x.to(device))
                if tuple(prediction.shape) != (1, 1, HEIGHT, WIDTH):
                    raise ValueError(f"Unexpected prediction shape: {tuple(prediction.shape)}")
                if not torch.isfinite(prediction).all():
                    raise FloatingPointError(f"Nonfinite prediction at row {i}")
                predictions[i] = prediction[0, 0].cpu().numpy()
                row = data.rows[i]
                writer.writerow({"row": i, "target_start": row["target_start"],
                                 "target_end": row["target_end"]})
                if (i + 1) % 100 == 0:
                    predictions.flush()
                    print(f"predicted {i + 1}/{len(data)}", flush=True)
    predictions.flush()
    del predictions
    prediction_sha256 = sha256(out / "predictions.npy")
    (out / "manifest.json").write_text(json.dumps({
        "complete": True,
        "split": args.split,
        "count": len(data),
        "shape": [len(data), HEIGHT, WIDTH],
        "prediction_file": "predictions.npy",
        "prediction_dtype": "float32",
        "predictions_sha256": prediction_sha256,
        "prediction_scale": "raw model output against targets normalized by 255; no clipping",
        "target_dates_file": "sample_index.csv",
        "dataset_manifest_sha256": config["dataset_sha256"]["manifest"],
        "best_checkpoint_sha256": sha256(checkpoint_path),
        "best_epoch": checkpoint["epoch"],
        "selected_channels": list(channels),
        **seed_metadata(config),
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--run", required=True, help="Completed training run directory")
    parser.add_argument("--split", required=True, choices=("validation", "test"))
    parser.add_argument("--out", required=True, help="New output directory")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    predict(parser.parse_args())


if __name__ == "__main__":
    main()
