"""Paired MAG outside-mask weight experiment using a completed w1=0.03 run.

No test targets are loaded during fitting. Run --smoke on a PyTorch platform
before starting the full 40-epoch training job.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from cl10_data import CHANNEL_ORDER, CL10Data, HEIGHT, WIDTH, parse_channels, sha256, training_mask
from model_cl10 import build_model


BASELINE_WEIGHTS = (0.000177210579159563, 0.03, 1.0)
LOSS_SCALE = 1.0 / 1875.0
MAG4_GRAY = 100.0
LR = 1e-4
GAMMA = 0.7
MAX_EPOCHS = 40
PATIENCE = 3
MASK_RADIUS = 6  # Grid cells; fixed before using validation or test targets.


class ArchivedWeightedMseMae(nn.Module):
    """Archived MSE+MAE and 3-tier weighting, with an explicit fixed mask."""

    def __init__(self, mask: np.ndarray, weights: tuple[float, float, float]):
        super().__init__()
        if mask.shape != (HEIGHT, WIDTH) or not np.isin(mask, (0, 1)).all():
            raise ValueError("Mask must be a binary 340 x 200 grid")
        self.register_buffer("mask", torch.from_numpy(mask[None, None].astype(np.float32)))
        self.weights = weights

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if prediction.shape != target.shape or prediction.ndim != 4:
            raise ValueError(f"Predicted {tuple(prediction.shape)} vs target {tuple(target.shape)}")
        if prediction.shape[1:] != (1, HEIGHT, WIDTH):
            raise ValueError("Expected (batch,1,340,200) prediction and target")
        w0, w1, w2 = self.weights
        weight = w0 + self.mask * (w1 - w0)
        # Preserve the archived additive formula. A positive validation cell
        # outside the historical mask has weight w0 + w2 - w1, not exactly w2.
        weight = weight + (target * 255.0 >= MAG4_GRAY).to(target.dtype) * (w2 - w1)
        delta = prediction - target
        mse = (weight * delta.square()).sum(dim=(1, 2, 3))
        mae = (weight * delta.abs()).sum(dim=(1, 2, 3))
        return LOSS_SCALE * (mse.mean() + mae.mean())


def set_seeds(init_seed: int, order_seed: int) -> torch.Generator:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(init_seed)
    np.random.seed(init_seed)
    torch.manual_seed(init_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(init_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    generator = torch.Generator()
    generator.manual_seed(order_seed)
    return generator


def baseline_seeds(baseline: dict) -> tuple[int, int]:
    """Reuse the exact initialization and DataLoader seeds, without drawing new ones."""
    if baseline.get("seed_scheme") != "independent_model_initialization_and_loader_order":
        raise ValueError("Baseline is not a dual-seed run")
    init_seed, order_seed = baseline.get("init_seed"), baseline.get("order_seed")
    for name, seed in (("init_seed", init_seed), ("order_seed", order_seed)):
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError(f"{name} must be an integer in [0, 2^32)")
    return init_seed, order_seed


def check_baseline(baseline: dict, current: dict) -> None:
    """Reject a pairing if anything material besides w1 differs."""
    if baseline.get("selected_names") != ["MAG"] or current["selected_names"] != ["MAG"]:
        raise ValueError("The paired weight experiment requires MAG only")
    if baseline.get("training", {}).get("weights") != list(BASELINE_WEIGHTS):
        raise ValueError("Baseline does not use the archived three-tier weights")
    for field in ("dataset_sha256", "dataset_mapping_revision", "channel_order",
                  "selected_channels", "train_count", "validation_count", "mask",
                  "train_positive_pixel_fraction"):
        if baseline.get(field) != current[field]:
            raise ValueError(f"Baseline and experiment differ in {field}")
    for field in ("data_script_sha256", "model_script_sha256"):
        if baseline.get("software", {}).get(field) != current["software"][field]:
            raise ValueError(f"Baseline and experiment differ in {field}")
    baseline_training = dict(baseline["training"])
    baseline_training.pop("weights")
    current_training = dict(current["training"])
    current_training.pop("weights")
    if baseline_training != current_training:
        raise ValueError("Baseline and experiment training settings differ besides weights")


def rng_state(generator: torch.Generator) -> dict:
    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "loader": generator.get_state(),
    }


def restore_rng(state: dict, generator: torch.Generator) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("Cannot exactly resume a CUDA run without CUDA")
        torch.cuda.set_rng_state_all(state["cuda"])
    generator.set_state(state["loader"])


def atomic_torch_save(payload: dict, path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def atomic_json(payload: dict, path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def write_history(rows: list[dict], path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "validation_loss", "lr", "best_epoch"])
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def assert_batch(x: torch.Tensor, y: torch.Tensor, channels: int) -> None:
    if tuple(x.shape) != (1, 9, channels, HEIGHT, WIDTH):
        raise ValueError(f"Unexpected X shape: {tuple(x.shape)}")
    if tuple(y.shape) != (1, 1, HEIGHT, WIDTH):
        raise ValueError(f"Unexpected Y shape: {tuple(y.shape)}")
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise ValueError("Nonfinite input or target")
    if x.min() < 0 or x.max() > 1 or y.min() < 0 or y.max() > 1:
        raise ValueError("Inputs and targets must be normalized to [0,1]")


def check_single_cell_weights(criterion: ArchivedWeightedMseMae, mask: np.ndarray,
                              weights: tuple[float, float, float], device: torch.device) -> dict:
    """Check the actual loss formula on one inside and one outside grid cell."""
    inside = np.argwhere(mask != 0)
    outside = np.argwhere(mask == 0)
    if len(inside) == 0 or len(outside) == 0:
        raise ValueError("Weight check needs both inside and outside cells")
    cases = (
        ("inside_negative", tuple(inside[0]), 1.0, 0.0, weights[1]),
        ("outside_negative", tuple(outside[0]), 1.0, 0.0, weights[0]),
        ("inside_positive", tuple(inside[0]), 0.0, 1.0, weights[2]),
        ("outside_positive", tuple(outside[0]), 0.0, 1.0, weights[0] + weights[2] - weights[1]),
    )
    observed = {}
    for name, (row, column), predicted, target, expected in cases:
        p = torch.zeros((1, 1, HEIGHT, WIDTH), device=device)
        y = torch.zeros_like(p)
        p[0, 0, row, column] = predicted
        y[0, 0, row, column] = target
        actual = float(criterion(p, y)) / (2 * LOSS_SCALE)
        if not math.isclose(actual, expected, rel_tol=1e-5, abs_tol=1e-7):
            raise AssertionError(f"{name}: measured {actual} but expected {expected}")
        observed[name] = actual
    return observed


def create_run_config(data: CL10Data, channels, init_seed, order_seed, mask,
                      train_fraction, weights, baseline_path: Path):
    root = data.root
    return {
        "dataset_root": str(root),
        "dataset_sha256": {
            "manifest": sha256(root / "manifest.json"),
            "sample_index": sha256(root / "sample_index.csv"),
            "usgs_daily": sha256(root / "usgs_daily.npz"),
        },
        "dataset_mapping_revision": data.manifest.get("mapping_revision"),
        "channel_order": list(CHANNEL_ORDER),
        "selected_channels": list(channels),
        "selected_names": [CHANNEL_ORDER[i] for i in channels],
        "seed_scheme": "independent_model_initialization_and_loader_order",
        "init_seed": init_seed,
        "order_seed": order_seed,
        "baseline_run": str(baseline_path),
        "baseline_run_json_sha256": sha256(baseline_path / "run.json"),
        "train_count": len(data),
        "validation_count": 2473,
        "train_positive_pixel_fraction": train_fraction,
        "mask": {
            "source": "USGS daily events, 1978-01-01 through 2008-12-31 inclusive",
            "radius_grid_cells": MASK_RADIUS,
            "positive_grid_cells": int(mask.sum()),
            "sha256": hashlib.sha256(mask.tobytes()).hexdigest(),
        },
        "training": {
            "batch_size": 1, "optimizer": "Adam", "initial_lr": LR,
            "scheduler": "StepLR once per epoch", "gamma": GAMMA,
            "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
            "clip_grad_value": 50.0, "rotation": False,
            "loss": "Archived weighted MSE + MAE", "weights": list(weights),
            "mag4_gray_threshold": MAG4_GRAY, "loss_scale": LOSS_SCALE,
            "checkpoint_selection": "lowest full-validation weighted loss",
        },
        "software": {
            "python": sys.version.split()[0], "numpy": np.__version__,
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "train_script_sha256": sha256(Path(__file__)),
            "data_script_sha256": sha256(Path(__file__).with_name("cl10_data.py")),
            "model_script_sha256": sha256(Path(__file__).with_name("model_cl10.py")),
        },
    }


def train(args):
    channels = parse_channels(args.channels)
    out = Path(args.out).resolve() if args.out else None
    baseline_path = Path(args.baseline_run).resolve()
    if not all((baseline_path / name).is_file() for name in ("run.json", "summary.json", "best.pt")):
        raise FileNotFoundError(f"Baseline training run is incomplete: {baseline_path}")
    baseline = json.loads((baseline_path / "run.json").read_text(encoding="utf-8"))
    init_seed, order_seed = baseline_seeds(baseline)
    if not math.isfinite(args.w0) or not 0 <= args.w0 < BASELINE_WEIGHTS[1]:
        raise ValueError("--w0 must be finite, nonnegative, and lower than fixed w1=0.03")
    weights = (args.w0, BASELINE_WEIGHTS[1], BASELINE_WEIGHTS[2])
    generator = set_seeds(init_seed, order_seed)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    train_data = CL10Data(args.data, "train", channels)
    validation_data = CL10Data(args.data, "validation", channels)
    base_mask, mask = training_mask(train_data.day, train_data.key, MASK_RADIUS)
    positive_fraction = train_data.positive_fraction()
    config = create_run_config(train_data, channels, init_seed, order_seed, mask,
                               positive_fraction, weights, baseline_path)
    check_baseline(baseline, config)
    model = build_model(len(channels)).to(device)
    criterion = ArchivedWeightedMseMae(mask, weights).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=GAMMA)
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True, num_workers=0, generator=generator)
    validation_loader = DataLoader(validation_data, batch_size=1, shuffle=False, num_workers=0)

    if args.smoke:
        effective_weights = check_single_cell_weights(criterion, mask, weights, device)
        x, y = next(iter(train_loader))
        assert_batch(x, y, len(channels))
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(x.to(device)), y.to(device))
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            x_val, y_val = next(iter(validation_loader))
            assert_batch(x_val, y_val, len(channels))
            model.eval()
            validation_loss = criterion(model(x_val.to(device)), y_val.to(device))
        print(json.dumps({"device": str(device), "init_seed": init_seed, "order_seed": order_seed,
                          "weights": list(weights), "baseline_run": str(baseline_path),
                          "effective_weights": effective_weights,
                          "x_shape": list(x.shape), "y_shape": list(y.shape),
                          "train_loss": float(loss), "validation_loss": float(validation_loss),
                          "mask_cells": int(mask.sum())}, ensure_ascii=False))
        return

    assert out is not None
    if args.resume:
        if not out.is_dir():
            raise FileNotFoundError(f"Run directory does not exist: {out}")
        previous = json.loads((out / "run.json").read_text(encoding="utf-8"))
        if previous != config:
            raise ValueError("Run configuration, dataset, code, or environment changed; refusing resume")
        with np.load(out / "train_mask.npz") as archive:
            if not np.array_equal(archive["mask"], mask):
                raise ValueError("Saved training mask differs from recomputed mask")
        # Load only a checkpoint created by this run; it contains Python/NumPy RNG states.
        checkpoint = torch.load(out / "latest.pt", map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["next_epoch"])
        best = float(checkpoint["best_validation_loss"])
        best_epoch = int(checkpoint["best_epoch"])
        bad_epochs = int(checkpoint["bad_epochs"])
        restore_rng(checkpoint["rng"], generator)
        with (out / "history.csv").open(newline="", encoding="utf-8") as handle:
            history = list(csv.DictReader(handle))
        if len(history) < start_epoch:
            raise ValueError("History is shorter than the latest checkpoint")
        if len(history) > start_epoch:
            history = history[:start_epoch]  # Interrupted between history and checkpoint saves.
            write_history(history, out / "history.csv")
        if best_epoch == start_epoch and (
            not (out / "best.pt").exists()
            or torch.load(out / "best.pt", map_location="cpu", weights_only=False)["epoch"] != best_epoch
        ):
            atomic_torch_save({"model": model.state_dict(), "epoch": best_epoch,
                               "validation_loss": best, "run_config": config}, out / "best.pt")
        if bad_epochs >= PATIENCE or start_epoch >= MAX_EPOCHS:
            atomic_json({"best_epoch": best_epoch, "best_validation_loss": best,
                         "completed_epochs": len(history),
                         "early_stopped": bad_epochs >= PATIENCE}, out / "summary.json")
            print("Run already completed; no training steps remain")
            return
    else:
        out.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(out / "train_mask.npz", original=base_mask, mask=mask)
        atomic_json(config, out / "run.json")
        history = []
        start_epoch = 0
        best = float("inf")
        best_epoch = 0
        bad_epochs = 0

    for epoch in range(start_epoch, MAX_EPOCHS):
        model.train()
        total = 0.0
        lr_this_epoch = optimizer.param_groups[0]["lr"]
        for step, (x, y) in enumerate(train_loader, 1):
            if step == 1:
                assert_batch(x, y, len(channels))
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            prediction = model(x)
            loss = criterion(prediction, y)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite training loss at epoch {epoch + 1}, step {step}")
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=50.0)
            optimizer.step()
            total += float(loss.detach())
            if step % 100 == 0:
                print(f"epoch={epoch + 1} step={step}/{len(train_loader)} train_loss={total / step:.8f}", flush=True)
        train_loss = total / len(train_loader)

        model.eval()
        total_validation = 0.0
        with torch.no_grad():
            for step, (x, y) in enumerate(validation_loader, 1):
                if step == 1:
                    assert_batch(x, y, len(channels))
                loss = criterion(model(x.to(device)), y.to(device))
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite validation loss at epoch {epoch + 1}, step {step}")
                total_validation += float(loss)
        validation_loss = total_validation / len(validation_loader)
        improved = validation_loss < best
        if improved:
            best = validation_loss
            best_epoch = epoch + 1
            bad_epochs = 0
        else:
            bad_epochs += 1
        scheduler.step()
        history.append({"epoch": epoch + 1, "train_loss": train_loss,
                        "validation_loss": validation_loss, "lr": lr_this_epoch,
                        "best_epoch": best_epoch})
        write_history(history, out / "history.csv")
        atomic_torch_save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                           "scheduler": scheduler.state_dict(), "next_epoch": epoch + 1,
                           "best_validation_loss": best, "best_epoch": best_epoch,
                           "bad_epochs": bad_epochs, "rng": rng_state(generator)}, out / "latest.pt")
        if improved:
            atomic_torch_save({"model": model.state_dict(), "epoch": best_epoch,
                               "validation_loss": best, "run_config": config}, out / "best.pt")
        print(f"epoch={epoch + 1} train={train_loss:.8f} validation={validation_loss:.8f} "
              f"best_epoch={best_epoch} bad_epochs={bad_epochs}", flush=True)
        if bad_epochs >= PATIENCE:
            break
    atomic_json({"best_epoch": best_epoch, "best_validation_loss": best,
                 "completed_epochs": len(history), "early_stopped": bad_epochs >= PATIENCE}, out / "summary.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Frozen datasets/cl10_mapping_v2 directory")
    parser.add_argument("--out", help="New output directory for a formal run")
    parser.add_argument("--channels", default="MAG", help="Comma-separated names or 0-based indices; default MAG")
    parser.add_argument("--baseline-run", required=True,
                        help="Completed w1=0.03 MAG run whose two seeds will be reused")
    parser.add_argument("--w0", required=True, type=float,
                        help="Weight of negative grid cells outside the fixed six-cell mask")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--resume", action="store_true", help="Resume this run from latest.pt")
    parser.add_argument("--smoke", action="store_true", help="One train and validation step, without creating output")
    args = parser.parse_args()
    if not args.smoke and not args.out:
        parser.error("--out is required for a formal run")
    train(args)


if __name__ == "__main__":
    main()
