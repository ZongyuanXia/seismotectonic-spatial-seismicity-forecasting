import argparse
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import optim
from torch.utils.data import DataLoader

from config import CHECKPOINTS_DIR, MODEL_CHANNELS, RESULTS_DIR, STAGE4_PREFIX, Config
from dataset import MultiDataset
from early_stopping import EarlyStopping
from loss import WeightedMseMae
from model import EncoderForecaster


N_TRAIN = 7879
N_TEST = 806


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_channels(model_name, channels_text):
    if channels_text:
        return [int(value.strip()) for value in channels_text.split(",") if value.strip()]
    return MODEL_CHANNELS[model_name]


def train(args):
    channels = parse_channels(args.model, args.channels)
    seed = args.seed if args.seed is not None else 2019
    setup_seed(seed)
    cfg = Config(selected_channels=channels)
    device = cfg.device

    run_dir = CHECKPOINTS_DIR / args.model / f"run_{args.run:02d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "second_n_conv2d.pth"
    history_path = run_dir / "training_history.csv"
    figure_path = run_dir / "training_history.png"
    if checkpoint_path.exists() and not (args.resume or args.overwrite):
        raise FileExistsError(
            f"Checkpoint already exists: {checkpoint_path}. Use --resume or --overwrite explicitly."
        )
    if args.overwrite and checkpoint_path.exists():
        checkpoint_path.unlink()

    train_data = MultiDataset(STAGE4_PREFIX, N_TRAIN, "train", channels)
    test_data = MultiDataset(STAGE4_PREFIX, N_TEST, "test", channels)
    train_loader = DataLoader(
        train_data,
        batch_size=cfg.batch,
        shuffle=True,
        num_workers=cfg.num_workers,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=cfg.batch,
        shuffle=False,
        num_workers=cfg.num_workers,
    )

    model = EncoderForecaster(cfg).to(device)
    loss_fn = WeightedMseMae().to(device)
    learning_rate, step_size, gamma = cfg.pre_parameters
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    early_stopping = EarlyStopping(cfg.patience, path=checkpoint_path)

    start_epoch = 0
    if args.resume and checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state["state_dict"])
        optimizer_state = state.get("optimizer_state_dict", state.get("opt_state_dict"))
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
        if "scheduler_state_dict" in state:
            scheduler.load_state_dict(state["scheduler_state_dict"])
        start_epoch = int(state.get("epoch", -1)) + 1

    history = []
    for epoch in range(start_epoch, cfg.num_epochs):
        model.train()
        train_losses = []
        for images, target in train_loader:
            images = images.to(device=device, dtype=torch.float32)
            target = target.to(device=device, dtype=torch.float32)
            prediction = model(images)
            loss = loss_fn(prediction, target)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 50.0)
            optimizer.step()
            scheduler.step()
            train_losses.append(float(loss.item()))

        model.eval()
        validation_losses = []
        with torch.no_grad():
            for images, target in test_loader:
                images = images.to(device=device, dtype=torch.float32)
                target = target.to(device=device, dtype=torch.float32)
                prediction = model(images)
                validation_losses.append(float(loss_fn(prediction, target).item()))

        train_loss = float(np.mean(train_losses))
        validation_loss = float(np.mean(validation_losses))
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        print(
            f"Epoch {epoch + 1:02d}/{cfg.num_epochs}: "
            f"train={train_loss:.8f}, validation={validation_loss:.8f}"
        )

        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "model_name": args.model,
            "run": args.run,
            "seed": seed,
            "selected_channels": channels,
        }
        early_stopping(validation_loss, checkpoint)
        if early_stopping.early_stop:
            break

    frame = pd.DataFrame(history)
    frame.to_csv(history_path, index=False)
    if not frame.empty:
        plt.figure(figsize=(8, 8))
        plt.plot(frame["epoch"], frame["train_loss"], color="red", label="loss_train")
        plt.plot(frame["epoch"], frame["validation_loss"], color="black", label="loss_test")
        plt.legend(loc="upper right")
        plt.savefig(figure_path, dpi=100, bbox_inches="tight")
        plt.close()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Best checkpoint: {checkpoint_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train one ConvLSTM channel configuration.")
    parser.add_argument("--model", choices=MODEL_CHANNELS, default="M1")
    parser.add_argument("--channels", help="Optional comma-separated channel indices.")
    parser.add_argument("--run", type=int, choices=range(1, 11), default=1)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
