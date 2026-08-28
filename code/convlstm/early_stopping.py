from pathlib import Path

import numpy as np
import torch


class EarlyStopping:
    def __init__(self, patience=3, delta=0.0, path="checkpoint.pth"):
        self.patience = int(patience)
        self.delta = float(delta)
        self.path = Path(path)
        self.counter = 0
        self.best_score = None
        self.val_loss_min = np.inf
        self.early_stop = False

    def __call__(self, val_loss, checkpoint):
        score = -float(val_loss)
        if self.best_score is None or score >= self.best_score + self.delta:
            self.best_score = score
            self.counter = 0
            self.path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint, self.path)
            self.val_loss_min = float(val_loss)
            return

        self.counter += 1
        if self.counter >= self.patience:
            self.early_stop = True

