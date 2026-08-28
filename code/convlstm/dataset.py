from pathlib import Path

import numpy as np
from torch.utils.data import Dataset


class MultiDataset(Dataset):
    """Read the one-sample NPZ files produced by ``01_build_dataset.py``."""

    def __init__(
        self,
        path_prefix,
        n_samples,
        split,
        selected_channels,
        case_str="_n_",
        start=0,
        part_size=1,
    ):
        if split not in {"train", "test"}:
            raise ValueError("split must be 'train' or 'test'.")
        if not selected_channels:
            raise ValueError("selected_channels must contain at least one channel.")

        self.path_prefix = str(Path(path_prefix)) + "_half_"
        self.n_samples = int(n_samples)
        self.split = split
        self.channels = np.asarray(selected_channels, dtype=np.int64)
        self.case_str = case_str
        self.start = int(start)
        self.part_size = int(part_size)

        first_x, _ = self._paths(self.start)
        if not first_x.exists():
            raise FileNotFoundError(
                f"Dataset file not found: {first_x}. Run 01_build_dataset.py first "
                "or download the derived dataset from Zenodo."
            )
        with np.load(first_x) as data:
            shape = data["sequence_array"].shape
        if len(shape) != 5:
            raise ValueError(f"Expected X shape (N,T,C,H,W), got {shape} in {first_x}.")
        if self.channels.min() < 0 or self.channels.max() >= shape[2]:
            raise ValueError(f"Selected channels {self.channels.tolist()} are invalid for C={shape[2]}.")

    def _paths(self, index):
        part = index // self.part_size
        stem = f"{self.path_prefix}{part}{self.case_str}{self.split}"
        return Path(f"{stem}_x.npz"), Path(f"{stem}_y.npz")

    def __getitem__(self, index):
        absolute_index = self.start + int(index)
        x_path, y_path = self._paths(absolute_index)
        offset = absolute_index % self.part_size
        with np.load(x_path) as x_data, np.load(y_path) as y_data:
            x = x_data["sequence_array"][offset]
            y = y_data["sequence_array"][offset]
        return x[:, self.channels, :, :], y

    def __len__(self):
        return self.n_samples

