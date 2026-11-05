"""Compare the training adapter with the original CL10 dataset reader."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from cl10_data import CHANNEL_ORDER, CL10Data, HEIGHT, WIDTH, training_mask


def verify(root: Path, reader_dir: Path):
    original_reader_dir = reader_dir.resolve()
    if not (original_reader_dir / "build_cl10.py").exists():
        raise FileNotFoundError(f"Original CL10 reader missing: {original_reader_dir}")
    sys.path.insert(0, str(original_reader_dir))
    from build_cl10 import CLDataset  # Original data-generation reader.

    channels = tuple(range(len(CHANNEL_ORDER)))
    for split in ("train", "validation", "test"):
        new = CL10Data(root, split, channels)
        original = CLDataset(root, split)
        for index in (0, len(new) // 2, len(new) - 1):
            x, y = new[index]
            expected_x, expected_y = original[index]
            if not np.array_equal(x, expected_x):
                raise AssertionError(f"X differs at {split}[{index}]")
            if not np.array_equal(y, expected_y[0]):
                raise AssertionError(f"Y differs at {split}[{index}]")
            if x.shape != (9, 10, HEIGHT, WIDTH) or y.shape != (1, HEIGHT, WIDTH):
                raise AssertionError(f"Unexpected shape at {split}[{index}]")
        print(f"{split}: {len(new)} samples; first, middle, and last match exactly")

    base, expanded = training_mask(new.day, new.key, radius=6)
    if np.any(base > expanded):
        raise AssertionError("Expanded mask excludes an event cell")
    print(f"training mask: {int(base.sum())} event cells, {int(expanded.sum())} expanded cells")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--reader-dir", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    verify(Path(args.data), args.reader_dir)
