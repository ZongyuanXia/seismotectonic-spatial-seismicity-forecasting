"""Read the frozen Japan CL10 data without changing any source files."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import OrderedDict
from datetime import date, timedelta
from pathlib import Path

import numpy as np


CHANNEL_ORDER = (
    "MAG", "DEPTH_LOCAL_MEAN", "DEPTH_LOCAL_STD", "MAX_SPREAD_100KM_DEG",
    "STD_LOCAL", "REGIME_ENTROPY", "REGIME_GINI", "MEDIAN_LOCAL_DEPTH",
    "CONSISTENCY_R_MEAN", "MAX_ROTATION_DEG",
)
SPLITS = {
    "train": ("1989-02-02", "2008-12-31", 6824),
    "validation": ("2009-01-01", "2016-12-31", 2473),
    "test": ("2017-01-01", "2024-12-31", 2473),
}
START = date(1978, 1, 1)
TRAIN_LAST_DAY = date(2008, 12, 31)
HEIGHT, WIDTH = 340, 200


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_channels(spec: str, names=CHANNEL_ORDER) -> tuple[int, ...]:
    tokens = [part.strip() for part in spec.split(",")]
    if not tokens or any(not part for part in tokens):
        raise ValueError("--channels must be a comma-separated list of indices or channel names")
    result = []
    for token in tokens:
        index = int(token) if token.isdigit() else names.index(token.upper())
        if index < 0 or index >= len(names) or index in result:
            raise ValueError(f"Invalid or repeated channel: {token}")
        result.append(index)
    return tuple(result)


def _day_index(value: str) -> int:
    return (date.fromisoformat(value) - START).days


def _frame_mag(day, key, gray, first: int, last: int):
    """Return the maximum MAG gray value per grid cell for inclusive days."""
    lo = np.searchsorted(day, first, side="left")
    hi = np.searchsorted(day, last, side="right")
    if lo == hi:
        return np.empty(0, np.int32), np.empty(0, np.float32)
    order = np.argsort(key[lo:hi], kind="stable")
    sorted_key = key[lo:hi][order]
    unique, positions = np.unique(sorted_key, return_index=True)
    maxima = np.maximum.reduceat(gray[lo:hi][order], positions)
    return unique.astype(np.int32), maxima.astype(np.float32)


def _validate_index(rows: list[dict]) -> None:
    by_split = {name: [] for name in SPLITS}
    for row in rows:
        split = row["split"]
        if split not in by_split:
            raise ValueError(f"Unknown split: {split}")
        target_start = date.fromisoformat(row["target_start"])
        target_end = date.fromisoformat(row["target_end"])
        if target_end != target_start + timedelta(days=449):
            raise ValueError(f"Target is not 450 days: {row}")
        first_input = target_start - timedelta(days=4050)
        for j in range(9):
            expected_start = first_input + timedelta(days=450 * j)
            expected_end = expected_start + timedelta(days=449)
            if (row[f"x{j}_start"], row[f"x{j}_end"]) != (
                expected_start.isoformat(), expected_end.isoformat()
            ):
                raise ValueError(f"Input dates are not contiguous before target: {row}")
        if first_input < START:
            raise ValueError(f"Insufficient input history: {row}")
        by_split[split].append(row)
    for split, (first, final, count) in SPLITS.items():
        group = by_split[split]
        if len(group) != count:
            raise ValueError(f"{split}: expected {count} samples, got {len(group)}")
        first_day = date.fromisoformat(first)
        for i, row in enumerate(group):
            if row["target_start"] != (first_day + timedelta(days=i)).isoformat():
                raise ValueError(f"{split}: target starts are not daily and ordered")
        if group[-1]["target_end"] != final:
            raise ValueError(f"{split}: final target date differs from {final}")


class CL10Data:
    """Return X=(9,C,340,200), Y=(1,340,200), both float32 in [0,1]."""

    def __init__(self, root: str | Path, split: str, channels: tuple[int, ...], cache_size=4096):
        self.root = Path(root).resolve()
        if split not in SPLITS:
            raise ValueError(f"Invalid split: {split}")
        self.split = split
        self.channels = tuple(channels)
        self.cache_size = max(0, int(cache_size))
        self._cache: OrderedDict[str, tuple[np.ndarray, np.ndarray]] = OrderedDict()
        self.manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        if not self.manifest.get("complete") or self.manifest.get("dataset") != "CL-MAG+9":
            raise ValueError("CL10 dataset is incomplete or has an unexpected identity")
        if tuple(self.manifest.get("channel_order", ())) != CHANNEL_ORDER:
            raise ValueError("CL10 channel order has changed")
        if (self.manifest.get("input_frames"), self.manifest.get("frame_days"),
                self.manifest.get("normalized_divisor")) != (9, 450, 255):
            raise ValueError("CL10 timing or normalization does not match this trainer")
        if self.manifest.get("split_counts") != {k: v[2] for k, v in SPLITS.items()}:
            raise ValueError("CL10 split counts have changed")
        if not self.channels or len(set(self.channels)) != len(self.channels):
            raise ValueError("At least one unique channel is required")
        if min(self.channels) < 0 or max(self.channels) >= len(CHANNEL_ORDER):
            raise ValueError("Channel index out of range")
        with (self.root / "sample_index.csv").open(newline="", encoding="utf-8") as handle:
            all_rows = list(csv.DictReader(handle))
        _validate_index(all_rows)
        self.rows = [row for row in all_rows if row["split"] == split]
        with np.load(self.root / "usgs_daily.npz") as archive:
            self.day = archive["day"].astype(np.int32)
            self.key = archive["key"].astype(np.int32)
            self.gray = archive["gray"].astype(np.float32)
        if not (len(self.day) == len(self.key) == len(self.gray)):
            raise ValueError("USGS daily arrays have different lengths")
        if np.any(np.diff(self.day) < 0) or np.any(self.key < 0) or np.any(self.key >= HEIGHT * WIDTH):
            raise ValueError("USGS daily arrays have invalid dates or grid cells")
        if not np.isfinite(self.gray).all() or np.any((self.gray < 100) | (self.gray > 255)):
            raise ValueError("USGS daily MAG gray values are outside [100,255]")

    def __len__(self):
        return len(self.rows)

    def _load_frame(self, start: str):
        if start in self._cache:
            self._cache.move_to_end(start)
            return self._cache[start]
        path = self.root / "frames" / f"{start}.npz"
        with np.load(path) as archive:
            keys = archive["keys"].astype(np.int32)
            gray = archive["gray"][:, self.channels].astype(np.float32)
        if gray.shape != (len(keys), len(self.channels)):
            raise ValueError(f"Bad frame shape: {path}")
        if len(np.unique(keys)) != len(keys) or np.any(keys < 0) or np.any(keys >= HEIGHT * WIDTH):
            raise ValueError(f"Bad grid keys: {path}")
        if not np.isfinite(gray).all() or np.any((gray < 0) | (gray > 255)):
            raise ValueError(f"Bad gray values: {path}")
        value = keys, gray
        if self.cache_size:
            self._cache[start] = value
            if len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return value

    def target_keys(self, row):
        return _frame_mag(self.day, self.key, self.gray,
                          _day_index(row["target_start"]), _day_index(row["target_end"]))

    def positive_fraction(self):
        positives = sum(len(self.target_keys(row)[0]) for row in self.rows)
        return positives / (len(self.rows) * HEIGHT * WIDTH)

    def __getitem__(self, index):
        row = self.rows[index]
        x = np.zeros((9, len(self.channels), HEIGHT, WIDTH), dtype=np.float32)
        for j in range(9):
            keys, values = self._load_frame(row[f"x{j}_start"])
            yy, xx = np.divmod(keys, WIDTH)
            x[j, :, yy, xx] = values
        keys, values = self.target_keys(row)
        y = np.zeros((1, HEIGHT, WIDTH), dtype=np.float32)
        yy, xx = np.divmod(keys, WIDTH)
        y[0, yy, xx] = values
        x /= 255.0
        y /= 255.0
        return x, y


def training_mask(day, key, radius=6):
    """Union Mw>=4 event cells through 2008-12-31; square expansion in grid cells."""
    if radius < 0:
        raise ValueError("Mask radius must be nonnegative")
    cutoff = (TRAIN_LAST_DAY - START).days
    base = np.zeros((HEIGHT, WIDTH), dtype=bool)
    selected = key[(day >= 0) & (day <= cutoff)]
    yy, xx = np.divmod(selected, WIDTH)
    base[yy, xx] = True
    padded = np.pad(base, radius)
    mask = np.zeros_like(base)
    for dy in range(2 * radius + 1):
        for dx in range(2 * radius + 1):
            mask |= padded[dy:dy + HEIGHT, dx:dx + WIDTH]
    return base.astype(np.float32), mask.astype(np.float32)
