"""Sparse daily event counts for the local Japan PI baseline; NumPy only.

Count every event in the frozen USGS archive. Each event contributes to its
own grid cell and eight neighbours. Both local boundaries clip; no wrapping.
This preprocessing is independent of future targets and uses no WSM fields.
"""

from __future__ import annotations

import json
import os
from datetime import date, timedelta
from pathlib import Path

import numpy as np


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def atomic_npz(path, **arrays):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('wb') as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


class DailyCounts:
    """Interface expected by the unchanged original compute_pi()."""

    def __init__(self, path):
        with np.load(path, allow_pickle=False) as archive:
            self.day_ptr = archive['day_ptr'].copy()
            self.pixel_index = archive['pixel_index'].copy()
            self.event_count = archive['event_count'].copy()
            self.metadata = json.loads(str(archive['metadata'].item()))
        n = int(self.metadata['n_days'])
        size = int(self.metadata['height']) * int(self.metadata['width'])
        if (self.day_ptr.shape != (n + 1,) or self.day_ptr[0] != 0 or
                np.any(np.diff(self.day_ptr.astype(np.int64)) < 0) or
                self.day_ptr[-1] != len(self.pixel_index) or
                len(self.pixel_index) != len(self.event_count) or
                np.any(self.pixel_index >= size) or np.any(self.event_count == 0)):
            raise ValueError('Invalid sparse daily-count archive')

    def get(self, day):
        if not 0 <= day < self.metadata['n_days']:
            raise IndexError(day)
        first, last = int(self.day_ptr[day]), int(self.day_ptr[day + 1])
        return self.pixel_index[first:last], self.event_count[first:last]


def build_counts(data, path, expected_metadata, sha256):
    """Build once per output identity, with counts for every calendar day."""
    path = Path(path)
    if path.exists():
        cached = DailyCounts(path)
        if cached.metadata != expected_metadata:
            raise ValueError('Daily-count cache belongs to different data or rules')
        return sha256(path)
    n_days = expected_metadata['n_days']
    height, width = expected_metadata['height'], expected_metadata['width']
    day_ptr = np.zeros(n_days + 1, np.int64)
    all_keys, all_counts = [], []
    if np.any(data.day < 0) or np.any(data.day >= n_days):
        raise ValueError('USGS event days are outside the frozen calendar')
    for day in range(n_days):
        lo, hi = np.searchsorted(data.day, [day, day + 1], side='left')
        keys = data.key[lo:hi]
        pieces = []
        if len(keys):
            yy, xx = np.divmod(keys, width)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    y, x = yy + dy, xx + dx
                    valid = (y >= 0) & (y < height) & (x >= 0) & (x < width)
                    pieces.append((y[valid] * width + x[valid]).astype(np.int32))
            unique, counts = np.unique(np.concatenate(pieces), return_counts=True)
            all_keys.append(unique.astype(np.int32))
            all_counts.append(counts.astype(np.int64))
            day_ptr[day + 1] = day_ptr[day] + len(unique)
        else:
            day_ptr[day + 1] = day_ptr[day]
    atomic_npz(path, day_ptr=day_ptr,
               pixel_index=np.concatenate(all_keys) if all_keys else np.empty(0, np.int32),
               event_count=np.concatenate(all_counts) if all_counts else np.empty(0, np.int64),
               metadata=np.array(json.dumps(expected_metadata, sort_keys=True)))
    return sha256(path)


def sample_bounds(row, calendar_start):
    """Source sample_index uses inclusive ends; PI uses half-open intervals."""
    target = date.fromisoformat(row['target_start'])
    if date.fromisoformat(row['target_end']) != target + timedelta(days=449):
        raise ValueError('Target must contain exactly 450 days')
    if (date.fromisoformat(row['x0_start']) != target - timedelta(days=4050) or
            date.fromisoformat(row['x8_end']) != target - timedelta(days=1)):
        raise ValueError('PI history must match all nine ConvLSTM frames')
    return (target - timedelta(days=4050) - calendar_start).days
