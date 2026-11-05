"""Change only added input values; retain dates, MAG, grid and targets."""
import numpy as np
from cl10_data import CL10Data


def placeholder_values(gray, valid, channels, group):
    """gray is 0..255; valid has 9 columns (dataset channels 1..9)."""
    if group not in ('A', 'B', 'C'):
        raise ValueError('Unknown group')
    if channels[0] != 0 or len(channels) not in (2, 3):
        raise ValueError('Expected MAG plus one or two channels')
    values = gray[:, channels].astype(np.float32, copy=True)
    mask = valid[:, np.asarray(channels[1:]) - 1]
    if not np.array_equal(mask, values[:, 1:] > 0):
        raise ValueError('Original valid mask and gray support differ')
    values[:, 1:] = 0 if group == 'A' else np.where(mask, 100 if group == 'B' else 255, 0)
    return values


def dataset_class(group):
    class PlaceholderData(CL10Data):
        def _load_frame(self, start):
            if start in self._cache:
                self._cache.move_to_end(start)
                return self._cache[start]
            with np.load(self.root / 'frames' / f'{start}.npz', allow_pickle=False) as frame:
                keys = frame['keys'].astype(np.int32)
                values = placeholder_values(frame['gray'], frame['valid'], self.channels, group)
            if values.shape != (len(keys), len(self.channels)):
                raise ValueError('Wrong transformed frame shape')
            result = keys, values
            if self.cache_size:
                self._cache[start] = result
                if len(self._cache) > self.cache_size:
                    self._cache.popitem(last=False)
            return result
    return PlaceholderData
