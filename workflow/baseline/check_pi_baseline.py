"""Optional checks for the user to run; this file does not launch PI batches."""

import argparse
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np

from pi_core import compute_pi
from pi_data import DailyCounts, build_counts, sample_bounds
from pi_evaluation import load_training, score_sample, summaries

TRAINING = None


class TinyDaily:
    def __init__(self, frames, forbidden=None):
        self.frames = frames
        self.forbidden = forbidden
        self.metadata = {'height': frames.shape[1], 'width': frames.shape[2], 'n_days': len(frames)}
    def get(self, day):
        if self.forbidden is not None and day >= self.forbidden:
            raise AssertionError('Future target data accessed')
        x = self.frames[day].ravel()
        indices = np.flatnonzero(x)
        return indices, x[indices]


def direct_pi(frames, start, input_frames, days, epsilon=1e-12):
    t2 = start + input_frames * days
    t1 = t2 - days
    changes = []
    def zscore(a):
        std = a.std(ddof=0)
        return np.zeros_like(a) if std == 0 else (a - a.mean()) / (std + epsilon)
    for base in range(start, t1 - days + 1):
        first = frames[base:t1].sum(axis=0) / (t1 - base)
        second = frames[base:t2].sum(axis=0) / (t2 - base)
        changes.append(zscore(second) - zscore(first))
    squared = np.mean(changes, axis=0) ** 2
    return np.maximum(squared - squared.mean(), 0)


class PIChecks(unittest.TestCase):
    def test_original_core_against_direct_dense_formula(self):
        frames = np.random.default_rng(17).integers(0, 3, size=(8, 3, 4)).astype(np.int64)
        result = compute_pi(TinyDaily(frames, forbidden=6), 0, 3, 2)
        expected = direct_pi(frames, 0, 3, 2)
        np.testing.assert_allclose(result.image, expected, rtol=1e-5, atol=1e-7)
        self.assertEqual(result.base_days, 3)

    def test_constant_maps_give_zero(self):
        frames = np.ones((8, 3, 4), dtype=np.int64)
        np.testing.assert_array_equal(compute_pi(TinyDaily(frames), 0, 3, 2).image, 0)

    def test_local_neighbour_counts_preserve_events_and_do_not_wrap(self):
        _, _, scorer = load_training(TRAINING)
        with TemporaryDirectory() as tmp:
            # Two different events in the same corner cell must both be counted.
            data = SimpleNamespace(day=np.array([0, 0]), key=np.array([0, 0]))
            metadata = {'n_days': 2, 'height': 3, 'width': 3}
            file = Path(tmp) / 'daily.npz'
            build_counts(data, file, metadata, scorer.sha256)
            keys, counts = DailyCounts(file).get(0)
            self.assertEqual(keys.tolist(), [0, 1, 3, 4])
            self.assertEqual(counts.tolist(), [2, 2, 2, 2])
            self.assertEqual(len(DailyCounts(file).get(1)[0]), 0)

    def test_window_means_are_not_pooled_counts(self):
        _, _, scorer = load_training(TRAINING)
        rows = []
        for index, (tp, fp) in enumerate([(1, 0), (1, 9)]):
            row = {'row': index, 'target_start': '2017-01-01', 'target_end': '2018-03-26',
                   'nonoverlap': index == 0, 'TP': tp, 'FP': fp, 'FN': 0, 'TN': 1,
                   'pred_pos': tp + fp, 'target_pos': tp, 'precision': tp / (tp + fp),
                   'recall': 1, 'CSI': tp / (tp + fp),
                   'pred_area_ratio': .1, 'target_area_ratio': .1}
            rows.extend({**row, 'protocol': p} for p, _, _ in scorer.PROTOCOLS)
        _, result = summaries(rows, scorer)
        daily = next(r for r in result if r['cohort'] == 'daily')
        self.assertAlmostEqual(daily['precision'], .55)
        self.assertNotAlmostEqual(daily['precision'], 2 / 11)

    def test_scorer_reproduces_single_channel_protocol(self):
        _, _, scorer = load_training(TRAINING)
        pi = np.zeros((340, 200), np.float32)
        target = pi.copy()
        pi[142, 100] = 1e-8  # Positive PI score must not be compared with gray 100.
        target[145, 100] = 1
        result = {r['protocol']: r for r in score_sample(pi, target, scorer)}
        self.assertEqual(result['target_only']['precision'], 1)
        self.assertEqual(result['prediction_only']['recall'], 1)

    def test_undefined_precision_is_not_zero(self):
        _, _, scorer = load_training(TRAINING)
        pi = np.zeros((340, 200), np.float32)
        target = pi.copy()
        target[100, 100] = 1
        rows = [{**r, 'nonoverlap': True} for r in score_sample(pi, target, scorer)]
        _, result = summaries(rows, scorer)
        self.assertIsNone(result[0]['precision'])
        self.assertEqual(result[0]['precision_valid_count'], 0)
        self.assertEqual(result[0]['recall'], 0)

    def test_calendar_boundaries(self):
        row = {'target_start': '2009-01-01', 'target_end': '2010-03-26',
               'x0_start': '1997-11-30', 'x8_end': '2008-12-31'}
        # Construct the start rather than relying on a manually typed date.
        from datetime import timedelta
        row['x0_start'] = (date(2009, 1, 1) - timedelta(days=4050)).isoformat()
        self.assertEqual(sample_bounds(row, date(1978, 1, 1)),
                         (date(2009, 1, 1) - timedelta(days=4050) - date(1978, 1, 1)).days)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-dir', type=Path, required=True)
    args = parser.parse_args()
    TRAINING = args.training_dir.resolve()
    unittest.main(argv=['check_pi_baseline.py'], verbosity=2)
