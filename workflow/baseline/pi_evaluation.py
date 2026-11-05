"""Use the exact current ConvLSTM symmetric scorer; never scale PI by 255."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np


def load_training(training_dir):
    directory = Path(training_dir).resolve()
    required = ('cl10_data.py', 'spatial_symmetric.py', 'evaluate_cl10_symmetric.py')
    for name in required:
        if not (directory / name).is_file():
            raise FileNotFoundError(f'Missing shared ConvLSTM evaluator dependency: {directory / name}')
    sys.path.insert(0, str(directory))
    modules = [importlib.import_module(name[:-3]) for name in required]
    for name, module in zip(required, modules):
        if Path(module.__file__).resolve() != directory / name:
            raise ValueError(f'A different {name} is already imported')
    reader, geometry, scorer = modules
    if (geometry.EVALUATION_VERSION != 'symmetric_band_union_v1' or
            scorer.EXPAND_D != 3 or scorer.THRESHOLD_GRAY != 100 or
            scorer.NONOVERLAP_DAYS != 450 or scorer.EVALUATION_VERSION != geometry.EVALUATION_VERSION):
        raise ValueError('The shared evaluator is not the agreed d=3 symmetric version')
    return reader, geometry, scorer


def score_sample(pi, raw_target, scorer):
    """PI>0 is a binary alarm; true MAG is thresholded by the shared scorer."""
    if pi.shape != raw_target.shape or not np.isfinite(pi).all() or np.any(pi < 0):
        raise ValueError('Invalid PI score map')
    alarm = (pi > 0).astype(np.float32)  # 0 or 1, not a magnitude/PI-gray mapping.
    expanded_alarm = scorer.archived_expand(alarm)
    expanded_target = scorer.archived_expand(raw_target)
    rows = []
    for protocol, pe, te in scorer.PROTOCOLS:
        rows.append({'protocol': protocol, 'prediction_expanded': pe, 'target_expanded': te,
                     **scorer.counts(expanded_alarm if pe else alarm,
                                     expanded_target if te else raw_target)})
    return rows


def summaries(rows, scorer):
    """Exactly the same per-window mean/SD and valid-count handling as MAG."""
    detail = [scorer._summary(rows, protocol, cohort)
              for cohort in ('daily', 'nonoverlap')
              for protocol, _, _ in scorer.PROTOCOLS]
    official = []
    for cohort in ('daily', 'nonoverlap'):
        table = {r['protocol']: r for r in detail if r['cohort'] == cohort}
        row = {'model': 'PI', 'evaluation_version': scorer.EVALUATION_VERSION,
               'metric_protocol': 'symmetric_neighborhood_PR_and_expanded_both_CSI',
               'cohort': cohort, 'sample_count': table['both']['sample_count'],
               'pi_alarm': 'PI>0', 'target_threshold_gray': 100, 'expand_d': 3}
        for metric, protocol in [('precision', 'target_only'), ('recall', 'prediction_only'), ('CSI', 'both')]:
            for suffix in ('mean', 'std', 'valid_count'):
                key = metric if suffix == 'mean' else f'{metric}_{suffix}'
                row[key] = table[protocol][f'{metric}_{suffix}']
        row['CSI_expanded_both'] = row['CSI']
        row['raw_pred_area_ratio'] = table['neither']['pred_area_ratio_mean']
        row['expanded_pred_area_ratio'] = table['both']['pred_area_ratio_mean']
        official.append(row)
    return detail, official
