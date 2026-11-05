"""Generate and evaluate one PI map per frozen Japan ConvLSTM sample.

CPU/NumPy only. Default: validation split, all samples, PI>0, symmetric d=3.
No model training, WSM input, ETAS fitting or test-set threshold selection.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import date
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys

import numpy as np

from pi_core import compute_pi
from pi_data import DailyCounts, atomic_json, atomic_npz, build_counts, sample_bounds
from pi_evaluation import load_training, score_sample, summaries

VERSION = 'pi_japan_4050d_change450d_positive_v1'
BASE = Path(__file__).resolve().parent
_CONTEXT = {}


def digest_object(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':')).encode()).hexdigest()


def default_location(candidates, required):
    for candidate in candidates:
        if (candidate / required).is_file():
            return candidate.resolve()
    raise FileNotFoundError(f'Cannot find {required}; specify the directory explicitly')


def initialize_worker(training_dir, data_dir, split, cache, output, identity):
    reader, geometry, scorer = load_training(training_dir)
    data = reader.CL10Data(data_dir, split, (0,), cache_size=0)
    _CONTEXT.update(reader=reader, geometry=geometry, scorer=scorer, data=data,
                    daily=DailyCounts(cache), output=Path(output), identity=identity)


def process_sample(index):
    context = _CONTEXT
    data, reader, scorer = context['data'], context['reader'], context['scorer']
    row = data.rows[index]
    name = f'sample_{index:05d}_{row["target_start"]}'
    map_path = context['output'] / 'maps' / f'{name}.npz'
    record_path = context['output'] / 'records' / f'{name}.json'
    if record_path.exists():
        saved = json.loads(record_path.read_text())
        if (not saved.get('complete') or saved.get('identity_sha256') != context['identity'] or
                saved.get('row') != index or saved.get('target_start') != row['target_start'] or
                saved.get('target_end') != row['target_end'] or
                digest_object(saved.get('metrics')) != saved.get('metrics_sha256') or
                not map_path.is_file() or reader.sha256(map_path) != saved.get('map_sha256')):
            raise ValueError(f'Cached sample differs or is incomplete: {record_path}')
        return saved
    day0 = sample_bounds(row, reader.START)
    # compute_pi never reads day >= the target start; targets are loaded only AFTER it returns.
    result = compute_pi(context['daily'], day0, input_frames=9, days_per_frame=450)
    if result.base_days != 3151 or result.image.shape != (reader.HEIGHT, reader.WIDTH):
        raise ValueError('Unexpected PI base-day count or map shape')
    if not np.isfinite(result.image).all() or np.any(result.image < 0):
        raise ValueError('PI map is nonfinite or negative')
    keys, gray = data.target_keys(row)
    target = np.zeros((reader.HEIGHT, reader.WIDTH), np.float32)
    target.flat[keys] = gray / 255.0
    stats = score_sample(result.image, target, scorer)
    metadata = {'row': index, 'target_start': row['target_start'], 'target_end': row['target_end'],
                'history_start': row['x0_start'], 'change_start': row['x8_start'],
                'base_days': result.base_days, 'nonoverlap': index % 450 == 0,
                'pi_alarm_cells': int(np.count_nonzero(result.image > 0))}
    atomic_npz(map_path, pi_map=result.image, row=np.array(index),
               target_start=np.array(row['target_start']), target_end=np.array(row['target_end']),
               identity_sha256=np.array(context['identity']))
    # Record is committed last. If interrupted before it, this sample is recomputed.
    saved = {'complete': True, 'identity_sha256': context['identity'], **metadata,
             'map_sha256': reader.sha256(map_path), 'map_file': str(map_path.relative_to(context['output'])),
             'metrics': [{**metadata, **r} for r in stats]}
    saved['metrics_sha256'] = digest_object(saved['metrics'])
    atomic_json(record_path, saved)
    return saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, help='Frozen cl10_mapping_v2 directory')
    parser.add_argument('--training-dir', type=Path, help='Existing ConvLSTM training/ directory')
    parser.add_argument('--split', choices=('train', 'validation', 'test'), default='validation')
    parser.add_argument('--workers', type=int, default=1, help='CPU worker count; no GPU is used')
    parser.add_argument('--out', type=Path, help='Separate results directory, or same identity for resume')
    parser.add_argument('--limit', type=int, help='First N samples only, labeled as partial outputs')
    parser.add_argument('--dry-run', action='store_true', help='Read/check inputs and print plan only')
    args = parser.parse_args()
    if args.workers < 1 or args.limit is not None and args.limit < 1:
        parser.error('--workers and --limit must be positive integers')
    training = args.training_dir.resolve() if args.training_dir else default_location(
        [BASE.parent / 'training', BASE.parent / 'convlstm/training'], 'evaluate_single_symmetric.py')
    root = args.data.resolve() if args.data else default_location(
        [BASE.parent / 'cl10_mapping_v2', BASE.parent / 'datasets/cl10_mapping_v2'], 'manifest.json')
    reader, geometry, scorer = load_training(training)
    data = reader.CL10Data(root, args.split, (0,), cache_size=0)
    if data.manifest.get('region_sha256') != scorer.REGION_SHA256:
        raise ValueError('Dataset grid does not match the symmetric Japan evaluator')
    indices = list(range(min(len(data), args.limit))) if args.limit is not None else list(range(len(data)))
    partial = len(indices) < len(data)
    suffix = f'{args.split}_first{len(indices)}' if partial else args.split
    out = args.out.resolve() if args.out else BASE / 'results' / VERSION / suffix
    for protected in (root, training):
        if out == protected or out in protected.parents or protected in out.parents:
            raise ValueError(f'Results must not overlap source data/training directory: {protected}')
    dataset_hashes = {key: reader.sha256(root / file) for key, file in (
        ('manifest', 'manifest.json'), ('sample_index', 'sample_index.csv'), ('usgs_daily', 'usgs_daily.npz'))}
    source_hashes = {name: reader.sha256(BASE / name) for name in (
        'pi_core.py', 'pi_data.py', 'pi_evaluation.py', 'run_pi_baseline.py')}
    shared_hashes = {name: reader.sha256(training / name) for name in (
        'cl10_data.py', 'spatial_symmetric.py', 'evaluate_cl10_symmetric.py')}
    n_days = (date.fromisoformat(reader.SPLITS['test'][1]) - reader.START).days + 1
    cache_metadata = {'version': VERSION, 'height': reader.HEIGHT, 'width': reader.WIDTH,
                      'n_days': n_days, 'calendar_start': reader.START.isoformat(),
                      'daily_source_sha256': dataset_hashes['usgs_daily'],
                      'neighborhood': '3x3 event counts; local row/column boundaries clip; no wrap'}
    identity = {'baseline_version': VERSION, 'evaluation_version': geometry.EVALUATION_VERSION,
                'split': args.split, 'full_split_count': len(data), 'selected_indices': indices,
                'partial_run': partial, 'dataset_sha256': dataset_hashes,
                'baseline_code_sha256': source_hashes, 'shared_code_sha256': shared_hashes,
                'input_days': 4050, 'change_days': 450, 'base_day_step': 1, 'base_days': 3151,
                'pi_epsilon_events_per_day': 1e-12, 'alarm_rule': 'PI>0', 'target_Mw': 4.0,
                'expand_d': 3, 'daily_counts': cache_metadata,
                'spatial_normalization': 'all 340x200 Japan grid cells including zero counts'}
    identity_hash = digest_object(identity)
    print(json.dumps({'data': str(root), 'training_dir': str(training), 'out': str(out),
                      'split': args.split, 'samples': len(indices), 'partial_run': partial,
                      'nonoverlap_samples': sum(i % 450 == 0 for i in indices),
                      'workers': args.workers, 'identity_sha256': identity_hash}, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        return
    out.mkdir(parents=True, exist_ok=True)
    # Prevent concurrent jobs from writing the same output root on Linux/macOS.
    import fcntl
    with (out / '.run.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another PI job is using this --out directory') from exc
        identity_path = out / 'identity.json'
        if identity_path.exists():
            if json.loads(identity_path.read_text()) != identity:
                raise ValueError('Input data, samples or code changed; use a new --out directory')
        elif set(p.name for p in out.iterdir()) - {'.run.lock'}:
            raise ValueError('Output is nonempty and does not contain a matching identity.json')
        else:
            atomic_json(identity_path, identity)
        for folder in ('maps', 'records'):
            (out / folder).mkdir(exist_ok=True)
        previous_manifest = out / 'manifest.json'
        if previous_manifest.is_file():
            previous = json.loads(previous_manifest.read_text())
            if previous.get('identity_sha256') != identity_hash:
                raise ValueError('Existing run manifest belongs to a different experiment')
            expected_cache = previous.get('daily_cache_sha256')
            if expected_cache is not None and (
                    not (out / 'daily_counts.npz').is_file() or
                    reader.sha256(out / 'daily_counts.npz') != expected_cache):
                raise ValueError('Existing daily-count cache changed; use a new output directory')
        atomic_json(out / 'manifest.json', {'complete': False, **identity, 'identity_sha256': identity_hash})
        cache = out / 'daily_counts.npz'
        cache_hash = build_counts(data, cache, cache_metadata, reader.sha256)
        atomic_json(out / 'manifest.json', {'complete': False, **identity,
                    'identity_sha256': identity_hash, 'daily_cache_sha256': cache_hash})
        daily = DailyCounts(cache)
        for i in indices:
            start = sample_bounds(data.rows[i], reader.START)
            if not 0 <= start < start + 4050 <= daily.metadata['n_days']:
                raise ValueError(f'Invalid historical boundary at sample {i}')
        init_args = (str(training), str(root), args.split, str(cache), str(out), identity_hash)
        records = []
        if args.workers == 1:
            initialize_worker(*init_args)
            iterator = map(process_sample, indices)
            for count, record in enumerate(iterator, 1):
                records.append(record)
                if count % 25 == 0 or count == len(indices):
                    print(f'PI generated/evaluated {count}/{len(indices)}', flush=True)
        else:
            with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize_worker,
                                     initargs=init_args, mp_context=multiprocessing.get_context('spawn')) as pool:
                for count, record in enumerate(pool.map(process_sample, indices, chunksize=1), 1):
                    records.append(record)
                    if count % 25 == 0 or count == len(indices):
                        print(f'PI generated/evaluated {count}/{len(indices)}', flush=True)
        rows = [r for record in records for r in record['metrics']]
        detail, official = summaries(rows, scorer)
        # Use the original zero-based split index, not the ordinal in a partial run.
        intervals = [(date.fromisoformat(r['target_start']), date.fromisoformat(r['target_end']))
                     for r in rows if r['protocol'] == 'neither' and r['nonoverlap']]
        if any(a[1] >= b[0] for a, b in zip(intervals, intervals[1:])):
            raise ValueError('Nonoverlap target intervals intersect')
        exports = {'by_sample.csv': rows,
                   'nonoverlap_by_sample.csv': [r for r in rows if r['nonoverlap']],
                   'summary.csv': detail, 'official_summary.csv': official,
                   'sample_manifest.csv': [{k: v for k, v in r.items() if k not in (
                       'metrics', 'metrics_sha256', 'identity_sha256', 'complete')} for r in records]}
        for name, values in exports.items():
            temporary = out / (name + '.tmp')
            scorer._write_csv(temporary, values)
            os.replace(temporary, out / name)
        atomic_json(out / 'manifest.json', {'complete': True, **identity,
                    'identity_sha256': identity_hash, 'daily_cache_sha256': cache_hash,
                    'sample_count': len(indices), 'nonoverlap_sample_count': len(intervals),
                    'output_sha256': {name: reader.sha256(out / name) for name in exports},
                    'window_standard_deviation_ddof': 0,
                    'aggregation': 'arithmetic mean of per-window metrics; undefined values excluded with valid counts',
                    'interpretation': 'daily targets overlap; window SD is not an independent-sample confidence interval'})
    print(f'Finished PI baseline: {out}', flush=True)


if __name__ == '__main__':
    main()
