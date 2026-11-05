"""Extend the frozen one-window ETAS trial to six disjoint validation targets.

Uses the existing trusted fit and author simulator. Each forecast conditions
only on its preceding 4050 days; fitted parameters/background remain frozen.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

P = None
WINDOW_ROW = 0
SIMULATIONS = 2048


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(p, value):
    temp = Path(str(p) + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(p)


def initialize(root, fit, origin, window_row, simulations):
    global P, WINDOW_ROW, SIMULATIONS
    root = Path(root)
    sys.path.insert(0, str(root / 'baseline/etas'))
    import predict_etas_trial as engine
    P = engine
    WINDOW_ROW, SIMULATIONS = window_row, simulations
    P.initialize(fit, root / 'baseline/etas/region_meta.json')
    events = pd.read_csv(root / 'datasets/cl10_mapping_v2/usgs_event_audit.csv')
    events['time'] = pd.to_datetime(events['time'], format='ISO8601', utc=True).dt.tz_localize(None)
    start = pd.Timestamp(origin) - pd.Timedelta(days=4050)
    stop = pd.Timestamp(origin)
    history = events.loc[(events.time >= start) & (events.time < stop)].copy()
    history = history.rename(columns={'MW': 'magnitude'}).set_index('id')
    history = history[['latitude', 'longitude', 'time', 'magnitude']].sort_values('time')
    assert history.index.is_unique and len(history) and (history.magnitude >= 4).all()
    assert history.time.min() >= start and history.time.max() < stop
    history['xi_plus_1'] = 1.
    P.CTX.update(history=history, origin=stop, end=stop + pd.Timedelta(days=450), history_start=start)


def simulate(i):
    # The original first window uses row=0 and therefore preserves its seed sequence.
    global_index = WINDOW_ROW * SIMULATIONS + i
    _, unique, keys, record = P.generate_one(global_index)
    record.update(simulation=i, global_simulation_index=global_index, window_row=WINDOW_ROW)
    return i, unique, keys, record


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args()
    root, out = args.root.resolve(), args.out.resolve()
    if args.workers < 1:
        ap.error('workers must be positive')
    n = 2048
    sys.path.insert(0, str(root / 'convlstm/training'))
    import cl10_data as reader
    data_root = root / 'datasets/cl10_mapping_v2'
    data = reader.CL10Data(data_root, 'validation', (0,), cache_size=0)
    rows = [0, 450, 900, 1350, 1800, 2250]
    dates = [(pd.Timestamp(data.rows[i]['target_start']), pd.Timestamp(data.rows[i]['target_end'])) for i in rows]
    assert all(a[1] < b[0] for a, b in zip(dates, dates[1:]))
    fit_root = root / 'baseline/etas/results/validation_00000_20090101_v2'
    fit_meta = json.loads((fit_root / 'fit_manifest.json').read_text())
    prediction_meta = json.loads((fit_root / 'prediction_manifest.json').read_text())
    verification = json.loads((fit_root / 'verification.json').read_text())
    assert fit_meta['complete'] and prediction_meta['complete'] and prediction_meta['simulations'] == n
    assert pd.Timestamp(fit_meta['training_max_time']) < dates[0][0]
    assert sha(fit_root / 'fit.pkl') == verification['output_sha256']['fit.pkl']
    assert sha(fit_root / 'prediction.npz') == prediction_meta['prediction_sha256']
    assert sha(data_root / 'usgs_event_audit.csv') == fit_meta['sha256']['event_audit']
    assert sha(data_root / 'manifest.json') == fit_meta['sha256']['dataset_manifest']
    assert sha(root / 'baseline/etas/predict_etas_trial.py') == prediction_meta['script_sha256']
    assert sha(root / 'baseline/etas/vendor/etas-main/etas/simulation.py') == prediction_meta['author_simulation_sha256']
    identity = {'version': 'etas_six_validation_frozen_2009_v1', 'simulations': n,
                'selected_rows': rows, 'fit_sha256': sha(fit_root / 'fit.pkl'),
                'fit_manifest_sha256': sha(fit_root / 'fit_manifest.json'),
                'script_sha256': sha(__file__), 'original_adapter_sha256': prediction_meta['script_sha256'],
                'event_audit_sha256': sha(data_root / 'usgs_event_audit.csv'),
                'seed_formula': '202610290 + validation_row * 2048 + simulation_index',
                'fitted_parameters_and_background_updated': False, 'history_days': 4050,
                'all_generations': True, 'first_window_reused': True, 'ETAS_trial_only': True}
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'identity.json').exists():
        assert json.loads((out / 'identity.json').read_text()) == identity, 'Output identity changed'
    else:
        if any(out.iterdir()):
            raise FileExistsError('Nonempty output without identity')
        dump(out / 'identity.json', identity)
    dump(out / 'manifest.json', {'complete': False, **identity})
    windows = []
    for index in rows:
        row = data.rows[index]
        origin, end = pd.Timestamp(row['target_start']), pd.Timestamp(row['target_end']) + pd.Timedelta(days=1)
        name = f'ETAS_{index:05d}'
        dest, meta = out / (name + '.npz'), out / (name + '.json')
        if meta.exists():
            saved = json.loads(meta.read_text())
            assert saved['complete'] and saved['row'] == index and saved['target_start'] == row['target_start']
            assert saved['prediction_sha256'] == sha(dest)
            windows.append(saved)
            print(f'Reused completed ETAS row {index}', flush=True)
            continue
        print(f'ETAS row {index}: {origin.date()} to {(end-pd.Timedelta(days=1)).date()}', flush=True)
        if index == 0:
            shutil.copy2(fit_root / 'prediction.npz', dest)
            dg = json.loads((fit_root / 'diagnostics.json').read_text())
            stats = {'events_mean': dg['events_mean'], 'events_p05_p95': dg['events_p05_p95'],
                     'events_max': dg['events_max'], 'reused_source_sha256': prediction_meta['prediction_sha256']}
        else:
            presence = np.zeros((340, 200), np.int64)
            counts = np.zeros_like(presence)
            halves = np.zeros((2, 340, 200), np.int64)
            records = []
            with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('spawn'),
                    initializer=initialize, initargs=(str(root), str(fit_root/'fit.pkl'), str(origin), index, n)) as pool:
                for i, unique, keys, record in pool.map(simulate, range(n), chunksize=1):
                    presence.flat[unique] += 1
                    np.add.at(counts.ravel(), keys, 1)
                    halves[int(i >= n//2)].flat[unique] += 1
                    records.append(record)
                    if (i+1) % 256 == 0 or i+1 == n:
                        print(f'ETAS row {index}: {i+1}/{n}', flush=True)
            with Path(str(dest) + '.tmp').open('wb') as f:
                np.savez_compressed(f, probability=(presence/n).astype(np.float32),
                    expected_event_count=(counts/n).astype(np.float32), presence_count=presence,
                    event_count=counts, half_presence_count=halves, row=np.array(index),
                    origin=np.array(row['target_start']), end_exclusive=np.array(str(end.date())), simulations=np.array(n))
            Path(str(dest) + '.tmp').replace(dest)
            import csv
            with (out / (name + '_simulations.csv')).open('w', newline='', encoding='utf-8-sig') as f:
                w = csv.DictWriter(f, fieldnames=list(records[0])); w.writeheader(); w.writerows(records)
            values = [x['events_in_region'] for x in records]
            stats = {'events_mean': float(np.mean(values)), 'events_p05_p95': np.percentile(values, [5,95]).tolist(),
                     'events_max': int(max(values)), 'seed_first': records[0]['seed'], 'seed_last': records[-1]['seed']}
        history_start = origin - pd.Timedelta(days=4050)
        saved = {'complete': True, 'row': index, 'target_start': row['target_start'], 'target_end': row['target_end'],
                 'end_exclusive': str(end.date()), 'history_start': str(history_start.date()),
                 'history_end_exclusive': row['target_start'], 'prediction_file': dest.name,
                 'prediction_sha256': sha(dest), 'simulations': n, 'diagnostics': stats}
        dump(meta, saved)
        windows.append(saved)
    dump(out / 'manifest.json', {'complete': True, **identity, 'windows': windows,
                                'completed_at': datetime.now().isoformat()})
    print('Completed all six ETAS forecasts:', out, flush=True)


if __name__ == '__main__':
    main()
