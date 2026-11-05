"""Collect small verified scores; never read prediction arrays or checkpoints."""
import argparse
import fcntl
import zipfile
from pathlib import Path
from compare_common import (DEFAULT_PROJECT, DEFAULT_OUTPUT, HERE, frozen, plan,
                            PIN, read_csv, read_json, sha, write_csv, write_json,
                            software_identity)

METRICS = ('precision', 'recall', 'CSI')


def collect(project, out, f, tasks, aliases):
    experiment = read_json(out / 'experiment_manifest.json')
    if (experiment['selection_sha256'] != PIN or
            experiment['vendor_sha256'] != f['vendor_sha256'] or
            experiment['compare_code_sha256'] != software_identity()):
        raise ValueError('Experiment selection or code differs from this package')
    available, windows, protocols, histories, missing, receipts = {}, [], [], [], [], {}
    for task_id, task in tasks.items():
        run = out / 'models' / task_id
        if (run / 'training_done.json').is_file():
            receipts[task_id] = {'receipt': read_json(run / 'training_done.json'),
                                 'summary': read_json(run / 'summary.json'),
                                 'training_config': read_json(run / 'run.json')}
        if (run / 'history.csv').is_file():
            histories.extend({'task': task_id, **r} for r in read_csv(run / 'history.csv'))
        for split in ('validation', 'test'):
            folder = out / 'evaluations' / task_id / split / 'evaluation'
            marker = folder / 'evaluation_manifest.json'
            if not marker.is_file():
                missing.append(f'{task_id}/{split}')
                continue
            m = read_json(marker)
            if not m.get('complete') or m.get('split') != split:
                raise ValueError(f'Incomplete evaluation: {folder}')
            if m.get('evaluation_version') != f['evaluation_version']:
                raise ValueError('Spatial protocol changed')
            for field, name in [('evaluator_sha256', 'evaluate_cl10_symmetric.py'),
                                ('spatial_module_sha256', 'spatial_symmetric.py'),
                                ('data_reader_sha256', 'cl10_data.py')]:
                if m.get(field) != f['vendor_sha256'][name]:
                    raise ValueError('Scoring code changed')
            # Only read the small files. Daily by_sample and arrays remain remote.
            for name in ('official_summary.csv', 'summary.csv',
                         'nonoverlap_by_sample.csv', 'prediction_manifest.json'):
                if sha(folder / name) != m['output_sha256'][name]:
                    raise ValueError(f'Changed score file: {folder / name}')
            pred = read_json(folder / 'prediction_manifest.json')
            if pred.get('input_intervention', {}).get('group') != task['group']:
                raise ValueError('Scores belong to a different input group')
            if (pred['input_intervention']['selection_sha256'] != PIN or
                    pred['input_intervention']['reference_checkpoint_sha256'] != task['reference']['checkpoint_sha256'] or
                    pred['dataset_manifest_sha256'] != f['dataset_sha256']['manifest']):
                raise ValueError('Scores use a different reference or dataset')
            if pred['input_intervention']['compare_code_sha256'] != experiment['compare_code_sha256']:
                raise ValueError('Worker input transform code differs from batch')
            receipt = read_json(run / 'training_done.json')
            if pred['best_checkpoint_sha256'] != receipt['files_sha256']['best.pt']:
                raise ValueError('Scored checkpoint differs from completed training')
            rows = read_csv(folder / 'official_summary.csv')
            if len(rows) != 2 or {r['cohort'] for r in rows} != {'daily', 'nonoverlap'}:
                raise ValueError('Unexpected cohort rows')
            for r in rows:
                if int(r['sample_count']) != {'daily': 2473, 'nonoverlap': 6}[r['cohort']]:
                    raise ValueError('Unexpected score window count')
                available[task_id, split, r['cohort']] = r
            windows.extend({'task': task_id, 'group': task['group'], 'split': split, **r}
                           for r in read_csv(folder / 'nonoverlap_by_sample.csv'))
            protocols.extend({'task': task_id, 'group': task['group'], 'split': split, **r}
                             for r in read_csv(folder / 'summary.csv'))
    reports = []
    for m in f['models']:
        for split in ('validation', 'test'):
            for cohort in ('daily', 'nonoverlap'):
                real = f['reference_scores'][m['model']][split][cohort]
                for group in ('REAL', 'A', 'B', 'C'):
                    task_id = '' if group == 'REAL' else aliases[m['model']][group]
                    row = real if group == 'REAL' else available.get((task_id, split, cohort))
                    if row is None:
                        continue
                    record = {'model': m['model'], 'selected_names': m['selected_names'],
                              'selected_seed': m['seed'], 'init_seed': m['init_seed'],
                              'order_seed': m['order_seed'], 'group': group, 'task': task_id,
                              'split': split, 'cohort': cohort, 'sample_count': row['sample_count'],
                              'logical_selected_channels': ','.join(map(str, m['selected_channels'])),
                              'trained_selected_channels': ','.join(map(str, m['selected_channels'] if group == 'REAL' else tasks[task_id]['reference']['selected_channels']))}
                    for metric in METRICS:
                        value = float(row[metric]) if row[metric] not in ('', None) else None
                        sd_key = metric + ('_window_sd' if group == 'REAL' else '_std')
                        sd = float(row[sd_key]) if row[sd_key] not in ('', None) else None
                        reference = float(real[metric])
                        record[metric] = value
                        record[metric + '_window_sd'] = sd
                        record[metric + '_valid_count'] = row[metric + '_valid_count']
                        record[metric + '_delta_vs_REAL_pp'] = 100 * (value - reference) if value is not None else None
                    reports.append(record)
    compact = out / 'compact_compare_summary'
    compact.mkdir(exist_ok=True)
    files = {'comparison.csv': reports, 'nonoverlap_by_protocol.csv': windows,
             'protocol_summary.csv': protocols, 'training_history.csv': histories}
    for name, rows in files.items():
        write_csv(compact / name, rows)
    write_json(compact / 'frozen_reference.json', f)
    write_json(compact / 'experiment_manifest.json', read_json(out / 'experiment_manifest.json'))
    write_json(compact / 'task_aliases.json', aliases)
    write_json(compact / 'training_receipts.json', receipts)
    names = [n for n in files if files[n]] + ['frozen_reference.json', 'experiment_manifest.json', 'task_aliases.json', 'training_receipts.json']
    write_json(compact / 'summary_manifest.json', {
        'complete': not missing, 'missing_evaluations': missing,
        'expected_unique_training_tasks': len(tasks), 'logical_controls': len(f['models']) * 3,
        'available_evaluations': len(available) // 2,
        'original_real_feature_models_retrained': False,
        'A_reuse': 'Same channel count, seeds, training fingerprint and recorded initialization; aliases retained',
        'metric_definition': 'P target_only; R prediction_only; CSI expanded_both; window SD ddof=0',
        'hash_verification': 'Small evaluation outputs only; no independent daily score recomputation',
        'files_sha256': {name: sha(compact / name) for name in names}})
    archive = out / 'compact_compare_summary.zip'
    temporary = archive.with_suffix('.zip.partial')
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as z:
        for name in names + ['summary_manifest.json']:
            z.write(compact / name, arcname=name)
    temporary.replace(archive)
    print('Complete:', not missing, '; download:', archive, flush=True)
    return not missing


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root', type=Path, default=DEFAULT_PROJECT)
    p.add_argument('--out', type=Path)
    a = p.parse_args()
    f = frozen()
    tasks, aliases = plan(f)
    out = a.out.resolve() if a.out else a.project_root.resolve() / DEFAULT_OUTPUT
    if not (out / 'experiment_manifest.json').is_file():
        p.error('No recognized experiment output')
    with (out / '.run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        complete = collect(a.project_root.resolve(), out, f, tasks, aliases)
    if not complete:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
