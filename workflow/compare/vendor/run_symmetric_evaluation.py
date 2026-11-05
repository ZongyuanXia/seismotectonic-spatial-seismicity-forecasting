"""Re-evaluate frozen single/dual/triple-channel CL10 models, without training.

Default split is validation. Test requires an explicit --split test argument.
All completed seeds are reported; this script never selects a test maximum.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from cl10_data import CHANNEL_ORDER, sha256
from spatial_symmetric import EVALUATION_VERSION

DEFAULT_PROJECT = Path('/share/home/22460734/project_MP_1978_2024')
PAIRS = ((1, 2), (1, 7), (2, 7), (3, 4), (3, 8), (3, 9),
         (4, 8), (4, 9), (8, 9), (5, 6))
METRICS = ('precision', 'recall', 'CSI')


@dataclass(frozen=True)
class Case:
    model: str
    seed: str
    run: Path
    channels: tuple[int, ...]
    predictions: Path | None = None


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def read_csv(path):
    with path.open(newline='', encoding='utf-8-sig') as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    if not rows:
        raise ValueError(f'No rows to write: {path}')
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def default_cases(project, family, seeds):
    cases = []
    for seed in seeds:
        cases.append(Case('MAG', seed, project / 'runs/mag_w0_000175' / seed, (0,)))
        if family in ('dual', 'all'):
            for index in range(1, 10):
                if seed == 'run_03':
                    run = project / 'runs/mag_plus_one_w0_000175' / f'channel_{index:02d}' / seed
                elif index == 6:
                    run = project / 'runs/mag_plus_gini_seed5_w0_000175' / seed
                else:
                    run = project / 'runs/mag_plus_one_seed5_w0_000175' / f'channel_{index:02d}' / seed
                cases.append(Case(f'channel_{index:02d}', seed, run, (0, index)))
        if family in ('triple', 'all'):
            for first, second in PAIRS:
                model = f'pair_{first:02d}_{second:02d}'
                run = project / 'runs/mag_plus_two_same_family_seed5_w0_000175' / seed / model
                cases.append(Case(model, seed, run, (0, first, second)))
    return cases


def custom_cases(path, project, family):
    """CSV columns: model,seed,run_dir; optional predictions (absolute or project-relative)."""
    cases = []
    for row in read_csv(path):
        run = Path(row['run_dir'])
        run = run if run.is_absolute() else project / run
        channels = tuple(read_json(run / 'run.json')['selected_channels'])
        allowed = {'single': (1,), 'dual': (1, 2), 'triple': (1, 3), 'all': (1, 2, 3)}[family]
        if len(channels) not in allowed:
            raise ValueError(f'Wrong channel count for {family}: {run}')
        pred = Path(row['predictions']) if row.get('predictions') else None
        if pred is not None and not pred.is_absolute():
            pred = project / pred
        cases.append(Case(row['model'], row['seed'], run, channels, pred))
    return cases


def _safe_label(value):
    if not value or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in value):
        raise ValueError(f'Use letters, numbers, underscores or hyphens in case labels: {value!r}')


def preflight(cases, data):
    if not cases:
        raise ValueError('No cases were specified')
    seen = set()
    data_hashes = {key: sha256(data / file) for key, file in (
        ('manifest', 'manifest.json'), ('sample_index', 'sample_index.csv'),
        ('usgs_daily', 'usgs_daily.npz'))}
    configs = {}
    baselines = {}
    for case in cases:
        for label in (case.model, case.seed):
            _safe_label(label)
        key = case.model, case.seed
        if key in seen:
            raise ValueError(f'Duplicate case: {key}')
        seen.add(key)
        for filename in ('run.json', 'summary.json', 'best.pt'):
            if not (case.run / filename).is_file():
                raise FileNotFoundError(f'Missing frozen model file: {case.run / filename}')
        config = read_json(case.run / 'run.json')
        channels = tuple(config['selected_channels'])
        if (channels != case.channels or not 1 <= len(channels) <= 3 or
                channels[0] != 0 or len(set(channels)) != len(channels) or
                any(type(i) is not int or not 0 <= i < len(CHANNEL_ORDER) for i in channels)):
            raise ValueError(f'Unexpected selected channels: {case.run}')
        if config.get('dataset_sha256') != data_hashes:
            raise ValueError(f'Training dataset differs: {case.run}')
        best_epoch = read_json(case.run / 'summary.json').get('best_epoch')
        if type(best_epoch) is not int or best_epoch < 1:
            raise ValueError(f'Invalid completed training summary: {case.run}')
        if (case.model == 'MAG') != (channels == (0,)):
            raise ValueError('The MAG label is reserved for the single-channel baseline')
        configs[key] = config
        if channels == (0,):
            baselines[case.seed] = config
    for case in cases:
        if case.seed not in baselines:
            raise ValueError(f'Missing same-seed MAG baseline: {case.seed}')
        base, current = baselines[case.seed], configs[case.model, case.seed]
        # Same seed label must describe the same actual initialization/order seeds.
        for field in ('init_seed', 'order_seed', 'seed'):
            if current.get(field) != base.get(field):
                raise ValueError(f'{field} differs from MAG: {case.model}/{case.seed}')
        if current.get('training', {}).get('weights') != base.get('training', {}).get('weights'):
            raise ValueError(f'Loss weights differ from MAG: {case.model}/{case.seed}')
    return data_hashes


def _run(script, *arguments):
    print('Running:', script.name, *map(str, arguments), flush=True)
    subprocess.run([sys.executable, str(script), *map(str, arguments)], check=True)


def evaluation_identity(case, data, split, scripts):
    return {'evaluation_version': EVALUATION_VERSION, 'split': split,
            'prediction_checkpoint_sha256': sha256(case.run / 'best.pt'),
            'prediction_best_epoch': read_json(case.run / 'summary.json')['best_epoch'],
            'selected_channels': list(case.channels),
            'dataset_manifest_sha256': sha256(data / 'manifest.json'),
            'evaluator_sha256': sha256(scripts / 'evaluate_cl10_symmetric.py'),
            'spatial_module_sha256': sha256(scripts / 'spatial_symmetric.py'),
            'data_reader_sha256': sha256(scripts / 'cl10_data.py')}


def verify_evaluation(path, expected):
    report = read_json(path / 'evaluation_manifest.json')
    if not report.get('complete') or any(report.get(k) != v for k, v in expected.items()):
        raise ValueError(f'Evaluation version, model or data changed: {path}')
    hashes = report.get('output_sha256', {})
    for name in ('by_sample.csv', 'nonoverlap_by_sample.csv', 'summary.csv',
                 'official_summary.csv', 'prediction_manifest.json'):
        if sha256(path / name) != hashes.get(name):
            raise ValueError(f'Evaluation output changed: {path / name}')
    return report


def evaluate_case(case, data, split, out, device, scripts):
    folder = out / 'cases' / case.model / case.seed
    result = folder / 'evaluation'
    expected = evaluation_identity(case, data, split, scripts)
    if result.exists():
        verify_evaluation(result, expected)
        print('Verified and reused NEW evaluation:', case.model, case.seed, flush=True)
    else:
        predictions = case.predictions
        if predictions is None:
            old = case.run / f'{split}_predictions'
            predictions = old if (old / 'manifest.json').is_file() else folder / 'predictions'
        if not predictions.exists():
            _run(scripts / 'predict_cl10.py', '--data', data, '--run', case.run,
                 '--split', split, '--out', predictions, '--device', device)
        elif not (predictions / 'manifest.json').is_file():
            raise ValueError(f'Incomplete prediction export; preserve or relocate before retrying: {predictions}')
        # Existing exports are fully verified by the evaluator before scoring.
        _run(scripts / 'evaluate_cl10_symmetric.py', '--data', data, '--run', case.run,
             '--predictions', predictions, '--split', split, '--out', result)
        verify_evaluation(result, expected)
    rows = read_csv(result / 'official_summary.csv')
    if {r['cohort'] for r in rows} != {'daily', 'nonoverlap'} or len(rows) != 2:
        raise ValueError(f'Missing or duplicated evaluation cohort: {result}')
    config = read_json(case.run / 'run.json')
    return [{**r, 'model': case.model, 'seed': case.seed,
             'input_channel_count': len(case.channels),
             'selected_channels': ','.join(map(str, case.channels)),
             'selected_names': ','.join(CHANNEL_ORDER[i] for i in case.channels),
             'init_seed': config.get('init_seed', ''), 'order_seed': config.get('order_seed', ''),
             'checkpoint_sha256': expected['prediction_checkpoint_sha256'],
             'run_json_sha256': sha256(case.run / 'run.json'),
             'evaluation_manifest_sha256': sha256(result / 'evaluation_manifest.json')}
            for r in rows]


def _number(value):
    return float(value) if value not in ('', None) else None


def paired_table(rows):
    base = {(r['seed'], r['cohort']): r for r in rows if r['model'] == 'MAG'}
    result = []
    for row in rows:
        if row['model'] == 'MAG':
            continue
        ref = base[row['seed'], row['cohort']]
        pair = {'evaluation_version': EVALUATION_VERSION, 'model': row['model'],
                'seed': row['seed'], 'cohort': row['cohort'],
                'sample_count': row['sample_count']}
        for metric in METRICS:
            a, b = _number(ref[metric]), _number(row[metric])
            pair[f'MAG_{metric}'], pair[metric] = a, b
            pair[f'{metric}_delta_pp'] = 100 * (b - a) if a is not None and b is not None else None
        result.append(pair)
    return result


def aggregate_table(rows):
    result = []
    for model, cohort in sorted({(r['model'], r['cohort']) for r in rows}):
        selected = [r for r in rows if (r['model'], r['cohort']) == (model, cohort)]
        entry = {'evaluation_version': EVALUATION_VERSION, 'model': model,
                 'cohort': cohort, 'total_seed_count': len(selected)}
        for metric in METRICS:
            values = [_number(r[metric]) for r in selected]
            values = [x for x in values if x is not None]
            entry[f'{metric}_valid_seed_count'] = len(values)
            entry[f'{metric}_mean'] = statistics.mean(values) if values else None
            entry[f'{metric}_sample_sd'] = statistics.stdev(values) if len(values) > 1 else None
        result.append(entry)
    return result


def main(fixed_family=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=DEFAULT_PROJECT)
    parser.add_argument('--data', type=Path, help='Defaults to project/cl10_mapping_v2')
    parser.add_argument('--family', choices=('single', 'dual', 'triple', 'all'), default=fixed_family or 'all')
    parser.add_argument('--split', choices=('validation', 'test'), default='validation')
    parser.add_argument('--seeds', nargs='+', default=[f'run_{i:02d}' for i in range(1, 6)])
    parser.add_argument('--cases-csv', type=Path, help='Custom frozen run locations; model,seed,run_dir[,predictions]')
    parser.add_argument('--out', type=Path, help='New isolated output root; repeat to verify and resume completed cases')
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--dry-run', action='store_true', help='Validate sources and list cases, without prediction/evaluation')
    args = parser.parse_args()
    if fixed_family is not None and args.family != fixed_family:
        parser.error(f'This entry point is for {fixed_family} models')
    if len(args.seeds) != len(set(args.seeds)):
        parser.error('Duplicate seeds')
    project = args.project_root.resolve()
    data = args.data.resolve() if args.data else project / 'cl10_mapping_v2'
    cases = (custom_cases(args.cases_csv, project, args.family) if args.cases_csv else
             default_cases(project, args.family, args.seeds))
    dataset_sha = preflight(cases, data)
    out = (args.out.resolve() if args.out else project / 'runs' / EVALUATION_VERSION / args.family / args.split)
    # Do not allow an output tree to contain source data or trained checkpoints.
    for source in [data, *(case.run.resolve() for case in cases)]:
        if out == source or out in source.parents or source in out.parents:
            raise ValueError(f'Output overlaps source data/model directory: {source}')
    print(f'{len(cases)} frozen models; split={args.split}; output={out}', flush=True)
    for case in cases:
        print(case.model, case.seed, case.run, flush=True)
    if args.dry_run:
        return
    scripts = Path(__file__).resolve().parent
    identity = {'evaluation_version': EVALUATION_VERSION, 'split': args.split,
                'dataset_sha256': dataset_sha, 'family': args.family,
                'evaluator_sha256': sha256(scripts / 'evaluate_cl10_symmetric.py'),
                'spatial_module_sha256': sha256(scripts / 'spatial_symmetric.py'),
                'cases': [{'model': c.model, 'seed': c.seed, 'run': str(c.run.resolve()),
                           'checkpoint_sha256': sha256(c.run / 'best.pt'),
                           'run_json_sha256': sha256(c.run / 'run.json')} for c in cases]}
    out.mkdir(parents=True, exist_ok=True)
    identity_path = out / 'batch_identity.json'
    if identity_path.exists():
        if read_json(identity_path) != identity:
            raise ValueError('Batch inputs changed: use a NEW --out directory')
    elif any(out.iterdir()):
        raise ValueError('Output is nonempty and is not a recognized new evaluation batch')
    else:
        identity_path.write_text(json.dumps(identity, ensure_ascii=False, indent=2) + '\n')
    rows = []
    for case in cases:
        rows.extend(evaluate_case(case, data, args.split, out, args.device, scripts))
    write_csv(out / 'per_run_summary.csv', rows)
    write_csv(out / 'between_runs_summary.csv', aggregate_table(rows))
    pairs = paired_table(rows)
    if pairs:
        write_csv(out / 'paired_vs_MAG.csv', pairs)
    files = ['per_run_summary.csv', 'between_runs_summary.csv'] + (['paired_vs_MAG.csv'] if pairs else [])
    (out / 'batch_manifest.json').write_text(json.dumps({
        **identity, 'complete': True, 'model_count': len(cases),
        'summary_sha256': {f: sha256(out / f) for f in files},
        'selection': 'none; all seeds retained; no best-of-test selection',
    }, ensure_ascii=False, indent=2) + '\n')
    print('Completed:', out, flush=True)


if __name__ == '__main__':
    main()
