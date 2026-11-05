"""Frozen source identity, reusable A controls, and safe output handling."""
from pathlib import Path
import csv
import hashlib
import json

HERE = Path(__file__).resolve().parent
DEFAULT_PROJECT = Path('/share/home/22460734/project_MP_1978_2024')
DEFAULT_OUTPUT = Path('runs/compare_abc_v1')
SELECTION_FILE = HERE / 'frozen_compare.json'
PIN = '2e371699d189ffc56696b158c3336c9f62cf01ec80e157171075ccb9a399fec1'
DATA_FILES = {'manifest': 'manifest.json', 'sample_index': 'sample_index.csv',
              'usgs_daily': 'usgs_daily.npz'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, data):
    path = Path(path)
    tmp = path.with_name(path.name + '.partial')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    tmp.replace(path)


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    tmp = path.with_name(path.name + '.partial')
    with tmp.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)


def frozen():
    if sha(SELECTION_FILE) != PIN:
        raise ValueError('Fixed selection or archived reference scores changed')
    f = read_json(SELECTION_FILE)
    for name, digest in f['vendor_sha256'].items():
        if sha(HERE / 'vendor' / name) != digest:
            raise ValueError(f'Archived dependency changed: {name}')
    return f


def software_identity():
    return {p.name: sha(p) for p in sorted(HERE.glob('*.py'))}


def source_run(project, model, seed):
    if model.startswith('channel_'):
        index = int(model.split('_')[1])
        if seed == 'run_03':
            return project / 'runs/mag_plus_one_w0_000175' / model / seed
        if index == 6:
            return project / 'runs/mag_plus_gini_seed5_w0_000175' / seed
        return project / 'runs/mag_plus_one_seed5_w0_000175' / model / seed
    return project / 'runs/mag_plus_two_same_family_seed5_w0_000175' / seed / model


def plan(f):
    tasks, aliases, a_keys = {}, {}, {}
    for m in f['models']:
        aliases[m['model']] = {}
        for group in ('A', 'B', 'C'):
            if group == 'A':
                key = (len(m['selected_channels']), m['init_seed'], m['order_seed'])
                task_id = a_keys.setdefault(key, f'A_c{key[0]}_{m["seed"]}')
            else:
                task_id = f'{group}_{m["model"]}_{m["seed"]}'
            tasks.setdefault(task_id, {'task_id': task_id, 'group': group, 'reference': m})
            aliases[m['model']][group] = task_id
    return tasks, aliases


def fingerprint(config):
    fields = ('dataset_sha256', 'dataset_mapping_revision', 'channel_order', 'train_count',
              'validation_count', 'mask', 'train_positive_pixel_fraction', 'training',
              'seed_scheme', 'init_seed', 'order_seed')
    return {k: config[k] for k in fields} | {
        'channel_count': len(config['selected_channels']),
        'model_code': config['software']['model_script_sha256'],
        'data_code': config['software']['data_script_sha256']}


def preflight(project, data, f, tasks, aliases):
    for key, name in DATA_FILES.items():
        if sha(data / name) != f['dataset_sha256'][key]:
            raise ValueError(f'Dataset differs from real-feature reference: {name}')
    configs = {}
    for m in f['models']:
        run = source_run(project, m['model'], m['seed'])
        config = read_json(run / 'run.json')
        if sha(run / 'best.pt') != m['checkpoint_sha256']:
            raise ValueError(f'Real-feature checkpoint changed: {run}')
        if config['selected_channels'] != m['selected_channels']:
            raise ValueError(f'Wrong source channels: {run}')
        for key in ('init_seed', 'order_seed'):
            if config[key] != m[key]:
                raise ValueError(f'Wrong source {key}: {run}')
        if config['dataset_sha256'] != f['dataset_sha256']:
            raise ValueError(f'Source training data differ: {run}')
        for key, name in [('data_script_sha256', 'cl10_data.py'),
                          ('model_script_sha256', 'model_cl10.py')]:
            if config['software'][key] != f['vendor_sha256'][name]:
                raise ValueError(f'Source implementation differs: {run}/{name}')
        if config['training']['weights'] != [0.000175, 0.03, 1.0]:
            raise ValueError(f'Unexpected source loss weights: {run}')
        if read_json(run / 'summary.json')['best_epoch'] != m['best_epoch']:
            raise ValueError(f'Source best epoch changed: {run}')
        base = project / 'runs/mag_w0_000175' / m['seed']
        for name in ('run.json', 'summary.json', 'best.pt'):
            if not (base / name).is_file():
                raise FileNotFoundError(base / name)
        configs[m['model']] = config
    for model, group_map in aliases.items():
        representative = tasks[group_map['A']]['reference']['model']
        if fingerprint(configs[model]) != fingerprint(configs[representative]):
            raise ValueError('A-sharing training conditions differ; cannot reuse the control')
        a, b = configs[model].get('initialization'), configs[representative].get('initialization')
        if a is not None or b is not None:
            if a != b:
                raise ValueError('A-sharing source initial states differ')
    return configs


def task_run(out, task):
    return out / 'models' / task['task_id']


def intervention(task, project):
    m = task['reference']
    ref = source_run(project, m['model'], m['seed'])
    return {'version': 'compare_abc_v1', 'group': task['group'],
            'gray_value': {'A': 0, 'B': 100, 'C': 255}[task['group']],
            'support': 'all zero' if task['group'] == 'A' else 'stored frame valid mask',
            'mag_unchanged': True, 'targets_unchanged': True,
            'reference_checkpoint_sha256': m['checkpoint_sha256'],
            'reference_run_json_sha256': sha(ref / 'run.json'),
            'selection_sha256': PIN, 'compare_code_sha256': software_identity()}


def completed_run(run, expected):
    receipt = run / 'training_done.json'
    if not receipt.exists():
        return False
    j = read_json(receipt)
    if j['intervention'] != expected:
        raise ValueError(f'Training identity changed: {run}')
    for name, digest in j['files_sha256'].items():
        if sha(run / name) != digest:
            raise ValueError(f'Training output changed: {run / name}')
    return True
