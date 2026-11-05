"""Export frozen MAG run_03 for one window, then compare with frozen ETAS/PI.

Requires the existing project's PyTorch and NumPy environment. No fitting.
--check-reference only checks small reference maps and shared NumPy scoring.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import platform
from pathlib import Path
import sys
import zipfile

import numpy as np

BASE = Path(__file__).resolve().parent
SHAPE = (340, 200)
FRACTIONS = (.005, .01, .02, .03)
THRESHOLDS = (.001, .005, .01, .02, .05, .1)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n', encoding='utf-8')


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    with Path(path).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def locate(project, supplied, relative_paths, filename):
    if supplied is not None:
        candidates = [supplied.resolve()]
    else:
        candidates = [project / rel for rel in relative_paths]
    for p in candidates:
        if (p / filename).is_file():
            return p
    raise FileNotFoundError(f'Missing {filename}; searched {candidates}')


def load_context(args):
    refs = BASE / 'reference'
    bundle = read_json(refs / 'reference_manifest.json')
    for name, expected in bundle['files_sha256'].items():
        if sha(refs / name) != expected:
            raise ValueError(f'Reference file changed: {name}')
    frozen = read_json(refs / 'frozen_mag.json')
    project = args.project_root.resolve()
    training = locate(project, args.training_dir,
                      ('training', 'convlstm/training'), 'model_cl10.py')
    data_root = locate(project, args.data,
                       ('cl10_mapping_v2', 'datasets/cl10_mapping_v2'), 'manifest.json')
    for name, expected in frozen['dataset_sha256'].items():
        filename = {'manifest': 'manifest.json', 'sample_index': 'sample_index.csv',
                    'usgs_daily': 'usgs_daily.npz'}[name]
        if sha(data_root / filename) != expected:
            raise ValueError(f'Dataset differs from frozen MAG/ETAS/PI: {filename}')
    for name, expected in frozen['code_sha256'].items():
        if sha(training / name) != expected:
            raise ValueError(f'Shared model/data/scoring code changed: {name}')
    sys.path.insert(0, str(training))
    reader = importlib.import_module('cl10_data')
    scorer = importlib.import_module('evaluate_cl10_symmetric')
    for mod in (reader, scorer):
        if Path(mod.__file__).resolve().parent != training.resolve():
            raise ValueError('A different project module was already imported')
    if (scorer.EVALUATION_VERSION != frozen['evaluation_version'] or
            scorer.EXPAND_D != 3 or scorer.THRESHOLD_GRAY != 100):
        raise ValueError('Unexpected evaluation protocol')
    dataset = reader.CL10Data(data_root, 'validation', (0,), cache_size=0)
    row = dataset.rows[0]
    if (row['target_start'], row['target_end']) != ('2009-01-01', '2010-03-26'):
        raise ValueError('The requested first validation window changed')
    region_path = refs / 'region_meta.json'
    if sha(region_path) != dataset.manifest['region_sha256']:
        raise ValueError('Reference grid differs from dataset')
    with np.load(refs / 'ETAS_prediction.npz', allow_pickle=False) as z:
        etas = z['probability'].copy()
        if (str(z['origin'].item()), str(z['end_exclusive'].item())) != (
                '2009-01-01', '2010-03-27'):
            raise ValueError('ETAS dates do not match the requested window')
    with np.load(refs / 'PI_prediction.npz', allow_pickle=False) as z:
        pi = z['pi_map'].copy()
        if int(z['row']) != 0 or str(z['target_start'].item()) != row['target_start']:
            raise ValueError('PI dates do not match the requested window')
    for a in (etas, pi):
        if a.shape != SHAPE or not np.isfinite(a).all() or np.any(a < 0):
            raise ValueError('Invalid ETAS/PI map')
    if np.any(etas > 1):
        raise ValueError('ETAS probabilities exceed one')
    return frozen, training, dataset, scorer, row, read_json(region_path), etas, pi


def cell_areas(region):
    """Spherical cell area in km²; radius matches the preceding ETAS trial."""
    edges = (np.arange(341) + region['crop_box_closed']['y_min']) / region['global_H'] * 180 - 90
    area = 6378.1 ** 2 * np.radians(360 / region['global_W']) * np.diff(np.sin(np.radians(edges)))
    return np.broadcast_to(area[:, None], SHAPE).copy()


def ranked_alarm(score, area, fraction):
    """Same frozen score ranking and seed-2901 tie order as the ETAS trial.

    Choose the nearest prefix to the physical area budget. The discrepancy is
    at most one grid cell. Do not normalize or clip the raw MAG output.
    """
    tie = np.random.default_rng(2901).random(score.size)
    order = np.lexsort((tie, -score.ravel().astype(float)))
    cumulative = np.cumsum(area.ravel()[order])
    budget = float(area.sum() * fraction)
    hi = int(np.searchsorted(cumulative, budget))
    choices = [max(0, hi), min(len(order), hi + 1)]
    k = min(choices, key=lambda j: abs((cumulative[j - 1] if j else 0) - budget))
    mask = np.zeros(score.size, bool)
    mask[order[:k]] = True
    return mask.reshape(SHAPE)


def score_alarms(scorer, target, area, specifications):
    rows, protocols = [], []
    expanded_target = scorer.archived_expand(target)
    for model, setting, value, alarm in specifications:
        expanded = scorer.archived_expand(alarm.astype(np.float32))
        stats = {}
        for protocol, pe, te in scorer.PROTOCOLS:
            s = scorer.counts(expanded if pe else alarm.astype(np.float32),
                              expanded_target if te else target)
            protocols.append({'model': model, 'setting': setting, 'value': value,
                              'protocol': protocol, **s})
            stats[protocol] = s
        rows.append({'model': model, 'setting': setting, 'value': value, 'sample_count': 1,
                     'precision': stats['target_only']['precision'],
                     'recall': stats['prediction_only']['recall'], 'CSI': stats['both']['CSI'],
                     'precision_exact': stats['neither']['precision'],
                     'recall_exact': stats['neither']['recall'],
                     'alarm_cells': int(alarm.sum()), 'alarm_grid_fraction': float(alarm.mean()),
                     'alarm_km2': float(area[alarm].sum()),
                     'alarm_area_fraction': float(area[alarm].sum() / area.sum()),
                     'expanded_alarm_area_fraction': float(area[expanded > 0].sum() / area.sum()),
                     'target_cells': int(np.count_nonzero(target * 255 >= 100))})
    return rows, protocols


def base_specifications(etas, pi, area):
    specs = [('ETAS', 'probability_threshold', t, etas >= t) for t in THRESHOLDS]
    specs.append(('PI', 'original_PI_positive', 0, pi > 0))
    for fraction in FRACTIONS:
        for model, score in (('ETAS', etas), ('PI', pi)):
            specs.append((model, 'equal_actual_area', fraction, ranked_alarm(score, area, fraction)))
    return specs


def locate_run(project, supplied, frozen):
    item = frozen['model']
    if supplied is not None:
        candidates = [supplied.resolve()]
    else:
        candidates = [project / 'runs/mag_w0_000175/run_03',
                      project / 'runs/mag_plus_one_matched_init_run03/mag/run_03']
        if (project / 'runs').is_dir():
            candidates.extend(p.parent for p in sorted((project / 'runs').rglob('run.json')))
    seen = set()
    for run in candidates:
        if run in seen:
            continue
        seen.add(run)
        if not all((run / n).is_file() for n in ('run.json', 'summary.json', 'best.pt')):
            continue
        cfg = read_json(run / 'run.json')
        if (cfg.get('selected_channels') != [0] or
                cfg.get('dataset_sha256') != frozen['dataset_sha256'] or
                any(cfg.get(k) != item[k] for k in ('init_seed', 'order_seed'))):
            continue
        if read_json(run / 'summary.json').get('best_epoch') != item['best_epoch']:
            continue
        print(f'Checking MAG checkpoint: {run}', flush=True)
        if sha(run / 'best.pt') == item['checkpoint_sha256']:
            return run, cfg
    raise FileNotFoundError('Frozen MAG run_03 not found. Pass --run with the completed run directory; '
                            f'expected checkpoint SHA256={item["checkpoint_sha256"]}')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project-root', type=Path, default=Path('/share/home/22460734/project_MP_1978_2024'))
    ap.add_argument('--data', type=Path)
    ap.add_argument('--training-dir', type=Path)
    ap.add_argument('--run', type=Path, help='Optional completed MAG run; identity must match frozen selection')
    ap.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    ap.add_argument('--out', type=Path, default=BASE / 'results/MAG_ETAS_PI_20090101')
    ap.add_argument('--check-reference', action='store_true', help='Reproduce ETAS/PI scores only; no model inference or output')
    ap.add_argument('--dry-run', action='store_true', help='Check frozen run/data/code only; no inference or output')
    args = ap.parse_args()
    frozen, training, data, scorer, row, region, etas, pi = load_context(args)
    if args.check_reference:
        keys, gray = data.target_keys(row)
        target = np.zeros(SHAPE, np.float32)
        target.flat[keys] = gray / 255
        actual, _ = score_alarms(scorer, target, cell_areas(region),
                                 base_specifications(etas, pi, cell_areas(region)))
        expected = read_csv(BASE / 'reference/ETAS_PI_evaluation.csv')
        if len(actual) != len(expected):
            raise ValueError('Reference rule count differs')
        for a, b in zip(actual, expected):
            for k, v in a.items():
                if isinstance(v, (float, int)) and not isinstance(v, bool):
                    if not np.isclose(v, float(b[k]), rtol=1e-12, atol=1e-12):
                        raise ValueError(f'Reference reproduction failed: {a["model"]}/{k}')
                elif str(v) != b[k]:
                    raise ValueError(f'Reference reproduction failed: {k}')
        print(json.dumps({'reference_check_passed': True, 'alarm_rules_reproduced': len(actual),
                          'MAG_inference_executed': False}, indent=2))
        return
    run, cfg = locate_run(args.project_root.resolve(), args.run, frozen)
    plan = {'run': str(run), 'data': str(data.root), 'training_dir': str(training),
            'split': 'validation', 'row': 0, 'target_start': row['target_start'],
            'target_end': row['target_end'], 'selected_channels': [0],
            'expected_checkpoint_sha256': frozen['model']['checkpoint_sha256']}
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        return
    out = args.out.resolve()
    archive = out.parent / (out.name + '.zip')
    for protected in (run, data.root, training, BASE / 'reference'):
        if out == protected or out in protected.parents or protected in out.parents:
            raise ValueError(f'Output overlaps protected source: {protected}')
    if out.exists() or archive.exists():
        raise FileExistsError('Result folder/zip exists; use a new --out to avoid overwriting')
    import torch
    from model_cl10 import build_model
    device = torch.device(args.device if args.device != 'auto' else
                          ('cuda' if torch.cuda.is_available() else 'cpu'))
    model = build_model(1).to(device)
    # Load only the user's trusted training checkpoint after exact SHA256 matching.
    checkpoint = torch.load(run / 'best.pt', map_location=device, weights_only=False)
    if checkpoint['run_config'] != cfg or checkpoint['epoch'] != frozen['model']['best_epoch']:
        raise ValueError('Checkpoint metadata differs from the completed frozen run')
    model.load_state_dict(checkpoint['model'], strict=True)
    model.eval()
    del checkpoint
    # Match CL10Data input construction; target is not read by this stage.
    x = np.zeros((1, 9, 1, *SHAPE), np.float32)
    input_hashes = {}
    for j in range(9):
        keys, gray = data._load_frame(row[f'x{j}_start'])
        x[0, j, 0].flat[keys] = gray[:, 0]
        name = row[f'x{j}_start'] + '.npz'
        input_hashes[name] = sha(data.root / 'frames' / name)
    x /= 255
    with torch.no_grad():
        result = model(torch.from_numpy(x).to(device))
    if tuple(result.shape) != (1, 1, *SHAPE) or not torch.isfinite(result).all():
        raise ValueError('Unexpected or nonfinite single-window output')
    raw = result[0, 0].detach().cpu().numpy().astype(np.float32)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / 'manifest.json', {'complete': False, **plan})
    np.savez_compressed(out / 'MAG_prediction.npz', raw_prediction=raw,
                        row=np.array(0), target_start=np.array(row['target_start']),
                        target_end=np.array(row['target_end']),
                        checkpoint_sha256=np.array(frozen['model']['checkpoint_sha256']))
    pred_hash = sha(out / 'MAG_prediction.npz')
    print('Saved one MAG map; now reading target and calculating comparisons.', flush=True)
    keys, gray = data.target_keys(row)
    target = np.zeros(SHAPE, np.float32)
    target.flat[keys] = gray / 255
    area = cell_areas(region)
    specs = base_specifications(etas, pi, area)
    specs.append(('ConvLSTM_MAG', 'original_gray_threshold', 100, raw * 255 >= 100))
    specs += [('ConvLSTM_MAG', 'equal_actual_area', f, ranked_alarm(raw, area, f)) for f in FRACTIONS]
    rows, protocols = score_alarms(scorer, target, area, specs)
    equal = sorted([r for r in rows if r['setting'] == 'equal_actual_area'],
                   key=lambda r: (r['value'], r['model']))
    write_csv(out / 'equal_area_comparison.csv', equal)
    write_csv(out / 'all_rules.csv', rows)
    write_csv(out / 'all_protocols.csv', protocols)
    np.savez_compressed(out / 'comparison_maps.npz', ETAS_probability=etas, PI_score=pi,
                        ConvLSTM_MAG_raw=raw, target_normalized_MAG=target, cell_area_km2=area)
    native = next(r for r in rows if r['model'] == 'ConvLSTM_MAG' and r['setting'] == 'original_gray_threshold')
    archived = frozen['archived_first_window']
    matched = all(np.isclose(native[k], float(archived[k]), rtol=0, atol=1e-6)
                  for k in ('precision', 'recall', 'CSI'))
    write_json(out / 'archived_window_check.json', {'matches_within_1e_6': bool(matched),
               'archived': archived, 'current': native,
               'interpretation': 'Frozen checkpoint is matched exactly; backend/threshold differences must be examined if this flag is false.'})
    lines = ['# 同一450天窗口的ETAS、PI与ConvLSTM比较', '',
             '目标期：2009-01-01至2010-03-26。MAG为已选定的单通道run_03，使用已有最佳权重，仅预测一次。', '',
             '| 实际报警面积 | 模型 | Precision | Recall | CSI | 扩张后面积 |',
             '|---|---|---:|---:|---:|---:|']
    for r in equal:
        lines.append(f'| {100*r["value"]:.1f}% | {r["model"]} | {100*r["precision"]:.2f}% | {100*r["recall"]:.2f}% | {100*r["CSI"]:.2f}% | {100*r["expanded_alarm_area_fraction"]:.2f}% |')
    lines += ['', '面积按原ETAS试验的球面网格实际面积计算，离散误差不超过一个网格。分数并列采用固定随机排序，独立于真实目标。',
              'P为target_only，R为prediction_only，CSI为both；它们不共用一张列联表。未扩张指标与所有匹配计数见CSV。',
              'MAG原始输出用于排序，未裁剪或归一化为概率；MAG不使用ETAS的概率阈值。',
              '这是单个验证窗口的比较。ETAS仍是试算，其目录完整性、边界处理和分支比触及0.99约束的限制仍然存在。',
              '相同原始报警面积不代表扩张后面积相同，应一起解读面积和未扩张指标。',
              f'已有默认阈值成绩复现标记：{matched}，详见archived_window_check.json。', '']
    (out / 'comparison_report.md').write_text('\n'.join(lines), encoding='utf-8')
    if sha(out / 'MAG_prediction.npz') != pred_hash:
        raise ValueError('Saved prediction changed during scoring')
    metadata = {'complete': True, **plan, 'device': str(device),
                'torch': torch.__version__, 'numpy': np.__version__, 'python': platform.python_version(),
                'training_executed': False, 'inference_window_count': 1,
                'raw_output_range': [float(raw.min()), float(raw.max())],
                'input_frame_sha256': input_hashes, 'run_json_sha256': sha(run / 'run.json'),
                'code_sha256': frozen['code_sha256'], 'dataset_sha256': frozen['dataset_sha256'],
                'reference_manifest_sha256': sha(BASE / 'reference/reference_manifest.json'),
                'export_script_sha256': sha(Path(__file__)), 'archived_native_scores_match': bool(matched),
                'output_sha256': {p.name: sha(p) for p in sorted(out.iterdir()) if p.name != 'manifest.json'}}
    write_json(out / 'manifest.json', metadata)
    temporary = archive.with_suffix('.zip.partial')
    with zipfile.ZipFile(temporary, 'x', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.iterdir()):
            z.write(p, arcname=out.name + '/' + p.name)
    temporary.replace(archive)
    print('Finished one-window inference and comparison. Download only:', archive, flush=True)
    if not matched:
        print('NOTICE: native scores differ from the archived window; return the result zip for inspection.', flush=True)


if __name__ == '__main__':
    main()
