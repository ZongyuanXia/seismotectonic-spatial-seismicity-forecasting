"""Shared frozen inputs and exact-hit summaries for six validation windows."""
import argparse
import importlib
from pathlib import Path
import sys

import numpy as np
import core_one_window as core

BASE = Path(__file__).resolve().parent
ROWS = (0, 450, 900, 1350, 1800, 2250)


def load_inputs(args):
    ref = BASE / 'reference'
    manifest = core.read_json(ref / 'reference_manifest.json')
    if not manifest['complete']:
        raise ValueError('ETAS/PI references incomplete')
    for name, expected in manifest['files_sha256'].items():
        if core.sha(ref / name) != expected:
            raise ValueError(f'Reference changed: {name}')
    frozen = core.read_json(ref / 'frozen_mag.json')
    project = args.project_root.resolve()
    training = core.locate(project, args.training_dir, ('training', 'convlstm/training'), 'model_cl10.py')
    data_root = core.locate(project, args.data, ('cl10_mapping_v2', 'datasets/cl10_mapping_v2'), 'manifest.json')
    for key, name in [('manifest','manifest.json'),('sample_index','sample_index.csv'),('usgs_daily','usgs_daily.npz')]:
        if core.sha(data_root / name) != frozen['dataset_sha256'][key]:
            raise ValueError(f'Dataset changed: {name}')
    for name, expected in frozen['code_sha256'].items():
        if core.sha(training / name) != expected:
            raise ValueError(f'Shared model/scoring source changed: {name}')
    sys.path.insert(0, str(training))
    reader = importlib.import_module('cl10_data')
    scorer = importlib.import_module('evaluate_cl10_symmetric')
    if scorer.EVALUATION_VERSION != frozen['evaluation_version'] or scorer.EXPAND_D != 3:
        raise ValueError('Scoring protocol changed')
    data = reader.CL10Data(data_root, 'validation', (0,), cache_size=0)
    windows = core.read_json(ref / 'windows.json')
    if [w['row'] for w in windows] != list(ROWS):
        raise ValueError('Unexpected sample selection')
    for w in windows:
        r = data.rows[w['row']]
        if any(w[k] != r[k] for k in ('target_start','target_end')):
            raise ValueError('Reference target dates differ from dataset')
    region = core.read_json(ref / 'region_meta.json')
    if core.sha(ref / 'region_meta.json') != data.manifest['region_sha256']:
        raise ValueError('Grid changed')
    return frozen, training, data, scorer, windows, core.cell_areas(region)


def reference_maps(window):
    ref = BASE / 'reference'
    with np.load(ref / window['ETAS_file'], allow_pickle=False) as z:
        etas = z['probability'].copy()
        if str(z['origin'].item()) != window['target_start']:
            raise ValueError('ETAS origin differs')
    with np.load(ref / window['PI_file'], allow_pickle=False) as z:
        pi = z['pi_map'].copy()
        if str(z['target_start'].item()) != window['target_start'] or int(z['row']) != window['row']:
            raise ValueError('PI origin differs')
    for a in (etas, pi):
        if a.shape != (340,200) or not np.isfinite(a).all() or np.any(a < 0):
            raise ValueError('Invalid reference map')
    return {'ETAS': etas, 'PI': pi}


def analyze_window(window, maps, data, scorer, area):
    index = window['row']
    keys, gray = data.target_keys(data.rows[index])
    target = np.zeros((340,200), np.float32)
    target.flat[keys] = gray / 255
    specs = [(name, 'equal_actual_area', f, core.ranked_alarm(score, area, f))
             for f in core.FRACTIONS for name, score in maps.items()]
    scores, protocols = core.score_alarms(scorer, target, area, specs)
    metadata = {k: window[k] for k in ('row','target_start','target_end')}
    details = []
    for s in scores:
        exact = next(p for p in protocols if p['model']==s['model'] and p['value']==s['value'] and p['protocol']=='neither')
        details.append({**metadata, **s, 'exact_hits': exact['TP'], 'exact_FP': exact['FP'],
                        'exact_FN': exact['FN'], 'exact_recall': exact['recall'],
                        'exact_precision': exact['precision']})
    return details, [{**metadata, **r} for r in protocols], target


def write_reports(out, details, protocols):
    models = sorted({r['model'] for r in details})
    selected = sorted({r['row'] for r in details})
    if selected != list(ROWS) or len(details) != 6 * len(models) * 4:
        raise ValueError('Incomplete window/model/area table')
    core.write_csv(out / 'exact_hits_by_window.csv', details)
    core.write_csv(out / 'all_protocols.csv', protocols)
    summaries, gains = [], []
    metrics = ('exact_hits','exact_recall','exact_precision','precision','recall','CSI',
               'alarm_area_fraction','expanded_alarm_area_fraction')
    for model in models:
        for f in core.FRACTIONS:
            group = [r for r in details if r['model']==model and r['value']==f]
            summary = {'model': model, 'alarm_area_fraction_requested': f, 'window_count': 6,
                       'exact_hits_total_grid_occurrences': sum(r['exact_hits'] for r in group),
                       'target_total_grid_occurrences': sum(r['target_cells'] for r in group)}
            for metric in metrics:
                values = np.array([r[metric] for r in group], dtype=float)
                summary[metric+'_mean'] = float(values.mean())
                summary[metric+'_std_ddof0'] = float(values.std(ddof=0))
            summary['pooled_exact_recall'] = summary['exact_hits_total_grid_occurrences'] / summary['target_total_grid_occurrences']
            summaries.append(summary)
        for first, last in zip(core.FRACTIONS, core.FRACTIONS[1:]):
            delta = [next(r['exact_hits'] for r in details if r['row']==index and r['model']==model and r['value']==last)
                     - next(r['exact_hits'] for r in details if r['row']==index and r['model']==model and r['value']==first)
                     for index in ROWS]
            if min(delta) < 0:
                raise ValueError('Non-nested alarm masks or decreasing exact hits')
            gains.append({'model': model, 'from_area_fraction': first, 'to_area_fraction': last,
                          'additional_hits_total_grid_occurrences': sum(delta),
                          'additional_hits_mean_per_window': float(np.mean(delta))})
    core.write_csv(out / 'exact_hits_summary.csv', summaries)
    core.write_csv(out / 'exact_hits_area_gains.csv', gains)
    wide = []
    for f in core.FRACTIONS:
        r = {'alarm_area_percent': 100*f}
        for model in models:
            s = next(s for s in summaries if s['model']==model and s['alarm_area_fraction_requested']==f)
            r[model+'_mean_exact_hits'] = s['exact_hits_mean']
            r[model+'_total_grid_occurrences'] = s['exact_hits_total_grid_occurrences']
            r[model+'_macro_exact_recall'] = s['exact_recall_mean']
        wide.append(r)
    core.write_csv(out / 'exact_hits_summary_wide.csv', wide)
    lines = ['# 六个目标期不重叠的验证窗口：同面积比较', '',
             '样本索引：0、450、900、1350、1800、2250。历史输入仍有重叠，不能称为严格独立试验。', '',
             '完全同格命中是未扩张的TP，每个窗口先计数，再汇总。跨窗口总数单位为格次，同一位置在不同窗口出现可重复计数。', '',
             '| 报警面积 | 模型 | 平均命中格数 | 总命中格次 | 完全同格召回率均值 |',
             '|---|---|---:|---:|---:|']
    for f in core.FRACTIONS:
        for model in models:
            s = next(s for s in summaries if s['model']==model and s['alarm_area_fraction_requested']==f)
            lines.append(f'| {100*f:.1f}% | {model} | {s["exact_hits_mean"]:.2f} | {s["exact_hits_total_grid_occurrences"]} | {100*s["exact_recall_mean"]:.2f}% |')
    lines += ['', '完全同格召回率均值为六个窗口TP/真实有震格数的算术平均；另存按有震格次数加权的pooled_exact_recall，二者不可混用。',
              '相同实际面积采用与单窗口相同的排名、并列处理和网格面积；预算差异不超过一个网格。',
              '保留d=3的P target_only、R prediction_only、CSI both和扩张后面积，不能将三者反算为同一列联表。',
              'ETAS参数及背景率固定为2009年前拟合值；仅随预测起点更新此前4050天的触发历史，包含未来各代余震模拟。',
              'ETAS仍为试算：目录完整性未确认、边界处理有限、分支比触及0.99上限；2048次模拟的离散噪声也应考虑。', '']
    (out/'comparison_report.md').write_text('\n'.join(lines), encoding='utf-8')
    return summaries


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project-root', type=Path, required=True)
    ap.add_argument('--data', type=Path)
    ap.add_argument('--training-dir', type=Path)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    _, _, data, scorer, windows, area = load_inputs(args)
    details, protocols = [], []
    args.out.mkdir(parents=True, exist_ok=False)
    for w in windows:
        d,p,_ = analyze_window(w, reference_maps(w), data, scorer, area)
        details.extend(d); protocols.extend(p)
    write_reports(args.out, details, protocols)
    core.write_json(args.out/'manifest.json', {'complete':True,'models':['ETAS','PI'],
                     'MAG_pending_cloud_inference':True,'selected_rows':list(ROWS),
                     'reference_manifest_sha256':core.sha(BASE/'reference/reference_manifest.json')})
    print('Completed ETAS/PI statistics only; MAG awaits cloud inference:', args.out)


if __name__ == '__main__':
    main()
