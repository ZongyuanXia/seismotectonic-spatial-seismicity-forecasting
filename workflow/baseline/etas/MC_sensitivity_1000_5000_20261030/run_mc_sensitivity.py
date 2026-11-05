"""ETAS Monte Carlo count sensitivity: nested 1000/2048/5000 simulations.

No refitting. Frozen seed stride 2048 preserves the existing six-window runs.
History=4050 days; horizon=450 days; spatial background bandwidth=25 km.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time

import numpy as np


ENGINE = None


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    temp = Path(str(path)+'.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def initialize(root, origin, row):
    global ENGINE
    sys.path.insert(0, str(Path(root)/'baseline/validation_six_windows'))
    import run_etas_six_windows as engine
    fit = Path(root)/'baseline/etas/results/validation_00000_20090101_v2/fit.pkl'
    # Keep the original seed sequence, independent of the new simulation count.
    engine.initialize(root, str(fit), origin, row, 2048)
    ENGINE = engine


def simulate(i):
    return ENGINE.simulate(i)


def run_window(args, window, old_dir):
    row = window['row']
    out = args.out/f'window_{row:05d}'
    out.mkdir(exist_ok=True)
    identity = {'row':row, 'script_sha256':sha(__file__),
                'seed_formula':'202610290 + validation_row * 2048 + simulation_index'}
    done = out/'complete.json'
    if done.exists():
        meta = json.loads(done.read_text())
        assert meta['identity'] == identity and meta['complete']
        for name,h in meta['output_sha256'].items():
            assert sha(out/name)==h
        print(f'Reusing complete row {row}', flush=True)
        return meta
    presence = np.zeros((340,200), np.int64)
    events = np.zeros_like(presence)
    # Separate halves are available for each of the requested sample sizes.
    checkpoints = (1000,2048,5000)
    halves = {n:np.zeros((2,340,200),np.int64) for n in checkpoints}
    records = []
    start = time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('spawn'),
            initializer=initialize, initargs=(str(args.root),window['target_start'],row)) as pool:
        for i, unique, keys, record in pool.map(simulate, range(5000), chunksize=4):
            assert i == len(records)
            records.append(record)
            presence.flat[unique] += 1
            np.add.at(events.ravel(),keys,1)
            for n in checkpoints:
                if i<n:
                    halves[n][int(i>=n//2)].flat[unique]+=1
            if i+1 in checkpoints:
                n=i+1
                if n==2048:
                    with np.load(old_dir/f'ETAS_{row:05d}.npz') as z:
                        assert np.array_equal(presence,z['presence_count']), 'Existing presence not reproduced'
                        assert np.array_equal(events,z['event_count']), 'Existing event counts not reproduced'
                        assert np.array_equal(halves[n],z['half_presence_count']), 'Existing half counts not reproduced'
                path=out/f'ETAS_N{n}.npz'
                np.savez_compressed(path, probability=(presence/n).astype(np.float32),
                    expected_event_count=(events/n).astype(np.float32), presence_count=presence,
                    event_count=events, half_presence_count=halves[n], simulations=np.array(n),
                    origin=np.array(window['target_start']), row=np.array(row))
            if (i+1)%250==0 or i+1 in checkpoints:
                print(f'row={row} simulations={i+1}/5000 elapsed={time.monotonic()-start:.1f}s',flush=True)
    import csv
    with (out/'simulation_counts.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    meta={'complete':True,'identity':identity,'window':window,'old_2048_reproduced_bitwise':True,
          'elapsed_seconds':time.monotonic()-start,
          'output_sha256':{p.name:sha(p) for p in out.iterdir() if p.suffix in ('.npz','.csv')}}
    dump(done,meta)
    return meta


def analyze(args, common, data, scorer, windows, area):
    import csv
    details,protocols,stability,probability_changes=[],[],[],[]
    old_input=args.root/'baseline/MAG_ETAS_PI_validation_six'
    returned=json.loads((old_input/'manifest.json').read_text())
    for name,h in returned['output_sha256'].items():
        assert sha(old_input/name)==h
    for w in windows:
        row=w['row']
        with np.load(old_input/'maps'/f'comparison_{row:05d}.npz') as z:
            maps={'ConvLSTM_MAG':z['ConvLSTM_MAG_raw'].copy(),'PI':z['PI_score'].copy()}
            original_etas=z['ETAS_probability'].copy()
            original_target=z['target_normalized_MAG'].copy()
        probabilities={}
        for n in (1000,2048,5000):
            with np.load(args.out/f'window_{row:05d}'/f'ETAS_N{n}.npz') as z:
                p=z['probability'].copy(); half=z['half_presence_count'].copy()
            probabilities[n]=p
            maps[f'ETAS_N{n}']=p
            if n==2048:
                assert np.array_equal(p,original_etas)
            for fraction in common.core.FRACTIONS:
                a,b=[common.core.ranked_alarm(h/(n//2),area,fraction) for h in half]
                stability.append({'row':row,'simulations':n,'alarm_area_percent':100*fraction,
                    'half_size':n//2,'halves_grid_Jaccard':float((a&b).sum()/(a|b).sum()),
                    'halves_area_Jaccard':float(area[a&b].sum()/area[a|b].sum())})
        d,p,target=common.analyze_window(w,maps,data,scorer,area)
        assert np.array_equal(target,original_target)
        details.extend(d);protocols.extend(p)
        for n in (1000,2048):
            a,b=probabilities[n],probabilities[5000]
            for fraction in common.core.FRACTIONS:
                alarm_a,alarm_b=[common.core.ranked_alarm(p,area,fraction) for p in (a,b)]
                probability_changes.append({'row':row,'compared_simulations':n,'reference_simulations':5000,
                    'alarm_area_percent':100*fraction,
                    'probability_MAE':float(np.abs(a-b).mean()),'probability_max_abs_change':float(np.abs(a-b).max()),
                    'alarm_grid_Jaccard':float((alarm_a&alarm_b).sum()/(alarm_a|alarm_b).sum()),
                    'alarm_area_Jaccard':float(area[alarm_a&alarm_b].sum()/area[alarm_a|alarm_b].sum()),
                    'changed_alarm_cells':int(np.count_nonzero(alarm_a!=alarm_b))})
    common.write_reports(args.out,details,protocols)
    common.core.write_csv(args.out/'split_half_stability.csv',stability)
    common.core.write_csv(args.out/'probability_and_alarm_changes.csv',probability_changes)
    wide=[]
    for row in common.ROWS:
        for fraction in common.core.FRACTIONS:
            group=[r for r in details if r['row']==row and r['value']==fraction]
            line={k:group[0][k] for k in ('row','target_start','target_end','target_cells')}
            line['alarm_area_percent']=100*fraction
            for r in group:
                line[r['model']+'_exact_hits']=r['exact_hits']
            line['ETAS_5000_minus_1000_hits']=line['ETAS_N5000_exact_hits']-line['ETAS_N1000_exact_hits']
            wide.append(line)
    common.core.write_csv(args.out/'per_window_hits_wide.csv',wide)
    # Replace the legacy report, whose fixed wording referred only to N=2048.
    (args.out/'comparison_report.md').write_text(
        '# ETAS 模拟次数敏感性检查\n\n'
        '六个窗口各生成 5000 次模拟，前 1000/2048 次构成对应结果；前 2048 次逐格复现原结果。'
        '采用相同实际报警面积和原有评分代码。没有重新拟合参数或重跑 ConvLSTM、PI。\n\n'
        'exact_hits_summary.csv 为六窗口均值；per_window_hits_wide.csv 为每窗口同格命中数。'
        'split_half_stability.csv 比较两个不重叠半批次的报警区域，半批次大小分别为 500、1024、2500。'
        'probability_and_alarm_changes.csv 比较前缀 1000/2048 与 5000 的概率及报警区域。\n\n'
        '各次数使用嵌套随机序列，不是独立重复试验。5000 次不是真值，单次嵌套比较不构成严格收敛证明。'
        '目标期不重叠，但历史重叠。ETAS 仍有目录完整性、边界与分支比触及约束上限的限制。\n',encoding='utf-8')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--workers',type=int,default=4)
    args=ap.parse_args();args.root=args.root.resolve();args.out=args.out.resolve()
    if args.workers<1:ap.error('--workers must be positive')
    args.out.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(args.root/'baseline/validation_six_windows'))
    import common_six as common
    context=argparse.Namespace(project_root=args.root,data=None,training_dir=None)
    _,_,data,scorer,windows,area=common.load_inputs(context)
    old_dir=args.root/'baseline/etas/results/validation_six_frozen_2009_20261030'
    old_meta=json.loads((old_dir/'manifest.json').read_text())
    assert old_meta['complete'] and old_meta['simulations']==2048
    fit=args.root/'baseline/etas/results/validation_00000_20090101_v2/fit.pkl'
    assert sha(fit)==old_meta['fit_sha256']
    assert sha(args.root/'datasets/cl10_mapping_v2/usgs_event_audit.csv')==old_meta['event_audit_sha256']
    assert sha(args.root/'baseline/etas/predict_etas_trial.py')==old_meta['original_adapter_sha256']
    assert sha(args.root/'baseline/validation_six_windows/run_etas_six_windows.py')==old_meta['script_sha256']
    for w in old_meta['windows']:
        assert sha(old_dir/w['prediction_file'])==w['prediction_sha256']
    provenance={'complete':False,'started_at':datetime.now().astimezone().isoformat(),
        'root':str(args.root),'simulation_counts':[1000,2048,5000],'fit_sha256':sha(fit),
        'old_manifest_sha256':sha(old_dir/'manifest.json'),'script_sha256':sha(__file__),
        'seed_formula':'202610290 + validation_row * 2048 + simulation_index',
        'nested_simulations':True,'parameter_refitting':False,'history_days':4050,'forecast_days':450}
    dump(args.out/'manifest.json',provenance)
    completed=[run_window(args,w,old_dir) for w in windows]
    analyze(args,common,data,scorer,windows,area)
    outputs={str(p.relative_to(args.out)):sha(p) for p in args.out.rglob('*')
             if p.is_file() and p.name!='manifest.json'}
    dump(args.out/'manifest.json',{**provenance,'complete':True,'windows':completed,
        'finished_at':datetime.now().astimezone().isoformat(),'output_sha256':outputs})
    print('COMPLETE:',args.out,flush=True)


if __name__=='__main__':main()
