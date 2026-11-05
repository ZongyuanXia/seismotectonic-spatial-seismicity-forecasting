"""Six frozen-parameter ETAS test forecasts; no parameter fitting or target use.

History: preceding 4050 days; forecast: 450 days; processed Mw >= 4.
Uses the existing all-generation author simulator and its 25-km background.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import argparse, csv, hashlib, json, multiprocessing as mp, sys, time
import numpy as np

ROOT = Path('/Users/xiazongyuan/1 课题/7 地震震级预测/xzy-小论文1-文件汇总/代码与结果（归档）/九项指标重构归档')
ENGINE = None
ROW = None

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def dump(p, data):
    tmp = Path(str(p)+'.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    tmp.replace(p)

def initialize(root, origin, row):
    global ENGINE, ROW
    root = Path(root)
    sys.path.insert(0, str(root/'baseline/validation_six_windows'))
    import run_etas_six_windows as engine
    engine.initialize(root, root/'baseline/etas/results/validation_00000_20090101_v2/fit.pkl', origin, row, 10000)
    ENGINE, ROW = engine, row

def simulate(i):
    # Separate seed block from all previous validation simulations.
    global_index = 100000000 + ROW*10000 + i
    _, unique, keys, record = ENGINE.P.generate_one(global_index)
    record.update(simulation=i, global_simulation_index=global_index, test_row=ROW)
    return i, unique, keys, record

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=ROOT)
    ap.add_argument('--out', type=Path, default=Path(__file__).resolve().parent/'etas_test')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--simulations', type=int, default=5000)
    args = ap.parse_args()
    assert 2 <= args.simulations <= 10000 and args.workers > 0
    root, out = args.root.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    fitdir = root/'baseline/etas/results/validation_00000_20090101_v2'
    fitmeta = json.loads((fitdir/'fit_manifest.json').read_text())
    verification = json.loads((fitdir/'verification.json').read_text())
    assert fitmeta['complete'] and fitmeta['training_max_time'] < '2009-01-01'
    assert sha(fitdir/'fit.pkl') == verification['output_sha256']['fit.pkl']
    assert sha(root/'datasets/cl10_mapping_v2/usgs_event_audit.csv') == fitmeta['sha256']['event_audit']
    grid = json.loads((root/'convlstm/runs/compact_core_summary/grid_and_protocol.json').read_text())
    windows = grid['test_target_dates']
    identity = dict(version='etas_six_test_frozen2009_v1', simulations=args.simulations,
        fit_sha256=sha(fitdir/'fit.pkl'), script_sha256=sha(__file__),
        engine_sha256=sha(root/'baseline/etas/predict_etas_trial.py'),
        adapter_sha256=sha(root/'baseline/validation_six_windows/run_etas_six_windows.py'),
        author_sha256=sha(root/'baseline/etas/vendor/etas-main/etas/simulation.py'),
        audit_sha256=sha(root/'datasets/cl10_mapping_v2/usgs_event_audit.csv'),
        seed_formula='202610290 + 100000000 + test_row*10000 + simulation_index',
        history_days=4050, forecast_days=450, windows=windows, parameter_refit=False,
        extra_forecast_map_smoothing=False)
    identpath = out/'identity.json'
    if identpath.exists():
        assert json.loads(identpath.read_text()) == identity
    else:
        assert not any(out.iterdir())
        dump(identpath, identity)
    done = []
    for w in windows:
        row = int(w['row'])
        dest = out/f'ETAS_{row:05d}.npz'
        meta = out/f'ETAS_{row:05d}.json'
        if meta.exists():
            m = json.loads(meta.read_text())
            assert m['complete'] and m['prediction_sha256'] == sha(dest)
            done.append(m)
            print('REUSE', row, flush=True)
            continue
        presence = np.zeros((340,200), np.int64)
        eventcounts = np.zeros_like(presence)
        records = []
        start = time.monotonic()
        print('START', row, w['target_start'], flush=True)
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('spawn'),
                initializer=initialize, initargs=(str(root),w['target_start'],row)) as pool:
            for i, unique, keys, record in pool.map(simulate, range(args.simulations), chunksize=4):
                assert i == len(records)
                presence.flat[unique] += 1
                np.add.at(eventcounts.ravel(), keys, 1)
                records.append(record)
                if (i+1)%500 == 0 or i+1 == args.simulations:
                    print(f'row={row} simulations={i+1}/{args.simulations} elapsed={time.monotonic()-start:.1f}s', flush=True)
        with Path(str(dest)+'.tmp').open('wb') as f:
            np.savez_compressed(f, probability=(presence/args.simulations).astype(np.float32),
                expected_event_count=(eventcounts/args.simulations).astype(np.float32),
                presence_count=presence, event_count=eventcounts,
                origin=np.array(w['target_start']), end=np.array(w['target_end']),
                row=np.array(row), simulations=np.array(args.simulations))
        Path(str(dest)+'.tmp').replace(dest)
        with (out/f'ETAS_{row:05d}_simulations.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
        m=dict(complete=True,**w,prediction_sha256=sha(dest),simulations=args.simulations,
            elapsed_seconds=time.monotonic()-start,mean_simulated_events=float(np.mean([r['events_in_region'] for r in records])))
        dump(meta,m);done.append(m)
    dump(out/'manifest.json',dict(complete=True,**identity,completed_windows=done))
    print('COMPLETE',out,flush=True)

if __name__ == '__main__':
    main()
