"""Export maps and sensitivity for frozen MAG + max direction angle + s0.

No training or seed selection. Reuse the completed core package for other
models and the mask-equivalent B=100 direction control. Large arrays stay on
the cloud; only compact_core_direction_s0_summary needs downloading.
"""
from pathlib import Path
import argparse, fcntl, hashlib, json, sys
import numpy as np

HERE=Path(__file__).resolve().parent
PROJECT=Path('/share/home/22460734/project_MP_1978_2024')
FROZEN='core_direction_s0_frozen.json'
FROZEN_SHA='dff3d4f41be1f9fb7b78b39e6f8ce36620244388a9c402361ec93a24fc89c1e4'
MODELS=('MAG','pair_01_02','pair_03_04','pair_05_06')
MODEL='pair_03_04'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))

def check(project,data,source):
    if sha(HERE/FROZEN)!=FROZEN_SHA:raise ValueError('Frozen direction settings changed')
    f=load(HERE/FROZEN)
    if sha(HERE/'run_core_completion.py')!=f['previous_runner_sha256']:
        raise ValueError('Original core runner changed; preserve original run_core_completion.py')
    for n,h in f['dependency_sha256'].items():
        if sha(HERE/n)!=h:raise ValueError('Original dependency changed: '+n)
    for n,p in [('manifest','manifest.json'),('sample_index','sample_index.csv'),('usgs_daily','usgs_daily.npz')]:
        if sha(data/p)!=f['dataset_sha256'][n]:raise ValueError('Dataset changed: '+p)
    if sha(source/'summary_manifest.json')!=f['previous_summary_manifest_sha256']:
        raise ValueError('Prior compact core summary changed; set --source to the completed original package')
    previous=load(source/'summary_manifest.json')
    if not previous['analysis_complete'] or not previous['direction_B_complete']:
        raise ValueError('Original core package is incomplete')
    for n,h in previous['files_sha256'].items():
        if sha(source/n)!=h:raise ValueError('Original core output changed: '+n)
    audit=f['direction_mask_audit']
    if not (audit['B_reusable'] and audit['stats']['frames']==16268
            and audit['stats']['differing_entries']==0
            and audit['dataset_manifest_sha256']==f['dataset_sha256']['manifest']):
        raise ValueError('Missing complete support-mask equivalence audit')
    import run_symmetric_evaluation as engine
    import run_core_completion as core
    if engine.EVALUATION_VERSION!=f['evaluation_version']:raise ValueError('Spatial protocol changed')
    available={c.model:c for c in engine.default_cases(project,'triple',['run_03'])}
    engine.preflight([available['MAG'],available[MODEL]],data)
    for m in ['MAG',MODEL]:
        c=available[m];item=next(x for x in f['models'] if x['model']==m)
        config=load(c.run/'run.json')
        if (list(c.channels)!=item['selected_channels'] or sha(c.run/'best.pt')!=item['checkpoint_sha256']
                or load(c.run/'summary.json')['best_epoch']!=item['best_epoch']
                or any(config.get(k)!=item[k] for k in ['init_seed','order_seed'])):
            raise ValueError('Frozen checkpoint/seed changed: '+m)
    # Reuse identical existing scoring functions with this delivery's frozen identity.
    core.FROZEN_SHA=FROZEN_SHA
    return f,available[MODEL],core,previous

def merge_rows(old,new):
    rows=[r.copy() for r in old if r['model'] in MODELS and r['model']!=MODEL]
    rows.extend(r.copy() for r in new)
    lookup={(r['model'],r['split'],r['cohort'],str(r['pred_threshold']),str(r['d'])):r for r in rows}
    if len(lookup)!=len(rows):raise ValueError('Duplicated sensitivity summaries')
    for r in rows:
        b=lookup['MAG',r['split'],r['cohort'],str(r['pred_threshold']),str(r['d'])]
        for metric in ['precision','recall','CSI']:
            r[metric+'_delta_vs_MAG_pp']=100*(float(r[metric+'_mean'])-float(b[metric+'_mean']))
    return rows

def collect(out,source,f,core,previous):
    new_rows=[];new_windows=[];new_receipts={}
    for split in ['validation','test']:
        folder=out/'analysis_cases'/MODEL/split
        if not (folder/'done.json').is_file():raise ValueError('Missing direction analysis: '+split)
        new_receipts[MODEL+'/'+split]=core.verify_done(folder)
        new_rows.extend(core.read_csv(folder/'summary.csv'))
        new_windows.extend(core.read_csv(folder/'nonoverlap.csv'))
    rows=merge_rows(core.read_csv(source/'sensitivity_summary.csv'),new_rows)
    windows=[r for r in core.read_csv(source/'sensitivity_nonoverlap.csv') if r['model'] in MODELS]
    windows.extend(new_windows)
    if len(rows)!=160 or len(windows)!=480:raise ValueError('Incomplete four-model summary')
    compact=out/'compact_core_direction_s0_summary';compact.mkdir(exist_ok=True)
    core.write_csv(compact/'sensitivity_summary.csv',rows)
    core.write_csv(compact/'sensitivity_nonoverlap.csv',windows)
    with np.load(source/'six_window_maps.npz',allow_pickle=False) as z:
        maps={k:z[k].copy() for k in z.files if k=='targets' or any(k==m+'_predictions' for m in MODELS if m!=MODEL)}
    with np.load(out/'analysis_cases'/MODEL/'test/six_maps.npz',allow_pickle=False) as z:
        if not np.array_equal(z['targets'],maps['targets']):raise ValueError('New test targets differ from original package')
        maps[MODEL+'_predictions']=z['predictions'].copy()
    if set(maps)!={'targets',*(m+'_predictions' for m in MODELS)}:raise ValueError('Missing map arrays')
    np.savez_compressed(compact/'six_window_maps.npz',**maps)
    # The A/B architecture and numerical inputs are unchanged, as documented by
    # the complete frame audit. Original receipt/task names remain traceable.
    bfiles=['direction_B_metrics.csv','direction_B_nonoverlap.csv']
    for name in bfiles:
        b=core.read_csv(source/name)
        if any(r['model']!='pair_03_08' for r in b):raise ValueError('Unexpected prior direction B model')
        for r in b:r['model']=MODEL
        core.write_csv(compact/name,b)
    core.save(compact/'direction_B_receipt.json',dict(reused=True,model=MODEL,
        source_model='pair_03_08',source_receipt=load(source/'direction_B_receipt.json'),
        reason='All 16268 frames have identical s0/R support; max-angle channel unchanged; B inputs identical',
        mask_audit=f['direction_mask_audit'],
        files_sha256={n:sha(compact/n) for n in bfiles}))
    core.save(compact/'grid_and_protocol.json',load(source/'grid_and_protocol.json'))
    core.save(compact/FROZEN,f)
    files=[p for p in compact.iterdir() if p.is_file() and p.name!='summary_manifest.json']
    receipts={k:v for k,v in previous['case_receipts'].items() if k.split('/')[0] in MODELS}
    receipts.update(new_receipts)
    core.save(compact/'summary_manifest.json',dict(analysis_complete=True,direction_B_complete=True,
        missing_analysis_cases=[],models=list(MODELS),seed='run_03',no_training=True,
        source_manifest_sha256=f['previous_summary_manifest_sha256'],source=str(source),
        runner_sha256=sha(Path(__file__)),frozen_sha256=FROZEN_SHA,
        reused_models=['MAG','pair_01_02','pair_05_06'],newly_scored_model=MODEL,
        case_receipts=receipts,files_sha256={p.name:sha(p) for p in files},archives_created=False))
    print('Complete. Download only:',compact,flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['check','analysis','summary'],default='check')
    p.add_argument('--project-root',type=Path,default=PROJECT)
    p.add_argument('--data',type=Path);p.add_argument('--source',type=Path);p.add_argument('--out',type=Path)
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    a=p.parse_args();project=a.project_root.resolve()
    data=a.data.resolve() if a.data else project/'cl10_mapping_v2'
    source=a.source.resolve() if a.source else project/'runs/core_completion_v1/compact_core_summary'
    out=a.out.resolve() if a.out else project/'runs/core_direction_s0_v1'
    f,case,core,previous=check(project,data,source)
    for src in [HERE,data,source,case.run.resolve()]:
        if out==src or out in src.parents or src in out.parents:raise ValueError('Output overlaps input: '+str(src))
    if a.stage=='check':
        print('Preflight passed. No training. Only pair_03_04/run_03 requires scoring; A/B controls reused.');return
    out.mkdir(parents=True,exist_ok=True)
    with (out/'.run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        ident=dict(version=f['version'],runner_sha256=sha(Path(__file__)),frozen_sha256=FROZEN_SHA,
                   project=str(project),data=str(data),source=str(source))
        marker=out/'batch_identity.json'
        if marker.is_file():
            if load(marker)!=ident:raise ValueError('Batch identity changed; choose a new --out')
        elif any(x.name!='.run.lock' for x in out.iterdir()):raise ValueError('Unrecognized nonempty output directory')
        else:core.save(marker,ident)
        if a.stage=='analysis':
            for split in ['validation','test']:core.analyze_case(case,split,data,project,out,a.device,f)
        collect(out,source,f,core,previous)

if __name__=='__main__':main()
