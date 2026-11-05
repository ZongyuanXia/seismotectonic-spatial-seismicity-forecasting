"""Complete one B=100 control, six-window exports and sensitivity scores.

Only --stage control trains, and only B_pair_03_08_run_03 is trained.
--stage analysis reuses frozen real-feature weights and compatible predictions.
No archives are created and no source code/data/predictions are deleted.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
FROZEN_NAME = 'core_completion_frozen.json'
FROZEN_SHA = 'cc82b6ef55bd178f1c5b8bc5735e834c35487bcaf8b590b76a81981f4613bd43'
PROJECT = Path('/share/home/22460734/project_MP_1978_2024')
MODELS = ('MAG', 'pair_01_02', 'pair_03_08', 'pair_05_06')
METRICS = ('precision', 'recall', 'CSI')
SHAPE = (340, 200)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    path = Path(path); tmp = path.with_name(path.name+'.partial')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    tmp.replace(path)


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f: return list(csv.DictReader(f))


def write_csv(path, rows):
    if not rows: raise ValueError(f'Empty output: {path}')
    path = Path(path); tmp = path.with_name(path.name+'.partial')
    with tmp.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    tmp.replace(path)


def command(*args):
    print('Running:', *map(str,args), flush=True)
    subprocess.run([str(a) for a in args], check=True)


def check(project, data, compare):
    if sha(HERE/FROZEN_NAME) != FROZEN_SHA: raise ValueError('Frozen settings changed')
    f = load(HERE/FROZEN_NAME)
    for name, expected in f['dependency_sha256'].items():
        if sha(HERE/name) != expected: raise ValueError(f'Original dependency changed: {name}')
    for name, filename in [('manifest','manifest.json'),('sample_index','sample_index.csv'),('usgs_daily','usgs_daily.npz')]:
        if sha(data/filename) != f['dataset_sha256'][name]: raise ValueError(f'Dataset changed: {filename}')
    import run_symmetric_evaluation as engine
    if engine.EVALUATION_VERSION != f['evaluation_version']: raise ValueError('Spatial version changed')
    available = {c.model:c for c in engine.default_cases(project,'triple',['run_03'])}
    cases = [available[m] for m in MODELS]
    engine.preflight(cases,data)
    for c in cases:
        item = next(m for m in f['models'] if m['model']==c.model)
        config = load(c.run/'run.json')
        if list(c.channels)!=item['selected_channels'] or sha(c.run/'best.pt')!=item['checkpoint_sha256']:
            raise ValueError(f'Frozen model changed: {c.model}')
        if load(c.run/'summary.json')['best_epoch']!=item['best_epoch']:
            raise ValueError(f'Best epoch changed: {c.model}')
        if any(config.get(k)!=item[k] for k in ('init_seed','order_seed')):
            raise ValueError(f'Initialization/order seed changed: {c.model}')
    if sha(compare/'frozen_compare.json')!=f['compare_frozen_sha256']:
        raise ValueError('ABC control selection changed')
    for name,h in f['compare_code_sha256'].items():
        if sha(compare/name)!=h: raise ValueError(f'ABC code changed: {name}')
    for name,h in f['compare_vendor_sha256'].items():
        if sha(compare/'vendor'/name)!=h: raise ValueError(f'ABC vendor changed: {name}')
    print('Verified four run_03 models, dataset, original evaluation code and ABC code.',flush=True)
    return f,cases,engine


def prepare_root(out, project, data, f):
    out.mkdir(parents=True,exist_ok=True)
    identity = dict(version=f['version'],runner_sha256=sha(Path(__file__)),frozen_sha256=FROZEN_SHA,
                    project=str(project),data=str(data),models=list(MODELS),seed='run_03')
    p = out/'batch_identity.json'
    if p.exists():
        if load(p)!=identity: raise ValueError('Batch inputs changed; use a NEW --out')
    elif any(p.name!='.run.lock' for p in out.iterdir()):
        raise ValueError('Unrecognized nonempty output directory')
    else: save(p,identity)
    (out/'compact_core_summary').mkdir(exist_ok=True)


def control(project,data,compare,compare_out,out,device,f):
    """Use the original ABC worker and preserve its training fingerprint/init."""
    marker = compare_out/'experiment_manifest.json'
    if not marker.is_file(): raise FileNotFoundError('Original ABC output missing; set --compare-output')
    spec = importlib.util.spec_from_file_location('_core_abc_common',compare/'compare_common.py')
    common = importlib.util.module_from_spec(spec); spec.loader.exec_module(common)
    original = common.frozen(); tasks,aliases = common.plan(original)
    common.preflight(project,data,original,tasks,aliases)
    identity = load(marker)
    if (identity['selection_sha256']!=common.PIN or identity['compare_code_sha256']!=common.software_identity()
            or identity['dataset_sha256']!=f['dataset_sha256']):
        raise ValueError('Original ABC batch identity changed')
    task = tasks[aliases['pair_03_08']['B']]
    assert task['task_id']=='B_pair_03_08_run_03'
    with (compare_out/'.run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        base = [sys.executable,'-u',compare/'compare_worker.py','--task',task['task_id'],
                '--project',project,'--data',data,'--out',compare_out,'--device',device]
        command(*base,'--action','train')
        run = common.task_run(compare_out,task)
        if not common.completed_run(run,common.intervention(task,project)):
            raise ValueError('Control training receipt incomplete')
        import run_symmetric_evaluation as engine
        reports=[]; windows=[]; receipts={}
        for split in ('validation','test'):
            command(*base,'--action','predict','--split',split)
            folder = compare_out/'evaluations'/task['task_id']/split
            result = folder/'evaluation'
            case = engine.Case(task['task_id'],'run_03',run,(0,3,8),folder/'predictions')
            expected = engine.evaluation_identity(case,data,split,compare/'vendor')
            if not result.exists():
                command(sys.executable,'-u',compare/'vendor/evaluate_cl10_symmetric.py','--data',data,
                        '--run',run,'--predictions',folder/'predictions','--split',split,'--out',result)
            engine.verify_evaluation(result,expected)
            receipts[split]=sha(result/'evaluation_manifest.json')
            for row in read_csv(result/'official_summary.csv'):
                reports.append({'model':'pair_03_08','group':'B','seed':'run_03','split':split,**row})
            for row in read_csv(result/'nonoverlap_by_sample.csv'):
                windows.append({'model':'pair_03_08','group':'B','seed':'run_03','split':split,**row})
        compact=out/'compact_core_summary'
        write_csv(compact/'direction_B_metrics.csv',reports)
        write_csv(compact/'direction_B_nonoverlap.csv',windows)
        save(compact/'direction_B_receipt.json',dict(task=task,training_receipt=load(run/'training_done.json'),
             evaluation_manifest_sha256=receipts,compare_batch_sha256=sha(marker),compare_code_sha256=f['compare_code_sha256'],
             files_sha256={n:sha(compact/n) for n in ['direction_B_metrics.csv','direction_B_nonoverlap.csv']}))


def locate_predictions(case,split,project,out,device,data):
    candidates=[case.run/(split+'_predictions')]
    if split=='test':
        candidates.append(project/'runs/symmetric_band_union_v1/selected_r60_precision/test/cases'/case.model/case.seed/'predictions')
    for family in ('triple','single','all'):
        candidates.append(project/'runs/symmetric_band_union_v1'/family/split/'cases'/case.model/case.seed/'predictions')
    candidates.append(out/'prediction_cache'/case.model/split)
    for p in candidates:
        if (p/'manifest.json').is_file(): return p
    p=candidates[-1]
    if p.exists(): raise ValueError(f'Incomplete new prediction export preserved: {p}')
    p.parent.mkdir(parents=True,exist_ok=True)
    command(sys.executable,'-u',HERE/'predict_cl10.py','--data',data,'--run',case.run,
            '--split',split,'--out',p,'--device',device)
    return p


def expanded(mask,d):
    from spatial_symmetric import symmetric_expand
    return symmetric_expand(np.asarray(mask,dtype=np.uint8),d=d)>0


def score_masks(pred,truth,pred_expanded,truth_expanded):
    """Grid counts: prediction-side precision, truth-side recall, expanded CSI."""
    npred=int(pred.sum()); ntrue=int(truth.sum())
    hitp=int(np.count_nonzero(pred&truth_expanded))
    hitt=int(np.count_nonzero(truth&pred_expanded))
    inter=int(np.count_nonzero(pred_expanded&truth_expanded))
    union=int(np.count_nonzero(pred_expanded|truth_expanded))
    return dict(precision=hitp/npred if npred else None,recall=hitt/ntrue if ntrue else None,
                CSI=inter/union if union else None,raw_pred_grid_count=npred,raw_true_grid_count=ntrue,
                expanded_pred_grid_count=int(pred_expanded.sum()),expanded_true_grid_count=int(truth_expanded.sum()),
                hit_pred_grids=hitp,covered_true_grids=hitt,false_positive_pred_grids=npred-hitp,
                false_negative_true_grids=ntrue-hitt,raw_pred_grid_fraction=npred/pred.size,
                expanded_pred_grid_fraction=float(pred_expanded.mean()))


def summarize(rows,cohort):
    selected=[r for r in rows if cohort=='daily' or r['nonoverlap']]
    s={'cohort':cohort,'sample_count':len(selected)}
    for key in ('precision','recall','CSI','raw_pred_grid_fraction','expanded_pred_grid_fraction',
                'false_positive_pred_grids','false_negative_true_grids'):
        v=np.asarray([r[key] for r in selected if r[key] is not None],dtype=np.float64)
        s[key+'_mean']=float(v.mean()) if len(v) else None
        s[key+'_sd']=float(v.std(ddof=0)) if len(v) else None
        s[key+'_valid_count']=len(v)
    return s


def verify_done(folder):
    receipt=load(folder/'done.json')
    for name,h in receipt['files_sha256'].items():
        if sha(folder/name)!=h: raise ValueError(f'Case output changed: {folder/name}')
    return receipt


def analyze_case(case,split,data_root,project,out,device,f):
    import evaluate_cl10_symmetric as old
    from cl10_data import CL10Data
    data=CL10Data(data_root,split,(0,),cache_size=0)
    pred_dir=locate_predictions(case,split,project,out,device,data_root)
    manifest,predictions,pred_sha=old._validate_inputs(data,pred_dir,split)
    item=next(m for m in f['models'] if m['model']==case.model)
    if (manifest['best_checkpoint_sha256']!=item['checkpoint_sha256'] or
        manifest['selected_channels']!=item['selected_channels'] or manifest['best_epoch']!=item['best_epoch'] or
        any(manifest.get(k)!=item[k] for k in ('init_seed','order_seed'))):
        raise ValueError('Prediction export does not match frozen model')
    folder=out/'analysis_cases'/case.model/split
    identity=dict(model=case.model,seed=case.seed,split=split,prediction_sha256=pred_sha,
                  checkpoint_sha256=item['checkpoint_sha256'],frozen_sha256=FROZEN_SHA,runner_sha256=sha(Path(__file__)))
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'identity.json').exists():
        if load(folder/'identity.json')!=identity: raise ValueError('Case identity changed')
    elif any(folder.iterdir()): raise ValueError('Unrecognized case output')
    else: save(folder/'identity.json',identity)
    if (folder/'done.json').exists():
        verify_done(folder);print('Verified completed sensitivity:',case.model,split,flush=True);return
    grouped={}; summaries=[]; nonoverlap=[]; pred_maps=[]; targets=[]
    indices=set(f['nonoverlap_indices'])
    for i,sample in enumerate(data.rows):
        raw=np.asarray(predictions[i])
        if not np.isfinite(raw).all(): raise ValueError(f'Nonfinite prediction: {case.model}/{split}/{i}')
        keys,gray=data.target_keys(sample)
        target=np.zeros(SHAPE,dtype=np.float32);target.flat[keys]=gray/255.0
        # Multiply before thresholding to reproduce original float32 boundary behavior.
        truth=target*255.0>=f['target_threshold']
        truth_exp={d:expanded(truth,d) for d in (1,2,3,4,5)}
        raw_gray=raw*255.0
        masks={t:raw_gray>=t for t in (50,60,70,80,90,100)}
        for setting in f['settings']:
            t,d=setting['pred_threshold'],setting['d'];p=masks[t];ep=expanded(p,d)
            stats=score_masks(p,truth,ep,truth_exp[d])
            row=dict(model=case.model,seed=case.seed,split=split,pred_threshold=t,target_threshold=100,d=d,
                     row=i,target_start=sample['target_start'],target_end=sample['target_end'],nonoverlap=i in indices,**stats)
            grouped.setdefault((t,d),[]).append(row)
            if i in indices: nonoverlap.append(row)
        if split=='test' and i in indices:
            pred_maps.append(raw.copy());targets.append(target.copy())
        if (i+1)%100==0: print('scored',case.model,split,i+1,'/',len(data),flush=True)
    for (t,d),rows in grouped.items():
        for cohort in ('daily','nonoverlap'):
            summary=dict(model=case.model,seed=case.seed,split=split,pred_threshold=t,target_threshold=100,d=d,
                         **summarize(rows,cohort))
            if (t,d)==(100,3):
                anchor=f['anchors'][case.model][split][cohort]
                for metric in METRICS:
                    if (not math.isclose(summary[metric+'_mean'],anchor[metric],rel_tol=0,abs_tol=1e-10)
                        or not math.isclose(summary[metric+'_sd'],anchor[metric+'_sd'],rel_tol=0,abs_tol=1e-10)):
                        raise ValueError(f'Formal-score anchor mismatch: {case.model}/{split}/{cohort}/{metric}')
            summaries.append(summary)
    write_csv(folder/'summary.csv',summaries);write_csv(folder/'nonoverlap.csv',nonoverlap)
    write_csv(folder/'per_window.csv',[r for rows in grouped.values() for r in rows])
    names=['identity.json','summary.csv','nonoverlap.csv','per_window.csv']
    if split=='test':
        if len(pred_maps)!=6: raise ValueError('Expected six exported test maps')
        np.savez_compressed(folder/'six_maps.npz',predictions=np.stack(pred_maps),targets=np.stack(targets))
        names.append('six_maps.npz')
    save(folder/'done.json',dict(complete=True,identity=identity,prediction_source=str(pred_dir),anchor_verified=True,
         files_sha256={name:sha(folder/name) for name in names}))


def collect(out,f,require_analysis=False):
    compact=out/'compact_core_summary';summaries=[];windows=[];missing=[];maps={};reference_targets=None;receipts={}
    if (compact/'direction_B_receipt.json').is_file():
        for name,h in load(compact/'direction_B_receipt.json')['files_sha256'].items():
            if sha(compact/name)!=h: raise ValueError('Direction control summary changed')
    for model in MODELS:
        for split in ('validation','test'):
            folder=out/'analysis_cases'/model/split
            if not (folder/'done.json').is_file(): missing.append(model+'/'+split);continue
            receipts[model+'/'+split]=verify_done(folder)
            summaries.extend(read_csv(folder/'summary.csv'));windows.extend(read_csv(folder/'nonoverlap.csv'))
            if split=='test':
                with np.load(folder/'six_maps.npz',allow_pickle=False) as z:
                    maps[model+'_predictions']=z['predictions'].copy();target=z['targets'].copy()
                if reference_targets is None: reference_targets=target
                elif not np.array_equal(reference_targets,target): raise ValueError('Exported targets differ by model')
    if summaries:
        lookup={(r['model'],r['split'],r['cohort'],r['pred_threshold'],r['d']):r for r in summaries}
        for r in summaries:
            base=lookup.get(('MAG',r['split'],r['cohort'],r['pred_threshold'],r['d']))
            for metric in METRICS:
                a,b=r[metric+'_mean'],base[metric+'_mean'] if base else ''
                r[metric+'_delta_vs_MAG_pp']=100*(float(a)-float(b)) if a!='' and b!='' else None
        write_csv(compact/'sensitivity_summary.csv',summaries)
        write_csv(compact/'sensitivity_nonoverlap.csv',windows)
    if len(maps)==4:
        maps['targets']=reference_targets
        np.savez_compressed(compact/'six_window_maps.npz',**maps)
    save(compact/'grid_and_protocol.json',dict(region=f['region'],row_orientation='row 0 is south; columns west to east',
         prediction_scale='raw normalized model output; multiply by 255 for gray',target_threshold=100,
         settings=f['settings'],nonoverlap_indices=f['nonoverlap_indices'],
         test_target_dates=[{'row':r['row'],'target_start':r['target_start'],'target_end':r['target_end']}
          for r in windows if r['model']=='MAG' and r['split']=='test' and r['pred_threshold']=='100' and r['d']=='3'],
         metric_definition=f['metric_definition'],standard_deviation='window population SD ddof=0; not confidence intervals'))
    save(compact/'core_completion_frozen.json',f)
    receipt_files=[p for p in compact.iterdir() if p.is_file() and p.name!='summary_manifest.json']
    save(compact/'summary_manifest.json',dict(analysis_complete=not missing,missing_analysis_cases=missing,
         direction_B_complete=all((compact/n).is_file() for n in ['direction_B_metrics.csv','direction_B_nonoverlap.csv','direction_B_receipt.json']),
         frozen_sha256=FROZEN_SHA,runner_sha256=sha(Path(__file__)),case_receipts=receipts,
         files_sha256={p.name:sha(p) for p in receipt_files},archives_created=False))
    print('Download folder:',compact,'; missing analysis cases:',missing,flush=True)
    if require_analysis and missing: raise ValueError('Analysis is incomplete; rerun --stage analysis')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['check','control','analysis','summary'],default='check')
    p.add_argument('--project-root',type=Path,default=PROJECT)
    p.add_argument('--data',type=Path)
    p.add_argument('--compare-dir',type=Path)
    p.add_argument('--compare-output',type=Path)
    p.add_argument('--out',type=Path)
    p.add_argument('--models',nargs='+',choices=MODELS,default=list(MODELS))
    p.add_argument('--splits',nargs='+',choices=['validation','test'],default=['validation','test'])
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    a=p.parse_args()
    if len(set(a.models))!=len(a.models) or len(set(a.splits))!=len(a.splits): p.error('Duplicate model or split')
    project=a.project_root.resolve();data=a.data.resolve() if a.data else project/'cl10_mapping_v2'
    compare=a.compare_dir.resolve() if a.compare_dir else project/'training/compare'
    compare_out=a.compare_output.resolve() if a.compare_output else project/'runs/compare_abc_v1'
    out=a.out.resolve() if a.out else project/'runs/core_completion_v1'
    f,cases,engine=check(project,data,compare)
    for source in [HERE,data,compare,compare_out,*(c.run.resolve() for c in cases)]:
        if out==source or out in source.parents or source in out.parents: raise ValueError(f'Output overlaps input: {source}')
    if a.stage=='check':
        print('Preflight passed. Only control trains B_pair_03_08_run_03. No jobs or outputs created.');return
    out.mkdir(parents=True,exist_ok=True)
    with (out/'.run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        prepare_root(out,project,data,f)
        if a.stage=='control': control(project,data,compare,compare_out,out,a.device,f)
        if a.stage=='analysis':
            for c in cases:
                if c.model in a.models:
                    for split in a.splits: analyze_case(c,split,data,project,out,a.device,f)
        collect(out,f,require_analysis=a.stage=='analysis' and set(a.models)==set(MODELS) and len(a.splits)==2)


if __name__=='__main__': main()
