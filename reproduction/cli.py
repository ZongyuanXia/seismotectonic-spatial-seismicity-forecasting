"""One path-aware entry point; scientific computations require --execute."""
import argparse
import csv
from decimal import Decimal
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

from .common import Layout, MODELS, PROTOCOL, SOFTWARE, dump, import_cloud, load, read, sha

COMPUTE={'build-rf','build-cl10','rf-accuracy','rf-pairs','rf-ablation','train',
         'predict','evaluate','sensitivity','baseline-pi','baseline-etas','baseline-compare','full'}
STEPS=sorted(COMPUTE|{'check','prepare','import-cloud','export','figures','verify','reproduce','rf-summary'})


def job_plan(layout,args,step):
    project=layout.project
    model_job=step in ('train','predict','evaluate','sensitivity','baseline-compare','baseline-pi','baseline-etas')
    python=getattr(args,'python_model' if model_job else 'python_rf',None) or sys.executable
    def source(script,arguments,overrides=None):
        cmd=[python,str(SOFTWARE/'reproduction/run_source.py'),'--script',str(project/script)]
        for k,v in (overrides or {}).items():cmd+=['--path',k+'='+str(v)]
        return [*cmd,'--',*map(str,arguments)]
    if step=='build-rf':
        if not args.wsm:raise ValueError('--wsm is required for raw WSM reconstruction')
        return [source('rf/build_rf_wsm.py',['--wsm',Path(args.wsm).resolve(),'--config',project/'reference/region_config.py','--output',project/'datasets/rf_wsm'])]
    if step=='build-cl10':
        if not args.wsm or not args.usgs:raise ValueError('--wsm and --usgs are required for raw frame reconstruction')
        return [source('convlstm/build_cl10.py',['--wsm',Path(args.wsm).resolve(),'--usgs',Path(args.usgs).resolve(),
            '--region',project/'datasets/region_meta.json','--output',project/'datasets/cl10_mapping_v2'])]
    if step=='rf-accuracy':return [source('rf_analysis/run_rf_archive_replay.py',
        ['--source-dir',project/'datasets/rf_wsm','--out-dir',project/'rf_analysis/results_1000','--repeats','1000'],
        {'ARCHIVED_SCRIPT':project/'reference/04a_rf_full_and_stability.py'})]
    if step in ('rf-pairs','rf-ablation'):
        folder='rf_pairs' if step=='rf-pairs' else 'rf_ablation'
        script='run_pairs.py' if step=='rf-pairs' else 'run_ablation.py'
        return [source(folder+'/'+script,['--out-dir',project/folder/'results','--repeats','1000','--workers',args.workers])]
    if step=='rf-summary':return [source(folder+'/'+script,[]) for folder,script in
        [('rf_pairs','analyze_results.py'),('rf_ablation','analyze_ablation.py')]]
    if step in ('train','predict','evaluate','sensitivity','baseline-compare'):
        base=[python,str(SOFTWARE/'reproduction/worker.py'),step,'--data-root',str(layout.data),
              '--output-root',str(layout.out),'--weights',args.weights,'--device',args.device]
        if step=='baseline-compare':return [base]
        models=args.models or list(MODELS)
        if step=='sensitivity':models=[m for m in models if m in ('M1','M2','M3','M4')]
        return [[*base,'--model',m,*([] if step=='train' else ['--split',s])]
                for m in models for s in ([''] if step=='train' else args.splits)]
    if step=='baseline-pi':return [source('baseline/run_pi_baseline.py',[
        '--data',project/'datasets/cl10_mapping_v2','--training-dir',project/'convlstm/training','--split',s,
        '--out',project/'baseline/results/pi_japan_4050d_change450d_positive_v1'/s,'--workers',args.workers]) for s in args.splits]
    if step=='baseline-etas':
        commands=[]
        if 'validation' in args.splits:
            commands.append(source('baseline/etas/MC_sensitivity_1000_5000_20261030/run_mc_sensitivity.py',
                ['--root',project,'--out',project/'baseline/etas/MC_sensitivity_1000_5000_20261030/results','--workers',args.workers]))
        if 'test' in args.splits:
            commands.append(source('baseline/final_comparison/run_etas_test.py',
                ['--root',project,'--out',project/'baseline/final_comparison/etas_test','--workers',args.workers,'--simulations','5000']))
        return commands
    return []


def check(layout):
    import ast
    data=layout.data/'datasets/cl10_mapping_v2'
    files={'manifest':'manifest.json','sample_index':'sample_index.csv','usgs_daily':'usgs_daily.npz'}
    dataset={k:(data/n).is_file() and sha(data/n)==PROTOCOL['expected_dataset_sha256'][k] for k,n in files.items()}
    missing=[]
    for ident in MODELS:
        try:layout.check_model(ident)
        except (OSError,ValueError,KeyError) as e:missing.append(dict(model=ident,reason=str(e)))
    syntax=[]
    for p in (SOFTWARE/'reproduction').glob('*.py'):ast.parse(p.read_text());syntax.append(p.name)
    modules={n:importlib.util.find_spec(n) is not None for n in ['numpy','pandas','matplotlib','cartopy','torch','sklearn','scipy']}
    report=dict(source=layout.source,output=str(layout.out),dataset_hashes=dataset,
        missing_models=missing,syntax_checked=syntax,
        modules=modules,formal_model_files_ready=all(dataset.values()) and not missing,
        model_execution=False,full_model_reproduction_ready=all(dataset.values()) and not missing and all(modules.values()),
        environment_check_scope='Current interpreter only; separate RF/model/figure interpreters may be supplied for computational jobs')
    dump(layout.out/'preflight.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return report


def verify(layout):
    differences=[]
    for n in range(1,6):
        actual=layout.tables/('Table%d_display.csv'%n);reference=layout.data/('manuscript/tables/Table%d_display.csv'%n)
        with actual.open(encoding='utf-8-sig',newline='') as f:a=list(csv.reader(f))[1:]
        with reference.open(encoding='utf-8-sig',newline='') as f:b=list(csv.reader(f))[1:]
        fields={1:[0,1,2,3],2:[2,3,4,5,6],3:[2,3,4,5,6],4:[2,3],5:[2,3]}[n]
        if len(a)!=len(b):differences.append(dict(table=n,reason='row count'));continue
        for i,(x,y) in enumerate(zip(a,b)):
            for j in fields:
                numbers=lambda s:[Decimal(t) for t in re.findall(r'-?\d+(?:\.\d+)?',s)]
                if numbers(x[j])!=numbers(y[j]):differences.append(dict(table=n,row=i+1,column=j+1,actual=x[j],expected=y[j]))
        if n==1:
            for i,(x,y) in enumerate(zip(a,b)):
                if re.sub(r'\s|\$','',x[4])!=re.sub(r'\s|\$','',y[4]):differences.append(dict(table=n,row=i+1,reason='top four indicators'))
    figures={}
    for p in layout.figures.glob('*.png'):
        reference=layout.data/'manuscript/figures'/p.name
        figures[p.name]=dict(sha256=sha(p),byte_identical=reference.exists() and sha(p)==sha(reference))
    report=dict(source=layout.source,tables_match_manuscript=not differences,differences=differences,
        figures=figures,figures_exported=len(figures),all_five_figures_identical=len(figures)==5 and all(v['byte_identical'] for v in figures.values()),
        model_execution=False,note='Image bytes can differ across font, OS and dependency versions; inspect maps/data if they differ.')
    dump(layout.out/'verification.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2))
    if differences:raise ValueError('Generated tables differ from the manuscript; see verification.json')
    return report


def figures(layout,python=None):
    if not (layout.out/'export_manifest.json').is_file():raise ValueError('Run export before figures')
    env=os.environ.copy();env['MPLCONFIGDIR']=str(layout.out/'cache/matplotlib')
    env['PAPER_CARTOPY_DATA']=str(layout.project/'cartopy')
    subprocess.run([python or sys.executable,str(layout.project/'plotting/reproduce_figures.py')],check=True,env=env)
    layout.figures.mkdir(parents=True,exist_ok=True)
    for p in (layout.project/'outputs/figures').glob('figure*.png'):shutil.copy2(p,layout.figures/p.name)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('step',choices=STEPS)
    p.add_argument('--data-root',type=Path);p.add_argument('--output-root',type=Path)
    p.add_argument('--source',choices=['archived','records','computed'])
    p.add_argument('--weights',choices=['frozen','retrained'],default='frozen')
    p.add_argument('--cloud-return',type=Path)
    p.add_argument('--models',nargs='+',choices=list(MODELS))
    p.add_argument('--splits',nargs='+',choices=['validation','test'],default=['validation','test'])
    p.add_argument('--workers',type=int,default=4);p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    p.add_argument('--python-rf',help='RF/data Python executable; defaults to the current interpreter')
    p.add_argument('--python-model',help='PyTorch/ETAS Python executable; defaults to the current interpreter')
    p.add_argument('--python-figures',help='Figure Python executable; defaults to the current interpreter')
    p.add_argument('--wsm',type=Path);p.add_argument('--usgs',type=Path)
    p.add_argument('--execute',action='store_true')
    a=p.parse_args();source=a.source or ('records' if a.step=='import-cloud' else 'computed' if a.step in COMPUTE else 'archived')
    if a.step in COMPUTE and source!='computed':p.error('Computational jobs require --source computed')
    if a.weights=='retrained' and source!='computed':p.error('Retrained weights require --source computed')
    layout=Layout(a.data_root,a.output_root,source,a.weights)
    if a.step=='full' and (a.models and set(a.models)!=set(MODELS) or set(a.splits)!={'validation','test'}):
        p.error('full requires M1--M7 and both splits; use individual steps for subsets')
    steps=(['rf-accuracy','rf-pairs','rf-ablation','rf-summary','predict','evaluate','sensitivity','baseline-pi','baseline-etas','baseline-compare','export','figures','verify']
           if a.step=='full' else ['export','figures','verify'] if a.step=='reproduce' else [a.step])
    plan=[dict(step=s,commands=job_plan(layout,a,s)) for s in steps]
    if not a.execute and a.step!='check':
        print(json.dumps(dict(source=source,weights=a.weights,data=str(layout.data),output=str(layout.out),
              execute=False,steps=plan,cloud_import=str(a.cloud_return) if a.cloud_return else None),ensure_ascii=False,indent=2));return
    layout.prepare()
    if a.cloud_return:print('Imported cloud files:',import_cloud(layout,a.cloud_return,include_results=a.step!='full'))
    if a.step=='full':
        # Fail before starting RF fits if the model-dependent steps cannot run.
        for ident in MODELS:layout.check_model(ident)
    for s in steps:
        if s in ('prepare','import-cloud'):
            if s=='import-cloud' and not a.cloud_return:p.error('import-cloud requires --cloud-return')
            print('Prepared',layout.project);continue
        if s=='check':check(layout);continue
        if s=='export':
            from .exports import export_all
            export_all(layout);continue
        if s=='figures':figures(layout,a.python_figures);continue
        if s=='verify':verify(layout);continue
        commands=job_plan(layout,a,s)
        dump(layout.out/('jobs/'+s+'.json'),dict(commands=commands,execution_requested=True))
        for command in commands:subprocess.run(command,check=True)


if __name__=='__main__':main()
