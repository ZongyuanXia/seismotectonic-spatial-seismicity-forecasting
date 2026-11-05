"""Explicit computational jobs, invoked only by run_workflow.py --execute."""
import argparse
from collections import defaultdict
from pathlib import Path
import sys
from types import SimpleNamespace

if __package__ in (None, ''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from reproduction.common import Layout, MODELS, PROTOCOL, dump, load, module, read, sha, write


def sources(layout):
    training=layout.project/'convlstm/training'
    sys.path.insert(0,str(layout.project/'compare'))
    sys.path.insert(0,str(training))
    return training


def train(layout, ident, device):
    training=sources(layout);entry=MODELS[ident]
    path=training/('train_cl10_dualseed.py' if ident=='M1' else 'train_cl10_mag_plus_two.py')
    trainer=module('paper_trainer_'+ident,path)
    trainer.WEIGHTS=tuple(PROTOCOL['training_weights'])
    original=trainer.create_run_config
    def create(*a,**kw):
        config=original(*a,**kw)
        config['reproduction_adapter']=dict(sha256=sha(__file__),protocol_sha256=sha(Path(__file__).with_name('protocol.json')),
                                           weights=list(PROTOCOL['training_weights']))
        if entry['intervention']:
            config['input_intervention']=dict(group=entry['intervention'],
                gray_value=0 if entry['intervention']=='A' else 100,
                support='stored frame valid mask',mag_unchanged=True,targets_unchanged=True)
        return config
    trainer.create_run_config=create
    if entry['intervention']:
        from compare_data import dataset_class
        trainer.CL10Data=dataset_class(entry['intervention'])
    args=SimpleNamespace(data=str(layout.project/'datasets/cl10_mapping_v2'),
        out=str(layout.project/'trained'/ident),channels=','.join(map(str,entry['selected_channels'])),
        baseline_run=str(layout.project/'trained/M1'),device=device,resume=False,smoke=False,
        init_seed=PROTOCOL['init_seed'],order_seed=PROTOCOL['order_seed'])
    trainer.train(args)


def predict(layout, ident, split, device):
    training=sources(layout);run=layout.check_model(ident)
    predictor=module('paper_predict_'+ident,training/'predict_cl10.py')
    intervention=MODELS[ident]['intervention']
    if intervention:
        from compare_data import dataset_class
        predictor.CL10Data=dataset_class(intervention)
    dest=layout.predictions(ident,split)
    predictor.predict(SimpleNamespace(data=str(layout.project/'datasets/cl10_mapping_v2'),run=str(run),
        split=split,out=str(dest),device=device))
    if intervention:
        manifest=load(dest/'manifest.json')
        manifest['input_intervention']=load(run/'run.json').get('input_intervention',
            dict(group=intervention,gray_value=0 if intervention=='A' else 100))
        dump(dest/'manifest.json',manifest)


def evaluate(layout, ident, split):
    training=sources(layout);run=layout.check_model(ident)
    scorer=module('paper_evaluate',training/'evaluate_cl10_symmetric.py')
    scorer.evaluate(SimpleNamespace(data=str(layout.project/'datasets/cl10_mapping_v2'),
        run=str(run),split=split,predictions=str(layout.predictions(ident,split)),
        out=str(layout.evaluation(ident,split)),cleanup_predictions=False))


def sensitivity(layout, ident, split):
    """Use the original matching and aggregation functions, for current s0 models."""
    import numpy as np
    training=sources(layout);run=layout.check_model(ident)
    scorer=module('paper_evaluate',training/'evaluate_cl10_symmetric.py')
    core=module('paper_core',training/'run_core_completion.py')
    from cl10_data import CL10Data
    data=CL10Data(layout.project/'datasets/cl10_mapping_v2',split,(0,),cache_size=0)
    pm,predictions,pred_sha=scorer._validate_inputs(data,layout.predictions(ident,split),split)
    if pm['best_checkpoint_sha256']!=sha(run/'best.pt'):raise ValueError('Predictions use a different model')
    folder=layout.sensitivity(ident,split)
    if folder.exists():raise FileExistsError('Choose a fresh output for sensitivity: '+str(folder))
    folder.mkdir(parents=True)
    identity=dict(model=MODELS[ident]['source_model'],seed='run_03',split=split,
        checkpoint_sha256=sha(run/'best.pt'),prediction_sha256=pred_sha,
        dataset_manifest_sha256=pm['dataset_manifest_sha256'],adapter_sha256=sha(__file__))
    dump(folder/'identity.json',identity)
    grouped=defaultdict(list);nonoverlap=[];saved_pred=[];saved_targets=[]
    for i,sample in enumerate(data.rows):
        raw=np.asarray(predictions[i]);keys,gray=data.target_keys(sample)
        target=np.zeros((340,200),dtype=np.float32);target.flat[keys]=gray/255.0
        truth=target>0
        if not np.array_equal(truth,target*255>=100):raise ValueError('Unexpected target encoding')
        truths={d:core.expanded(truth,d) for d in range(1,6)}
        masks={t:raw*255>=t for t in [50,60,70,80,90,100]}
        for s in PROTOCOL['sensitivity_settings']:
            t,d=s['pred_threshold'],s['d'];pred=masks[t]
            stats=core.score_masks(pred,truth,core.expanded(pred,d),truths[d])
            row=dict(model=MODELS[ident]['source_model'],seed='run_03',split=split,pred_threshold=t,d=d,
                row=i,target_start=sample['target_start'],target_end=sample['target_end'],
                nonoverlap=i in PROTOCOL['nonoverlap_indices'],**stats)
            grouped[t,d].append(row)
            if row['nonoverlap']:nonoverlap.append(row)
        if split=='test' and i in PROTOCOL['nonoverlap_indices']:
            saved_pred.append(raw.copy());saved_targets.append(target.copy())
        if (i+1)%100==0:print('sensitivity',ident,split,i+1,'/',len(data),flush=True)
    summaries=[]
    for (t,d),rows in grouped.items():
        for cohort in ('daily','nonoverlap'):
            summaries.append(dict(model=MODELS[ident]['source_model'],split=split,pred_threshold=t,d=d,
                                  **core.summarize(rows,cohort)))
    write(folder/'per_window.csv',[r for rows in grouped.values() for r in rows])
    write(folder/'summary.csv',summaries);write(folder/'nonoverlap.csv',nonoverlap)
    names=['identity.json','per_window.csv','summary.csv','nonoverlap.csv']
    if split=='test':
        np.savez_compressed(folder/'six_maps.npz',predictions=np.stack(saved_pred),targets=np.stack(saved_targets))
        names.append('six_maps.npz')
    dump(folder/'done.json',dict(complete=True,identity=identity,
         files_sha256={n:sha(folder/n) for n in names},adapter=True))


def baseline_compare(layout):
    """M2 validation-recall matching only; no test-set threshold selection."""
    import numpy as np
    import pandas as pd
    sources(layout)
    folder=layout.project/'baseline/final_comparison'
    core=module('evaluate_baselines',folder/'evaluate_baselines.py')
    matching=module('paper_matching',folder/'reevaluate_recall_matched.py')
    core.setup(layout.project)
    from cl10_data import CL10Data
    from spatial_symmetric import symmetric_expand
    region=load(layout.project/'baseline/etas/region_meta.json');area=core.areas(region)
    audit=pd.read_csv(layout.project/'datasets/cl10_mapping_v2/usgs_event_audit.csv')
    audit['time']=pd.to_datetime(audit.time,format='ISO8601',utc=True).dt.tz_localize(None)
    windows={};scores={};reference={};output=layout.results/'baseline';output.mkdir(parents=True,exist_ok=True)
    lineage={}
    for split in ('validation','test'):
        data=CL10Data(layout.project/'datasets/cl10_mapping_v2',split,(0,),cache_size=0)
        rows=read(layout.sensitivity('M2',split)/'per_window.csv')
        anchor=[r for r in rows if int(r['row']) in PROTOCOL['nonoverlap_indices'] and int(r['d'])==3 and float(r['pred_threshold'])==100]
        if len(anchor)!=6:raise ValueError('Missing M2 six-window anchor')
        reference[split]=anchor
        windows[split]=[core.target(data,row) for row in PROTOCOL['nonoverlap_indices']]
        for model in ('PI','ETAS','SSS'):
            arr=[]
            for i,row in enumerate(PROTOCOL['nonoverlap_indices']):
                origin=data.rows[row]['target_start']
                if model=='PI':
                    p=layout.project/('baseline/results/pi_japan_4050d_change450d_positive_v1/%s/maps/sample_%05d_%s.npz'%(split,row,origin));key='pi_map'
                elif model=='ETAS':
                    p=(layout.project/('baseline/etas/MC_sensitivity_1000_5000_20261030/results/window_%05d/ETAS_N5000.npz'%row)
                       if split=='validation' else folder/('etas_test/ETAS_%05d.npz'%row));key='probability'
                else:
                    a,_=core.historical_probability(audit,pd.Timestamp(origin),region)
                    p=output/('SSS_%s_%05d.npz'%(split,row));key='probability'
                    np.savez_compressed(p,probability=a,origin=np.array(origin))
                with np.load(p,allow_pickle=False) as z:
                    if model=='ETAS' and int(z['simulations'])!=5000:raise ValueError('ETAS must use 5000 simulations')
                    arr.append(z[key].astype(np.float64))
                lineage[str(p)]=sha(p)
            scores[model,split]=arr
    target_recall=float(np.mean([float(r['recall']) for r in reference['validation']]))
    thresholds=[];details=[];summary=[]
    for model in ('PI','ETAS','SSS'):
        ts,rs,means=matching.attainable_recall(scores[model,'validation'],windows['validation'],symmetric_expand)
        k=matching.select(ts,means,target_recall);threshold=float(ts[k])
        thresholds.append(dict(model=model,threshold=threshold,reference_mean_recall=target_recall,
                               validation_mean_recall=float(means[k]),test_targets_used=False))
        observed=[]
        for i,row in enumerate(PROTOCOL['nonoverlap_indices']):
            stats=core.stats(scores[model,'test'][i]>=threshold,windows['test'][i],area)
            observed.append(stats);details.append(dict(model=model,row=row,threshold=threshold,**stats))
        summary.append(dict(anchor='pair_01_02',model=model,windows=6,
            precision_mean=float(np.mean([r['precision'] for r in observed])),precision_sd=float(np.std([r['precision'] for r in observed])),
            recall_mean=float(np.mean([r['recall'] for r in observed])),recall_sd=float(np.std([r['recall'] for r in observed]))))
    r=reference['test']
    summary.insert(0,dict(anchor='pair_01_02',model='pair_01_02',windows=6,
        precision_mean=float(np.mean([float(x['precision']) for x in r])),precision_sd=float(np.std([float(x['precision']) for x in r])),
        recall_mean=float(np.mean([float(x['recall']) for x in r])),recall_sd=float(np.std([float(x['recall']) for x in r]))))
    write(output/'recall_matched_test_summary.csv',summary);write(output/'test_by_window.csv',details)
    write(output/'validation_thresholds.csv',thresholds)
    dump(output/'manifest.json',dict(complete=True,inputs_sha256=lineage,threshold_reference='M2 validation mean recall',
        test_threshold_reselected=False,files_sha256={p.name:sha(p) for p in output.glob('*.csv')}))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('job',choices=['train','predict','evaluate','sensitivity','baseline-compare'])
    p.add_argument('--data-root',required=True);p.add_argument('--output-root',required=True)
    p.add_argument('--weights',choices=['frozen','retrained'],default='frozen')
    p.add_argument('--model',choices=list(MODELS));p.add_argument('--split',choices=['validation','test'])
    p.add_argument('--device',default='auto',choices=['auto','cpu','cuda'])
    a=p.parse_args();layout=Layout(a.data_root,a.output_root,'computed',a.weights)
    if a.job=='baseline-compare':baseline_compare(layout)
    elif a.job=='train':train(layout,a.model,a.device)
    elif a.job=='predict':predict(layout,a.model,a.split,a.device)
    elif a.job=='evaluate':evaluate(layout,a.model,a.split)
    else:sensitivity(layout,a.model,a.split)


if __name__=='__main__':main()
