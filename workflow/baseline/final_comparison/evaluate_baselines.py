"""Common P/R evaluation of seven models on six nonoverlapping test targets.

No training, no extra smoothing of PI/ETAS/ConvLSTM maps, and no CSI.
The added historical-activity baseline is itself a fixed 25-km Gaussian model.
"""
from pathlib import Path
from datetime import datetime, timedelta
import argparse, csv, hashlib, json, sys
import numpy as np
from scipy.ndimage import gaussian_filter

ROOT = Path('/Users/xiazongyuan/1 课题/7 地震震级预测/xzy-小论文1-文件汇总/代码与结果（归档）/九项指标重构归档')
HERE = Path(__file__).resolve().parent
SHAPE = (340, 200)
ROWS = [0, 450, 900, 1350, 1800, 2250]
CONV = ['MAG', 'pair_01_02', 'pair_03_08', 'pair_05_06']
MODELS = CONV + ['PI', 'ETAS', 'SSS']
LABEL = {'MAG':'震级单通道', 'pair_01_02':'震级＋深度均值＋标准差',
    'pair_03_08':'震级＋最大偏转角＋R', 'pair_05_06':'震级＋机制H＋D',
    'PI':'PI', 'ETAS':'ETAS（5000次）', 'SSS':'历史地震活动平滑分布'}
THRESHOLDS = [.0002, .0005, .001, .002, .005, .01, .02, .05, .1, .2, .5]
FRACTIONS = [.0025, .005, .0075, .01, .02, .03]

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def dump(p, obj):
    Path(p).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def read(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))

def write(p, rows):
    assert rows
    with Path(p).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def setup(root):
    sys.path.insert(0,str(root/'convlstm/training'))
    from spatial_symmetric import symmetric_expand
    from cl10_data import CL10Data
    global expand, Reader
    expand=lambda m: symmetric_expand(np.asarray(m,np.uint8),d=3)>0
    Reader=CL10Data

def areas(region):
    edges=(np.arange(341)+region['crop_box_closed']['y_min'])/region['global_H']*180-90
    a=6378.1**2*np.radians(360/region['global_W'])*np.diff(np.sin(np.radians(edges)))
    return np.broadcast_to(a[:,None],SHAPE).copy()

def target(dataset,row):
    keys,gray=dataset.target_keys(dataset.rows[row])
    t=np.zeros(SHAPE,bool);t.flat[keys[gray>=100]]=True
    return t

def stats(alarm,t,area,et=None):
    """P counts alarm cells; R counts true occupied cells; they have different numerators."""
    a=np.asarray(alarm,bool);ep=expand(a);et=expand(t) if et is None else et
    n=int(a.sum());nt=int(t.sum());ph=int((a&et).sum());rh=int((t&ep).sum());tp=int((a&t).sum())
    return dict(precision=ph/n if n else None,recall=rh/nt if nt else None,
        precision_exact=tp/n if n else None,recall_exact=tp/nt if nt else None,
        predicted_hit_cells=ph,covered_true_cells=rh,exact_hit_cells=tp,
        false_alarm_cells=n-ph,missed_true_cells=nt-rh,
        alarm_cells=n,expanded_alarm_cells=int(ep.sum()),target_cells=nt,
        alarm_km2=float(area[a].sum()),alarm_area_fraction=float(area[a].sum()/area.sum()),
        expanded_alarm_km2=float(area[ep].sum()),expanded_area_fraction=float(area[ep].sum()/area.sum()))

def ranking(score,area):
    tie=np.random.default_rng(2901).random(score.size)
    order=np.lexsort((tie,-score.ravel().astype(float)))
    return order,np.r_[0.,np.cumsum(area.ravel()[order])]

def prefix(order,k):
    a=np.zeros(np.prod(SHAPE),bool);a[order[:k]]=True
    return a.reshape(SHAPE)

def raw_match(rank,budget):
    order,cum=rank
    k=int(np.searchsorted(cum,budget));choices={max(0,k-1),min(len(order),k)}
    return prefix(order,min(choices,key=lambda j:(abs(cum[j]-budget),j)))

def expanded_match(rank,budget,area):
    order,_=rank;lo,hi=0,len(order);cache={}
    def covered(k):
        if k not in cache: cache[k]=float(area[expand(prefix(order,k))].sum())
        return cache[k]
    while lo<hi:
        mid=(lo+hi)//2
        if covered(mid)<budget:lo=mid+1
        else:hi=mid
    k=min({max(0,lo-1),lo},key=lambda j:(abs(covered(j)-budget),j))
    return prefix(order,k)

def historical_probability(audit,origin,region):
    """4050-day event counts -> 25-km kernel -> 450-day Poisson occupancy.

    Counts include repeated events in the same cell. Boundary truncation is
    normalized per source cell. Geographic distances use central-latitude
    grid spacing; sigma is km, not grid cells. It is fixed without test tuning.
    """
    start=origin-timedelta(days=4050)
    h=audit[(audit.time>=np.datetime64(start))&(audit.time<np.datetime64(origin))]
    count=np.bincount(h.key.to_numpy(),minlength=np.prod(SHAPE)).reshape(SHAPE).astype(float)
    b=region['effective_bounds'];lat=(b['lat_min']+b['lat_max'])/2
    dy=6378.1*np.radians(180/region['global_H'])
    dx=6378.1*np.cos(np.radians(lat))*np.radians(360/region['global_W'])
    sigma=(25/dy,25/dx)
    normalizer=gaussian_filter(np.ones(SHAPE),sigma,mode='constant',cval=0,truncate=4)
    smoothed=gaussian_filter(count/normalizer,sigma,mode='constant',cval=0,truncate=4)
    assert np.isclose(smoothed.sum(),count.sum(),rtol=1e-10)
    expected=smoothed/9 # 450 / 4050; expected event count per future window.
    return -np.expm1(-expected),dict(history_start=str(start.date()),
        history_end_exclusive=str(origin.date()),history_events=len(h),
        sigma_km=25,sigma_grid=list(sigma),probability_assumption='Poisson occupancy from a stationary smoothed rate')

def prepare(root,out):
    import pandas as pd
    setup(root);out.mkdir(parents=True,exist_ok=True)
    region=json.loads((root/'baseline/etas/region_meta.json').read_text());area=areas(region)
    fitdir=root/'baseline/etas/results/validation_00000_20090101_v2'
    fm=json.loads((fitdir/'fit_manifest.json').read_text())
    assert sha(root/'datasets/cl10_mapping_v2/usgs_event_audit.csv')==fm['sha256']['event_audit']
    audit=pd.read_csv(root/'datasets/cl10_mapping_v2/usgs_event_audit.csv')
    audit['time']=pd.to_datetime(audit.time,format='ISO8601',utc=True).dt.tz_localize(None)
    assert (audit.MW>=4).all() and audit.key.between(0,np.prod(SHAPE)-1).all()
    vd=Reader(root/'datasets/cl10_mapping_v2','validation',(0,),cache_size=0)
    td=Reader(root/'datasets/cl10_mapping_v2','test',(0,),cache_size=0)
    calibration=[];histmeta=[];prediction_sources={}
    for split,data in [('validation',vd),('test',td)]:
        for row in ROWS:
            origin=pd.Timestamp(data.rows[row]['target_start'])
            score,meta=historical_probability(audit,origin,region)
            p=out/f'SSS_{split}_{row:05d}.npz'
            np.savez_compressed(p,probability=score,origin=np.array(str(origin.date())))
            histmeta.append(dict(split=split,row=row,**meta,prediction_sha256=sha(p)))
            if split=='validation':
                etasp=root/f'baseline/etas/MC_sensitivity_1000_5000_20261030/results/window_{row:05d}/ETAS_N5000.npz'
                completion=json.loads((etasp.parent/'complete.json').read_text())
                assert completion['complete'] and completion['output_sha256'][etasp.name]==sha(etasp)
                with np.load(etasp,allow_pickle=False) as z:
                    assert str(z['origin'].item())==str(origin.date()) and int(z['simulations'])==5000
                    ep=z['probability'].copy()
                prediction_sources[str(etasp)]=sha(etasp)
                t=target(data,row);et=expand(t)
                for model,s in [('ETAS',ep),('SSS',score)]:
                    for th in THRESHOLDS:
                        calibration.append(dict(model=model,row=row,threshold=th,**stats(s>=th,t,area,et)))
    selections={};summary=[]
    for model in ['ETAS','SSS']:
        candidates=[]
        for th in THRESHOLDS:
            rs=[r for r in calibration if r['model']==model and r['threshold']==th]
            ps=[r['precision'] for r in rs if r['precision'] is not None]
            c=dict(model=model,threshold=th,validation_mean_precision=float(np.mean(ps)) if ps else None,
                validation_mean_recall=float(np.mean([r['recall'] for r in rs])),
                precision_valid_windows=len(ps),validation_mean_area_fraction=float(np.mean([r['alarm_area_fraction'] for r in rs])))
            c['eligible']=len(ps)==6 and c['validation_mean_recall']>=.60
            candidates.append(c)
        valid=[r for r in candidates if r['eligible']]
        assert valid,f'No threshold achieves the prespecified validation recall constraint: {model}'
        best=max(valid,key=lambda r:(r['validation_mean_precision'],r['validation_mean_recall'],-r['threshold']))
        selections[model]=best
        summary.extend(candidates)
    write(out/'validation_threshold_details.csv',calibration)
    write(out/'validation_threshold_summary.csv',summary)
    dump(out/'threshold_selection.json',dict(criterion='maximize mean validation P subject to mean validation R >= 0.60 and six defined P values',
        thresholds=THRESHOLDS,selections=selections,test_targets_used=False,
        note='60% follows the existing CL-stage recall safeguard; it is not a newly proven physical threshold.'))
    dump(out/'historical_baseline_manifest.json',dict(model='SSS',bandwidth_km=25,
        history_days=4050,forecast_days=450,bandwidth_tuned=False,windows=histmeta,
        validation_etas_sources=prediction_sources))
    print('PREPARED',json.dumps(selections,ensure_ascii=False),flush=True)

def aggregate(rows,keys):
    groups={}
    for r in rows:groups.setdefault(tuple(r[k] for k in keys),[]).append(r)
    result=[]
    fields=['precision','recall','precision_exact','recall_exact','alarm_km2','alarm_area_fraction',
        'expanded_alarm_km2','expanded_area_fraction','alarm_cells','false_alarm_cells','missed_true_cells',
        'exact_hit_cells','covered_true_cells']
    for group,rs in groups.items():
        a=dict(zip(keys,group));a['windows']=len(rs)
        for f in fields:
            values=[r[f] for r in rs if r[f] is not None]
            a[f+'_mean']=float(np.mean(values)) if values else None
            a[f+'_sd']=float(np.std(values)) if values else None
        result.append(a)
    return result

def complete(root,out):
    setup(root)
    selection=json.loads((out/'threshold_selection.json').read_text())
    assert selection['test_targets_used'] is False
    src=root/'convlstm/runs/compact_core_summary'
    cloud=json.loads((src/'summary_manifest.json').read_text())
    assert cloud['analysis_complete']
    for name,h in cloud['files_sha256'].items():assert sha(src/name)==h,name
    grid=json.loads((src/'grid_and_protocol.json').read_text());windows=grid['test_target_dates']
    region=grid['region'];area=areas(region)
    data=Reader(root/'datasets/cl10_mapping_v2','test',(0,),cache_size=0)
    with np.load(src/'six_window_maps.npz',allow_pickle=False) as z: cm={k:z[k].copy() for k in z.files}
    em=json.loads((out/'etas_test/manifest.json').read_text());assert em['complete'] and em['simulations']==5000
    pi_root=root/'baseline/results/pi_japan_4050d_change450d_positive_v1/test'
    pid=json.loads((pi_root/'identity.json').read_text())
    for alias,filename in [('manifest','manifest.json'),('sample_index','sample_index.csv'),('usgs_daily','usgs_daily.npz')]:
        assert sha(root/'datasets/cl10_mapping_v2'/filename)==pid['dataset_sha256'][alias]
    alarms=[];equal=[];curves=[];threshold_results=[];sources={};matches=[]
    save_maps={};thresholds={m:100/255 for m in CONV}
    thresholds.update(PI=0,ETAS=selection['selections']['ETAS']['threshold'],SSS=selection['selections']['SSS']['threshold'])
    cloud_anchor=read(src/'sensitivity_nonoverlap.csv')
    for i,w in enumerate(windows):
        row=int(w['row']);assert row==ROWS[i] and data.rows[row]['target_start']==w['target_start']
        t=target(data,row);assert np.array_equal(t,cm['targets'][i]*255>=100)
        et=expand(t);scores={m:cm[m+'_predictions'][i] for m in CONV}
        pip=pi_root/f'maps/sample_{row:05d}_{w["target_start"]}.npz'
        etp=out/f'etas_test/ETAS_{row:05d}.npz';ssp=out/f'SSS_test_{row:05d}.npz'
        for m,p,key in [('PI',pip,'pi_map'),('ETAS',etp,'probability'),('SSS',ssp,'probability')]:
            with np.load(p,allow_pickle=False) as z:
                assert str(z['target_start'].item() if m=='PI' else z['origin'].item())==w['target_start']
                scores[m]=z[key].copy()
            sources[str(p)]=sha(p)
        assert all(s.shape==SHAPE and np.isfinite(s).all() for s in scores.values())
        native={m:(s*255>=100 if m in CONV else s>0 if m=='PI' else s>=thresholds[m]) for m,s in scores.items()}
        ranks={m:ranking(s,area) for m,s in scores.items()}
        for m in MODELS:
            st=stats(native[m],t,area,et)
            alarms.append(dict(model=m,label=LABEL[m],**w,threshold=100 if m in CONV else thresholds[m],**st))
            if m in CONV:
                old=next(a for a in cloud_anchor if a['model']==m and a['split']=='test' and a['row']==str(row) and a['d']=='3' and a['pred_threshold']=='100')
                assert abs(st['precision']-float(old['precision']))<1e-12
                assert abs(st['recall']-float(old['recall']))<1e-12
            for fraction in FRACTIONS:
                mask=raw_match(ranks[m],area.sum()*fraction)
                curves.append(dict(model=m,**w,requested_area_fraction=fraction,**stats(mask,t,area,et)))
            save_maps[f'{m}_scores_{row:05d}']=scores[m]
            save_maps[f'{m}_native_alarm_{row:05d}']=native[m]
        for m in ['ETAS','SSS']:
            for th in THRESHOLDS:
                threshold_results.append(dict(model=m,**w,threshold=th,selected_on_validation=th==thresholds[m],**stats(scores[m]>=th,t,area,et)))
        for anchor in CONV:
            ref=stats(native[anchor],t,area,et)
            for protocol,budget in [('equal_raw_area',ref['alarm_km2']),('equal_expanded_area',ref['expanded_alarm_km2'])]:
                for m in MODELS:
                    a=native[anchor] if m==anchor else raw_match(ranks[m],budget) if protocol=='equal_raw_area' else expanded_match(ranks[m],budget,area)
                    st=stats(a,t,area,et)
                    achieved=st['alarm_km2'] if protocol=='equal_raw_area' else st['expanded_alarm_km2']
                    equal.append(dict(protocol=protocol,anchor=anchor,model=m,**w,
                        requested_km2=budget,area_error_km2=achieved-budget,**st))
                    if m!=anchor:
                        matches.append(dict(protocol=protocol,anchor=anchor,model=m,row=row,
                            absolute_error_km2=abs(achieved-budget),relative_error=abs(achieved-budget)/budget))
        save_maps[f'target_{row:05d}']=t
        print('EVALUATED',row,flush=True)
    save_maps['cell_area_km2']=area
    np.savez_compressed(out/'comparison_maps.npz',**save_maps)
    write(out/'unrestricted_by_window.csv',alarms)
    native_summary=aggregate(alarms,['model']);write(out/'unrestricted_summary.csv',native_summary)
    native_differences=[];native_wins=[]
    for anchor in CONV:
        for baseline in ['PI','ETAS','SSS']:
            ds=[]
            for row in ROWS:
                a=next(r for r in alarms if r['model']==anchor and int(r['row'])==row)
                b=next(r for r in alarms if r['model']==baseline and int(r['row'])==row)
                d=dict(anchor=anchor,baseline=baseline,row=row,
                    precision_delta_pp=100*(a['precision']-b['precision']),
                    recall_delta_pp=100*(a['recall']-b['recall']))
                ds.append(d);native_differences.append(d)
            native_wins.append(dict(anchor=anchor,baseline=baseline,
                precision_wins=sum(d['precision_delta_pp']>0 for d in ds),
                recall_wins=sum(d['recall_delta_pp']>0 for d in ds),
                precision_delta_pp_mean=float(np.mean([d['precision_delta_pp'] for d in ds])),
                recall_delta_pp_mean=float(np.mean([d['recall_delta_pp'] for d in ds]))))
    write(out/'unrestricted_paired_differences.csv',native_differences)
    write(out/'unrestricted_paired_summary.csv',native_wins)
    write(out/'equal_area_by_window.csv',equal)
    equal_summary=aggregate(equal,['protocol','anchor','model']);write(out/'equal_area_summary.csv',equal_summary)
    write(out/'area_curves_by_window.csv',curves)
    write(out/'area_curves_summary.csv',aggregate(curves,['requested_area_fraction','model']))
    write(out/'test_threshold_results.csv',threshold_results)
    write(out/'area_matching_errors.csv',matches)
    differences=[]
    for r in equal:
        if r['model'] not in ['PI','ETAS','SSS']:continue
        ref=next(a for a in equal if a['protocol']==r['protocol'] and a['anchor']==r['anchor'] and a['model']==r['anchor'] and a['row']==r['row'])
        differences.append(dict(protocol=r['protocol'],anchor=r['anchor'],baseline=r['model'],row=r['row'],
            precision_delta_pp=100*(ref['precision']-r['precision']),recall_delta_pp=100*(ref['recall']-r['recall']),
            precision_exact_delta_pp=100*(ref['precision_exact']-r['precision_exact']),
            recall_exact_delta_pp=100*(ref['recall_exact']-r['recall_exact'])))
    write(out/'paired_differences.csv',differences)
    wins=[]
    for protocol in ['equal_raw_area','equal_expanded_area']:
        for anchor in CONV:
            for baseline in ['PI','ETAS','SSS']:
                rs=[r for r in differences if r['protocol']==protocol and r['anchor']==anchor and r['baseline']==baseline]
                wins.append(dict(protocol=protocol,anchor=anchor,baseline=baseline,windows=6,
                    precision_wins=sum(r['precision_delta_pp']>0 for r in rs),recall_wins=sum(r['recall_delta_pp']>0 for r in rs),
                    both_wins=sum(r['precision_delta_pp']>0 and r['recall_delta_pp']>0 for r in rs),
                    precision_delta_pp_mean=float(np.mean([r['precision_delta_pp'] for r in rs])),
                    recall_delta_pp_mean=float(np.mean([r['recall_delta_pp'] for r in rs]))))
    write(out/'paired_summary.csv',wins)
    import importlib.metadata as metadata
    verification=dict(complete=True,cloud_native_P_R_reproduced_24_windows=True,
        cloud_targets_match_local_catalog=True,threshold_selection_test_targets_used=False,
        region_km2=float(area.sum()),models=MODELS,test_windows=windows,
        max_equal_raw_error_km2=max(r['absolute_error_km2'] for r in matches if r['protocol']=='equal_raw_area'),
        max_equal_expanded_error_km2=max(r['absolute_error_km2'] for r in matches if r['protocol']=='equal_expanded_area'),
        max_equal_expanded_relative_error=max(r['relative_error'] for r in matches if r['protocol']=='equal_expanded_area'),
        etas_fit_training_end='2008-12-31',etas_parameter_refit=False,extra_map_smoothing=False,
        source_sha256={**sources,str(src/'six_window_maps.npz'):sha(src/'six_window_maps.npz'),
            str(root/'convlstm/training/spatial_symmetric.py'):sha(root/'convlstm/training/spatial_symmetric.py')},
        checkpoint_sha256={m:cloud['case_receipts'][m+'/test']['identity']['checkpoint_sha256'] for m in CONV},
        environment={name:metadata.version(name) for name in ['numpy','scipy','pandas','matplotlib','shapely']},
        script_sha256=sha(__file__),timestamp=datetime.now().astimezone().isoformat())
    dump(out/'verification.json',verification)
    plots_and_report(out,alarms,native_summary,equal_summary,wins,selection,verification)
    dump(out/'manifest.json',dict(complete=True,files_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name not in ['manifest.json','etas_run.log','evaluation_run.log']}))
    print('COMPLETE',out,flush=True)

def plots_and_report(out,rows,summary,equal_summary,wins,selection,verification):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'Hiragino Sans GB','axes.unicode_minus':False,'svg.fonttype':'none','pdf.fonttype':42})
    colors=['#64748b','#0072b2','#d55e00','#7b4ab0','#009e73','#d89100','#444444']
    fig,axs=plt.subplots(1,3,figsize=(14,5.5))
    for ax,f,title in zip(axs,['precision','recall','alarm_area_fraction'],['精准度（%）','召回率（%）','原始报警面积占比（%）']):
        vals=[100*next(r for r in summary if r['model']==m)[f+'_mean'] for m in MODELS]
        ax.barh(range(7),vals,color=colors)
        ax.set_yticks(range(7),[LABEL[m] for m in MODELS] if ax is axs[0] else [])
        ax.invert_yaxis();ax.set_title(title);ax.spines[['top','right']].set_visible(False)
        for i,v in enumerate(vals):ax.text(v+.5,i,f'{v:.2f}',va='center',fontsize=9)
        ax.set_xlim(0,max(vals)*1.22)
    fig.suptitle('六个测试窗口：不限制报警面积，d=3；阈值在验证集固定')
    fig.tight_layout();fig.savefig(out/'不限制面积比较.png',dpi=180,bbox_inches='tight');plt.close(fig)
    fig,axs=plt.subplots(2,3,figsize=(13,8),sharex=True,sharey=True)
    for i,ax in enumerate(axs.flat):
        rs=[r for r in rows if int(r['row'])==ROWS[i]]
        for m,color in zip(MODELS,colors):
            r=next(a for a in rs if a['model']==m)
            ax.scatter(100*r['recall'],100*r['precision'],s=55,color=color,label=LABEL[m])
        ax.set_title(rs[0]['target_start']+'—'+rs[0]['target_end']);ax.grid(alpha=.2)
        ax.set_xlabel('召回率（%）');ax.set_ylabel('精准度（%）')
    axs[0,0].legend(fontsize=8,loc='best');fig.suptitle('不限制报警面积：逐窗口精准度与召回率')
    fig.tight_layout();fig.savefig(out/'六窗口精准度召回率.png',dpi=180,bbox_inches='tight');plt.close(fig)
    def table(rs):
        text='| 模型 | 精准度% | 召回率% | 同格精准度% | 同格召回率% | 原始面积% | 拓展面积% |\n|---|---:|---:|---:|---:|---:|---:|\n'
        for r in rs:
            text+='| '+LABEL[r['model']]+' | '+' | '.join(f'{100*r[f+"_mean"]:.2f}' for f in ['precision','recall','precision_exact','recall_exact','alarm_area_fraction','expanded_area_fraction'])+' |\n'
        return text
    report='# 七个模型在六个测试窗口的比较\n\n'
    report+='本轮已在本地完成。四个ConvLSTM与PI复用已有预测；ETAS补算每窗5000次、共30000次模拟；新增历史活动平滑基线。未训练ConvLSTM，未修改ETAS拟合参数。仅计算精准度、召回率及其同格版本。\n\n'
    report+='## 主要发现\n\n'
    report+='1. **当前结果尚未建立增加通道ConvLSTM相对ETAS的精准度优势。** 不限制面积时，ETAS与增加通道ConvLSTM的平均召回率都约61%，ETAS精准度72.32%，三个增加通道模型为63.92%—67.28%。ETAS原始面积更大，但拓展后覆盖范围相近。\n'
    report+='2. **相同原始面积下存在明显取舍。** 增加通道ConvLSTM比ETAS多召回约14—17个百分点，但精准度低约11—14个百分点；ETAS的拓展后覆盖面积只有约4.5%—5.1%，ConvLSTM约8.6%—9.1%。因此这项召回优势需要结合更大拓展覆盖范围解释。\n'
    report+='3. **匹配拓展后覆盖面积后，ETAS精准度仍较高。** ConvLSTM的平均召回优势缩小到约0.65—1.91个百分点，精准度低约6.13—9.97个百分点。不能将其表述为同时提高精准度和召回率。\n'
    report+='4. **相对PI和简单历史活动基线，增加通道模型具有一定竞争力。** 匹配拓展面积时，三个增加通道模型的平均精准度与召回率都高于这两项基线；相对PI六窗两项均领先，相对历史活动基线的逐窗精准度优势并非全部出现。\n'
    report+='5. **完全同格结果未显示增加通道ConvLSTM领先ETAS。** 原生阈值下ETAS同格召回率36.76%，增加通道模型为10.00%—13.34%；匹配原始面积后，ETAS同格精准度和召回率的六窗均值也均高于三个增加通道模型。以上结论仅针对本轮冻结配置与目标时段。\n\n'
    report+='## 不限制报警面积\n\n'+table(summary)+'\n'
    report+='ConvLSTM阈值100；PI使用原始正值热点。ETAS和历史活动基线在六个验证窗口上，从预先固定阈值中选取“平均召回率≥60%且平均精准度最高”的阈值，再固定用于测试。此组不匹配报警面积。\n\n'
    for m,s in selection['selections'].items():
        report+=f'- {LABEL[m]}：阈值{s["threshold"]}；验证集精准度{100*s["validation_mean_precision"]:.2f}%，召回率{100*s["validation_mean_recall"]:.2f}%。\n'
    report+='\n## 相同报警面积\n\n每个ConvLSTM的阈值100区域分别作为面积参照。所有模型按各自原始分数排序；固定并列顺序种子2901。先匹配原始实际面积，再单独匹配d=3拓展后的实际面积。后一种允许原始面积不同。所有百分比均以整个日本研究矩形实际面积为分母，包含陆地和海域。\n\n'
    for protocol,title in [('equal_raw_area','相同原始报警面积'),('equal_expanded_area','相同拓展后面积')]:
        report+='### '+title+'\n\n'
        for anchor in CONV[1:]:
            report+='**以'+LABEL[anchor]+'为面积参照：**\n\n'
            rs=[r for r in equal_summary if r['protocol']==protocol and r['anchor']==anchor and r['model'] in [anchor,'PI','ETAS','SSS']]
            report+=table(rs)+'\n'
            for w in wins:
                if w['protocol']==protocol and w['anchor']==anchor:
                    report+=f'- 相对{LABEL[w["baseline"]]}：平均精准度差{w["precision_delta_pp_mean"]:+.2f}个百分点，召回率差{w["recall_delta_pp_mean"]:+.2f}个百分点；精准度领先{w["precision_wins"]}/6窗、召回率领先{w["recall_wins"]}/6窗。\n'
            report+='\n'
    report+='## 六个窗口的不限制面积结果\n\n'
    for row in ROWS:
        rs=[r for r in rows if int(r['row'])==row]
        report+='### '+rs[0]['target_start']+'—'+rs[0]['target_end']+'\n\n'
        report+=table([{**r,**{k+'_mean':v for k,v in r.items() if isinstance(v,(float,int))}} for r in rs])+'\n'
    report+='## 解释边界与核验\n\n'
    report+='- d=3保留当前分纬度带对称网格规则，不是恒定公里半径。精准度统计报警格，召回率统计真实有震格，两者分子不同；完全同格版本共享命中格。\n'
    report+='- Mw≥4指真实事件筛选；一个网格多次有震只算一个有震格。改变预测阈值不改变真实目标。\n'
    report+='- 历史活动基线仅使用起点之前4050天的事件次数，采用固定25km高斯核；将450天期望次数通过1-exp(-lambda)转换为Poisson占用分数。25km属于这个基线模型本身；没有对ConvLSTM、PI、ETAS预测图增加平滑。\n'
    report+='- ETAS使用2008年底之前拟合并冻结的参数，逐窗口更新4050天已观测触发历史；保留全世代模拟。分支比接近0.99、目录完整性和边界处理仍有限制，是现有试验版基线，不是最优ETAS的上限。\n'
    report+='- 六个目标窗口不重叠，历史和地震活动仍可能相关。均值为六窗等权平均，标准差为窗口总体标准差，不是置信区间。测试集已有通道比较和敏感性结果，本轮不称全新盲测。\n'
    report+='- 相同面积曲线中的ConvLSTM可包含低于100的预测分数；它是排序比较，与阈值100的原生报警方式分开解释。测试阈值表仅展示结果，不据此重选主阈值。\n'
    report+=f'- 已逐窗复现24个ConvLSTM原生P/R，并核对真实目标。研究区总面积{verification["region_km2"]:,.3f}km²。原始面积匹配最大误差{verification["max_equal_raw_error_km2"]:.3f}km²；拓展面积匹配最大误差{verification["max_equal_expanded_error_km2"]:.3f}km²（最大相对误差{100*verification["max_equal_expanded_relative_error"]:.3f}%）。\n'
    report+='\n## 文件与运行\n\n代码和全部结果就在本文件所在目录。主要结果为unrestricted_summary.csv、unrestricted_by_window.csv、equal_area_summary.csv、equal_area_by_window.csv、paired_summary.csv；面积曲线见area_curves_*.csv，全部阈值见test_threshold_results.csv。comparison_maps.npz保存原始分数、原生报警区和目标。etas_test保存六窗5000次模拟的概率图、事件次数和诊断。\n\n本轮无需上传云平台，无需云端运行或下载。复现命令见README_运行与文件.md。\n'
    (out/'结果报告.md').write_text(report)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=ROOT)
    ap.add_argument('--out',type=Path,default=HERE)
    ap.add_argument('--prepare',action='store_true')
    args=ap.parse_args()
    if args.prepare:prepare(args.root,args.out)
    else:complete(args.root,args.out)

if __name__=='__main__':main()
