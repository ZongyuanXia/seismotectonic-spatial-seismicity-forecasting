"""Freeze baseline thresholds by validation recall, then evaluate test P/R.

ConvLSTM remains gray>=100. No area matching, no ConvLSTM ranking,
no probability interpretation of magnitude codes, no new simulations.
"""
from pathlib import Path
from datetime import datetime
import argparse, csv, hashlib, json, sys
import numpy as np
import evaluate_baselines as core

HERE=Path(__file__).resolve().parent
BASELINES=['PI','ETAS','SSS']
ANCHORS=core.CONV[1:]
PREFIX='recall_matched_'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return core.read(p)
def write(p,rs):core.write(p,rs)
def dump(p,v):core.dump(p,v)

def validate_inputs(root,source):
    """Verify the already completed input delivery before reusing predictions."""
    manifest=json.loads((source/'manifest.json').read_text())
    assert manifest['complete']
    for name,h in manifest['files_sha256'].items():
        assert sha(source/name)==h,name
    src=root/'convlstm/runs/compact_core_summary'
    cloud=json.loads((src/'summary_manifest.json').read_text())
    assert cloud['analysis_complete']
    for name,h in cloud['files_sha256'].items():assert sha(src/name)==h,name
    return src,cloud

def load_score(root,source,model,split,row,origin,sources):
    if model=='PI':
        p=root/f'baseline/results/pi_japan_4050d_change450d_positive_v1/{split}/maps/sample_{row:05d}_{origin}.npz'
        key='pi_map';datekey='target_start'
    elif model=='ETAS':
        p=(root/f'baseline/etas/MC_sensitivity_1000_5000_20261030/results/window_{row:05d}/ETAS_N5000.npz'
            if split=='validation' else source/f'etas_test/ETAS_{row:05d}.npz')
        key='probability';datekey='origin'
        if split=='validation':
            meta=json.loads((p.parent/'complete.json').read_text())
            assert meta['complete'] and meta['output_sha256'][p.name]==sha(p)
    else:
        p=source/f'SSS_{split}_{row:05d}.npz';key='probability';datekey='origin'
    with np.load(p,allow_pickle=False) as z:
        assert str(z[datekey].item())==origin
        if model=='ETAS':assert int(z['simulations'])==5000
        s=z[key].astype(np.float64)
    assert s.shape==core.SHAPE and np.isfinite(s).all() and (s>=0).all()
    sources[str(p)]=sha(p)
    return s

def attainable_recall(scores,targets,expander):
    """Enumerate every attainable six-window mean R for a positive threshold.

    For a symmetric neighborhood, a true cell is covered iff the maximum
    original score among its neighbors is >= threshold. Therefore the mean
    recall can change only at the observed maxima at true occupied cells.
    This is exact threshold enumeration, not a numerical grid search.
    Entire score ties are included. Among identical-recall thresholds, the
    largest threshold is retained; validation P is never optimized here.
    """
    activations=[expander(s,d=3)[t] for s,t in zip(scores,targets)]
    assert all(len(x)>0 for x in activations)
    positive=np.unique(np.concatenate([x[x>0] for x in activations]))
    empty=float(np.nextafter(max(float(s.max()) for s in scores),np.inf))
    thresholds=np.r_[positive,empty]
    recall=np.array([(len(x)-np.searchsorted(np.sort(x),thresholds,side='left'))/len(x)
        for x in activations])
    means=recall.mean(axis=0)
    assert (np.diff(means)<=1e-15).all()
    return thresholds,recall,means

def select(thresholds,recalls,reference):
    errors=np.abs(recalls-reference);minimum=float(errors.min())
    tied=np.flatnonzero(np.isclose(errors,minimum,rtol=0,atol=1e-15))
    # If equally close on either side, avoid choosing a lower recall target.
    k=max(tied,key=lambda j:(recalls[j]>=reference-1e-15,thresholds[j]))
    return int(k)

def freeze_validation(root,source,src,out,area,sources):
    from spatial_symmetric import symmetric_expand
    data=core.Reader(root/'datasets/cl10_mapping_v2','validation',(0,),cache_size=0)
    rows=read(src/'sensitivity_nonoverlap.csv')
    anchors={};anchor_rows=[]
    for anchor in ANCHORS:
        rs=[r for r in rows if r['model']==anchor and r['split']=='validation' and r['d']=='3' and r['pred_threshold']=='100']
        rs=sorted(rs,key=lambda r:int(r['row']))
        assert [int(r['row']) for r in rs]==core.ROWS and all(r['seed']=='run_03' for r in rs)
        reference=float(np.mean([float(r['recall']) for r in rs]))
        frozen=json.loads((src/'core_completion_frozen.json').read_text())['anchors'][anchor]['validation']['nonoverlap']
        assert abs(reference-frozen['recall'])<1e-12
        anchors[anchor]=reference
        for r in rs:
            anchor_rows.append(dict(anchor=anchor,row=int(r['row']),target_start=r['target_start'],
                target_end=r['target_end'],precision=float(r['precision']),recall=float(r['recall'])))
    targets=[core.target(data,row) for row in core.ROWS]
    scoremaps={m:[load_score(root,source,m,'validation',row,data.rows[row]['target_start'],sources)
        for row in core.ROWS] for m in BASELINES}
    selected=[];search=[];details=[]
    for model in BASELINES:
        ts,rs,means=attainable_recall(scoremaps[model],targets,symmetric_expand)
        for j,t in enumerate(ts):
            search.append(dict(model=model,threshold=float(t),mean_recall=float(means[j]),
                **{f'recall_row_{row:05d}':float(rs[k,j]) for k,row in enumerate(core.ROWS)}))
        for anchor,reference in anchors.items():
            k=select(ts,means,reference);threshold=float(ts[k])
            ss=[]
            for i,row in enumerate(core.ROWS):
                alarm=scoremaps[model][i]>=threshold
                st=core.stats(alarm,targets[i],area)
                assert abs(st['recall']-rs[i,k])<1e-12
                ss.append(st)
                details.append(dict(anchor=anchor,model=model,row=row,
                    target_start=data.rows[row]['target_start'],target_end=data.rows[row]['target_end'],
                    threshold=threshold,**st))
            delta=100*(float(means[k])-reference)
            selected.append(dict(anchor=anchor,model=model,threshold=threshold,
                alarm_operator='score >= threshold (float64 comparison; all score ties included)',
                reference_mean_recall=reference,matched_mean_recall=float(means[k]),
                recall_difference_pp=delta,absolute_recall_difference_pp=abs(delta),
                attainable_min_recall=float(means.min()),attainable_max_recall=float(means.max()),
                reference_attainable_range=bool(means.min()<=reference<=means.max()),
                candidate_count=len(ts),validation_precision_mean=float(np.mean([x['precision'] for x in ss])),
                validation_alarm_fraction_mean=float(np.mean([x['alarm_area_fraction'] for x in ss])),
                validation_expanded_fraction_mean=float(np.mean([x['expanded_area_fraction'] for x in ss]))))
    sources[str(src/'sensitivity_nonoverlap.csv')]=sha(src/'sensitivity_nonoverlap.csv')
    sources[str(src/'core_completion_frozen.json')]=sha(src/'core_completion_frozen.json')
    write(out/(PREFIX+'validation_reference.csv'),anchor_rows)
    write(out/(PREFIX+'threshold_search.csv'),search)
    write(out/(PREFIX+'validation_by_window.csv'),details)
    write(out/(PREFIX+'thresholds.csv'),[{k:v for k,v in x.items() if k!='alarm_operator'} for x in selected])
    frozen=dict(complete=True,version='validation_recall_matching_v1',
        matching_protocol='symmetric_band_union_v1 d=3; equal-weight six-window mean recall',
        criterion='minimize absolute validation mean recall difference; equal-distance ties prefer recall not below reference, then larger threshold',
        validation_precision_optimized=False,test_targets_used_for_selection=False,
        area_constraint=False,convlstm_rule='original normalized prediction * 255 >= 100',
        baseline_rule='score >= positive threshold; all score ties included',
        exact_evaluation='same fixed alarms; no separate threshold optimization for exact recall',
        selections=selected,validation_source_sha256=dict(sources),
        script_sha256=sha(__file__),frozen_at=datetime.now().astimezone().isoformat())
    dump(out/(PREFIX+'thresholds.json'),frozen)
    print('FROZEN validation thresholds BEFORE loading test targets',flush=True)
    for s in selected:
        print(s['anchor'],s['model'],f'threshold={s["threshold"]:.17g}',f'Rgap={s["recall_difference_pp"]:+.6f}pp',flush=True)
    return frozen

def evaluate_test(root,source,src,out,area,frozen,sources):
    threshold_hash=sha(out/(PREFIX+'thresholds.json'))
    # Test targets are first read after the thresholds have been written.
    data=core.Reader(root/'datasets/cl10_mapping_v2','test',(0,),cache_size=0)
    grid=json.loads((src/'grid_and_protocol.json').read_text());windows=grid['test_target_dates']
    with np.load(src/'six_window_maps.npz',allow_pickle=False) as z:cm={k:z[k].copy() for k in z.files}
    sources[str(src/'six_window_maps.npz')]=sha(src/'six_window_maps.npz')
    old=read(source/'unrestricted_by_window.csv')
    selections={(s['anchor'],s['model']):s['threshold'] for s in frozen['selections']}
    output=[];alarms={};replications=0
    for i,w in enumerate(windows):
        row=int(w['row']);t=core.target(data,row)
        assert np.array_equal(t,cm['targets'][i]*255>=100)
        scores={m:load_score(root,source,m,'test',row,w['target_start'],sources) for m in BASELINES}
        for anchor in ANCHORS:
            for model in [anchor,'MAG',*BASELINES]:
                if model in core.CONV:
                    a=cm[model+'_predictions'][i]*255>=100;threshold=100.;unit='gray'
                else:
                    threshold=selections[anchor,model];unit='own score'
                    a=scores[model]>=threshold
                st=core.stats(a,t,area)
                if model in core.CONV:
                    ref=next(r for r in old if r['model']==model and int(r['row'])==row)
                    for metric in ['precision','recall','precision_exact','recall_exact','alarm_km2','expanded_alarm_km2']:
                        assert np.isclose(st[metric],float(ref[metric]),rtol=1e-12,atol=1e-12)
                    replications+=1
                output.append(dict(anchor=anchor,model=model,label=core.LABEL[model],**w,
                    threshold=threshold,threshold_unit=unit,**st))
                alarms[f'{anchor}__{model}__row{row:05d}']=a
    assert threshold_hash==sha(out/(PREFIX+'thresholds.json'))
    write(out/(PREFIX+'test_by_window.csv'),output)
    summary=core.aggregate(output,['anchor','model']);write(out/(PREFIX+'test_summary.csv'),summary)
    differences=[];paired=[]
    for anchor in ANCHORS:
        for baseline in BASELINES:
            rows=[]
            for row in core.ROWS:
                a=next(r for r in output if r['anchor']==anchor and r['model']==anchor and int(r['row'])==row)
                b=next(r for r in output if r['anchor']==anchor and r['model']==baseline and int(r['row'])==row)
                d=dict(anchor=anchor,baseline=baseline,row=row)
                for f in ['precision','recall','precision_exact','recall_exact','alarm_area_fraction','expanded_area_fraction']:
                    d[f+'_difference_pp']=100*(a[f]-b[f]) if a[f] is not None and b[f] is not None else None
                rows.append(d);differences.append(d)
            p=dict(anchor=anchor,baseline=baseline,windows=6)
            for f in ['precision','recall','precision_exact','recall_exact','alarm_area_fraction','expanded_area_fraction']:
                v=[r[f+'_difference_pp'] for r in rows if r[f+'_difference_pp'] is not None]
                p[f+'_difference_pp_mean']=float(np.mean(v)) if v else None
                if f in ['precision','recall','precision_exact','recall_exact']:
                    p[f+'_wins']=sum(x>0 for x in v)
            paired.append(p)
    write(out/(PREFIX+'paired_by_window.csv'),differences)
    write(out/(PREFIX+'paired_summary.csv'),paired)
    np.savez_compressed(out/(PREFIX+'test_alarms.npz'),**alarms)
    return output,summary,paired,dict(test_targets_match=True,convlstm_repeated_rows_reproduced=replications,
        unique_convlstm_test_windows_reproduced=24,thresholds_unchanged_after_testing=True,
        threshold_file_sha256=threshold_hash,test_source_sha256=dict(sources))

def make_report(out,frozen,rows,summary,paired):
    def table(rs):
        text='| 模型 | 精准度% | 召回率% | 同格精准度% | 同格召回率% | 原始面积% | 拓展面积% |\n|---|---:|---:|---:|---:|---:|---:|\n'
        for r in rs:
            values=[r[k+'_mean'] for k in ['precision','recall','precision_exact','recall_exact','alarm_area_fraction','expanded_area_fraction']]
            text+='| '+core.LABEL[r['model']]+' | '+' | '.join('未定义' if v is None else f'{100*v:.2f}' for v in values)+' |\n'
        return text
    report='# 按验证集召回率匹配的基线比较\n\n'
    report+='本轮已在本地重新评估，只复用已有预测图，不重新训练或模拟。以三个融合ConvLSTM分别作参照；ConvLSTM固定灰度≥100，不作震级排序，不解释为发生概率；报警面积不作相等约束。只计算精准度和召回率。\n\n'
    report+='## 主要结果\n\n'
    report+='- 九组基线与对应融合模型的验证集平均召回率差，最大为0.72个百分点；ETAS最大差0.20个百分点，历史活动基线最大差0.05个百分点。PI存在并列分数造成的不可分割变化，保留实际匹配误差。\n'
    report+='- 测试集出现召回率分离：融合模型约60.8%—61.0%，ETAS约46.1%—48.0%，PI约70.0%—70.3%，历史活动基线约47.1%—48.2%。这不再是测试集召回率相同的比较。\n'
    report+='- 相对ETAS，三个融合模型的精准度均在六个窗口中较低，召回率均在六个窗口中较高。均值上精准度低约10.0—13.5个百分点，召回率高约13.0—14.9个百分点，形成明显取舍，不能判定全面优于ETAS。\n'
    report+='- 相对PI，三个融合模型在六个窗口均提高精准度，同时在六个窗口均降低召回率；不能省略约9个百分点的平均召回损失。\n'
    report+='- 相对历史活动基线，深度组与机制组的平均精准度、召回率均提高；方向组的平均召回率提高，但平均精准度低1.30个百分点。逐窗是否同时领先见差值表。\n'
    report+='- 本轮改变的是报警阈值选择规则；ETAS的拟合、预测与模拟次数均未改变。验证集选出的ETAS概率阈值约为2.76%—2.94%，没有沿用旧的2%阈值，也没有按测试召回率重新调整。\n\n'
    report+='## 阈值选择与冻结\n\n'
    report+='匹配的是六个验证窗口的d=3平均召回率，不再统一使用60%下限，也不再根据验证集精准度最大化选阈值。所有基线使用分数≥正阈值报警，并列分数全部纳入。遍历能改变真实格检出情况的全部分数位置，得到每一个可实现的平均召回率；以绝对召回率差最小为准，距离相同时优先不低于参照，再取较高阈值。\n\n'
    report+='同一组基线阈值固定用于全部六个测试窗口。完全同格评价使用相同报警区，不单独重选阈值，故其验证集同格召回率并未被匹配。\n\n'
    report+='| 参照模型 | 基线 | 固定阈值 | 参照验证R% | 基线验证R% | 差值（百分点） |\n|---|---|---:|---:|---:|---:|\n'
    for s in frozen['selections']:
        report+=f'| {core.LABEL[s["anchor"]]} | {core.LABEL[s["model"]]} | {s["threshold"]:.17g} | {100*s["reference_mean_recall"]:.4f} | {100*s["matched_mean_recall"]:.4f} | {s["recall_difference_pp"]:+.4f} |\n'
    report+='\n完整精度的阈值以recall_matched_thresholds.json为准；ETAS仍为5000次模拟给出的网格占用概率，概率分辨率为1/5000，不能通过加密阈值制造更细的预测概率。\n\n'
    report+='## 六个测试窗口的等权均值\n\n'
    for anchor in ANCHORS:
        report+='### 以'+core.LABEL[anchor]+'为参照\n\n'
        report+=table([r for r in summary if r['anchor']==anchor])+'\n'
        for p in paired:
            if p['anchor']==anchor:
                report+=f'- 融合模型减{core.LABEL[p["baseline"]]}：平均精准度{p["precision_difference_pp_mean"]:+.2f}个百分点，召回率{p["recall_difference_pp_mean"]:+.2f}个百分点；精准度领先{p["precision_wins"]}/6窗、召回率领先{p["recall_wins"]}/6窗。\n'
        report+='\n'
    report+='## 六个测试窗口的逐窗结果\n\n'
    for anchor in ANCHORS:
        report+='### '+core.LABEL[anchor]+'组\n\n'
        for row in core.ROWS:
            rs=[r for r in rows if r['anchor']==anchor and int(r['row'])==row]
            report+='**'+rs[0]['target_start']+'—'+rs[0]['target_end']+'**\n\n'
            report+=table([{**r,**{k+'_mean':v for k,v in r.items() if isinstance(v,(int,float))}} for r in rs])+'\n'
    report+='## 解释与限制\n\n'
    report+='- 只匹配验证集召回率；测试集出现召回率分离时如实报告，不再修改阈值。精准度高但召回率明显低，不能单凭精准度判为全面更好。\n'
    report+='- d=3使用原有对称分纬度带网格拓展规则，精准度统计报警格，召回率统计真实格；两者命中数不同。完全同格两项共享同格命中数。\n'
    report+='- Mw≥4仅定义真实有震格。报警面积按整个研究矩形实际平方公里计算，包括陆地和海域。逐窗重复地震只计一个有震格。\n'
    report+='- ETAS沿用2008年底之前拟合的参数与5000次预测。历史活动基线保持原有4050天历史、固定25km核及Poisson占用分数。没有新增额外的输出图平滑。\n'
    report+='- 六个测试目标不重叠，但历史和地震活动仍有关联。六窗均值不是六个完全独立试验的统计证明；标准差是窗口总体标准差，不是置信区间。测试集已在此前分析中查看，本轮不称为全新盲测。\n'
    report+='- 阈值搜索中没有使用面积预算、ConvLSTM排序或测试精准度，没有根据测试结果挑选主阈值。\n\n'
    report+='## 文件和复现\n\n'
    report+='本目录中所有recall_matched_开头的文件属于本轮，原有结果不覆盖。代码为reevaluate_recall_matched.py。主表为recall_matched_test_summary.csv，逐窗表为recall_matched_test_by_window.csv，差值见recall_matched_paired_summary.csv及recall_matched_paired_by_window.csv。阈值搜索与冻结分别保存为recall_matched_threshold_search.csv及recall_matched_thresholds.json；本轮报警掩膜为recall_matched_test_alarms.npz。\n\n'
    report+='本轮无需上传云端、云端运行或从云端下载结果。原始预测输入仍在项目归档目录和本地既有baseline_test_six_20261103目录；复现所需路径、命令和依赖见recall_matched_README.md。\n'
    (out/(PREFIX+'结果报告.md')).write_text(report)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'Hiragino Sans GB','font.weight':300,'axes.unicode_minus':False})
    fig,axs=plt.subplots(1,3,figsize=(14,5))
    colors=['#0072b2','#64748b','#009e73','#d89100','#444444']
    for ax,anchor in zip(axs,ANCHORS):
        rs=[r for r in summary if r['anchor']==anchor]
        for i,r in enumerate(rs):
            ax.scatter(100*r['recall_mean'],100*r['precision_mean'],s=90,color=colors[i],label=core.LABEL[r['model']])
        ax.set_title(core.LABEL[anchor]);ax.set_xlabel('测试集平均召回率（%）');ax.set_ylabel('测试集平均精准度（%）')
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('验证集匹配召回率，固定阈值后评估六个测试窗口（d=3；不限制面积）')
    fig.tight_layout();fig.savefig(out/(PREFIX+'比较.png'),dpi=180,bbox_inches='tight');plt.close(fig)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=core.ROOT)
    ap.add_argument('--source',type=Path,default=HERE)
    ap.add_argument('--out',type=Path,default=HERE)
    args=ap.parse_args();root=args.root.resolve();source=args.source.resolve();out=args.out.resolve()
    out.mkdir(parents=True,exist_ok=True);core.setup(root)
    src,cloud=validate_inputs(root,source)
    region=json.loads((src/'grid_and_protocol.json').read_text())['region'];area=core.areas(region)
    sources={}
    frozen=freeze_validation(root,source,src,out,area,sources)
    output,summary,paired,checks=evaluate_test(root,source,src,out,area,frozen,{})
    make_report(out,frozen,output,summary,paired)
    checks.update(complete=True,script_sha256=sha(__file__),
        scoring_helper_sha256=sha(Path(core.__file__)),
        spatial_module_sha256=sha(root/'convlstm/training/spatial_symmetric.py'),
        exact_threshold_enumeration_verified_by_binary_scoring=True,
        area_constraint=False,convlstm_outputs_ranked=False,convlstm_outputs_used_as_probabilities=False,
        new_etas_simulations=0,main_threshold_matching='validation d=3 recall',
        max_validation_recall_gap_pp=max(s['absolute_recall_difference_pp'] for s in frozen['selections']),
        checkpoints_sha256={m:cloud['case_receipts'][m+'/test']['identity']['checkpoint_sha256'] for m in core.CONV})
    dump(out/(PREFIX+'verification.json'),checks)
    dump(out/(PREFIX+'manifest.json'),dict(complete=True,files_sha256={p.name:sha(p) for p in out.iterdir()
        if p.is_file() and p.name.startswith(PREFIX) and p.name!=PREFIX+'manifest.json'},
        script_sha256=sha(__file__)))
    print('COMPLETE',out/(PREFIX+'结果报告.md'),flush=True)

if __name__=='__main__':main()
