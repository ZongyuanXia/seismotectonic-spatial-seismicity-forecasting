"""Verify the completed single forecast and create figures and a Chinese report."""
from pathlib import Path
import argparse,csv,json,pickle,sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import predict_etas_trial as p

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--out',type=Path,default=p.BASE/'results/validation_00000_20090101_v2')
    args=ap.parse_args();out=args.out
    fm=json.loads((out/'fit_manifest.json').read_text());pm=json.loads((out/'prediction_manifest.json').read_text())
    dg=json.loads((out/'diagnostics.json').read_text())
    with (out/'evaluation.csv').open(encoding='utf-8-sig',newline='') as f:table=list(csv.DictReader(f))
    with (out/'prediction.npz').open('rb') as f:
        z=np.load(f);prob=z['probability'].copy();mean=z['expected_event_count'].copy()
        presence=z['presence_count'].copy();counts=z['event_count'].copy();half=z['half_presence_count'].copy();n=int(z['simulations'])
    assert p.sha(out/'prediction.npz')==pm['prediction_sha256']
    assert n==pm['simulations'] and np.all(presence<=counts) and np.array_equal(half.sum(axis=0),presence)
    np.testing.assert_array_equal(prob,(presence/n).astype(np.float32))
    assert np.isfinite(prob).all() and np.all((prob>=0)&(prob<=1))
    with (out/'fit.pkl').open('rb') as f:state=pickle.load(f)
    assert state['catalog'].time.max()<pd.Timestamp('2009-01-01')
    # Full simulator reproducibility for two representative catalog seeds.
    p.initialize(out/'fit.pkl',p.BASE/'region_meta.json')
    with (out/'simulation_counts.csv').open(encoding='utf-8-sig',newline='') as f:simrecords=list(csv.DictReader(f))
    for i in (0,n-1):
        a=p.generate_one(i);b=p.generate_one(i)
        np.testing.assert_array_equal(a[1],b[1]);np.testing.assert_array_equal(a[2],b[2]);assert a[3]==b[3]
        assert all(('' if v is None else str(v))==simrecords[i][k] for k,v in a[3].items())
    # Rescore all exported alarm rules from frozen maps.
    region=json.loads((p.BASE/'region_meta.json').read_text());area=p.areas(region)
    training=args.root/'convlstm/training'
    if not training.exists():training=args.root/'training'
    sys.path.insert(0,str(training));import cl10_data as reader;import evaluate_cl10_symmetric as scorer
    data=args.root/'datasets/cl10_mapping_v2'
    if not data.exists():data=args.root/'cl10_mapping_v2'
    d=reader.CL10Data(data,'validation',(0,),cache_size=0);keys,gray=d.target_keys(d.rows[0]);target=np.zeros((340,200),np.float32);target.flat[keys]=gray/255
    ref=args.root/'baseline/results/pi_japan_4050d_change450d_positive_v1/validation'
    if not ref.exists():ref=p.BASE/'reference/pi_validation'
    with np.load(ref/'maps/sample_00000_2009-01-01.npz') as z:pi=z['pi_map'].copy()
    for row in table:
        if row['setting']=='probability_threshold':alarm=prob>=float(row['value'])
        elif row['setting']=='original_PI_positive':alarm=pi>0
        else:alarm=p.ranked_alarm(prob if row['model']=='ETAS' else pi,area,float(row['value']))
        ae=scorer.archived_expand(alarm.astype(np.float32));te=scorer.archived_expand(target)
        for key,stats in [('precision',scorer.counts(alarm,te)),('recall',scorer.counts(ae,target)),('CSI',scorer.counts(ae,te))]:
            assert np.isclose(float(row[key]),stats[key],rtol=1e-12,atol=1e-12)
        assert abs(float(row['alarm_km2'])-area[alarm].sum())<1e-6
        if row['setting']=='equal_actual_area':assert abs(area[alarm].sum()-float(row['value'])*area.sum())<=area.max()
    b=region['effective_bounds'];extent=[b['lon_min'],b['lon_max'],b['lat_min'],b['lat_max']]
    centers_lat=(np.arange(340)+region['crop_box_closed']['y_min']+.5)/3840*180-90
    centers_lon=(np.arange(200)+region['crop_box_closed']['x_min']+.5)/3840*360-180
    ty,tx=np.where(target>0);lon=centers_lon[tx];lat=centers_lat[ty]
    fig,axes=plt.subplots(2,2,figsize=(11,10),layout='constrained')
    im=axes[0,0].imshow(np.where(prob>0,prob,np.nan),origin='lower',extent=extent,
        norm=LogNorm(vmin=1/n,vmax=float(prob.max())),cmap='magma_r',interpolation='nearest')
    fig.colorbar(im,ax=axes[0,0],label='P(at least one M>=4 event / grid cell)')
    axes[0,0].set_title(f'ETAS forecast | {n:,} simulations')
    axes[0,1].scatter(lon,lat,s=7,c='black');axes[0,1].set_title(f'Observed target | {len(keys)} occupied cells')
    for ax,label,score,color in [(axes[1,0],'ETAS',prob,'Reds'),(axes[1,1],'PI',pi,'Blues')]:
        alarm=p.ranked_alarm(score,area,.01)
        ax.imshow(np.where(alarm,1.,np.nan),origin='lower',extent=extent,vmin=0,vmax=1,cmap=color,interpolation='nearest')
        ax.scatter(lon,lat,s=6,c='black',alpha=.7,label='Observed occupied cells')
        row=next(x for x in table if x['model']==label and x['setting']=='equal_actual_area' and float(x['value'])==.01)
        ax.set_title(f'{label} | 1% area | P={100*float(row["precision"]):.1f}%, R={100*float(row["recall"]):.1f}%')
        ax.legend(loc='lower left',fontsize=7)
    for ax in axes.flat:
        ax.set_xlim(extent[:2]);ax.set_ylim(extent[2:]);ax.set_xlabel('Longitude (deg E)');ax.set_ylabel('Latitude (deg N)')
        ax.set_aspect(1/np.cos(np.radians(38)));ax.grid(alpha=.15)
    fig.suptitle('Single 450-day forecast: 2009-01-01 to 2010-03-26 | Pilot, not a full baseline',fontsize=12)
    fig.savefig(out/'ETAS_single_forecast.png',dpi=180);fig.savefig(out/'ETAS_single_forecast.pdf');plt.close(fig)
    lines=['# ETAS单窗口试预测结果','','## 预测设置','','- 预测起点：2009-01-01；目标期：2009-01-01—2010-03-26，共450天，验证集第0个样本。',
        f'- 时空ETAS：作者实现，空间变化背景率、共享触发参数。拟合使用2009年之前的记录，主拟合目标{fm["fit_target_count"]}条，历史来源{fm["fit_source_count"]}条。',
        '- 拟合主时间段：1989-02-02至2008-12-31；此前记录作为初始化历史。预测触发历史仅使用起点前4050天。',
        f'- 模拟次数：{n}；每个模拟独立种子，包含背景地震及未来逐代余震。4级门槛与当前数据一致，不使用目标期真实地震更新这次450天预测。',
        '- 输出为每格450天内至少一次目标地震的模拟比例；背景位置按推断权重采样，采用25公里Gaussian平滑。完整性、边界及平滑尺度未完成正式验证。','',
        '## 固定概率阈值结果','','以下阈值在计算评分前固定，全部保留，未根据本窗口表现挑选。P、R使用当前对称邻域口径，CSI为双方扩张后的交并比，d=3。','',
        '| 概率阈值 | 报警面积占比 | Precision | Recall | CSI |','|---|---:|---:|---:|---:|']
    for row in table:
        if row['setting']=='probability_threshold':lines.append(f'| {100*float(row["value"]):.1f}% | {100*float(row["alarm_area_fraction"]):.2f}% | {100*float(row["precision"]):.2f}% | {100*float(row["recall"]):.2f}% | {100*float(row["CSI"]):.2f}% |')
    lines+=['','## 相同实际报警面积对照','','| 指定面积 | 模型 | Precision | Recall | CSI |','|---|---|---:|---:|---:|']
    for row in table:
        if row['setting']=='equal_actual_area':lines.append(f'| {100*float(row["value"]):.1f}% | {row["model"]} | {100*float(row["precision"]):.2f}% | {100*float(row["recall"]):.2f}% | {100*float(row["CSI"]):.2f}% |')
    lines+=['','面积按球面网格实际面积计算，离散网格导致的预算差异小于一个网格面积。完整四种匹配组合、未扩张指标和扩张面积见CSV。','',
        '## 模拟与拟合诊断','',f'- 模拟目标事件数：均值{dg["events_mean"]:.1f}，中位数{dg["events_median"]:.1f}，5%—95%范围{dg["events_p05_p95"]}；实际目录为{dg["observed_events"]}次、{dg["observed_occupied_cells"]}个有震格。',
        f'- EM迭代{fm["EM_iterations"]}轮，分支比{fm["branching_ratio"]:.6f}，触及本轮0.99稳定性约束上限。不能因数值收敛就认定模型充分拟合；目录年代变化和参数约束仍需诊断。',
        f'- 初次拟合有{fm.get("initial_fit_optimizer_warnings",fm["optimizer_failed_steps"])}次M步线搜索告警。随后在同一训练数据、同一目标函数与约束下，以1e-6数值容差精修；最终精修阶段失败步骤为{fm["optimizer_failed_steps"]}，精修迭代{fm.get("refinement_EM_iterations",0)}轮。该修复未使用目标期成绩。',
        '- 两半模拟的相同面积报警交并比：'+str(dg['split_half_stability'])+'。这不是独立预测样本，也不是跨窗口稳定性。',
        '- 本次是一个历史窗口的试运行，不是完整验证集/测试集的ETAS成绩。概率尚未完成校准，不把0.99分支比解释成地震实际触发比例。',
        '- 作者触发似然使用无限平面空间积分；观测历史仅限日本窗口，边界影响仍是限制。当前同目录试验不替代目录完整性与震级换算审查。',
        '- 本轮只能与同一个窗口的PI空间图比较；未运行或重建ConvLSTM单窗口预测，不把已有整集均值拿来比较。','',
        '## 已完成的核验','','- 5项接口检查通过：全部事件网格映射、面积预算、排序并列处理、背景采样复现、触发时间逆变换。',
        '- 概率由模拟有震次数复算一致；两半次数相加一致；所有报警规则的评分与面积复核一致。',
        '- 首末两个完整模拟按相同种子重算一致；拟合目录最大时间早于预测起点。预测文件先保存后加载目标评分，评分后哈希不变。','',
        '## 结果文件','','- prediction.npz：概率、期望事件数、模拟有震计数及两半计数。',
        '- evaluation.csv / evaluation_protocols.csv：报警规则与同面积对照，以及四种匹配组合。',
        '- simulation_counts.csv / diagnostics.json：每次模拟事件数、随机种子和诊断。',
        '- fit_manifest.json / fit_config.json / optimizer_steps.json / fit.log：参数、拟合过程与来源。',
        '- ETAS_single_forecast.png / ETAS_single_forecast.pdf：空间图。',
        '- fit.pkl / historical_catalog.csv：本地继续模拟及追溯材料，pickle仅限本次可信本地生成文件。','']
    (out/'ETAS单窗口结果说明.md').write_text('\n'.join(lines),encoding='utf-8')
    verification={'complete':True,'trial_only':True,'interface_tests':5,'full_simulation_seeds_reproduced':[0,n-1],
        'all_exported_alarm_rules_rescored':len(table),'all_probabilities_reconstructed':True,
        'fit_maximum_event_time':str(state['catalog'].time.max()),'prediction_unchanged_after_scoring':True,
        'output_sha256':{x.name:p.sha(x) for x in out.iterdir() if x.is_file() and x.name!='verification.json'}}
    (out/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    print(json.dumps({'complete':True,'outputs':str(out),'diagnostics':dg},indent=2))

if __name__=='__main__':main()
