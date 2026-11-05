"""Export validation results using already frozen recall-matched thresholds.

No threshold selection, model execution, or simulation. Preserve old outputs.
Unavailable ConvLSTM exact/physical-area fields stay missing, not zero.
"""
from pathlib import Path
import csv, hashlib, json
import numpy as np
import evaluate_baselines as core

HERE=Path(__file__).resolve().parent
ROOT=core.ROOT
ANCHORS=core.CONV[1:]

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return core.read(p)

def main():
    frozenpath=HERE/'recall_matched_thresholds.json'
    frozen=json.loads(frozenpath.read_text());frozen_hash=sha(frozenpath)
    for name,h in frozen['validation_source_sha256'].items():assert sha(name)==h,name
    corepath=ROOT/'convlstm/runs/compact_core_summary/sensitivity_nonoverlap.csv'
    stats=read(corepath)
    baselinepath=HERE/'recall_matched_validation_by_window.csv'
    baselines=read(baselinepath)
    manifest=json.loads((HERE/'recall_matched_manifest.json').read_text())
    assert manifest['files_sha256'][baselinepath.name]==sha(baselinepath)
    selected={(r['anchor'],r['model']):r for r in frozen['selections']}
    records=[]
    for anchor in ANCHORS:
        for model in [anchor,'MAG','PI','ETAS','SSS']:
            for row in core.ROWS:
                isconv=model in core.CONV
                if isconv:
                    a=next(r for r in stats if r['model']==model and r['split']=='validation' and int(r['row'])==row and r['d']=='3' and r['pred_threshold']=='100')
                    r=dict(anchor=anchor,model=model,label=core.LABEL[model],row=row,
                        target_start=a['target_start'],target_end=a['target_end'],threshold=100.,threshold_unit='gray',
                        precision=float(a['precision']),recall=float(a['recall']),precision_exact=None,recall_exact=None,
                        alarm_cells=int(a['raw_pred_grid_count']),expanded_alarm_cells=int(a['expanded_pred_grid_count']),
                        target_cells=int(a['raw_true_grid_count']),predicted_hit_cells=int(a['hit_pred_grids']),
                        covered_true_cells=int(a['covered_true_grids']),false_alarm_cells=int(a['false_positive_pred_grids']),
                        missed_true_cells=int(a['false_negative_true_grids']),exact_hit_cells=None,
                        alarm_grid_fraction=float(a['raw_pred_grid_fraction']),expanded_grid_fraction=float(a['expanded_pred_grid_fraction']),
                        alarm_km2=None,alarm_area_fraction=None,expanded_alarm_km2=None,expanded_area_fraction=None,
                        source='verified cloud statistics',missing='exact P/R and physical area require original validation predictions')
                else:
                    a=next(r for r in baselines if r['anchor']==anchor and r['model']==model and int(r['row'])==row)
                    assert float(a['threshold'])==selected[anchor,model]['threshold']
                    r=dict(anchor=anchor,model=model,label=core.LABEL[model],row=row,
                        target_start=a['target_start'],target_end=a['target_end'],threshold=float(a['threshold']),threshold_unit='own score')
                    for key in ['precision','recall','precision_exact','recall_exact','alarm_cells','expanded_alarm_cells',
                        'target_cells','predicted_hit_cells','covered_true_cells','false_alarm_cells','missed_true_cells',
                        'exact_hit_cells','alarm_km2','alarm_area_fraction','expanded_alarm_km2','expanded_area_fraction']:
                        r[key]=float(a[key]) if a[key] else None
                    r.update(alarm_grid_fraction=r['alarm_cells']/68000,expanded_grid_fraction=r['expanded_alarm_cells']/68000,
                        source='previous local scoring of original validation maps',missing='')
                    # Match the ConvLSTM column order in this unified export.
                    r={key:r[key] for key in records[0]} if records else r
                assert abs(r['precision']-r['predicted_hit_cells']/r['alarm_cells'])<1e-12
                assert abs(r['recall']-r['covered_true_cells']/r['target_cells'])<1e-12
                records.append(r)
    assert len(records)==90
    core.write(HERE/'recall_matched_validation_all_by_window.csv',records)
    summary=core.aggregate(records,['anchor','model'])
    for s in summary:
        rs=[r for r in records if r['anchor']==s['anchor'] and r['model']==s['model']]
        for f in ['alarm_grid_fraction','expanded_grid_fraction']:
            s[f+'_mean']=float(np.mean([r[f] for r in rs]));s[f+'_sd']=float(np.std([r[f] for r in rs]))
        s['exact_available_windows']=sum(r['precision_exact'] is not None for r in rs)
        s['physical_area_available_windows']=sum(r['alarm_km2'] is not None for r in rs)
        if s['model'] not in core.CONV:
            ref=selected[s['anchor'],s['model']]
            assert abs(s['recall_mean']-ref['matched_mean_recall'])<1e-12
        elif s['model']==s['anchor']:
            ref=next(x for x in frozen['selections'] if x['anchor']==s['anchor'])
            assert abs(s['recall_mean']-ref['reference_mean_recall'])<1e-12
    core.write(HERE/'recall_matched_validation_summary.csv',summary)
    paired=[]
    for anchor in ANCHORS:
        ref=next(r for r in summary if r['anchor']==anchor and r['model']==anchor)
        for baseline in ['PI','ETAS','SSS']:
            b=next(r for r in summary if r['anchor']==anchor and r['model']==baseline)
            paired.append(dict(anchor=anchor,baseline=baseline,
                precision_difference_pp=100*(ref['precision_mean']-b['precision_mean']),
                recall_difference_pp=100*(ref['recall_mean']-b['recall_mean'])))
    core.write(HERE/'recall_matched_validation_paired_summary.csv',paired)
    def table(rs,mean=True):
        text='| 模型 | 精准度% | 召回率% | 同格精准度% | 同格召回率% | 原始实际面积% | 拓展实际面积% |\n|---|---:|---:|---:|---:|---:|---:|\n'
        for r in rs:
            text+='| '+core.LABEL[r['model']]+' | '
            values=[r[k+('_mean' if mean else '')] for k in ['precision','recall','precision_exact','recall_exact','alarm_area_fraction','expanded_area_fraction']]
            text+=' | '.join('缺原始预测图' if v is None else f'{100*v:.2f}' for v in values)+' |\n'
        return text
    report='# 验证集六窗口结果：召回率匹配方案\n\n'
    report+='使用本轮已经固定的九组基线阈值，不重新选阈值。ConvLSTM固定灰度≥100；主评价为原有对称d=3；不限制报警面积。以下均为六个不重叠验证目标窗口的等权统计，不是2473个滑动样本的均值。仅报告精准度和召回率。\n\n'
    report+='**验证集用于确定基线阈值，因此这些是阈值匹配与早期表现的记录，不能代替后期测试结果。**\n\n'
    report+='## 六窗口均值\n\n'
    for anchor in ANCHORS:
        report+='### '+core.LABEL[anchor]+'组\n\n'
        report+=table([r for r in summary if r['anchor']==anchor])+'\n'
        for p in paired:
            if p['anchor']==anchor:
                report+=f'- 融合模型减{core.LABEL[p["baseline"]]}：精准度{p["precision_difference_pp"]:+.2f}个百分点，召回率{p["recall_difference_pp"]:+.2f}个百分点。\n'
        report+='\n'
    report+='## 逐窗口结果\n\n'
    for anchor in ANCHORS:
        report+='### '+core.LABEL[anchor]+'组\n\n'
        for row in core.ROWS:
            rs=[r for r in records if r['anchor']==anchor and r['row']==row]
            report+='**'+rs[0]['target_start']+'—'+rs[0]['target_end']+'**\n\n'+table(rs,False)+'\n'
    report+='## 覆盖范围与缺项\n\n'
    report+='四个ConvLSTM的验证集原始预测图未包含在本地compact_core_summary交付中，现有统计支持d=3精准度、召回率、报警/覆盖网格数及网格比例；不能据此恢复完全同格命中数或按纬度加权的实际平方公里面积。以上缺项留空，不填0。三个外部基线有原始分数图，相关指标完整。\n\n'
    report+='所有模型可统一比较的网格覆盖比例另列如下。**网格比例不等于实际面积比例**，两者没有混用。\n\n'
    report+='| 参照组 | 模型 | 原始报警网格占比% | 拓展覆盖网格占比% |\n|---|---|---:|---:|\n'
    for r in summary:
        report+=f'| {core.LABEL[r["anchor"]]} | {core.LABEL[r["model"]]} | {100*r["alarm_grid_fraction_mean"]:.2f} | {100*r["expanded_grid_fraction_mean"]:.2f} |\n'
    report+='\n## 文件与复现\n\n'
    report+='新增代码export_validation_results.py；均值recall_matched_validation_summary.csv；90条逐窗结果recall_matched_validation_all_by_window.csv；差值recall_matched_validation_paired_summary.csv。原有recall_matched_validation_by_window.csv仍保留，为54条基线记录。\n\n'
    report+='本地位置：`'+str(HERE)+'`。使用既有Python环境（NumPy、SciPy），并保留同目录evaluate_baselines.py及原项目统计文件。\n\n'
    report+=f"```bash\ncd '{HERE}'\n'/Users/xiazongyuan/Documents/ChatGPT/论文撰写/etas_trial_20261029/.venv/bin/python' export_validation_results.py\n```\n\n"
    report+='本轮已完成本地输出，所需上传文件：无；云端运行：无；云端下载：无。补齐ConvLSTM验证集缺项需另取原始验证预测图，本轮未执行该云端工作。\n'
    (HERE/'recall_matched_validation_结果报告.md').write_text(report)
    assert sha(frozenpath)==frozen_hash
    names=['recall_matched_validation_all_by_window.csv','recall_matched_validation_summary.csv',
        'recall_matched_validation_paired_summary.csv','recall_matched_validation_结果报告.md']
    core.dump(HERE/'recall_matched_validation_export_manifest.json',dict(complete=True,
        thresholds_unchanged=True,thresholds_sha256=frozen_hash,
        source_sha256={str(corepath):sha(corepath),str(baselinepath):sha(baselinepath)},
        script_sha256=sha(__file__),records=len(records),summary_rows=len(summary),
        incomplete_fields='ConvLSTM validation exact metrics and latitude-weighted physical area',
        files_sha256={name:sha(HERE/name) for name in names}))
    for anchor in ANCHORS:
        print(anchor,[(r['model'],round(100*r['precision_mean'],4),round(100*r['recall_mean'],4)) for r in summary if r['anchor']==anchor])
    print('COMPLETE validation export',flush=True)

if __name__=='__main__':main()
