"""Summarize completed MC sensitivity outputs; no new forecasts."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--results',type=Path,required=True)
    args=ap.parse_args();out=args.results
    meta=json.loads((out/'manifest.json').read_text())
    assert meta['complete'] and all(w['old_2048_reproduced_bitwise'] for w in meta['windows'])
    for name,h in meta['output_sha256'].items():assert sha(out/name)==h
    summary=read(out/'exact_hits_summary.csv')
    details=read(out/'exact_hits_by_window.csv')
    wide=read(out/'per_window_hits_wide.csv')
    changes=read(out/'probability_and_alarm_changes.csv')
    halves=read(out/'split_half_stability.csv')
    original=read(args.root/'baseline/MAG_ETAS_PI_validation_six/exact_hits_by_window.csv')
    old={(int(r['row']),float(r['value']),r['model']):r for r in original}
    metrics=('exact_hits','exact_FP','exact_FN','exact_precision','exact_recall','precision','recall','CSI')
    checked=0
    for r in details:
        if r['model'] not in ('PI','ConvLSTM_MAG','ETAS_N2048'):continue
        model='ETAS' if r['model']=='ETAS_N2048' else r['model']
        before=old[int(r['row']),float(r['value']),model]
        for m in metrics:assert float(r[m])==float(before[m]),(r['row'],model,m)
        checked+=1
    assert checked==72
    fractions=(.005,.01,.02,.03)
    models=('ETAS_N1000','ETAS_N2048','ETAS_N5000','ConvLSTM_MAG','PI')
    lookup={(r['model'],float(r['alarm_area_fraction_requested'])):r for r in summary}
    get=lambda model,f,key:float(lookup[model,f][key])
    lines=['# ETAS 模拟次数检查结果','',
        '六个窗口完成。每个窗口前 2048 次逐格复现原有事件数、占据格数和分半结果；PI、ConvLSTM 及 ETAS-2048 的 72 条评分与原表完全一致。','',
        '## 六窗口平均完全同格命中数','',
        '| 实际报警面积 | ETAS 1000 | ETAS 2048 | ETAS 5000 | ConvLSTM MAG | PI |',
        '|---|---:|---:|---:|---:|---:|']
    for f in fractions:
        lines.append('| '+f'{100*f:g}%'+' | '+' | '.join(f'{get(m,f,"exact_hits_mean"):.2f}' for m in models)+' |')
    lines+=['','## 各窗口完全同格命中数','','各单元格依次为 1000 次 / 2048 次 / 5000 次。','',
        '| 预测起点 | 真实有震格数 | 0.5% | 1% | 2% | 3% |','|---|---:|---:|---:|---:|---:|']
    for row in (0,450,900,1350,1800,2250):
        group=[next(r for r in wide if int(r['row'])==row and float(r['alarm_area_percent'])==100*f) for f in fractions]
        values=[' / '.join(r[m+'_exact_hits'] for m in models[:3]) for r in group]
        lines.append('| '+group[0]['target_start']+' | '+group[0]['target_cells']+' | '+' | '.join(values)+' |')
    lines+=['','## 报警区域重合程度','','Jaccard 为交集除以并集，按格数计算；下表为六窗口均值。','',
        '| 报警面积 | 1000 与 5000 | 2048 与 5000 | 1000 内部分半 | 2048 内部分半 | 5000 内部分半 |',
        '|---|---:|---:|---:|---:|---:|']
    for f in fractions:
        vals=[np.mean([float(r['alarm_grid_Jaccard']) for r in changes if int(r['compared_simulations'])==n and float(r['alarm_area_percent'])==100*f]) for n in (1000,2048)]
        vals += [np.mean([float(r['halves_grid_Jaccard']) for r in halves if int(r['simulations'])==n and float(r['alarm_area_percent'])==100*f]) for n in (1000,2048,5000)]
        lines.append('| '+f'{100*f:g}%'+' | '+' | '.join(f'{v:.2%}' for v in vals)+' |')
    lines+=['','分半比较使用不重叠的 500+500、1024+1024、2500+2500 次模拟。1000/2048 与 5000 使用嵌套前缀，因此重合度不能按独立批次解释。',
        '','## 与 MAG 比较','','| 报警面积 | ETAS 1000 胜/平/负窗口数 | ETAS 5000 胜/平/负窗口数 |','|---|---:|---:|']
    for f in fractions:
        values=[]
        for model in ('ETAS_N1000','ETAS_N5000'):
            ds=[int(r[model+'_exact_hits'])-int(r['ConvLSTM_MAG_exact_hits']) for r in wide if float(r['alarm_area_percent'])==100*f]
            values.append(f'{sum(d>0 for d in ds)}/{sum(d==0 for d in ds)}/{sum(d<0 for d in ds)}')
        lines.append('| '+f'{100*f:g}%'+' | '+' | '.join(values)+' |')
    lines+=['','增加模拟次数是减少随机误差，并不保证对这六个真实目标的命中数逐次增加。不能按最高验证集分数挑选模拟次数。',
        '5000 次是更大的抽样估计，不是真实概率；本轮没有独立重复多批 5000 次模拟，不能据此声称完全收敛。ETAS 原有完整性、边界、分支比限制保留。','']
    (out/'结果解读.md').write_text('\n'.join(lines),encoding='utf-8')
    plt.rcParams['svg.fonttype']='none'
    fig,axs=plt.subplots(1,3,figsize=(15.5,4.7),layout='constrained')
    styles=[('ETAS_N1000','ETAS 1000','#db8b24','o'),('ETAS_N2048','ETAS 2048','#408aaf','s'),
            ('ETAS_N5000','ETAS 5000','#67458d','^'),('ConvLSTM_MAG','ConvLSTM MAG','#8b8b8b','x'),('PI','PI','#769474','d')]
    for m,label,color,marker in styles:
        for ax,key,scale in ((axs[0],'exact_hits_mean',1),(axs[1],'exact_recall_mean',100)):
            ax.plot(np.array(fractions)*100,[get(m,f,key)*scale for f in fractions],
                label=label,color=color,marker=marker,lw=2 if m.startswith('ETAS') else 1.3,
                linestyle='-' if m.startswith('ETAS') else '--')
    for n,color,marker in ((1000,'#db8b24','o'),(2048,'#408aaf','s')):
        vals=[100*np.mean([float(r['alarm_grid_Jaccard']) for r in changes if int(r['compared_simulations'])==n and float(r['alarm_area_percent'])==100*f]) for f in fractions]
        axs[2].plot(np.array(fractions)*100,vals,label=f'{n} vs 5000',color=color,marker=marker,lw=2)
    for ax in axs:
        ax.set_xticks(np.array(fractions)*100);ax.set_xlabel('Physical alarm area (% of region)')
        ax.grid(alpha=.2);ax.legend(frameon=False,fontsize=9)
    axs[0].set(title='Mean exact grid hits',ylabel='Count per target window',ylim=(0,None))
    axs[1].set(title='Mean exact recall',ylabel='Recall (%)',ylim=(0,None))
    axs[2].set(title='Alarm overlap with 5000 simulations',ylabel='Mean grid Jaccard (%)',ylim=(0,100))
    fig.suptitle('ETAS simulation count sensitivity | Six 450-day validation targets',fontsize=13)
    for ext in ('png','svg'):fig.savefig(out/f'MC_sensitivity_comparison.{ext}',dpi=180)
    plt.close(fig)
    checks={'complete':True,'all_simulation_result_hashes_verified':True,'original_72_rows_exactly_reproduced':True,
            'script_sha256':sha(__file__),
            'files_sha256':{p.name:sha(p) for p in [out/'结果解读.md',out/'MC_sensitivity_comparison.png',out/'MC_sensitivity_comparison.svg']}}
    (out/'analysis_verification.json').write_text(json.dumps(checks,indent=2)+'\n')
    print('\n'.join(lines))


if __name__=='__main__':main()
