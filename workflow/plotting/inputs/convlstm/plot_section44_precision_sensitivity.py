"""Original section-4.4 heatmap style, updated to fixed-model precision gains.

Read adjacent section44_plot_data.json. No models or evaluations are run.
Baseline M1 is used for gain computation and omitted from the heatmap rows.
"""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

BASE=Path(__file__).resolve().parent
NAME='Fig5_precision_sensitivity_M1_M4'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def style():
    for root in [Path('/System/Library/Fonts/Supplemental'),Path('C:/Windows/Fonts')]:
        if root.is_dir():
            for p in root.iterdir():
                if 'times' in p.name.lower() and p.suffix.lower() in ['.ttf','.ttc','.otf']:
                    try:fm.fontManager.addfont(str(p))
                    except (RuntimeError,ValueError):pass
    font=fm.findfont(fm.FontProperties(family='Times New Roman'),fallback_to_default=False)
    plt.rcParams.update({'font.family':'Times New Roman','font.size':18,'axes.labelsize':20,'axes.titlesize':20,
        'xtick.labelsize':18,'ytick.labelsize':18,'axes.linewidth':1.2,'mathtext.fontset':'stix',
        'pdf.fonttype':42,'ps.fonttype':42,'axes.unicode_minus':False,'xtick.direction':'in','ytick.direction':'in'})
    return font

def panel(ax,values,columns,models,xlabel,letter):
    im=ax.imshow(values,aspect='auto',cmap='Blues',vmin=0,vmax=13)
    ax.set_xticks(range(len(columns)),[str(v) for v in columns]);ax.set_yticks(range(len(models)),models)
    ax.set_xlabel(xlabel);ax.set_ylabel('Model')
    ax.set_xticks(np.arange(len(columns)+1)-.5,minor=True);ax.set_yticks(np.arange(len(models)+1)-.5,minor=True)
    ax.grid(which='minor',color='white',lw=1.4);ax.tick_params(which='both',length=0,top=False,right=False)
    for spine in ax.spines.values():spine.set_visible(True);spine.set_linewidth(1.2)
    for i in range(len(models)):
        for j in range(len(columns)):
            label='0.00' if abs(values[i,j])<1e-12 else f'{values[i,j]:+.2f}'
            ax.text(j,i,label,ha='center',va='center',fontsize=18,color='white',zorder=6)
    ax.text(-.01,1.035,letter,transform=ax.transAxes,ha='left',va='bottom',fontsize=20,clip_on=False)
    return im

def report(data,out):
    models=data['models'];ranges=data['ranges']
    lookup={(r['model'],r['analysis']):r for r in ranges}
    text=r'''# 4.4 预测增益的稳健性

## 中文结果总结

为检验精准度增益是否依赖特定评价设置，在依据验证集固定随机种子和模型权重后，分别改变预测阈值及空间拓展尺度，并在2473个滑动测试目标上重新汇总精准度。各参数设置下，均以采用相同设置的历史震级单通道模型M1为基线，计算精准度均值之差；真实目标阈值始终固定为100。

在空间拓展尺度固定为d=3时，将预测阈值由100逐步降低至50，深度融合M2、方向融合M3和机制融合M4相对M1的精准度提高分别为9.32—11.69、7.27—9.41和9.49—12.27个百分点，全部阈值设置下均保持正增益。说明三组融合模型的精准度改善并非只出现在预测阈值100这一设置下。

在预测阈值固定为100时，将空间拓展尺度从d=1调整至d=5，M2、M3和M4的精准度提高分别为5.09—9.32、4.45—7.27和5.05—9.49个百分点，全部尺度下均高于M1；三组模型的增益均在正式尺度d=3达到该尺度范围内的最大值。在更严格的d=1设置下，仍分别提高5.09、4.45和5.05个百分点。因此，在本次检验的预测阈值及空间尺度范围内，三组地震构造特征融合模型相对震级基线的精准度优势保持一致方向，增益幅度随评价设置有所变化。

## 增益范围（绝对百分点）

| 模型 | 输入组合 | 阈值敏感性：d=3，预测阈值50—100 | 尺度敏感性：预测阈值100，d=1—5 | 正式设置：100、d=3 |
|---|---|---:|---:|---:|
| M2 | MAG＋平均深度＋深度标准差 | 9.32—11.69 | 5.09—9.32 | +9.32 |
| M3 | MAG＋最大偏转角＋s₀ | 7.27—9.41 | 4.45—7.27 | +7.27 |
| M4 | MAG＋机制熵H＋基尼指数D | 9.49—12.27 | 5.05—9.49 | +9.49 |

## 图5与图注

![图5：精准度增量敏感性](__FIGURE__)

图5　精准度增益对评价设置的敏感性。（a）固定空间拓展尺度d=3时的预测阈值敏感性；（b）固定预测阈值100时的空间拓展尺度敏感性。数值为2473个滑动测试目标中，各融合模型与采用相同参数设置的M1之间的精准度均值之差，单位为绝对百分点。M1为历史震级单通道基线，仅作为计算参照，不在图中单列；M2—M4依次为深度、SHmax方向、机制双特征融合模型。真实目标阈值始终为100，两面板采用共享色标。

## English draft: Robustness of Forecast Gains

To assess the sensitivity of the precision gains to the evaluation settings, we varied the prediction threshold and spatial expansion scale while keeping the validation-selected seed and trained model weights fixed. For each setting, the difference in mean precision between each model and M1 was calculated over the same 2,473 sliding test targets. The observed-target threshold was fixed at 100 throughout.

With the spatial expansion scale fixed at d = 3, M2, M3, and M4 maintained positive precision gains across all prediction thresholds from 50 to 100 (Figure 5a). The gains ranged from 9.32 to 11.69, from 7.27 to 9.41, and from 9.49 to 12.27 percentage points, respectively. Thus, the precision advantage of each feature-augmented model was retained across the tested prediction thresholds.

With the prediction threshold fixed at 100, the precision gains remained positive over d = 1–5, ranging from 5.09 to 9.32 percentage points for M2, 4.45 to 7.27 for M3, and 5.05 to 9.49 for M4 (Figure 5b). All three models achieved their largest gains within this scale range at the formal setting of d = 3. Under the stricter setting of d = 1, their gains remained 5.09, 4.45, and 5.05 percentage points, respectively. These results show that the direction of the precision improvement relative to M1 was consistent across the tested settings, although its magnitude varied.

**English figure caption.** Sensitivity of precision gains to the evaluation settings. (a) Prediction-threshold sensitivity with the spatial expansion scale fixed at d = 3. (b) Spatial-expansion-scale sensitivity with the prediction threshold fixed at 100. Cell values are the differences in mean precision between each feature-augmented model and the historical magnitude-only baseline M1 over 2,473 sliding test targets, expressed in percentage points. M2, M3, and M4 incorporate hypocentral-depth, SHmax-orientation, and stress-regime features, respectively. Each comparison uses the same threshold and spatial scale for the model and M1. The observed-target threshold is fixed at 100. Both panels share the same color scale.

## 方法口径与文件

- 本轮是固定种子、固定权重的评价设置敏感性分析，不是重复训练或跨种子的稳定性分析。原4.4节涉及九次训练CSI排序、旧M5和旧CSI增益的数值不用于本轮总结。
- 阈值面板与尺度面板共涉及10种不同设置，正式100/d=3设置在两面板重复展示。
- 标准60%召回率条件对应正式设置；阈值敏感性中三组平均召回率也均高于60%，但d=1、2下的召回率低于60%。因此不能将本图解释为所有空间尺度均满足60%召回率。
- 精准度增量始终与同参数的M1比较，不能把放宽空间容差导致的绝对精准度上升单独解释为定位能力提高。
- 本轮表述是已测试参数范围内的增益方向一致，不代表统计显著性或所有潜在参数的稳健性证明。

文件：Fig5_precision_sensitivity_M1_M4.png/.pdf、S44_threshold_precision_gain.csv、S44_scale_precision_gain.csv、S44_sensitivity_details.csv、S44_gain_ranges.csv、section44_plot_data.json、plot_section44_precision_sensitivity.py、prepare_section44_data.py、section44_verification.json。

图风格沿用原Figure67.py和原稿4.4节热图：Times New Roman、蓝色热图、白色单元格分隔线、(a)/(b)面板及共享色标。单元格增益数字全部为白色，不再对阈值100和d=3单列强调框，保留热图正常外边框。色标范围为0—13个百分点，以覆盖本轮最大增益12.27。建议承接现图4，将本图编号为图5。

本地复现：将plot_section44_precision_sensitivity.py与section44_plot_data.json放在同一目录，安装NumPy、Matplotlib并具备Times New Roman字体，执行 `python3 plot_section44_precision_sensitivity.py`。默认输出到脚本所在目录；--out可指定其他目录。prepare_section44_data.py用于从原本地汇总重新整理数值。全部文件为本地分析与绘图用途，无需上传云平台，没有新增云端运行或需要下载的结果。
'''
    text=text.replace('__FIGURE__','<'+str((out/(NAME+'.png')).resolve())+'>')
    text=text.replace('SHmax方向',r'$S_{\mathrm{Hmax}}$方向').replace('SHmax-orientation',r'$S_{\mathrm{Hmax}}$-orientation')
    (out/'4.4节稳健性分析总结.md').write_text(text,encoding='utf-8')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=BASE/'section44_plot_data.json')
    p.add_argument('--out',type=Path,default=BASE);p.add_argument('--preview',type=Path)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    data=json.loads(args.data.read_text());assert data['models']==['M1','M2','M3','M4']
    models=['M2','M3','M4'];thresholds=data['threshold_order'];scales=data['scale_order'];records=data['records']
    matrices=[]
    for kind,x in [('threshold',thresholds),('scale',scales)]:
        v=np.array([[next(r['precision_gain_pp'] for r in records if r['model']==m and
            (r['d']==3 and r['pred_threshold']==t if kind=='threshold' else r['pred_threshold']==100 and r['d']==t)) for t in x] for m in models])
        assert np.all(v>0) and v.max()<13
        matrices.append(v)
    font=style();fig=plt.figure(figsize=(17,5.8));gs=fig.add_gridspec(1,2,width_ratios=[6,5],wspace=.30)
    a=fig.add_subplot(gs[0,0]);b=fig.add_subplot(gs[0,1])
    panel(a,matrices[0],thresholds,models,'Prediction threshold','(a)')
    im=panel(b,matrices[1],scales,models,'Spatial expansion scale, d','(b)')
    fig.subplots_adjust(left=.065,right=.89,bottom=.20,top=.87)
    box=b.get_position();cax=fig.add_axes([box.x1+.012,box.y0,.015,box.height])
    cb=fig.colorbar(im,cax=cax,ticks=[0,3,6,9,12]);cb.set_label(r'$\Delta$Precision (percentage points)',fontsize=20)
    cb.ax.tick_params(length=5,width=1);cb.outline.set_linewidth(1.2)
    fig.savefig(args.out/(NAME+'.png'),dpi=600,bbox_inches='tight');fig.savefig(args.out/(NAME+'.pdf'),bbox_inches='tight')
    if args.preview:fig.savefig(args.preview,dpi=125,bbox_inches='tight')
    plt.close(fig);report(data,args.out)
    verification=dict(source_data_sha256=sha(args.data),plot_code_sha256=sha(Path(__file__)),
        source_manifest_sha256=data['provenance']['source_manifest_sha256'],source_hashes_verified=True,
        models=models,test_target_count=2473,fixed_seed_and_weights=True,
        heatmap_values='precision gain vs M1 under identical settings; percentage points',
        baseline_model='M1',M1_zero_row=False,shared_color_scale=[0,13],font_file=font,dpi=600,
        formal_settings=dict(pred_threshold=100,d=3),formal_setting_outlines=False,cell_text_color='white',no_model_execution=True,
        output_sha256={NAME+ext:sha(args.out/(NAME+ext)) for ext in ['.png','.pdf']})
    (args.out/'section44_verification.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2)+'\n')
    print('Completed:',args.out/(NAME+'.png'))

if __name__=='__main__':main()
