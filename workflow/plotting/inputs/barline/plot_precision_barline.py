"""Local figure layout only: precision bars and connected precision means with window SD."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

BASE=Path(__file__).resolve().parent
BLUE='#4E79A7';GRAY='#686868'
GROUPS=[('depth','Depth',['mag','zero','presence_depth','real_depth'],['M1','M5','M6','M2']),
        ('direction',r'$S_{\mathrm{Hmax}}$ orientation',['mag','zero','presence_direction','real_direction'],['M1','M5','M7','M3']),
        ('regime','Stress regime',['mag','zero','presence_regime','real_regime'],['M1','M5','M6','M4'])]

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def style():
    for directory in [Path('/System/Library/Fonts/Supplemental'),Path('C:/Windows/Fonts')]:
        if directory.is_dir():
            for p in directory.iterdir():
                if 'times' in p.name.lower() and p.suffix.lower() in ['.ttf','.ttc','.otf']:
                    try:fm.fontManager.addfont(str(p))
                    except (ValueError,RuntimeError):pass
    font=fm.findfont(fm.FontProperties(family='Times New Roman'),fallback_to_default=False)
    plt.rcParams.update({'font.family':'Times New Roman','font.size':13,'mathtext.fontset':'stix',
        'pdf.fonttype':42,'ps.fonttype':42,'axes.unicode_minus':False,
        'xtick.direction':'in','ytick.direction':'in'})
    return font

def draw(ax,group,data,letter=None):
    key,title,ids,models=group;lookup={r['id']:r for r in data['rows']};r=[lookup[i] for i in ids]
    x=np.arange(4);p=np.array([z['precision'] for z in r]);ps=np.array([z['precision_sd'] for z in r])
    ax.bar(x,p,width=.56,color=BLUE,edgecolor='none',zorder=2)
    ax.errorbar(x,p,yerr=ps,fmt='o-',color='#2F4F70',ecolor='#2F4F70',mfc='white',mec='#2F4F70',
        mew=1.4,ms=6,lw=1.8,capsize=4,capthick=1.2,zorder=5)
    for xx,value,sd in zip(x,p,ps):
        gain=value-p[0]
        top_label='Baseline' if xx==0 else f'{gain:+.2f} pp'
        ax.text(xx,value+sd+1.8,top_label,ha='center',va='bottom',color='#2F4F70',fontsize=12,zorder=6)
    inputs=['MAG','MAG + 0','MAG + 100',{'depth':'MAG + Depth','direction':r'MAG + $S_{\mathrm{Hmax}}$'+'\norientation','regime':'MAG + Regime'}[key]]
    ax.set_xticks(x,[m+'\n'+i for m,i in zip(models,inputs)],fontsize=11.5)
    ax.tick_params(axis='x',pad=8,length=0);ax.set_xlim(-.55,3.55);ax.set_ylim(0,90);ax.set_yticks([0,20,40,60,80])
    ax.set_ylabel('Precision (%)',fontsize=13)
    ax.set_title((letter+' ' if letter else '')+title,fontsize=16,pad=11)
    ax.grid(axis='y',color='#D0D0D0',linestyle='--',lw=.65,alpha=.7,zorder=0)
    for spine in ax.spines.values():spine.set_color('#4D4D4D');spine.set_linewidth(.9)

def save(fig,name,args):
    fig.savefig(args.out/(name+'.png'),dpi=600,bbox_inches='tight')
    fig.savefig(args.out/(name+'.pdf'),bbox_inches='tight')
    if args.preview:
        args.preview.mkdir(parents=True,exist_ok=True);fig.savefig(args.preview/(name+'.png'),dpi=145,bbox_inches='tight')
    plt.close(fig)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=BASE);parser.add_argument('--preview',type=Path)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    data=json.loads((BASE/'barline_data.json').read_text());assert data['sample_count']==2473 and data['direction_channels']==[0,3,4]
    font=style();names=[]
    for group in GROUPS:
        fig,ax=plt.subplots(figsize=(6.3,5.3));draw(ax,group,data)
        fig.subplots_adjust(left=.13,right=.98,top=.90,bottom=.20)
        fig.text(.55,.035,'Labels: gain vs M1 (pp); error bars: window SD.',ha='center',fontsize=10.5)
        name='S43_precision_barline_'+group[0];save(fig,name,args);names.append(name)
    fig,axes=plt.subplots(1,3,figsize=(16.5,5.5),sharey=True)
    for i,(ax,group) in enumerate(zip(axes,GROUPS)):
        draw(ax,group,data,['(a)','(b)','(c)'][i])
        if i:ax.set_ylabel('')
    fig.subplots_adjust(left=.055,right=.99,top=.90,bottom=.22,wspace=.15)
    fig.text(.52,.04,'Labels: precision gain vs M1 (pp). Error bars: population SD across 2,473 test targets.',ha='center',fontsize=11)
    name='S43_precision_barline_combined';save(fig,name,args);names.append(name)
    report=dict(font_file=font,dpi=600,source_sha256=sha(BASE/'barline_data.json'),code_sha256=sha(Path(__file__)),
        precision_style='blue bars and connected precision means with SD',recall_shown=False,top_labels='precision increase relative to M1 in percentage points',orientation_label='SHmax orientation',shared_y_axis=True,
        model_orders=[g[3] for g in GROUPS],figures=names,no_model_execution=True,
        output_sha256={n+e:sha(args.out/(n+e)) for n in names for e in ['.png','.pdf']})
    (args.out/'precision_barline_verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    (args.out/'精准度柱线图说明.md').write_text('''# 4.3节分组柱线图

蓝色柱与深蓝折线均表示精准度均值，误差棒为2473个滑动测试目标的总体标准差（ddof=0），上方标签为相对M1的精准度增量（绝对百分点），M1标为Baseline；柱高和折线仍为原始精准度均值，误差棒仍为原始窗口标准差。不显示召回率。误差棒不是置信区间。

深度组：M1、M5、M6、M2；SHmax orientation组：M1、M5、M7、M3；机制组：M1、M5、M6、M4。
各图均固定验证集确定的种子，预测阈值100、空间拓展d=3。方向模型为最大偏转角+s0。M5为两个拓展通道置0；M6为深度/机制有效位置100占位；M7为方向有效位置100占位。“0”和“100”均为两个拓展通道共同设置的简写。

M1、M5为共享结果，M6由深度和机制复用；重复绘制用于组内比较，不是独立训练重复。各折线仅连接输入类别，横轴不是时间。

三张独立图及一张三面板合图，均提供600dpi PNG及嵌入字体PDF。此侧对话输出保存于独立的可视化目录，未覆盖主线程文件。

本地复现：将plot_precision_barline.py与barline_data.json放在同目录，安装NumPy、Matplotlib及Times New Roman字体后运行：`python3 plot_precision_barline.py`。无需上传云平台或补跑实验。
''',encoding='utf-8')
    print('Generated:',names)

if __name__=='__main__':main()
