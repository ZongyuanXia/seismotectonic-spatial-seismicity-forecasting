#!/usr/bin/env python3
"""Revised RF figures using the manuscript's Fig23_rf_reslust.py style.

Input summary CSVs are frozen copies. No model is trained or reevaluated.
"""
from pathlib import Path
import argparse
import json
import hashlib
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data'
BLUE='#3E517A'; BLUE_ERR='#7F90B0'; RED='#B22222'; RED_ERR='#D98C8C'
GRID='#D0D0D0'; SPINE='#4D4D4D'; BASELINE='#5F5F5F'
PERIODS=['1978–2024','1978–2008']
FEATURES=['MEAN_LOCAL_DEPTH','MEDIAN_LOCAL_DEPTH','STD_LOCAL_DEPTH',
          'REGIME_ENTROPY','REGIME_GINI','MAX_SPREAD_100KM_DEG',
          'CONSISTENCY_R_MEAN','STD_LOCAL','MAX_ROTATION_DEG']
LABELS={
 'MEAN_LOCAL_DEPTH':r'$\bar{h}$','MEDIAN_LOCAL_DEPTH':r'$\tilde{h}$',
 'STD_LOCAL_DEPTH':r'$\sigma_h$','REGIME_ENTROPY':r'$H$','REGIME_GINI':r'$D$',
 'MAX_SPREAD_100KM_DEG':r'$\Delta\theta_{\max}$','CONSISTENCY_R_MEAN':r'$R$',
 'STD_LOCAL':r'$s_0$','MAX_ROTATION_DEG':r'$\Delta\theta_{mi}$'}
REGIONS={'南加州':'Southern California','日本':'Japan','新西兰':'New Zealand',
 '希腊':'Greece','印度尼西亚':'Indonesia','伊朗':'Iran','意大利':'Italy',
 '墨西哥':'Mexico','中国台湾':'Taiwan','智利中部':'Central Chile'}
CHANNEL={1:'MEAN_LOCAL_DEPTH',2:'STD_LOCAL_DEPTH',3:'MAX_SPREAD_100KM_DEG',
 4:'STD_LOCAL',5:'REGIME_ENTROPY',6:'REGIME_GINI',7:'MEDIAN_LOCAL_DEPTH',
 8:'CONSISTENCY_R_MEAN',9:'MAX_ROTATION_DEG'}
PAIRS=['pair_01_02','pair_01_07','pair_02_07','pair_03_04','pair_03_08',
       'pair_03_09','pair_04_08','pair_04_09','pair_05_06','pair_08_09']
PRIMARY={'MEAN_LOCAL_DEPTH','STD_LOCAL_DEPTH','MAX_SPREAD_100KM_DEG',
         'CONSISTENCY_R_MEAN','REGIME_ENTROPY','REGIME_GINI'}


def style():
    # Register the actual manuscript font where available; do not silently substitute.
    for p in Path('/System/Library/Fonts/Supplemental').glob('Times New Roman*.ttf'):
        font_manager.fontManager.addfont(str(p))
    font_manager.findfont('Times New Roman',fallback_to_default=False)
    mpl.rcParams.update({'font.family':'Times New Roman','font.size':8,
       'axes.labelsize':8,'axes.titlesize':8,'xtick.labelsize':7,'ytick.labelsize':7,
       'legend.fontsize':7,'mathtext.fontset':'stix','pdf.fonttype':42,'ps.fonttype':42,
       'axes.unicode_minus':False,'savefig.facecolor':'white'})


def axes_style(ax,axis='x'):
    ax.tick_params(axis='both',direction='in',bottom=True,left=True,top=False,
                   right=False,labeltop=False,labelright=False,length=3,width=.7)
    for spine in ax.spines.values():spine.set_linewidth(.7);spine.set_color(SPINE)
    ax.set_axisbelow(True)
    ax.grid(axis=axis,color=GRID,linestyle='--',linewidth=.45,alpha=.8)


def err(ax,x,y,error,color,ecolor,horizontal=False,marker='o'):
    kw={'xerr':error} if horizontal else {'yerr':error}
    return ax.errorbar(x,y,fmt=marker,markersize=4.0,capsize=2.4,elinewidth=.9,
                      markeredgewidth=.7,color=color,ecolor=ecolor,zorder=4,**kw)


def pair_label(pair):
    a,b=map(int,pair.split('_')[1:])
    return LABELS[CHANNEL[a]][:-1]+' + '+LABELS[CHANNEL[b]][1:]


def period_legend(ax,loc='upper right'):
    handles=[Line2D([],[],marker='o',linestyle='none',markersize=4,color=c,label=p)
             for p,c in zip(PERIODS,[BLUE,RED])]
    ax.legend(handles=handles,loc=loc,frameon=False,handletextpad=.4,borderpad=.2)


def group_guides(ax):
    for x in [2.5,4.5]:ax.axvline(x,color='#BDBDBD',linestyle='--',lw=.6,zorder=1)
    trans=blended_transform_factory(ax.transData,ax.transAxes)
    for name,a,b in [('Depth',0,2),('Stress regime',3,4),(r'$S_{Hmax}$ orientation',5,8)]:
        ax.text((a+b)/2,.96,name,transform=trans,ha='center',va='top',fontsize=6.5,color=SPINE)


def importance_panel(ax,data,title,legend=True):
    for period,color,ecolor,offset in zip(PERIODS,[BLUE,RED],[BLUE_ERR,RED_ERR],[-.10,.10]):
        sub=data[data.period==period].set_index('feature_code').loc[FEATURES]
        err(ax,np.arange(9)+offset,sub.importance_mean.to_numpy(),
            sub.importance_sd.to_numpy(),color,ecolor)
    ax.set_xticks(np.arange(9),[LABELS[f] for f in FEATURES])
    ax.set_xlim(-.6,8.6);ax.set_ylim(0,.225)
    ax.set_yticks(np.arange(0,.201,.05));ax.set_ylabel('Mean feature importance')
    ax.set_title(title,pad=5);axes_style(ax,'y');group_guides(ax)
    if legend:period_legend(ax,'upper right')


def pair_panel(ax,data,title,drop=False,show_labels=True):
    for period,color,ecolor,offset in zip(PERIODS,[BLUE,RED],[BLUE_ERR,RED_ERR],[-.13,.13]):
        sub=data[data.period==period].set_index('pair').loc[PAIRS]
        if drop:x=sub.accuracy_drop_mean_pp.to_numpy();e=sub.accuracy_drop_sd_pp.to_numpy()
        else:x=sub.accuracy_mean_pct.to_numpy()/100;e=sub.accuracy_sd_pp.to_numpy()/100
        err(ax,x,np.arange(10)+offset,e,color,ecolor,horizontal=True)
        if not drop:
            ax.axvline(sub.nine_accuracy_mean_pct.iloc[0]/100,color=color,ls=':',lw=.75,zorder=1)
    ax.set_yticks(np.arange(10),[pair_label(p) for p in PAIRS] if show_labels else [])
    ax.set_ylim(9.6,-.6);ax.set_title(title,pad=5);axes_style(ax,'x')
    if drop:
        ax.axvline(0,color=BASELINE,ls='--',lw=.9)
        ax.set_xlim(-1.1,2.9);ax.set_xticks([-1,0,1,2]);ax.set_xlabel('Accuracy decrease (pp)')
    else:
        ax.axvline(.5,color=BASELINE,ls='--',lw=.9)
        ax.set_xlim(.49,.69);ax.set_xticks([.50,.55,.60,.65]);ax.set_xlabel('Accuracy')


def save(fig,out,stem,audit):
    fig.canvas.draw()
    # Detect text collisions with the physical figure edge before tight cropping.
    renderer=fig.canvas.get_renderer();width,height=fig.bbox.width,fig.bbox.height
    bad=[]
    for text in fig.findobj(mpl.text.Text):
        if text.get_visible() and text.get_text():
            box=text.get_window_extent(renderer)
            if box.width>0 and (box.x0<-2 or box.y0<-2 or box.x1>width+2 or box.y1>height+2):bad.append(text.get_text())
    assert not bad, (stem,'text outside figure',bad)
    fig.savefig(out/(stem+'.png'),dpi=600,bbox_inches='tight')
    fig.savefig(out/(stem+'.pdf'),bbox_inches='tight')
    audit.append({'figure':stem,'width_cm':fig.get_figwidth()*2.54,'height_cm':fig.get_figheight()*2.54,
                  'png_dpi':600,'errorbars':'SD','text_outside_figure':bad})
    plt.close(fig)


def multiregion(data,out,audit):
    fig,axes=plt.subplots(1,3,figsize=(18/2.54,9.5/2.54),sharex=True)
    for i,(ax,threshold) in enumerate(zip(axes,[4,5,6])):
        sub=data[data.threshold==threshold].sort_values('accuracy_mean_pct')
        for y,r in enumerate(sub.itertuples()):
            red=r.region=='日本'
            err(ax,r.accuracy_mean_pct/100,y,r.accuracy_sd_pp/100,
                RED if red else BLUE,RED_ERR if red else BLUE_ERR,horizontal=True)
        ax.axvline(.5,color=BASELINE,linestyle='--',lw=.9)
        ax.set_yticks(np.arange(len(sub)),[REGIONS[x] for x in sub.region])
        ax.set_xlim(.4,.96);ax.set_xticks(np.arange(.4,.91,.1));ax.set_xlabel('Accuracy')
        ax.set_ylim(-.7,len(sub)-.3)
        ax.set_title(rf'({chr(97+i)}) Threshold: $M_w\geq{threshold}.0$',pad=4)
        axes_style(ax,'x')
    fig.tight_layout(w_pad=1.2)
    save(fig,out,'Figure2_RF_multiregion_accuracy',audit)


def screening(imp,pair,drop,out,audit):
    fig=plt.figure(figsize=(18/2.54,17/2.54))
    gs=fig.add_gridspec(2,2,height_ratios=[.9,1.5],left=.12,right=.98,bottom=.08,top=.97,hspace=.34,wspace=.22)
    a=fig.add_subplot(gs[0,:]);importance_panel(a,imp,r'(a) Japan: $M_w\geq4.0$',legend=False)
    # One shared legend outside the axes avoids collision with category guides.
    handles=[Line2D([],[],marker='o',linestyle='none',markersize=4,color=c,label=p)
             for p,c in zip(PERIODS,[BLUE,RED])]
    fig.legend(handles=handles,loc='center',bbox_to_anchor=(.59,.62),ncol=2,frameon=False)
    b=fig.add_subplot(gs[1,0]);pair_panel(b,pair,'(b) Two-feature discrimination')
    c=fig.add_subplot(gs[1,1]);pair_panel(c,drop,'(c) Leave-two-out decrease',drop=True,show_labels=False)
    save(fig,out,'Figure3_RF_Japan_screening',audit)


def standalone(imp,pair,drop,out,audit):
    fig,ax=plt.subplots(figsize=(18/2.54,7.5/2.54))
    importance_panel(ax,imp,r'(a) Japan: $M_w\geq4.0$',legend=False)
    fig.legend(handles=[Line2D([],[],marker='o',linestyle='none',color=c,label=p)
         for p,c in zip(PERIODS,[BLUE,RED])],loc='upper right',bbox_to_anchor=(.98,.99),ncol=2,frameon=False)
    fig.subplots_adjust(left=.10,right=.98,bottom=.18,top=.86)
    save(fig,out,'Figure3a_Japan_importance',audit)
    for dat,isdrop,stem,title in [(pair,False,'Figure3b_Japan_pair_accuracy','(b) Two-feature discrimination'),
        (drop,True,'Figure3c_Japan_leave_two_out','(c) Leave-two-out decrease')]:
        fig,ax=plt.subplots(figsize=(12/2.54,11/2.54))
        pair_panel(ax,dat,title,drop=isdrop)
        period_legend(ax,'lower right')
        fig.subplots_adjust(left=.25,right=.97,bottom=.12,top=.94)
        save(fig,out,stem,audit)


def supplementary(data,out,audit):
    data=data[data.period=='1978_2024']
    fig,axes=plt.subplots(2,2,figsize=(18/2.54,11.5/2.54),sharey=True)
    for ax,threshold,panel in zip(axes.flat,['4','5','6','all'],'abcd'):
        sub=data[data.threshold.astype(str)==threshold].set_index('feature').loc[FEATURES]
        for i,(f,row) in enumerate(sub.iterrows()):
            red=threshold=='all' and f in PRIMARY
            err(ax,i,row.mean_importance,row.sd_between_region_tasks,
                RED if red else BLUE,RED_ERR if red else BLUE_ERR)
        ax.set_xticks(np.arange(9),[LABELS[f] for f in FEATURES]);ax.set_xlim(-.6,8.6)
        ax.set_ylim(0,.32);ax.set_ylabel('Mean feature importance')
        title=rf'({panel}) Threshold: $M_w\geq{threshold}.0$' if threshold!='all' else '(d) All 23 region–threshold tasks'
        ax.set_title(title,pad=4);axes_style(ax,'y');group_guides(ax)
    fig.tight_layout(w_pad=1.2,h_pad=1.4)
    save(fig,out,'FigureS1_RF_multiregion_importance',audit)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out-dir',type=Path,default=ROOT/'figures')
    args=ap.parse_args();args.out_dir.mkdir(exist_ok=True,parents=True);style()
    region=pd.read_csv(DATA/'图1_地区分类.csv');imp=pd.read_csv(DATA/'图2a_重要性.csv')
    pair=pd.read_csv(DATA/'图2b_双指标.csv');drop=pd.read_csv(DATA/'图2c_删除.csv')
    global_imp=pd.read_csv(DATA/'feature_importance_across_regions.csv')
    assert (len(region),len(imp),len(pair),len(drop))==(23,18,20,20)
    assert set(pair.pair)==set(drop.pair)==set(PAIRS)
    assert np.allclose(drop.nine_accuracy_mean_pct-drop.seven_accuracy_mean_pct,drop.accuracy_drop_mean_pp,atol=1e-10)
    # Every horizontal error bar must be fully visible.
    assert (region.accuracy_mean_pct+region.accuracy_sd_pp).max()<96
    audit=[];multiregion(region,args.out_dir,audit);screening(imp,pair,drop,args.out_dir,audit)
    standalone(imp,pair,drop,args.out_dir,audit);supplementary(global_imp,args.out_dir,audit)
    report={'figures':audit,'font':font_manager.findfont('Times New Roman',fallback_to_default=False),
       'matplotlib_version':mpl.__version__,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
       'model_training':False,'new_inference_or_evaluation':False,
       'data_counts':{'multiregion':23,'japan_importance':18,'japan_pair':20,'japan_leave_two_out':20}}
    (ROOT/'plot_verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
