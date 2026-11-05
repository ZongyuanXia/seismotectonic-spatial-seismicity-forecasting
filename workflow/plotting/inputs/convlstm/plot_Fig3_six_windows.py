"""Draw all six nonoverlapping target windows in the approved 2x2 style.

Only reads returned predictions. Original predicted cells are classified by
d=3 truth matching, then both classes are expanded for display. Purple has
priority where expansions overlap. P/R annotations retain raw-grid scoring.
"""
from pathlib import Path
import argparse,csv,hashlib,json,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.colors import ListedColormap
from matplotlib.path import Path as MPath
from matplotlib.patches import PathPatch,Patch
from matplotlib.lines import Line2D

BASE=Path(__file__).resolve().parent
ROOT=Path('/Users/xiazongyuan/1 课题/7 地震震级预测/xzy-小论文1-文件汇总/代码与结果（归档）')
INPUT=ROOT/'绘图/ConvLSTM重绘_20261104_s0'
SUMMARY=ROOT/'九项指标重构归档/convlstm/runs/compact_core_direction_s0_summary'
TRAINING=ROOT/'九项指标重构归档/convlstm/training'
SPATIAL_SHA='10518564f969a5a1381b6d3b69ca0e9186956397eb58ca5aa5700e7cddb7350c'
MODELS=['MAG','pair_01_02','pair_03_04','pair_05_06']
TITLES=['(a) M1: MAG',r'(b) M2: $\bar h+\sigma_h$',r'(c) M3: $\Delta\theta_{\max}+s_0$',r'(d) M4: $H+D$']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def geometry(bg):
    vertices=[];codes=[];pos=0
    for n in bg['land_sizes']:
        ring=bg['land_vertices'][pos:pos+n];pos+=n
        vertices.extend(ring);vertices.append(ring[0])
        codes.extend([MPath.MOVETO]+[MPath.LINETO]*(len(ring)-1)+[MPath.CLOSEPOLY])
    land=MPath(np.asarray(vertices),np.asarray(codes));coast=[];pos=0
    for n in bg['coast_sizes']:coast.append(bg['coast_vertices'][pos:pos+n]);pos+=n
    return land,coast
def style():
    for directory in [Path('/System/Library/Fonts/Supplemental'),Path('C:/Windows/Fonts')]:
        if directory.is_dir():
            for p in directory.iterdir():
                if 'times' in p.name.lower() and p.suffix.lower() in ['.ttf','.ttc','.otf']:
                    try:fm.fontManager.addfont(str(p))
                    except (RuntimeError,ValueError):pass
    font=fm.findfont(fm.FontProperties(family='Times New Roman'),fallback_to_default=False)
    plt.rcParams.update({'font.family':'Times New Roman','font.size':13,'mathtext.fontset':'stix',
        'pdf.fonttype':42,'ps.fonttype':42,'xtick.direction':'in','ytick.direction':'in'})
    return font

def plot_window(display,truth,stats,date,region,land,coast,args,name):
    b=region['effective_bounds'];extent=[b['lon_min'],b['lon_max'],b['lat_min'],b['lat_max']]
    lon=extent[0]+(np.arange(200)+.5)*region['dlon_deg'];lat=extent[2]+(np.arange(340)+.5)*region['dlat_deg']
    fig,axes=plt.subplots(2,2,figsize=(10.5,11.9));cmap=ListedColormap([(0,0,0,0),args.red,args.purple])
    yy,xx=np.where(truth)
    for i,(ax,mask,title,r) in enumerate(zip(axes.flat,display,TITLES,stats)):
        ax.set_facecolor('#DDEEF8');ax.add_patch(PathPatch(land,facecolor='white',edgecolor='none',zorder=0))
        ax.imshow(mask,cmap=cmap,vmin=0,vmax=2,origin='lower',extent=extent,interpolation='nearest',zorder=2,rasterized=True)
        for line in coast:ax.plot(line[:,0],line[:,1],c='#B0B0B0',lw=.4,zorder=3)
        ax.scatter(lon[xx],lat[yy],s=8,c='black',lw=0,zorder=6)
        ax.set_xlim(127.75,147.15);ax.set_ylim(29.70,46.30);ax.set_aspect(1/np.cos(np.deg2rad(38)))
        ax.set_xticks([130,135,140,145],['130°E','135°E','140°E','145°E'])
        ax.set_yticks([30,35,40,45],['30°N','35°N','40°N','45°N'])
        ax.tick_params(labelsize=12,length=4,width=.8,labelleft=i%2==0)
        for spine in ax.spines.values():spine.set_color('#4D4D4D');spine.set_linewidth(.9)
        ax.grid(color='#C8C8C8',ls='--',lw=.4,alpha=.8);ax.set_title(title,fontsize=17,pad=8)
        ax.text(.025,.975,f"Precision = {100*r['precision']:.2f}%\nRecall = {100*r['recall']:.2f}%",
            transform=ax.transAxes,ha='left',va='top',fontsize=11.5,linespacing=1.35,
            bbox={'facecolor':'white','alpha':.90,'edgecolor':'none','pad':2.5},zorder=8)
        if i%2==0:ax.set_ylabel('Latitude',fontsize=13)
        if i>=2:ax.set_xlabel('Longitude',fontsize=13)
    fig.subplots_adjust(left=.07,right=.985,top=.965,bottom=.11,wspace=.08,hspace=.15)
    fig.legend(handles=[Patch(facecolor=args.red,label='Matched prediction (d = 3)'),
        Patch(facecolor=args.purple,label='False-positive prediction (d = 3)'),
        Line2D([],[],color='black',marker='o',ls='none',ms=4,label='Observed target grid')],
        loc='lower center',bbox_to_anchor=(.52,.04),ncol=3,frameon=False,fontsize=12,columnspacing=1.2,handletextpad=.5)
    fig.text(.52,.015,f"W{date['window']}: {date['target_start']} – {date['target_end']}",ha='center',fontsize=12)
    fig.savefig(args.out/(name+'.png'),dpi=600,bbox_inches='tight');fig.savefig(args.out/(name+'.pdf'),dpi=600,bbox_inches='tight')
    if args.preview_dir:
        args.preview_dir.mkdir(parents=True,exist_ok=True);fig.savefig(args.preview_dir/(name+'.png'),dpi=120,bbox_inches='tight')
    plt.close(fig)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=INPUT);p.add_argument('--summary',type=Path,default=SUMMARY)
    p.add_argument('--out',type=Path,default=BASE);p.add_argument('--preview-dir',type=Path)
    p.add_argument('--window',type=int,choices=range(1,7))
    p.add_argument('--red',default='#E45756');p.add_argument('--purple',default='#C4AEDB')
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    manifest=load(args.summary/'summary_manifest.json')
    if not manifest['analysis_complete'] or manifest['models']!=MODELS:raise ValueError('Incomplete or wrong models')
    for n,h in manifest['files_sha256'].items():
        if sha(args.summary/n)!=h:raise ValueError('Source package changed: '+n)
    if sha(TRAINING/'spatial_symmetric.py')!=SPATIAL_SHA:raise ValueError('Spatial code changed')
    expected_background=load(args.input/'section42_verification.json')['output_sha256']['map_background.npz']
    if sha(args.input/'map_background.npz')!=expected_background:raise ValueError('Background changed')
    sys.path.insert(0,str(TRAINING));from spatial_symmetric import symmetric_expand
    def expand(m):return symmetric_expand(m.astype(np.uint8),d=3)>0
    with np.load(args.summary/'six_window_maps.npz',allow_pickle=False) as z:maps={k:z[k].copy() for k in z.files}
    if set(maps)!={'targets',*(m+'_predictions' for m in MODELS)}:raise ValueError('Wrong prediction models')
    if any(v.shape!=(6,340,200) or not np.isfinite(v).all() for v in maps.values()):raise ValueError('Invalid map arrays')
    references=[r for r in read(args.summary/'sensitivity_nonoverlap.csv') if r['split']=='test' and r['d']=='3' and r['pred_threshold']=='100']
    if len(references)!=24:raise ValueError('Missing formal window references')
    grid=load(args.summary/'grid_and_protocol.json');dates=grid['test_target_dates']
    if [int(x['row']) for x in dates]!=[0,450,900,1350,1800,2250]:raise ValueError('Unexpected target windows')
    with np.load(args.input/'map_background.npz',allow_pickle=False) as z:bg={k:z[k].copy() for k in z.files}
    land,coast=geometry(bg);font=style();all_stats=[];records=[]
    for i,date in enumerate(dates):
        if args.window is not None and i+1!=args.window:continue
        truth=maps['targets'][i]*255.0>=100;et=expand(truth);display=[];stats=[]
        for j,model in enumerate(MODELS):
            pred=maps[model+'_predictions'][i]*255.0>=100
            matched=pred&et;fp=pred&~et;ep=expand(pred);em=expand(matched);ef=expand(fp)
            if not np.array_equal(em|ef,ep):raise ValueError('Expanded partition union changed')
            c=np.zeros((340,200),dtype=np.uint8);c[em]=1;c[ef]=2;display.append(c)
            precision=float(matched.sum()/pred.sum());recall=float((truth&ep).sum()/truth.sum())
            ref=next(r for r in references if r['model']==model and int(r['row'])==int(date['row']))
            for k,v in [('precision',precision),('recall',recall),('false_positive_pred_grids',int(fp.sum())),('expanded_pred_grid_count',int(ep.sum())),('raw_pred_grid_count',int(pred.sum()))]:
                if abs(v-float(ref[k]))>1e-12:raise ValueError('Reference mismatch: '+model+'/'+str(i+1)+'/'+k)
            stats.append(dict(window=i+1,target_start=date['target_start'],target_end=date['target_end'],model='M'+str(j+1),source_model=model,
                precision=precision,recall=recall,FP=int(fp.sum()),raw_prediction_grid_count=int(pred.sum()),
                expanded_prediction_grid_count=int(ep.sum()),expanded_matched_grid_count=int(em.sum()),
                expanded_FP_grid_count=int(ef.sum()),expanded_overlap_grid_count=int((em&ef).sum())))
        name=f'Fig3_W{i+1:02d}_expanded_FP_purple_2x2'
        plot_window(display,truth,stats,dict(window=i+1,**date),grid['region'],land,coast,args,name)
        all_stats.extend(stats);records.append(name);print('Completed window:',i+1,flush=True)
    with (args.out/'Fig3_six_windows_metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(all_stats[0]));w.writeheader();w.writerows(all_stats)
    report=dict(source=str(args.summary),source_manifest_sha256=sha(args.summary/'summary_manifest.json'),
        plot_code_sha256=sha(Path(__file__)),red=args.red,purple=args.purple,opacity=1,
        d=3,prediction_threshold=100,target_threshold=100,layout='2x2: M1, M2, M3, M4',
        font_file=font,png_dpi=600,metrics_recounted_and_matched=len(all_stats),
        visualization='expand matched seeds and FP seeds separately; purple priority in overlaps; union equals full expanded prediction',
        metric_definition='P raw predictions matched to expanded truth; R raw truth covered by expanded predictions',
        no_training_or_inference=True,figures=records,
        output_sha256={name+ext:sha(args.out/(name+ext)) for name in records for ext in ['.png','.pdf']})
    (args.out/'Fig3_six_windows_verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    text='''六个不重叠目标窗口：红色非误报、浅紫色误报，四宫格版本

每个窗口分别生成600dpi PNG及字体嵌入PDF，从左上到右下依次为M1历史震级基线、M2深度融合、M3最大偏转角+s0方向融合、M4机制H+D融合。每幅子图左上角标注该窗口的精准度和召回率，黑点为原始真实目标网格中心。
配色统一为较明亮红色#E45756、较浅紫色#C4AEDB，淡蓝海洋、白色陆地、Times New Roman字体；所有窗口使用相同范围、尺寸与图例。

图注口径：先按原始预测网格是否匹配到真实目标的d=3拓展区域划分非误报和误报，再分别对这两类原始预测网格作d=3拓展显示。红色为非误报源网格的拓展区域；紫色为误报源网格的拓展区域。二者重叠部分优先显示紫色，完整并集与全部预测网格的拓展结果完全一致。不能将紫色区域中的每一个拓展网格另行视为一个误报；误报数仍按原始预测网格计数。P/R保留原正式评价口径，不按图中红紫面积重新计算。

窗口时段与文件名
'''
    for i,date in enumerate(dates):text+=f"W{i+1}: {date['target_start']}—{date['target_end']}；Fig3_W{i+1:02d}_expanded_FP_purple_2x2.png/.pdf\n"
    text+='''
本地代码plot_Fig3_six_windows.py默认读取原s0绘图目录的底图及convlstm/runs/compact_core_direction_s0_summary数据，不依赖模型权重。依赖NumPy、Matplotlib、Times New Roman和原training/spatial_symmetric.py。
本地运行：python3 plot_Fig3_six_windows.py
仅重绘第三个窗口：python3 plot_Fig3_six_windows.py --window 3
--red、--purple可指定颜色，--out指定输出目录，--input指定原绘图资料目录，--summary指定本轮九文件汇总目录。
Fig3_six_windows_metrics.csv保留所绘窗口的指标与计数；不加--window时为全部24行。
无需上传云平台，没有新增训练、推理、全数据集评分或需要下载的结果。不打包ZIP。
'''
    (args.out/'Fig3_六窗口绘图说明.txt').write_text(text,encoding='utf-8')

if __name__=='__main__':main()
