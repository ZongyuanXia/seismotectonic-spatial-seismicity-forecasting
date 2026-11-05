"""Derive paper tables/figure inputs from explicitly selected result files."""
from collections import defaultdict
from decimal import Decimal, ROUND_HALF_UP
import csv
from pathlib import Path
from statistics import fmean, pstdev, stdev

from .common import MODELS, PROTOCOL, dump, load, module, read, sha, write

REGIONS = {'southern_california':'南加州', 'japan':'日本', 'new_zealand':'新西兰',
           'greece':'希腊', 'indonesia':'印度尼西亚', 'iran':'伊朗', 'italy':'意大利',
           'mexico':'墨西哥', 'taiwan':'中国台湾', 'central_chile':'智利中部'}
SYMBOLS = {'MEAN_LOCAL_DEPTH':r'\bar h', 'STD_LOCAL_DEPTH':r'\sigma_h',
           'MEDIAN_LOCAL_DEPTH':r'\widetilde h', 'MAX_SPREAD_100KM_DEG':r'\Delta\theta_{\max}',
           'STD_LOCAL':r's_0', 'REGIME_ENTROPY':r'H', 'REGIME_GINI':r'D',
           'CONSISTENCY_R_MEAN':r'R', 'MAX_ROTATION_DEG':r'\Delta\theta_c'}
INPUTS = {'M1':'MAG', 'M2':r'$\mathrm{MAG}+\bar h+\sigma_h$',
          'M3':r'$\mathrm{MAG}+\Delta\theta_{\max}+s_0$', 'M4':r'$\mathrm{MAG}+H+D$',
          'M5':r'$\mathrm{MAG}+0$', 'M6':r'$\mathrm{MAG}+100$', 'M7':r'$\mathrm{MAG}+100$'}
CORE = ['M1', 'M2', 'M3', 'M4']
STAT_FIELDS = ['precision', 'recall', 'CSI', 'raw_pred_grid_fraction', 'expanded_pred_grid_fraction',
               'false_positive_pred_grids', 'false_negative_true_grids']


def moments(values, ddof=0):
    values = [float(x) for x in values if x not in ('', None)]
    if not values: raise ValueError('No defined observations for a reported metric')
    return fmean(values), (stdev(values) if ddof else pstdev(values))


def core_results(layout):
    if layout.source == 'archived':
        root = layout.project / 'convlstm/runs/compact_core_direction_s0_summary'
        manifest = load(root / 'summary_manifest.json')
        for name, h in manifest['files_sha256'].items():
            if sha(root / name) != h: raise ValueError('Archived core result changed: ' + name)
        return read(root / 'sensitivity_summary.csv'), read(root / 'sensitivity_nonoverlap.csv'), root
    summaries, windows, maps, truth = [], [], {}, None
    import numpy as np
    for ident in CORE:
        model = MODELS[ident]['source_model']
        for split, expected in PROTOCOL['expected_splits'].items():
            folder = layout.sensitivity(ident, split)
            receipt = load(folder / 'done.json')
            if not receipt['complete'] or receipt['identity']['checkpoint_sha256'] != sha(layout.check_model(ident) / 'best.pt'):
                raise ValueError('Incomplete or wrong sensitivity model: ' + ident)
            for name, h in receipt['files_sha256'].items():
                if sha(folder / name) != h: raise ValueError('Changed sensitivity result: ' + name)
            groups = defaultdict(list)
            for r in read(folder / 'per_window.csv'):
                groups[int(float(r['pred_threshold'])), int(r['d'])].append(r)
            required = {(s['pred_threshold'], s['d']) for s in PROTOCOL['sensitivity_settings']}
            if set(groups) != required: raise ValueError('Missing sensitivity settings: ' + ident)
            for (threshold, scale), rows in groups.items():
                if sorted(int(r['row']) for r in rows) != list(range(expected['sample_count'])):
                    raise ValueError('Incomplete or duplicate test samples: ' + ident)
                for cohort in ('daily', 'nonoverlap'):
                    chosen = rows if cohort == 'daily' else [r for r in rows if int(r['row']) in PROTOCOL['nonoverlap_indices']]
                    s = dict(model=model, seed='run_03', split=split, pred_threshold=threshold,
                             d=scale, cohort=cohort, sample_count=len(chosen))
                    for field in STAT_FIELDS:
                        mean, sd = moments([r[field] for r in chosen])
                        s.update({field+'_mean':mean, field+'_sd':sd,
                                  field+'_valid_count':sum(r[field] not in ('', None) for r in chosen)})
                    summaries.append(s)
                windows.extend(dict(r, model=model) for r in rows if int(r['row']) in PROTOCOL['nonoverlap_indices'])
            if split == 'test':
                with np.load(folder / 'six_maps.npz', allow_pickle=False) as z:
                    maps[model+'_predictions'] = z['predictions'].copy()
                    current = z['targets'].copy()
                if truth is not None and not np.array_equal(truth, current): raise ValueError('Different targets across models')
                truth = current
    root = layout.results / 'core_summary'; root.mkdir(parents=True, exist_ok=True)
    maps['targets'] = truth
    np.savez_compressed(root / 'six_window_maps.npz', **maps)
    dump(root / 'grid_and_protocol.json', load(layout.project / 'manuscript/source_summaries/reproduction_grid.json'))
    write(root / 'sensitivity_summary.csv', summaries)
    write(root / 'sensitivity_nonoverlap.csv', windows)
    dump(root / 'summary_manifest.json', dict(analysis_complete=True,
         files_sha256={n:sha(root/n) for n in ['six_window_maps.npz','grid_and_protocol.json','sensitivity_summary.csv','sensitivity_nonoverlap.csv']}))
    return summaries, windows, root


def rf_results(layout):
    root = layout.project / 'rf_analysis/results_1000'
    tasks = read(root / 'task_summary.csv'); table, figure = [], []
    for task in tasks:
        if task['status'] != 'completed': continue
        period, region, threshold = task['period'], task['region'], int(task['threshold'])
        if period=='1978_2008' and region!='japan':continue
        folder = root / ('%s__%s__ge%d' % (period, region, threshold))
        scores = read(folder / 'scores_per_repeat.csv')
        if len(scores) != 1000: raise ValueError('Expected 1000 RF repeats: ' + str(folder))
        mean, sd = moments([r['Accuracy'] for r in scores], ddof=1)
        if period == '1978_2024' and int(task['min_class_count']) >= 21:
            figure.append(dict(region=REGIONS[region], threshold=threshold,
                n_total=int(task['n_total']), n_positive=int(task['n_positive']), n_negative=int(task['n_negative']),
                accuracy_mean_pct=mean*100, accuracy_sd_pp=sd*100,
                repeats_above_50_pct=sum(float(r['Accuracy'])>.5 for r in scores)/10,
                random_baseline_pct=50, n_repeats=len(scores)))
        if region == 'japan' and period == '1978_2008':
            importance = read(folder / 'feature_importances_per_repeat.csv')
            ranked = sorted(SYMBOLS, key=lambda f:fmean(float(r[f]) for r in importance), reverse=True)
            table.append(dict(threshold=threshold, positive_events=int(task['n_positive']), negative_events=int(task['n_negative']),
                              accuracy_mean=mean, accuracy_sd=sd, accuracy_sd_ddof=1,
                              **{'top%d_indicator'%(i+1):f for i,f in enumerate(ranked[:4])}))
    if len(table) != 3 or len(figure) != 23: raise ValueError('Unexpected RF tasks for the current manuscript')
    write(layout.project / 'plotting/inputs/rf/data/图1_地区分类.csv', figure)
    return sorted(table, key=lambda r:r['threshold'])


def selected(rows, ident, cohort='daily', split='test', threshold=100, scale=3):
    matches = [r for r in rows if r['model']==MODELS[ident]['source_model'] and r['split']==split
               and r['cohort']==cohort and float(r['pred_threshold'])==threshold and int(r['d'])==scale]
    if len(matches)!=1: raise ValueError('Missing/duplicate result: '+str((ident,split,cohort,threshold,scale)))
    return matches[0]


def controls(layout):
    if layout.source == 'archived':
        # Precision and recall come from their respective tolerance protocols.
        rows = read(layout.project / 'compare/compact_compare_summary/protocol_summary.csv')
        rows += read(layout.project / 'convlstm/runs/compact_core_summary/direction_B_protocol_summary.csv') if (layout.project / 'convlstm/runs/compact_core_summary/direction_B_protocol_summary.csv').exists() else []
        output=[]
        for ident in ('M5','M6','M7'):
            task = MODELS[ident]['source_model']
            hit = [r for r in rows if r['task']==task and r['split']=='test' and r['cohort']=='daily']
            if hit:
                p=next(r for r in hit if r['protocol']=='target_only')
                q=next(r for r in hit if r['protocol']=='prediction_only')
                output.append(dict(model=ident,sample_count=int(p['sample_count']),
                    precision_mean_pct=float(p['precision_mean'])*100,precision_population_sd_pct=float(p['precision_std'])*100,
                    recall_mean_pct=float(q['recall_mean'])*100,recall_population_sd_pct=float(q['recall_std'])*100))
            elif ident=='M7':
                r=next(r for r in read(layout.project/'convlstm/runs/compact_core_summary/direction_B_metrics.csv')
                       if r['split']=='test' and r['cohort']=='daily')
                output.append(dict(model=ident,sample_count=int(r['sample_count']),precision_mean_pct=float(r['precision'])*100,
                    precision_population_sd_pct=float(r['precision_std'])*100,recall_mean_pct=float(r['recall'])*100,
                    recall_population_sd_pct=float(r['recall_std'])*100))
            else:raise ValueError('Missing archived control protocols: '+ident)
        return output
    output=[]
    for ident in ('M5','M6','M7'):
        folder=layout.evaluation(ident,'test'); manifest=load(folder/'evaluation_manifest.json')
        if manifest['prediction_checkpoint_sha256'] != sha(layout.check_model(ident)/'best.pt'): raise ValueError('Wrong control checkpoint')
        for n,h in manifest['output_sha256'].items():
            if sha(folder/n)!=h: raise ValueError('Changed control result: '+n)
        rows=read(folder/'by_sample.csv'); chosen={}
        for metric,protocol in [('precision','target_only'),('recall','prediction_only')]:
            group=[r for r in rows if r['protocol']==protocol]
            n=PROTOCOL['expected_splits']['test']['sample_count']
            if sorted(int(r['row']) for r in group)!=list(range(n)): raise ValueError('Incomplete control samples')
            chosen[metric]=moments([r[metric] for r in group])
        output.append(dict(model=ident,sample_count=n,precision_mean_pct=chosen['precision'][0]*100,
            precision_population_sd_pct=chosen['precision'][1]*100,recall_mean_pct=chosen['recall'][0]*100,
            recall_population_sd_pct=chosen['recall'][1]*100))
    return output


def numeric_tables(layout, core):
    output={1:rf_results(layout),4:controls(layout)}
    for n,cohort in [(2,'daily'),(3,'nonoverlap')]:
        base=float(selected(core,'M1',cohort)['precision_mean']); rows=[]
        for ident in CORE:
            r=selected(core,ident,cohort)
            rows.append(dict(model=ident,source_model=r['model'],sample_count=int(r['sample_count']),
                precision_mean_pct=float(r['precision_mean'])*100,precision_population_sd_pct=float(r['precision_sd'])*100,
                delta_precision_pp=(float(r['precision_mean'])-base)*100,
                mean_false_positive_grids=float(r['false_positive_pred_grids_mean']),
                recall_mean_pct=float(r['recall_mean'])*100,recall_population_sd_pct=float(r['recall_sd'])*100,
                predicted_area_pct=float(r['raw_pred_grid_fraction_mean'])*100))
        output[n]=rows
    root=(layout.results/'baseline' if layout.source=='computed' else layout.project/'baseline/final_comparison')
    if layout.source=='records':
        details=read(root/'recall_matched_test_by_window.csv');rows=[]
        for key in ['PI','ETAS','SSS']:
            group=[r for r in details if r['anchor']=='pair_01_02' and r['model']==key]
            if len(group)!=6:raise ValueError('Missing baseline six-window records')
            pm,ps=moments([r['precision'] for r in group]);rm,rs=moments([r['recall'] for r in group])
            rows.append(dict(anchor='pair_01_02',model=key,windows=6,precision_mean=pm,precision_sd=ps,recall_mean=rm,recall_sd=rs))
        anchor=selected(core,'M2','nonoverlap')
        rows.append(dict(anchor='pair_01_02',model='pair_01_02',windows=6,precision_mean=anchor['precision_mean'],
            precision_sd=anchor['precision_sd'],recall_mean=anchor['recall_mean'],recall_sd=anchor['recall_sd']))
    else:rows=read(root/'recall_matched_test_summary.csv')
    output[5]=[]
    for ident,key,name in [('M2','pair_01_02','ConvLSTM'),('M8','PI','PI'),('M9','ETAS','ETAS'),('M10','SSS','Smoothed historical seismicity')]:
        r=next(r for r in rows if r['anchor']=='pair_01_02' and r['model']==key)
        output[5].append(dict(model=ident,forecasting_model=name,target_windows=int(r['windows']),
            precision_mean_pct=float(r['precision_mean'])*100,precision_population_sd_pct=float(r['precision_sd'])*100,
            recall_mean_pct=float(r['recall_mean'])*100,recall_population_sd_pct=float(r['recall_sd'])*100))
    return output


def fmt(x, digits=2):
    return str(Decimal(str(x)).quantize(Decimal(10)**(-digits),rounding=ROUND_HALF_UP))


def export_tables(layout, tables):
    captions={c['number']:c['caption'] for c in load(layout.project/'manuscript/captions.json') if c['kind']=='table'}
    for n,rows in tables.items():
        write(layout.tables/('Table%d.csv'%n),rows)
        with (layout.project/('manuscript/tables/Table%d_display.csv'%n)).open(encoding='utf-8-sig',newline='') as stream:
            header=next(csv.reader(stream))
        body=[]
        for r in rows:
            if n==1:
                body.append([fmt(r['threshold'],1),str(r['positive_events']),str(r['negative_events']),
                    '$%s\\pm%s$'%(fmt(r['accuracy_mean'],4),fmt(r['accuracy_sd'],4)),
                    ', '.join('$%s$'%SYMBOLS[r['top%d_indicator'%i]] for i in range(1,5))])
            else:
                p='%s $\\pm$ %s'%(fmt(r['precision_mean_pct']),fmt(r['precision_population_sd_pct']))
                q='%s $\\pm$ %s'%(fmt(r['recall_mean_pct']),fmt(r['recall_population_sd_pct']))
                if n in (2,3):
                    body.append([r['model'],INPUTS[r['model']],p,
                        r'\textbf{---}' if r['model']=='M1' else r'\textbf{%s}'%fmt(r['delta_precision_pp']),
                        fmt(r['mean_false_positive_grids'],0),q,fmt(r['predicted_area_pct'],3)])
                elif n==4:body.append([r['model'],INPUTS[r['model']],p,q])
                else:body.append([r['model'],r['forecasting_model'],p,q])
        display=layout.tables/('Table%d_display.csv'%n)
        with display.open('w',encoding='utf-8-sig',newline='') as stream:csv.writer(stream).writerows([header,*body])
        columns='@{\\extracolsep{\\fill}}'+'c'*len(header)
        tex=['\\begin{table}',r'\centering',r'\fontsize{9}{11}\selectfont',r'\caption{%s}'%captions[n],
             r'\begin{tabular*}{\textwidth}{%s}'%columns,r'\toprule',' & '.join(header)+r' \\',r'\midrule']
        tex+=[' & '.join(row)+r' \\' for row in body]
        tex+=[r'\bottomrule',r'\end{tabular*}']
        if n==2:tex += [r'\par\fontsize{8}{10}\selectfont $\Delta P$ denotes the increase in precision relative to M1.']
        if n==4:tex += [r'\par\fontsize{8}{10}\selectfont M6 matches the valid locations of the hypocentral-depth and tectonic stress-regime features, whereas M7 matches those of the $S_{\mathrm{Hmax}}$-orientation features.']
        tex += [r'\end{table}']
        (layout.tables/('Table%d.tex'%n)).write_text('\n'.join(tex)+'\n')


def prepare_figure1(layout):
    """Use the archived filtered history, recompute all counts and months."""
    import pandas as pd
    folder=layout.project/'plotting/inputs/figure1/data'
    hist=pd.read_csv(folder/'wsm_global_history.csv')
    japan=hist[hist.LAT.between(30.05,45.95)&hist.LON.between(128.10,146.80)]
    usgs=pd.read_csv(layout.project/'datasets/cl10_mapping_v2/usgs_event_audit.csv')
    training=pd.read_csv(layout.project/'datasets/rf_wsm/rf_training_1978_2024.csv')
    japan.to_csv(folder/'wsm_japan_history.csv',index=False);usgs.to_csv(folder/'usgs_japan_events.csv',index=False)
    counts=training.groupby('region').size().rename('rf_event_count').reset_index()
    counts['display_name']=counts.region.map({'southern_california':'Southern California','japan':'Japan','new_zealand':'New Zealand',
        'greece':'Greece','indonesia':'Indonesia','iran':'Iran','italy':'Italy','mexico':'Mexico','taiwan':'Taiwan','central_chile':'Chile'})
    counts.sort_values('rf_event_count',ascending=False).to_csv(folder/'rf_region_counts.csv',index=False)
    months=pd.date_range('1976-01-01','2024-12-01',freq='MS')
    def monthly(series):
        return pd.to_datetime(series,utc=True).dt.tz_convert(None).dt.to_period('M').dt.to_timestamp().value_counts().reindex(months,fill_value=0).sort_index()
    u=monthly(usgs.time).astype(float);u.loc[u.index<'1978-01-01']=float('nan')
    pd.DataFrame(dict(month=months,usgs_japan_mw_ge_4=u.values,wsm_japan_records=monthly(japan.DAY).values)).to_csv(folder/'monthly_counts.csv',index=False)
    return dict(wsm_global_history=len(hist),usgs_japan=len(usgs),rf_all_regions=len(training))


def figure_inputs(layout, tables, core, windows, root):
    import numpy as np
    controls_by_id={r['model']:r for r in tables[4]}; real={r['model']:r for r in tables[2]}
    template=load(layout.project/'plotting/inputs/barline/barline_data.json')
    lookup={'mag':'M1','zero':'M5','presence_depth':'M6','presence_direction':'M7','presence_regime':'M6',
            'real_depth':'M2','real_direction':'M3','real_regime':'M4'}
    for r in template['rows']:
        s=(real|controls_by_id)[lookup[r['id']]]
        r.update(precision=s['precision_mean_pct'],precision_sd=s['precision_population_sd_pct'],
                 recall=s['recall_mean_pct'],recall_sd=s['recall_population_sd_pct'])
    dump(layout.project/'plotting/inputs/barline/barline_data.json',template)
    sensitivity=load(layout.project/'plotting/inputs/convlstm/section44_plot_data.json'); records=[]
    for ident in CORE:
        for setting in PROTOCOL['sensitivity_settings']:
            t,d=setting['pred_threshold'],setting['d'];r=selected(core,ident,threshold=t,scale=d);base=selected(core,'M1',threshold=t,scale=d)
            records.append(dict(model=ident,pred_threshold=t,d=d,precision_gain_pp=(float(r['precision_mean'])-float(base['precision_mean']))*100,
                precision_mean_pct=float(r['precision_mean'])*100,precision_sd_pct=float(r['precision_sd'])*100,
                recall_mean_pct=float(r['recall_mean'])*100,recall_sd_pct=float(r['recall_sd'])*100))
    sensitivity['records']=records;dump(layout.project/'plotting/inputs/convlstm/section44_plot_data.json',sensitivity)
    spatial=module('paper_spatial',layout.project/'convlstm/training/spatial_symmetric.py')
    expand=lambda a:spatial.symmetric_expand(np.asarray(a,dtype=np.uint8),d=3)>0
    stats=[]
    with np.load(root/'six_window_maps.npz',allow_pickle=False) as z:
        for i,row in enumerate(PROTOCOL['nonoverlap_indices']):
            truth=z['targets'][i]>0;et=expand(truth)
            for ident in CORE:
                name=MODELS[ident]['source_model'];pred=z[name+'_predictions'][i]*255>=100
                em,ef=expand(pred&et),expand(pred&~et)
                r=next(r for r in windows if r['model']==name and r['split']=='test' and int(r['row'])==row and int(r['d'])==3 and float(r['pred_threshold'])==100)
                stats.append(dict(window=i+1,target_start=r['target_start'],target_end=r['target_end'],model=ident,source_model=name,
                    precision=float(r['precision']),recall=float(r['recall']),FP=int(r['false_positive_pred_grids']),
                    raw_prediction_grid_count=int(pred.sum()),expanded_prediction_grid_count=int(expand(pred).sum()),
                    expanded_matched_grid_count=int(em.sum()),expanded_FP_grid_count=int(ef.sum()),expanded_overlap_grid_count=int((em&ef).sum())))
    write(layout.project/'plotting/inputs/convlstm/Fig3_six_windows_metrics.csv',stats)
    return prepare_figure1(layout)


def export_all(layout):
    core,windows,root=core_results(layout)
    tables=numeric_tables(layout,core);export_tables(layout,tables)
    counts=figure_inputs(layout,tables,core,windows,root)
    if layout.source!='archived':
        # The approved renderer reads this location. Use the newly built bundle.
        import shutil
        target=layout.project/'convlstm/runs/compact_core_direction_s0_summary'
        target.mkdir(parents=True,exist_ok=True)
        for p in root.iterdir():
            if p.is_file():shutil.copy2(p,target/p.name)
    ranges=[]
    for ident in CORE[1:]:
        for kind,settings in [('threshold',[s for s in PROTOCOL['sensitivity_settings'] if s['d']==3]),
                              ('scale',[s for s in PROTOCOL['sensitivity_settings'] if s['pred_threshold']==100])]:
            gains=[(float(selected(core,ident,threshold=s['pred_threshold'],scale=s['d'])['precision_mean'])-
                    float(selected(core,'M1',threshold=s['pred_threshold'],scale=s['d'])['precision_mean']))*100 for s in settings]
            ranges.append(dict(model=ident,analysis=kind,min_gain_pp=min(gains),max_gain_pp=max(gains)))
    write(layout.out/'statistics/sensitivity_ranges.csv',ranges)
    control={r['model']:r for r in tables[4]}
    contrasts=[]
    for r in tables[2][1:]:
        for key in ['M1','M5','M7' if r['model']=='M3' else 'M6']:
            base=tables[2][0] if key=='M1' else control[key]
            contrasts.append(dict(model=r['model'],reference=key,precision_gain_pp=r['precision_mean_pct']-base['precision_mean_pct'],
                recall_gain_pp=r['recall_mean_pct']-base['recall_mean_pct']))
    write(layout.out/'statistics/control_contrasts.csv',contrasts)
    drop=read(layout.project/'rf_ablation/results/summary.csv')
    write(layout.out/'statistics/rf_removal.csv',[r for r in drop if r['period']=='1978_2008' and r['pair'] in ['pair_01_02','pair_03_04','pair_05_06']])
    dump(layout.out/'export_manifest.json',dict(source=layout.source,weights=layout.weights,
        data_counts=counts,model_execution=False,full_sample_records_used=layout.source!='archived',
        tables={str(n):sha(layout.tables/('Table%d.csv'%n)) for n in tables},
        core_source=str(root),note='Archived mode uses frozen aggregate summaries; computed mode requires per-sample outputs.'))
    return tables
