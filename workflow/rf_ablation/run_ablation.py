#!/usr/bin/env python3
"""Leave two feature columns out, refit seven; compare to paired nine-feature RF.

The event tables, km/degree physical units, labels and split seeds are unchanged.
No features are replaced by zero, and original fitted forests are not edited.
"""
from pathlib import Path
import argparse
import json
import platform
import time
import numpy as np
import pandas as pd
import sklearn
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from paired_rf_core import (FEATURES, CHANNELS, PAIRS, METRICS, split_rows,
                            sha, fit_model)
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score, confusion_matrix)

ROOT = Path(__file__).resolve().parent


def worker(table, seed):
    z, y, tr, te, audit = split_rows(table, seed)
    rows=[]
    for a,b in PAIRS:
        removed=[CHANNELS[a],CHANNELS[b]]
        kept=[f for f in FEATURES if f not in removed]
        assert len(kept)==7 and not set(kept)&set(removed)
        x=z[kept].to_numpy(float)
        # 3 is the original nine-feature model's effective sqrt candidate count.
        model=RandomForestClassifier(n_estimators=100,max_depth=8,
            min_samples_leaf=5,max_features=3,n_jobs=1,random_state=seed)
        model.fit(x[tr],y[tr])
        pred=model.predict(x[te]);prob=model.predict_proba(x[te])[:,1]
        tn,fp,fn,tp=confusion_matrix(y[te],pred,labels=[0,1]).ravel()
        rows.append(dict(seed=seed,pair=f'pair_{a:02}_{b:02}',
            removed_1=removed[0],removed_2=removed[1],remaining_count=len(kept),
            remaining_importance=json.dumps(dict(zip(kept,map(float,model.feature_importances_)))),
            Accuracy=accuracy_score(y[te],pred),
            Precision=precision_score(y[te],pred,zero_division=0),
            Recall=recall_score(y[te],pred,zero_division=0),
            F1=f1_score(y[te],pred,zero_division=0),
            ROC_AUC=roc_auc_score(y[te],prob),
            PR_AUC=average_precision_score(y[te],prob),
            TN=int(tn),FP=int(fp),FN=int(fn),TP=int(tp)))
    return rows,audit


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--workers',type=int,default=6)
    ap.add_argument('--repeats',type=int,default=1000)
    ap.add_argument('--out-dir',type=Path,default=ROOT/'results')
    args=ap.parse_args();assert 1<=args.repeats<=1000
    args.out_dir.mkdir(parents=True,exist_ok=True)
    protocol=json.loads((ROOT/'protocol.json').read_text())
    assert sha(ROOT/'paired_rf_core.py')==protocol['paired_rf_core_sha256']
    protocol.update(repeats_executed=args.repeats,workers=args.workers,
        versions={'python':platform.python_version(),'numpy':np.__version__,
                  'pandas':pd.__version__,'sklearn':sklearn.__version__},
        run_script_sha256=sha(__file__))
    assert [np.__version__,pd.__version__,sklearn.__version__]==['2.5.4','2.2.3','1.9.1']
    summaries=[];verification=[];started=time.monotonic()
    for period in ['1978_2024','1978_2008']:
        source=ROOT/'data'/f'japan_{period}.csv'
        assert sha(source)==protocol['snapshot_sha256'][source.name]
        table=pd.read_csv(source,keep_default_na=False)
        assert table.region.eq('japan').all() and not table.WSM_ID.duplicated().any()
        assert np.isfinite(table[FEATURES].to_numpy(float)).all()
        assert np.array_equal(table.label_ge_4.to_numpy(int),table.MW.to_numpy(float)>=4)
        expected=protocol['counts'][period]
        assert [len(table),int(table.label_ge_4.sum())]==expected
        base=pd.read_csv(ROOT/'reference'/period/'scores_per_repeat.csv')
        audits=pd.read_csv(ROOT/'reference'/period/'split_audit.csv')
        for seed in [0,1,29]:
            z,y,tr,te,audit=split_rows(table,seed)
            ref=audits[audits.seed==seed].iloc[0]
            assert all(str(v)==str(ref[k]) for k,v in audit.items())
            scores,_=fit_model(z[FEATURES].to_numpy(float),y,tr,te,seed)
            row=base[base.seed==seed].iloc[0]
            assert max(abs(scores[k]-row[k]) for k in METRICS)<1e-12
            verification.append({'period':period,'seed':seed,'baseline_scores_match':True})
        print(f'BASELINE VERIFIED {period}: {expected}',flush=True)
        directory=args.out_dir/period;directory.mkdir(exist_ok=True)
        scores=[];splits=[]
        jobs=Parallel(n_jobs=args.workers,return_as='generator_unordered',backend='loky',batch_size=1)(
            delayed(worker)(table,seed) for seed in range(args.repeats))
        for rows,audit in jobs:
            scores.extend(rows);splits.append(audit)
            if len(splits)%50==0 or len(splits)==args.repeats:
                pd.DataFrame(scores).sort_values(['seed','pair']).to_csv(directory/'scores_per_repeat.csv',index=False)
                pd.DataFrame(splits).sort_values('seed').to_csv(directory/'split_audit.csv',index=False)
                print(f'{period}: {len(splits)}/{args.repeats}; {len(scores)} refits; {time.monotonic()-started:.1f}s',flush=True)
        for pair,group in pd.DataFrame(scores).groupby('pair'):
            group=group.merge(base[['seed','Accuracy']].rename(columns={'Accuracy':'full_nine_accuracy'}),on='seed',validate='many_to_one')
            delta=group.full_nine_accuracy-group.Accuracy
            row=dict(period=period,pair=pair,removed_1=group.removed_1.iloc[0],
                removed_2=group.removed_2.iloc[0],n_total=len(table),n_positive=expected[1],
                n_negative=len(table)-expected[1],n_repeats=len(group),
                full_nine_accuracy_mean=float(group.full_nine_accuracy.mean()),
                full_nine_accuracy_std=float(group.full_nine_accuracy.std(ddof=1)),
                accuracy_drop_mean=float(delta.mean()),accuracy_drop_std=float(delta.std(ddof=1)),
                accuracy_drop_positive_share=float((delta>0).mean()),
                accuracy_drop_zero_share=float((delta==0).mean()),
                accuracy_drop_negative_share=float((delta<0).mean()))
            for m in METRICS:
                row[m+'_mean']=float(group[m].mean());row[m+'_std']=float(group[m].std(ddof=1))
            summaries.append(row)
        pd.DataFrame(summaries).to_csv(args.out_dir/'summary.csv',index=False)
    protocol['duration_seconds']=time.monotonic()-started
    (args.out_dir/'run_config.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2))
    (args.out_dir/'baseline_verification.json').write_text(json.dumps(verification,indent=2))
    print(f'COMPLETE {len(summaries)} tasks; {20*args.repeats} seven-feature forests',flush=True)


if __name__=='__main__':
    main()
