#!/usr/bin/env python3
"""Japan Mw>=4 association tests for the ten existing CL feature pairs.

Inputs are the unchanged all-nine-valid RF tables. Depth is in km and
orientation differences are in degrees; no CL grayscale mapping is applied.
"""
from pathlib import Path
import argparse
import hashlib
import json
import platform
import time

import numpy as np
import pandas as pd
import sklearn
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, average_precision_score,
                             confusion_matrix)
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
FEATURES = ['MEAN_LOCAL_DEPTH', 'MEDIAN_LOCAL_DEPTH', 'STD_LOCAL_DEPTH',
            'STD_LOCAL', 'CONSISTENCY_R_MEAN', 'MAX_SPREAD_100KM_DEG',
            'MAX_ROTATION_DEG', 'REGIME_ENTROPY', 'REGIME_GINI']
CHANNELS = {1: 'MEAN_LOCAL_DEPTH', 2: 'STD_LOCAL_DEPTH',
            3: 'MAX_SPREAD_100KM_DEG', 4: 'STD_LOCAL',
            5: 'REGIME_ENTROPY', 6: 'REGIME_GINI',
            7: 'MEDIAN_LOCAL_DEPTH', 8: 'CONSISTENCY_R_MEAN',
            9: 'MAX_ROTATION_DEG'}
PAIRS = [(1, 2), (1, 7), (2, 7), (3, 4), (3, 8), (3, 9),
         (4, 8), (4, 9), (5, 6), (8, 9)]
METRICS = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC_AUC', 'PR_AUC']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def id_sha(values):
    return hashlib.sha256('\n'.join(sorted(map(str, values))).encode()).hexdigest()


def split_rows(table, seed):
    """Preserve the archived balance-before-split order exactly."""
    pos, neg = table[table.label_ge_4 == 1], table[table.label_ge_4 == 0]
    n = min(len(pos), len(neg))
    balanced = pd.concat([pos.sample(n=n, replace=False, random_state=seed),
                          neg.sample(n=n, replace=False, random_state=seed)])
    balanced = balanced.sample(frac=1, replace=False, random_state=seed).reset_index(drop=True)
    y = balanced.label_ge_4.to_numpy(dtype=int)
    tr, te = train_test_split(np.arange(len(y)), test_size=.5,
                              stratify=y, random_state=seed)
    assert not set(balanced.iloc[tr].WSM_ID) & set(balanced.iloc[te].WSM_ID)
    audit = {'seed': seed, 'n_selected': len(y), 'n_train': len(tr), 'n_test': len(te),
             'n_positive_train': int(y[tr].sum()), 'n_positive_test': int(y[te].sum()),
             'selected_wsm_ids_sha256': id_sha(balanced.WSM_ID),
             'train_wsm_ids_sha256': id_sha(balanced.iloc[tr].WSM_ID),
             'test_wsm_ids_sha256': id_sha(balanced.iloc[te].WSM_ID)}
    return balanced, y, tr, te, audit


def fit_model(x, y, tr, te, seed):
    # sqrt is the archived default: 1 feature/split for a pair, 3 for nine.
    model = RandomForestClassifier(n_estimators=100, max_depth=8,
        min_samples_leaf=5, max_features='sqrt', random_state=seed, n_jobs=1)
    model.fit(x[tr], y[tr])
    pred = model.predict(x[te])
    prob = model.predict_proba(x[te])[:, 1]
    tn, fp, fn, tp = confusion_matrix(y[te], pred, labels=[0, 1]).ravel()
    scores = dict(Accuracy=accuracy_score(y[te], pred),
        Precision=precision_score(y[te], pred, zero_division=0),
        Recall=recall_score(y[te], pred, zero_division=0),
        F1=f1_score(y[te], pred, zero_division=0),
        ROC_AUC=roc_auc_score(y[te], prob),
        PR_AUC=average_precision_score(y[te], prob),
        TN=int(tn), FP=int(fp), FN=int(fn), TP=int(tp))
    return scores, model.feature_importances_


def worker(table, seed):
    balanced, y, tr, te, audit = split_rows(table, seed)
    results = []
    for a, b in PAIRS:
        cols = [CHANNELS[a], CHANNELS[b]]
        score, imp = fit_model(balanced[cols].to_numpy(float), y, tr, te, seed)
        results.append(dict(seed=seed, pair=f'pair_{a:02}_{b:02}',
            feature_1=cols[0], feature_2=cols[1], importance_1=float(imp[0]),
            importance_2=float(imp[1]), **score))
    return results, audit


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--repeats', type=int, default=1000)
    ap.add_argument('--out-dir', type=Path, default=ROOT / 'results')
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((ROOT / 'protocol.json').read_text())
    assert args.repeats <= 1000
    protocol.update(repeats_executed=args.repeats, workers=args.workers,
        versions={'python': platform.python_version(), 'numpy': np.__version__,
                  'pandas': pd.__version__, 'sklearn': sklearn.__version__},
        run_script_sha256=sha(__file__))
    # Reproducing the existing baseline requires its original numerical runtime.
    assert protocol['versions']['numpy'] == '2.5.4'
    assert protocol['versions']['pandas'] == '2.2.3'
    assert protocol['versions']['sklearn'] == '1.9.1'
    verification = []
    summaries = []
    started = time.monotonic()
    for period in ['1978_2024', '1978_2008']:
        source = ROOT / 'data' / f'japan_{period}.csv'
        assert sha(source) == protocol['snapshot_sha256'][source.name]
        table = pd.read_csv(source, keep_default_na=False)
        assert table.region.eq('japan').all() and not table.WSM_ID.duplicated().any()
        assert np.isfinite(table[FEATURES].to_numpy(float)).all()
        assert (table.label_ge_4.to_numpy(int) == (table.MW.to_numpy(float) >= 4)).all()
        expected = protocol['counts'][period]
        assert [len(table), int(table.label_ge_4.sum())] == expected
        previous = ROOT / 'reference' / period
        base = pd.read_csv(previous / 'scores_per_repeat.csv')
        base_audit = pd.read_csv(previous / 'split_audit.csv')
        base_imp = pd.read_csv(previous / 'feature_importances_per_repeat.csv')
        # Three seeds check both exact membership and all baseline scores/importance.
        for seed in [0, 1, 29]:
            z, y, tr, te, audit = split_rows(table, seed)
            ref_a = base_audit[base_audit.seed == seed].iloc[0]
            assert all(str(v) == str(ref_a[k]) for k, v in audit.items())
            score, imp = fit_model(z[FEATURES].to_numpy(float), y, tr, te, seed)
            ref_s = base[base.seed == seed].iloc[0]
            assert max(abs(score[k] - ref_s[k]) for k in METRICS) < 1e-12
            ref_i = base_imp[base_imp.seed == seed].iloc[0]
            assert np.max(np.abs(imp - ref_i[FEATURES].to_numpy(float))) < 1e-12
            verification.append({'period': period, 'seed': seed,
                                 'baseline_scores_and_importance_match': True})
        directory = args.out_dir / period
        directory.mkdir(exist_ok=True)
        all_scores, audits = [], []
        print(f'BASELINE VERIFIED {period}; N={len(table)}; positives={expected[1]}', flush=True)
        tasks = Parallel(n_jobs=args.workers, return_as='generator_unordered',
                         backend='loky', batch_size=1)(
            delayed(worker)(table, seed) for seed in range(args.repeats))
        for scores, audit in tasks:
            all_scores.extend(scores)
            audits.append(audit)
            if len(audits) % 50 == 0 or len(audits) == args.repeats:
                pd.DataFrame(all_scores).sort_values(['seed','pair']).to_csv(
                    directory / 'scores_and_importances_per_repeat.csv', index=False)
                pd.DataFrame(audits).sort_values('seed').to_csv(directory / 'split_audit.csv', index=False)
                print(f'{period}: {len(audits)}/{args.repeats} seeds; {len(all_scores)} pair fits; '
                      f'{time.monotonic()-started:.1f}s', flush=True)
        scores = pd.DataFrame(all_scores).sort_values(['seed','pair'])
        for pair, group in scores.groupby('pair'):
            row = {'period': period, 'pair': pair, 'n_total': len(table),
                   'n_positive': expected[1], 'n_negative': len(table)-expected[1],
                   'n_repeats': len(group), 'feature_1': group.feature_1.iloc[0],
                   'feature_2': group.feature_2.iloc[0]}
            group = group.merge(base[['seed', 'Accuracy']].rename(columns={'Accuracy':'baseline_accuracy'}), on='seed', validate='many_to_one')
            for m in METRICS:
                row[m+'_mean'] = float(group[m].mean())
                row[m+'_std'] = float(group[m].std(ddof=1))
            delta = group.Accuracy-group.baseline_accuracy
            row.update(accuracy_above_chance_share=float((group.Accuracy>.5).mean()),
                delta_accuracy_vs_nine_mean=float(delta.mean()),
                delta_accuracy_vs_nine_std=float(delta.std(ddof=1)),
                delta_accuracy_vs_nine_positive_share=float((delta>0).mean()),
                importance_1_mean=float(group.importance_1.mean()),
                importance_1_std=float(group.importance_1.std(ddof=1)),
                importance_2_mean=float(group.importance_2.mean()),
                importance_2_std=float(group.importance_2.std(ddof=1)))
            summaries.append(row)
        pd.DataFrame(summaries).to_csv(args.out_dir / 'summary.csv', index=False)
    protocol['duration_seconds'] = time.monotonic()-started
    (args.out_dir / 'run_config.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2))
    (args.out_dir / 'baseline_verification.json').write_text(json.dumps(verification, indent=2))
    print(f'COMPLETE: {len(summaries)} tasks, {20*args.repeats} pair fits', flush=True)


if __name__ == '__main__':
    main()
