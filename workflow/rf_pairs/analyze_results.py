"""Audit completed fits and prepare compact, reproducible report data."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
NAMES = {'MEAN_LOCAL_DEPTH':'平均深度', 'STD_LOCAL_DEPTH':'深度标准差',
         'MEDIAN_LOCAL_DEPTH':'深度中位数', 'MAX_SPREAD_100KM_DEG':'最大局部方向差',
         'STD_LOCAL':'方向标准差', 'CONSISTENCY_R_MEAN':'方向一致性R',
         'MAX_ROTATION_DEG':'震中方向偏差', 'REGIME_ENTROPY':'机制熵',
         'REGIME_GINI':'机制Gini'}


def main():
    summary = pd.read_csv(ROOT/'results/summary.csv')
    assert len(summary) == 20 and summary.n_repeats.eq(1000).all()
    expected_pairs = {x['pair'] for x in json.loads((ROOT/'protocol.json').read_text())['CL_pair_mapping']}
    audits = []
    baseline_rows = []
    for period in ['1978_2024', '1978_2008']:
        scores = pd.read_csv(ROOT/f'results/{period}/scores_and_importances_per_repeat.csv')
        splits = pd.read_csv(ROOT/f'results/{period}/split_audit.csv')
        old = pd.read_csv(ROOT/f'reference/{period}/split_audit.csv')
        pd.testing.assert_frame_equal(splits, old, check_dtype=False)
        assert len(scores)==10000 and set(scores.pair)==expected_pairs
        assert not scores.duplicated(['seed','pair']).any()
        assert scores.groupby('pair').seed.nunique().eq(1000).all()
        counts = scores[['TN','FP','FN','TP']].sum(axis=1)
        assert np.allclose(scores.Accuracy, (scores.TN+scores.TP)/counts, atol=1e-14)
        assert np.allclose(scores.importance_1+scores.importance_2, 1, atol=1e-12)
        assert scores.Accuracy.between(0,1).all() and scores.ROC_AUC.between(0,1).all()
        base = pd.read_csv(ROOT/f'reference/{period}/scores_per_repeat.csv')
        baseline_rows.append({'period':period,'accuracy_mean':float(base.Accuracy.mean()),
                              'accuracy_std':float(base.Accuracy.std(ddof=1))})
        for pair, group in scores.groupby('pair'):
            row=summary[(summary.period==period)&(summary.pair==pair)].iloc[0]
            assert abs(group.Accuracy.mean()-row.Accuracy_mean)<1e-12
            assert abs(group.Accuracy.std(ddof=1)-row.Accuracy_std)<1e-12
            b=group.merge(base[['seed','Accuracy']],on='seed',suffixes=('','_baseline'))
            assert abs((b.Accuracy-b.Accuracy_baseline).mean()-row.delta_accuracy_vs_nine_mean)<1e-12
        audits.append({'period':period,'fitted_pair_forests':len(scores),
                       'all_1000_splits_match_original_baseline':True,
                       'metrics_and_importance_audited':True})
    full=pd.read_csv(ROOT/'data/japan_1978_2024.csv').set_index('WSM_ID')
    early=pd.read_csv(ROOT/'data/japan_1978_2008.csv').set_index('WSM_ID')
    common=[k for k in NAMES]+['MW','label_ge_4']
    assert np.allclose(full.loc[early.index,common].to_numpy(float),early[common].to_numpy(float),atol=1e-12)
    view=[]
    for row in summary.to_dict('records'):
        row['label']=NAMES[row['feature_1']]+'＋'+NAMES[row['feature_2']]
        view.append(row)
    d={'rows':view,'baselines':baseline_rows,'verification':audits,
       'notes':['Mw≥4 vs Mw<4; original Japan rectangle and all-nine-valid events.',
       'Each period is fitted separately; 1000 balanced random 50/50 splits.',
       'SD is between repeated seeds; no independent-sample significance claim.',
       'Within-pair importance is normalized to sum one, not directly comparable to nine-feature importance.',
       'sqrt max_features gives one split candidate for two inputs, three for nine inputs.',
       'Supplementary exploratory analysis proposed after CL results; no CL test-based RF tuning.',
       'The two periods share events and are not independent replications.']}
    (ROOT/'report_data.json').write_text(json.dumps(d,ensure_ascii=False,indent=2))
    lines=['日本CL十组双指标组合：RF区分实验结果','',
           '主任务：Mw≥4与Mw<4。各时期分别训练。准确率为均值±重复运行样本标准差（%）。',
           '正负类先按1:1抽样，再分层随机对半划分训练与测试。种子0—999，共1000次。',
           '100棵树，最大深度8，每叶至少5个样本，max_features=sqrt。',
           '每对指标与对应时期九指标基线使用完全相同的抽样事件和训练测试划分。','']
    for period in ['1978_2024','1978_2008']:
        b=next(x for x in baseline_rows if x['period']==period)
        lines += [period.replace('_','—'),f'九指标基线准确率：{b["accuracy_mean"]*100:.2f} ± {b["accuracy_std"]*100:.2f}%',
                  '组合：准确率；超过50%的重复比例；相对九指标基线的平均差值（百分点）']
        for r in view:
            if r['period']==period:
                lines.append(f'{r["label"]}：{r["Accuracy_mean"]*100:.2f} ± {r["Accuracy_std"]*100:.2f}%；'
                    f'{r["accuracy_above_chance_share"]*100:.1f}%；{r["delta_accuracy_vs_nine_mean"]*100:+.2f}')
        lines.append('')
    lines += ['解释边界：重复比例不是p值，标准差不是独立样本置信区间。组内重要性是二指标相对分配，不能与九指标重要性直接比较。',
              '两时期样本存在包含关系，不是两份独立数据。本轮为看过CL结果后的探索性补充分析，不将RF排序解释为CL增益排序。',
              '本轮未训练任何新CL模型，也未新增单指标RF实验。',
              '复现：python run_pairs.py --workers 6；随后 python analyze_results.py。',
              '依赖：Python 3.12、numpy==2.5.4、pandas==2.2.3、scikit-learn==1.9.1。']
    (ROOT/'实验结果说明.txt').write_text('\n'.join(lines)+'\n')
    (ROOT/'verification.json').write_text(json.dumps({'status':'complete','checks':audits,
        'new_pair_fits':20000,'baseline_check_refits':6,'early_subset_values_match_full':True},indent=2))
    print('\n'.join(lines))


if __name__=='__main__':
    main()
