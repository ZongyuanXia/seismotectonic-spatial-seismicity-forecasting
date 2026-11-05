"""Audit drop-pair refits and join the already completed pair-only tests."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from paired_rf_core import FEATURES, CHANNELS, PAIRS

ROOT=Path(__file__).resolve().parent
NAMES={'MEAN_LOCAL_DEPTH':'平均深度','STD_LOCAL_DEPTH':'深度标准差',
       'MEDIAN_LOCAL_DEPTH':'深度中位数','MAX_SPREAD_100KM_DEG':'最大局部方向差',
       'STD_LOCAL':'方向标准差','CONSISTENCY_R_MEAN':'方向一致性R',
       'MAX_ROTATION_DEG':'震中方向偏差','REGIME_ENTROPY':'机制熵','REGIME_GINI':'机制Gini'}


def main():
    summary=pd.read_csv(ROOT/'results/summary.csv')
    only=pd.read_csv(ROOT/'pair_only_reference.csv')
    assert len(summary)==20 and summary.n_repeats.eq(1000).all()
    assert not summary.duplicated(['period','pair']).any()
    expected_pairs={f'pair_{a:02}_{b:02}' for a,b in PAIRS}
    checks=[]
    for period in ['1978_2024','1978_2008']:
        scores=pd.read_csv(ROOT/f'results/{period}/scores_per_repeat.csv')
        splits=pd.read_csv(ROOT/f'results/{period}/split_audit.csv')
        original=pd.read_csv(ROOT/f'reference/{period}/split_audit.csv')
        pd.testing.assert_frame_equal(splits,original,check_dtype=False)
        assert len(scores)==10000 and set(scores.pair)==expected_pairs
        assert not scores.duplicated(['seed','pair']).any()
        assert scores.groupby('pair').seed.nunique().eq(1000).all()
        assert scores.remaining_count.eq(7).all()
        counts=scores[['TN','FP','FN','TP']].sum(axis=1)
        assert np.allclose((scores.TN+scores.TP)/counts,scores.Accuracy,atol=1e-14)
        assert scores.ROC_AUC.between(0,1).all()
        baseline=pd.read_csv(ROOT/f'reference/{period}/scores_per_repeat.csv')
        for r in scores.itertuples():
            imp=json.loads(r.remaining_importance)
            assert set(imp)==set(FEATURES)-{r.removed_1,r.removed_2}
            assert abs(sum(imp.values())-1)<1e-12
        for pair,g in scores.groupby('pair'):
            g=g.merge(baseline[['seed','Accuracy']],on='seed',suffixes=('','_nine'))
            s=summary[(summary.period==period)&(summary.pair==pair)].iloc[0]
            drop=g.Accuracy_nine-g.Accuracy
            assert abs(drop.mean()-s.accuracy_drop_mean)<1e-12
            assert abs(drop.std(ddof=1)-s.accuracy_drop_std)<1e-12
            assert abs((drop>0).mean()-s.accuracy_drop_positive_share)<1e-12
            assert abs(g.Accuracy.mean()-s.Accuracy_mean)<1e-12
        checks.append({'period':period,'refits':len(scores),
                       'all_splits_match_original':True,'all_retained_columns_verified':True})
    combined=summary.merge(only[['period','pair','Accuracy_mean','Accuracy_std']],
        on=['period','pair'],validate='one_to_one',suffixes=('','_pair_only'))
    rows=combined.to_dict('records')
    for r in rows:r['label']=NAMES[r['removed_1']]+'＋'+NAMES[r['removed_2']]
    combined['label']=[r['label'] for r in rows]
    combined.to_csv(ROOT/'results/combined_summary.csv',index=False,encoding='utf-8-sig')
    (ROOT/'report_data.json').write_text(json.dumps({'rows':rows,'checks':checks},ensure_ascii=False,indent=2))
    lines=['日本十组指标对删除实验','',
        '主任务：Mw≥4 vs Mw<4；两个时期分别训练，每组1000次重复。',
        '每次从原九项中删除对应两项，保留七项重新训练。',
        '下降=原九项准确率－删除后准确率；正值表示下降，负值表示删除后提高。',
        '下降幅度为同一种子配对差值，SD为这1000个差值的样本标准差，不是两组SD相减。',
        '正负类1:1抽样后随机分层对半划分；全部事件与划分均和原基线一致。',
        '100棵树、最大深度8、叶节点最少5样本、每次分裂候选特征固定为3。',
        '固定3项与原九指标sqrt的有效候选数相同，避免七指标自动sqrt变成2项。',
        '仅双指标结果沿用上一轮sqrt=1设置，作为不同问题的参考，不直接当作同模型容量的对照。','']
    for period in ['1978_2024','1978_2008']:
        period_rows=[r for r in rows if r['period']==period]
        lines += [period.replace('_','—'),
                  f'九指标准确率：{period_rows[0]["full_nine_accuracy_mean"]*100:.2f} ± {period_rows[0]["full_nine_accuracy_std"]*100:.2f}%',
                  '删除指标对：剩余七项准确率（%）；配对下降均值±SD（百分点）；下降重复比例']
        for r in period_rows:
            lines.append(f'{r["label"]}：{r["Accuracy_mean"]*100:.2f} ± {r["Accuracy_std"]*100:.2f}%；'
                f'{r["accuracy_drop_mean"]*100:+.2f} ± {r["accuracy_drop_std"]*100:.2f}；'
                f'{r["accuracy_drop_positive_share"]*100:.1f}%')
        lines.append('')
    lines += ['解释：下降大表示在这套RF及其余指标条件下，删除该组的损失较大。',
              '下降小可能由其余相关指标替代，不能直接认定被删指标不含信息。',
              '差值不是物理因果贡献。重复比例不是p值，SD不是独立样本置信区间。',
              '没有新增空间或时间分块，也没有修改数据、选择新地区或训练CL。',
              '两时期事件存在包含关系。本轮为已查看CL结果后的探索性补充分析。']
    (ROOT/'删除实验结果说明.txt').write_text('\n'.join(lines)+'\n')
    (ROOT/'verification.json').write_text(json.dumps({'status':'complete','new_seven_feature_fits':20000,
        'baseline_check_refits':6,'checks':checks},ensure_ascii=False,indent=2))
    print('\n'.join(lines))


if __name__=='__main__':main()
