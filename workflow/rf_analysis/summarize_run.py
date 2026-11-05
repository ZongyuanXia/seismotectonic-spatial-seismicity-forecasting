#!/usr/bin/env python3
"""Write a compact audit note for an archived-method RF replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()
    root = args.result_dir
    summary = pd.read_csv(root / "task_summary.csv")
    config = json.loads((root / "run_config.json").read_text(encoding="utf-8"))
    completed = summary[summary.status == "completed"]
    skipped = summary[summary.status == "skipped"]
    tiny = completed[completed.min_class_count < 10]
    japanese = completed[completed.region == "japan"].sort_values(["period", "threshold"])
    lines = [
        "# 归档方式 RF 首轮训练与数据核查",
        "",
        f"结果目录：`{root.name}`。重复次数：**{config['repeats']}**。",
        "",
        "## 运行口径",
        "",
        "每轮先按正负类中较少的一类做 1∶1 无放回抽样，再按标签分层将样本各半划入训练和测试，"
        "训练 100 棵树、最大深度 8、叶节点至少 5 条记录的随机森林。平衡抽样、划分和森林均使用同一轮编号作随机种子。"
        "九项原始指标与 Mw≥4/5/6 标签来自重构的 RF-WSM 表。此处只复现归档 RF 的统计关联分析。",
        "",
        "该测试集经过 1∶1 平衡抽样，Precision、F1 和 PR_AUC 仅代表这种人为比例；"
        "1000 次使用重叠事件，不能当成 1000 份独立数据。",
        "",
        "## 数据与执行检查",
        "",
        f"- 共 {len(summary)} 个时期、地区、阈值组合；完成 **{len(completed)}** 个，跳过 **{len(skipped)}** 个。",
        "- 输入表已检查九项特征有限、九项有效标记为真、标签与转换后 Mw 一致、事件 ID 在同一地区内不重复；"
        "每个组合的正负数与正式任务计数表一致。",
        "- 已保存每轮指标、九项特征重要性、训练与测试事件名单哈希；种子 0 另存完整事件名单。",
    ]
    for item in config["table_audits"]:
        lines.append(
            f"- {item['period']}：{item['rows']:,} 条九项均有效事件；"
            f"同地区重复 ISC_ID {item['duplicate_isc_id_within_region']} 个。"
        )
    if not skipped.empty:
        lines += ["", "无法按归档方法划分的组合：", "", "| 时期 | 地区 | Mw 阈值 | 正类 | 负类 |", "|---|---|---:|---:|---:|"]
        for _, row in skipped.iterrows():
            lines.append(f"| {row.period} | {row.region} | {row.threshold} | {row.n_positive} | {row.n_negative} |")
    if not tiny.empty:
        lines += [
            "",
            f"另有 **{len(tiny)}** 个已运行组合的较少类别不足 10 条。平衡后对半划分，"
            "训练集总量等于较少类别数量；在 `min_samples_leaf=5` 下，少于 10 条训练记录的树无法分裂。"
            f"这些组合的特征重要性全为 0，分数不能解释为九项特征的作用。"
            f"因此，40 个完成组合中有 {len(completed) - len(tiny)} 个至少具备分裂条件。",
        ]
    lines += [
        "",
        "## 日本地区首轮结果",
        "",
        "| 时期 | Mw 阈值 | 正/负事件 | Accuracy 均值±SD | ROC AUC 均值±SD | PR_AUC 均值±SD | 最重要特征（平均） |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in japanese.iterrows():
        lines.append(
            f"| {row.period} | {row.threshold} | {row.n_positive}/{row.n_negative} | "
            f"{row.Accuracy_mean:.3f}±{row.Accuracy_std:.3f} | "
            f"{row.ROC_AUC_mean:.3f}±{row.ROC_AUC_std:.3f} | "
            f"{row.PR_AUC_mean:.3f}±{row.PR_AUC_std:.3f} | {row.top1_feature} |"
        )
    lines += [
        "",
        "上表是描述性复算，尚不决定 CL 使用哪些通道。全地区逐任务汇总见 `task_summary.csv`，"
        "每轮原始结果见各任务子目录。",
        "",
        "## 下一步判读所需注意",
        "",
        "- 这轮把平衡抽样、训练/测试划分和森林种子同时改变，因此标准差反映三者合在一起的波动，"
        "不能单独归因于随机划分。",
        "- 随机拆分同一时期事件能检查归档分类任务的重复性，不能证明跨时期迁移能力。",
        "- 单一类别或极少数类别的地区阈值组合，应先标记为样本不足，不据此比较九项指标。",
        "",
    ]
    (root / "结果核查.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
