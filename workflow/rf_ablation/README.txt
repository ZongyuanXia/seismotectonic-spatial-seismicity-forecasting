日本十组CL指标对的RF删除实验

问题：从原九项输入中删去某两个指标，剩余七项重新训练后，准确率下降多少？
本轮不是只保留两个指标，也不是对原训练模型在测试时简单清零。
主任务Mw>=4 vs Mw<4。1978-2024与1978-2008分别训练。
原地区、原九项均有效的样本集合不变，种子0-999，1000次重复。
先正负1:1无放回抽样，再分层随机对半划分训练和测试。
每个删除组、原九指标基线、前一轮双指标实验共用相同抽样与划分。

模型
100棵树、最大深度8、叶节点最少5样本。
七指标模型max_features固定为3，与原九指标sqrt的有效候选数相同。
若直接对七指标使用sqrt将变成2，故本轮不采用该设置。
原九指标已保存的1000次成绩用作基线，另复现种子0、1、29确认一致。
前轮双指标模型使用sqrt=1，与本轮有效候选数不同，不能将两列直接解释为纯信息量比较。

输出
下降=原九指标准确率－七指标准确率；正值下降，负值提高。
差值按同种子配对计算，记录均值、样本标准差(ddof=1)及正/零/负比例。
同时保存七项模型Precision、Recall、F1、ROC AUC和average precision。
PR_AUC字段沿用旧脚本名称，实际计算为average_precision_score。
逐次记录含剩余七项的特征名及其重要性，核查删除是否正确。
重复比例不是p值；SD不是独立样本置信区间。
相关指标可以替代被删指标，删除损失小不等于该指标本身没有信息。
结果不解释为因果贡献，不要求RF删除损失排序与CL通道增益排序一致。
两时期样本存在包含关系。本轮为查看CL结果后的探索性分析。

本地复算
cd /Users/xiazongyuan/Documents/ChatGPT/论文撰写/rf_japan_cl_pair_ablation_20261102
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python run_ablation.py --workers 6
.venv/bin/python analyze_ablation.py
将完整重跑1000次，不自动跳过已有结果。
若要保留已交付结果，可将该文件夹复制到新目录后复算。

云端上传集
run_ablation.py、paired_rf_core.py、analyze_ablation.py、requirements.txt、
protocol.json、pair_only_reference.csv，以及完整data/、reference/。
目标示例：/share/home/22460734/project_MP_1978_2024/training/rf_japan_ablation/
cd /share/home/22460734/project_MP_1978_2024/training/rf_japan_ablation/
python -m pip install -r requirements.txt
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python run_ablation.py --workers 6
python analyze_ablation.py

必须下载
results/summary.csv
results/combined_summary.csv
results/run_config.json
results/baseline_verification.json
report_data.json
verification.json
删除实验结果说明.txt

可选下载
results/1978_2024/scores_per_repeat.csv
results/1978_2024/split_audit.csv
results/1978_2008/scores_per_repeat.csv
results/1978_2008/split_audit.csv
没有单个森林检查点，无须下载此类文件。

复现依赖
Python 3.12，numpy==2.5.4，pandas==2.2.3，scikit-learn==1.9.1。
Excel为本地交付汇总，重建需Codex内置artifact-tool Node环境，不属于训练依赖。
