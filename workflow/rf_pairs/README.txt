日本地区：对应CL十组双指标的随机森林区分实验

范围
仅日本原矩形研究区；主任务Mw>=4 vs Mw<4。
1978-2024与1978-2008分别训练，不合并评价。保留原九指标均有效的
事件集合，不因某一指标对条件较宽而增加样本，也不重新计算特征。
两时期样本有包含关系，不是两个独立样本的重复验证。

实验
十个指标对与CL模型编号一一对应，见protocol.json。
每次先无放回平衡正负类，再分层随机对半划分训练和测试。
种子0-999；每组1000次，共20000个新RF模型。
100棵树、最大深度8、最小叶样本数5、max_features=sqrt。
双指标每次分裂候选1项；九指标参照候选3项。
模型内n_jobs=1，只在种子之间并行，不改变随机森林结果。
不做空间或时间分块，不调整模型参数，不新增CL训练。

数据与核对
data/是从原训练表按region=japan原顺序提取的快照。
protocol.json记录原完整数据文件SHA256、快照SHA256和样本量。
reference/保留对应时期九指标基线1000次成绩、重要性及划分哈希。
正式运行前每时期复现种子0、1、29的全部九指标成绩和重要性。
分析时核对每时期全部1000个划分与原基线完全相同。
对照重要性仅在二指标组内归一化，不能与九指标重要性直接比较。

本地复算（在交付文件夹内）
cd /Users/xiazongyuan/Documents/ChatGPT/论文撰写/rf_japan_cl_pairs_20261102
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python run_pairs.py --workers 6
.venv/bin/python analyze_results.py
以上命令会重跑完整1000次，不自动跳过旧结果。
建议复算时加 --out-dir results_rerun 保留现有结果；分析脚本默认读取results。

云端复算
上传本文件夹中的run_pairs.py、analyze_results.py、requirements.txt、
protocol.json，以及完整data/、reference/。无需上传旧results/或Excel。
目标目录示例：/share/home/22460734/project_MP_1978_2024/training/rf_japan_cl_pairs/
cd /share/home/22460734/project_MP_1978_2024/training/rf_japan_cl_pairs/
python -m pip install -r requirements.txt
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python run_pairs.py --workers 6
python analyze_results.py
每50个种子写入一次中间结果。

必须下载
results/summary.csv
results/run_config.json
results/baseline_verification.json
verification.json
report_data.json
实验结果说明.txt

可选下载（完整逐次记录与复核）
results/1978_2024/scores_and_importances_per_repeat.csv
results/1978_2024/split_audit.csv
results/1978_2008/scores_and_importances_per_repeat.csv
results/1978_2008/split_audit.csv
无须下载模型检查点，本实验不持久化单个森林。

输出解释
准确率均值、种子间样本标准差(ddof=1)、超过50%的重复比例。
同时保留Precision、Recall、F1、ROC_AUC、average precision。
PR_AUC字段沿用原脚本命名，实际计算为average_precision_score。
相对九指标差值按同一种子配对计算，不是独立样本效应。
重复比例不是p值，标准差不是独立样本置信区间。
本实验在查看CL结果后提出，属于探索性补充分析。

Excel
交付的Excel有两个时期工作表，列出准确率、SD、配对差值和组内重要性。
重建Excel需要Codex内置artifact-tool Node环境，不是训练依赖。
