# ETAS：1000、2048、5000 次模拟比较

本次已获授权在本地运行。每个验证窗口生成 5000 次随机未来目录，前 1000 次和前 2048 次分别形成相应预测。六个验证窗口、4050 天历史、450 天预测期、Mw≥4.0、固定背景及触发参数均沿用原实验。没有重新拟合模型，也没有重新运行 PI 或 ConvLSTM。

随机种子固定为 `202610290 + validation_row * 2048 + simulation_index`。其中 2048 是原来的种子间隔，此处保持不变，避免改变随机序列。六个所选窗口行号相差 450，因而各窗口使用的 5000 个种子没有重叠。每个窗口前 2048 次模拟必须逐格复现原有 presence_count、event_count 和 half_presence_count，否则脚本停止。

各次数使用同一随机序列的前缀，属于配对检查。5000 次是更大样本估计，不是真实概率。单次嵌套比较不能证明严格收敛，也不能按验证集命中数挑选最有利的模拟次数。

## 本地存储

工作副本：`/Users/xiazongyuan/Documents/ChatGPT/论文撰写/outputs/ETAS_MC_sensitivity_20261030/`

项目归档副本：`/Users/xiazongyuan/1 课题/7 地震震级预测/xzy-小论文1-文件汇总/代码与结果（归档）/九项指标重构归档/baseline/etas/MC_sensitivity_1000_5000_20261030/`

## 文件

- `run_mc_sensitivity.py`：生成、核对并统计六窗口结果，支持按已完成窗口恢复。
- `summarize_mc_sensitivity.py`：读取完成结果，生成比较图和汇总说明。
- `run.log`：执行进度及异常。
- `results/manifest.json`：`complete=true` 才表示全部窗口和统计完成，记录随机种子、模型和文件哈希。
- `results/window_*/ETAS_N1000.npz`、`ETAS_N2048.npz`、`ETAS_N5000.npz`：网格概率、平均次数、整数原始计数及分半计数。
- `results/window_*/simulation_counts.csv`：每次模拟的事件数、种子等诊断。
- `results/window_*/complete.json`：单窗口完成状态、文件哈希及原有 2048 次复现状态。
- `results/per_window_hits_wide.csv`：各窗口、各报警面积下三种次数的同格命中数，以及已有 PI、ConvLSTM 命中数。
- `results/exact_hits_summary.csv`：各模型六窗口均值与标准差；包含同格及原 d=3 指标。
- `results/all_protocols.csv`：未扩展、单边扩展和双边扩展的全部原始计数。
- `results/probability_and_alarm_changes.csv`：1000/2048 相对 5000 次的概率变化和报警区重合程度。
- `results/split_half_stability.csv`：各次数内部两个不重叠半批次的报警区重合程度；比较的半批次大小为 500/1024/2500。
- `results/结果解读.md`：平均命中数、逐窗口命中数、报警区域重合程度与原 MAG 的比较。
- `results/MC_sensitivity_comparison.png`、`.svg`：比较图，SVG 为可编辑矢量版。
- `results/analysis_verification.json`：原有 PI、ConvLSTM、ETAS-2048 共 72 条评分的复现校验。

`Jaccard` 表示两个报警区域的交集除以并集，1 表示完全相同。分别记录按格数及实际面积计算的数值。

同面积评估使用原有概率排序、并列种子 2901 和实际面积预算 0.5%、1%、2%、3%。容差评分的 P/R/CSI 使用原先不同扩展组合，不当成同一混淆矩阵推导。均值先按窗口计算，再取六窗口算术平均。

## 复现命令

依赖现有本地 ETAS 虚拟环境，以及项目原有 baseline、training、数据集和模型拟合文件。脚本运行成功后，相同代码再次运行会校验并复用已完成窗口。

```bash
cd '/Users/xiazongyuan/Documents/ChatGPT/论文撰写'
MPLCONFIGDIR='/Users/xiazongyuan/Documents/ChatGPT/论文撰写/etas_trial_20261029/.mplconfig' \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  '/Users/xiazongyuan/Documents/ChatGPT/论文撰写/etas_trial_20261029/.venv/bin/python' -u \
  '/Users/xiazongyuan/Documents/ChatGPT/论文撰写/outputs/ETAS_MC_sensitivity_20261030/run_mc_sensitivity.py' \
  --root '/Users/xiazongyuan/1 课题/7 地震震级预测/xzy-小论文1-文件汇总/代码与结果（归档）/九项指标重构归档' \
  --out '/Users/xiazongyuan/Documents/ChatGPT/论文撰写/outputs/ETAS_MC_sensitivity_20261030/results' \
  --workers 4
```

本次不需要上传云平台或下载云端文件，全部结果保存在本地。若汇报时只保留小文件，保留两个脚本、README、run.log、results 中所有 CSV/JSON/MD 和比较图；预测数组 NPZ 可作为可选的大文件另存，但后续重新比较报警位置时需要它们。
