# 验证集六个不重叠目标窗口的完全同格命中统计

## 本轮范围

| 验证集索引 | 目标开始 | 目标结束（含） |
|---:|---|---|
| 0 | 2009-01-01 | 2010-03-26 |
| 450 | 2010-03-27 | 2011-06-19 |
| 900 | 2011-06-20 | 2012-09-11 |
| 1350 | 2012-09-12 | 2013-12-05 |
| 1800 | 2013-12-06 | 2015-02-28 |
| 2250 | 2015-03-01 | 2016-05-23 |

每个目标期450天，目标期之间不重叠，历史输入仍重叠，不能把它们称为严格独立试验。这是与此前PI、ConvLSTM非重叠评估完全一致的选窗；没有为了覆盖2016年底另外移动最后一个窗口。

## 已在本地完成的部分

- PI：采用此前已计算的六张分数图。
- ETAS：保持原试验的2009年前拟合参数、空间背景率、25 km背景平滑、作者逐代模拟及每窗口2048次模拟。复用第0个窗口，补算其余五个。
- 对每个新预测起点，从事件审计表重新截取此前4050天的真实地震记录；较晚窗口可以使用此前已经发生的验证期地震。不会用目标期真实事件更新这次450天预测，也不重新估计ETAS参数。
- ETAS随机种子为202610290 + 验证集索引×2048 + 模拟编号；第0个窗口保留原有种子与预测。
- ETAS仍有目录完整性、区域边界、分支比触及0.99上限和模拟离散误差等限制，属于沿用原设置的试比较。

## 本地保存与上传

代码正式目录：

```text
/Users/xiazongyuan/1 课题/7 地震震级预测/xzy-小论文1-文件汇总/代码与结果（归档）/九项指标重构归档/baseline/validation_six_windows/
```

上传整个`Validation_six_windows_20261030.zip`，其中包含`validation_six_windows/`文件夹、六窗口ETAS/PI参考图和固定模型身份信息。解压至：

```text
/share/home/22460734/project_MP_1978_2024/baseline/validation_six_windows/
```

本地ETAS模拟记录另存于baseline/etas/results/validation_six_frozen_2009_20261030，保留在本地即可；云端无需重新运行ETAS。

## 云端运行

使用此前ConvLSTM的Python环境（Python>=3.10、PyTorch、NumPy）。导出和统计过程不需要SciPy、ETAS、GeoPandas或Matplotlib。

项目中原有的`training/`和`cl10_mapping_v2/`保持原位置，模型为`runs/mag_w0_000175/run_03/`。脚本也会搜索其他目录，但必须匹配已选定权重：

```text
468aa2919415eae1131c30cbc71f7cc8ec40a9e7d522638126ba4741a6c471ac
```

激活原环境后执行：

```bash
cd /share/home/22460734/project_MP_1978_2024/baseline
unzip Validation_six_windows_20261030.zip
cd validation_six_windows
export PYTHONDONTWRITEBYTECODE=1
python -u export_mag_six_windows.py --device auto
```

只核对输入而不推理，可用：

```bash
python export_mag_six_windows.py --dry-run
```

仅使用已有最佳权重推理六个窗口，不训练模型。默认输出若已存在会停止，重跑请使用新的`--out`目录。

## 运行后下载一个文件

```text
/share/home/22460734/project_MP_1978_2024/baseline/validation_six_windows/results/MAG_ETAS_PI_validation_six.zip
```

包中包含：

- `exact_hits_by_window.csv`：6窗口×3模型×4面积，共72行；含完全同格命中数、FP、FN、未扩张P/R和原d=3口径P/R/CSI及扩张面积。
- `exact_hits_summary.csv`：3模型×4面积，共12行；含命中格数均值、总命中格次、窗口标准差、逐窗口P/R均值及按有震格次数加权的召回率。
- `exact_hits_summary_wide.csv`：四档面积的三个模型汇总，便于直接对照。
- `exact_hits_area_gains.csv`：相邻面积档位增加的总命中格次和平均命中格数，共9行。
- `all_protocols.csv`：完整四种扩张组合的TP/FP/FN/TN，共288行。
- `native_MAG_window_check.csv`：六窗口MAG在默认灰度100阈值下与已有归档评分的复现检查。
- `maps/`：六张原始MAG预测图和六套三模型参考图、目标图与面积，供本地核查和绘图。
- `comparison_report.md`、`manifest.json`：可读汇总、完成状态、代码/数据/输入帧/模型及结果哈希。

不需要下载模型权重或2473个日滑动窗口的大型预测数组。

## 统计口径

面积档位固定0.5%、1%、2%、3%，按实际km²预算选取高分网格。并列排序、面积离散误差处理与上一轮一致。

完全同格命中数取未扩张列联表的TP。每个窗口分别统计；汇总总数的单位是“格次”，同一空间位置在不同时间窗口可以重复出现。平均命中格数为六个TP的算术平均。

完全同格Recall先对每窗口计算TP/该窗口真实有震格数，再求六窗口均值。它与总TP除以总真实有震格次数不同，后者单独列为`pooled_exact_recall`。

P、R、CSI的原d=3口径仍分别对应target_only、prediction_only、both；它们不共用同一个TP。相同原始报警面积也不代表相同扩张面积，需一起解释。

## 脚本用途

- `export_mag_six_windows.py`：云端六窗口MAG推理、三模型比较及结果打包。
- `common_six.py`：公共读取与统计；本地可只统计已完成的ETAS和PI。
- `core_one_window.py`：原单窗口脚本的原样副本，复用经过核对的面积排序、评分和模型身份核对函数。
- `run_etas_six_windows.py`：本轮本地ETAS补算入口，依赖现有baseline/etas及本地ETAS环境；无需在云端运行。

本地检查与执行状态见`delivery_check.json`。MAG六窗口推理在用户执行云端脚本后才完成，当前ETAS/PI本地统计不代表三模型最终汇总。
