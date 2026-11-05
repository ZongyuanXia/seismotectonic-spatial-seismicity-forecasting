# ETAS单窗口试预测（2026-10-29）

本次任务为一次450天预测，固定采用验证集第0个样本：2009-01-01至2010-03-26。最终结果为 `results/validation_00000_20090101_v2/`。它不是整个训练/验证/测试集的ETAS均值。

## 本地保存位置

正式代码目录：
`/Users/xiazongyuan/1 课题/7 地震震级预测/xzy-小论文1-文件汇总/代码与结果（归档）/九项指标重构归档/baseline/etas/`

本次已在本地执行，不需要再上传云端或下载结果。`.venv/`只在本次工作目录中，未复制进正式代码目录；正式目录的代码可在安装依赖后运行。

## 方法与本次边界

- 使用Mizrahi等作者开源实现的时空ETAS，背景位置可变、触发参数共享。作者原始归档及SHA256保留在vendor和source_manifest.json中。
- 拟合只使用2009-01-01之前的事件。1978-01-01起作为辅助历史，1989-02-02起进入主拟合期。模拟触发源只用预测起点前4050天。
- 沿用当前处理后的Mw>=4目录；目录完整性与震级转换尚未完成正式审查。本次不降低阈值，也不以目标真实事件调整参数。
- 背景平滑带宽固定25 km，本次未调优。背景模拟适配器按P_background精确加权，采用25 km经纬度换算的Gaussian偏移和区域内拒绝采样；不调用作者原函数中无参重置随机种子的路径。
- 逐代生成背景地震和未来余震，2048个种子（202610290+i）。模拟震级上界不截断。预测值为网格中至少一次目标地震的模拟比例。
- 固定概率阈值：0.1%、0.5%、1%、2%、5%、10%，全部报告；相同实际报警面积：0.5%、1%、2%、3%。未依据本窗口成绩选择阈值。
- 复用当前ConvLSTM对称评分器d=3：P target_only，R prediction_only，CSI both。另存完整四种protocol及未扩张指标。
- 作者空间触发积分使用无限平面，观测父事件仅限日本区域，边界影响仍需后续处理。拟合分支比触及0.99约束上限，不能将本次试算当作完成模型适用性验证。
- 初次拟合的线搜索告警通过同一训练数据的数值精修处理；v2的最终精修步骤均需优化成功，未使用目标评分反馈。

## 代码文件

- fit_etas_trial.py：预起点数据筛选、初次约束EM拟合，输出v1拟合状态。
- refine_etas_fit.py：从v1继续，修正过紧求解容差，保存v2参数及诊断。
- predict_etas_trial.py：生成2048套未来目录、概率图、概率阈值及与PI的同面积评分。
- finish_etas_trial.py：复算结果、核验种子、生成空间图和中文说明。
- check_etas_trial.py：5项接口核验。
- region_meta.json、reference/pi_validation/：固定网格和同窗口PI参照。目标数据不包含在PI分数中。
- requirements_ETAS.txt、vendor/：依赖版本、作者原始实现及归档。

脚本目前只针对上述单个预测起点；批量起点需要另行编写入口，不能改日期后沿用旧结果目录。

## 如果需要上传云端重现

上传完整 `baseline/etas/` 代码目录（包含vendor、reference、region_meta.json；无需上传results和代码/结果zip），放到：
`/share/home/22460734/project_MP_1978_2024/baseline/etas/`

云端还必须存在：

- `/share/home/22460734/project_MP_1978_2024/cl10_mapping_v2/` 中的manifest.json、sample_index.csv、usgs_event_audit.csv、usgs_daily.npz。
- `/share/home/22460734/project_MP_1978_2024/training/` 中的cl10_data.py、spatial_symmetric.py、evaluate_cl10_symmetric.py。

本次运行环境为Python 3.12。作者此版本声明要求Python>=3.12；已有旧版云Python环境不能直接假定兼容。安装NumPy、pandas、SciPy、GeoPandas、Shapely、PyProj、SeismoStats、Matplotlib等，版本见requirements_ETAS.txt。

在没有旧输出的独立云端代码目录运行：

```bash
cd /share/home/22460734/project_MP_1978_2024/baseline/etas
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements_ETAS.txt
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$PWD/.mplconfig"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

.venv/bin/python check_etas_trial.py --root /share/home/22460734/project_MP_1978_2024
.venv/bin/python fit_etas_trial.py --root /share/home/22460734/project_MP_1978_2024
.venv/bin/python refine_etas_fit.py
.venv/bin/python predict_etas_trial.py --root /share/home/22460734/project_MP_1978_2024 --simulations 2048 --workers 4
.venv/bin/python finish_etas_trial.py --root /share/home/22460734/project_MP_1978_2024
```

已存在的拟合或预测不会自动覆盖。如需重跑，请为所有步骤一致指定新的`--out`，并将refine的`--from-fit`指向新初次拟合目录。不要在本地已完成的目录直接重复上述默认命令。

## 云端运行后应下载的文件

从最终 `results/validation_00000_20090101_v2/` 下载：

**必要结果：**
- ETAS单窗口结果说明.md
- evaluation.csv、evaluation_protocols.csv
- diagnostics.json、verification.json
- fit_manifest.json、fit_config.json、optimizer_steps.json、fit.log
- prediction_manifest.json、simulation_counts.csv
- ETAS_single_forecast.png、ETAS_single_forecast.pdf

**空间分析需要：**prediction.npz（建议一并下载）。

**继续模拟或追溯时需要：**fit.pkl、historical_catalog.csv。pickle只读取本次可信代码自己生成的文件。

作者源码参考：https://github.com/lmizrahi/etas 。数据完整性方法参考：https://doi.org/10.1029/2021JB022379 。本次改动和核验细节以source_manifest.json及verification.json为准。
