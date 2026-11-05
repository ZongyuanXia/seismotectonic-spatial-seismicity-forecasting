# Seismotectonic Characteristics and Spatial Seismicity Forecasting

Version **2.0.0** for *Deep-Learning Forecasts of Spatial Seismicity Considering Hypocentral Depth and Stress Regime*. Code: MIT. Companion derived data: CC BY 4.0; upstream ETAS code and Natural Earth assets retain their respective terms.

The portable entry point is **`run_workflow.py`**. Historical scientific scripts remain unchanged, including their frozen hash checks. The adapters supply current paths and the final protocol; there is no need to edit the historical absolute-path defaults. Outputs go outside `software/` and `data/`.

## Directory layout

Keep the two release directories together:

```text
release/
  software/
    run_workflow.py
    reproduction/              portable paths, protocol, exporters and job adapters
    workflow/                  preserved scientific and plotting sources
  data/                        supplied processed inputs and archived results
  reproduced/                  created when a workflow executes
```

The new entry point reads `../data` directly. Running the earlier `install_data.py` is unnecessary. Override locations with `--data-root /absolute/path/data` and `--output-root /absolute/path/new_output`.

The v2.0.0 upload files are `seismotectonic-spatial-seismicity-forecasting-v2.0.0.zip` (software) and `seismotectonic-spatial-seismicity-derived-results-v2.0.0.zip` (data). Extract both into the same parent directory to obtain `software/` and `data/`.

The slim release excludes unreported early-period non-Japan RF outputs, redundant M1—M4 standard evaluations, superseded manuscript exporters and historical search/report runners. Formal models, input frames, full M1—M4 sensitivity records (including the formal setting), M5—M7 evaluations, selection provenance and the actual trainer sources recorded by the checkpoints are retained. Exact duplicate data are stored once; `data/file_aliases.json` restores byte-identical copies at required working paths automatically. No external local backup or symlink is required.

Except `check`, commands preview their plan by default. Add **`--execute`** to perform the requested step. No training, inference or rescoring was run while preparing this release.

## 1. Export from archived result snapshots

This currently works without neural checkpoints. It recalculates RF statistics from repeat records, derives the five tables, rebuilds the plot inputs, renders the five figures and compares them with the manuscript. Neural full-test statistics in this mode come from the archived compact summaries; this is not a new inference run.

Use Python 3.12 and the tested figure environment:

```bash
cd /absolute/path/release/software
python3 -m pip install -r requirements-figures-tested.txt
python3 run_workflow.py reproduce --source archived --execute
```

Outputs: `../reproduced/archived/tables/Table1.csv`—`Table5.csv`, matching `_display.csv` and `.tex` fragments; `figures/` with five PNGs; `statistics/` with sensitivity ranges, control contrasts and RF removal results; `verification.json`.

All table text is centered and 9 pt; table notes are 8 pt. Captions retain the JGR template style. Times New Roman is a separately installed font and is not redistributed. Exact PNG bytes depend on the font, OS and library versions. Required Natural Earth shapefiles are bundled in `data/cartopy/`; the renderer refuses online map downloads.

## 2. Reproduce from individual-sample records (recommended)

The companion data now include all seven formal model groups in `checkpoints/`, and the selected validation/test individual-sample records in `paper_results/`. After downloading and arranging only the software and data directories, run:

```bash
python3 run_workflow.py reproduce --source records --execute
```

This creates `../reproduced/records/`. The seven checkpoint hashes and dataset identities are checked without loading a neural model. Neural means and population SDs are recalculated from individual-sample records rather than copied from compact summaries. RF uses the supplied 1,000-repeat records; Table 5 uses the supplied six-window baseline records and the M2 records. Baseline scores are not recomputed in this mode.

Missing records cause an error. The final M1 validation delivery is identified by the prediction-array hash pinned in the manuscript receipt; two historical same-checkpoint deliveries are retained only in the original local download, not used as final results. Other conflicting deliveries are rejected. Selection provenance is in `data/paper_results/import_manifest.json`.

For a future separate cloud collection, the optional maintenance command is `python3 run_workflow.py import-cloud --cloud-return /path/formal_artifacts_for_archive --execute`. It is unnecessary for the current complete data release.

## 3. Rerun models and regenerate results

This is a computational workflow for a user who chooses to rerun the experiments. Formal neural weights must first be available. It uses supplied processed WSM features and CL10 frames, refits the RF experiments, runs neural prediction/evaluation/sensitivity and PI/ETAS/smoothed baselines, then exports the tables and figures. It does not retrain the neural networks automatically.

Preview first:

```bash
python3 run_workflow.py full --source computed
```

Execute only on a suitable computing environment:

```bash
python3 run_workflow.py full --source computed \
  --python-rf /absolute/path/rf_environment/bin/python \
  --python-model /absolute/path/model_environment/bin/python \
  --python-figures /absolute/path/figure_environment/bin/python \
  --workers 4 --device cuda --execute
```

The three interpreter options default to the current Python, but the original RF and PyTorch environments differ. Do not assume that one environment can reproduce both. RF versions are in `requirements-rf-recorded.txt`; PyTorch/ETAS dependencies are in `requirements-training.txt` and `workflow/baseline/etas/requirements_ETAS.txt`. Original model receipts record Python 3.10.6 and PyTorch 2.1.0a0+b5021ba; the figure export was tested with Python 3.12.14 and `requirements-figures-tested.txt`. Full model execution has not been validated in this packaging pass.

Outputs: `../reproduced/computed/`. This mode copies model files but excludes archived evaluation results from the working project, so old records cannot be mistaken for newly computed results. Missing model files fail before any RF fitting starts. Use a new `--output-root` for a fresh calculation; source snapshots and released data are never overwritten.

The final steps are:

```text
RF accuracy -> pair experiments -> pair removal -> RF summaries
M1--M7 prediction -> individual-sample evaluation
M1--M4 fixed-weight sensitivity -> six-window map bundle
PI -> ETAS (5,000 simulations) -> smoothed baseline
M2 validation-recall threshold matching -> six test windows
five numeric/English LaTeX tables -> five figures -> comparison report
```

Every stage is also available independently: `rf-accuracy`, `rf-pairs`, `rf-ablation`, `rf-summary`, `predict`, `evaluate`, `sensitivity`, `baseline-pi`, `baseline-etas`, `baseline-compare`, `export`, `figures`, `verify`. Select model subsets with `--models M2 M3` and splits with `--splits test`. `full` always requires all seven models and both splits. Do not use subsets while claiming complete manuscript reproduction.

Full predictions occupy about 0.63 GiB per model per split, about 8.8 GiB for seven models and two splits. The workflow leaves these generated arrays in place. They are not part of the default cloud download set.

## Retrain at the final frozen settings (separate operation)

```bash
python3 run_workflow.py train --source computed --weights retrained \
  --output-root ../reproduced/retrained --device cuda --execute
```

This trains M1 first, then M2—M7, using initialization seed 1438401796, sample-order seed 60007756, and weights `[0.000175, 0.03, 1.0]`. M1 uses the archived dual-seed trainer with these explicitly recorded final weights. Added-channel models use the archived paired trainer. Controls use the original zero/100 input transform. The adapter hash and actual weights are written into the new run configurations. Test targets do not select weights or seeds.

Subsequent stages must use the same output root and `--weights retrained`. Retrained checkpoint bytes can differ across GPU/software environments. This command reproduces the final configuration, not the entire historical hyperparameter search. Necessary validation-selection records and original formal trainer implementations are retained; unreported historical search runners are local backups, outside the upload packages.

## Raw-input reconstruction and its boundary

The release supplies the processed observations and sparse frames used for the paper. Original full WSM/USGS downloads are identified by hashes but are not included as new online downloads. Sources: WSM2025, https://doi.org/10.5880/wsm.2025.001; USGS, https://doi.org/10.5066/F7MS3QZH.

Optional raw reconstruction is available with `build-rf --wsm /path/WSM_Database_2025.csv` and `build-cl10 --wsm /path/WSM_Database_2025.csv --usgs /path/usgs_monthly`. Use a separate `--output-root` and add `--execute` to run. Raw reconstruction can change provenance bytes in dataset manifests even when physical inputs match. The frozen-weight path requires the original processed-data identity and rejects mismatches; newly rebuilt inputs require their own verified training/data identity. Figure 1 uses the supplied filtered global WSM history, with counts and months recalculated by the exporter.

## Formal model mapping

| ID | Source identity | Input |
| --- | --- | --- |
| M1 | MAG, run_03 | MAG |
| M2 | pair_01_02, run_03 | MAG + mean depth + depth SD |
| M3 | pair_03_04, run_03 | MAG + maximum orientation deviation + s0 |
| M4 | pair_05_06, run_03 | MAG + H + D |
| M5 | A_c3_run_03 | MAG + two zero channels |
| M6 | B_pair_04_08_run_03 | MAG + 100 at depth/regime support |
| M7 | B_pair_03_08_run_03 | MAG + 100 at direction support |
| M8 / M9 / M10 | PI / ETAS / SSS | traditional/smoothed baselines |

M6 retains its actual trained channels `[0,4,8]`. M7 retains `[0,3,8]`; its support is reusable for the s0 pair because all stored s0/R support masks match. These control identities do not substitute R values for s0 in M3. Precision uses `target_only` tolerance; recall uses `prediction_only`. Their matching counts are kept separate.

## Validation and release status

Both the archived-result route and the individual-sample route reproduced the five tables and byte-identical PNGs. Fourteen synthetic orchestration tests cover paths, missing inputs, shared-file materialization, cloud-byte checks, final prediction identity, per-sample aggregation, distinct tolerance protocols and multiprocessing. The seven formal checkpoints/configurations and the individual-sample records are present and verified. No real neural training, prediction, full-sample scoring or baseline simulation was executed during preparation; fresh model execution remains a separate unperformed check.

Earlier release identifiers are https://doi.org/10.5281/zenodo.22142076 (software) and https://doi.org/10.5281/zenodo.22140383 (data). They are not the revised release identifiers. This staging directory has not been compressed, uploaded or published.

## Version identifiers

Software version 2.0.0: https://doi.org/10.5281/zenodo.23983140 (reserved identifier; resolves after publication). Companion data concept DOI: https://doi.org/10.5281/zenodo.22140382; select dataset version 2.0.0.
