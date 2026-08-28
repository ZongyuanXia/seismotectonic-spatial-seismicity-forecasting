# Seismotectonic Characteristics and Spatial Seismicity Forecasting

This repository contains the two-stage analysis for the manuscript
*Hypocentral Depth and Stress Regime Improve Deep-Learning Forecasts of Spatial
Seismicity*. Nine event-level seismotectonic characteristics describing
hypocentral depth, tectonic stress regime, and maximum horizontal stress
orientation are constructed from WSM2025 records. Random forests are then used
to evaluate magnitude-class discrimination across 10 tectonic regions.

The first stage constructs event-level seismotectonic characteristics and uses
random forests to examine their association with earthquake magnitude. The
second stage constructs seven-channel spatiotemporal samples and evaluates the
spatial identification ability of ConvLSTM models.

## Repository structure

```text
code/
  rf_association/   Data preparation, feature construction, RF experiments,
                    result summarization, and the RF pipeline entry point
  convlstm/          Seven-channel data generation, ConvLSTM training,
                    evaluation, robustness analysis, and pipeline entry point
  figures/          Scripts for Figures 1-7
data/
  raw/              Local WSM2025 and USGS inputs
  derived/          Regional records, event features, and magnitude labels
results/
  random_forest/    Per-repeat outputs and RF summary tables
  convlstm/         Repeated-run, area, and robustness summary tables
  figures/          Figures and figure-source CSV files
```

## Data preparation

Download the original data and place them at:

```text
data/raw/WSM_Database_2025.csv
data/raw/usgs_csv/usgsYYYY_MM.csv
```

- WSM2025: <https://doi.org/10.5880/WSM.2025.001>
- USGS Earthquake Catalog: <https://earthquake.usgs.gov/earthquakes/search/>

WSM2025 is required for the RF pipeline. The monthly USGS files are used only
for Figure 1(a) and Figure 1(e). Additional data details are provided in
[`data/README.md`](data/README.md).

## Environment

The code was tested with Python 3.12.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Random-forest pipeline

Preview the complete command sequence:

```bash
python code/rf_association/run_rf_pipeline.py --dry-run
```

Run the complete five-stage analysis:

```bash
python code/rf_association/run_rf_pipeline.py
```

The pipeline executes:

1. Filter WSM2025 records and extract the 10 study regions.
2. Construct the nine event-level seismotectonic characteristics.
3. Convert event magnitudes to approximate `Mw` and create threshold labels.
4. Run the 23 formal region-threshold RF tasks with 1,000 repeats.
5. Generate the RF summary tables used by the manuscript.

Existing RF run directories can be replaced with:

```bash
python code/rf_association/run_rf_pipeline.py --overwrite
```

Use `--from-step` and `--to-step` to run part of the workflow. The
`--n-repeats` option is intended for short tests in a separate working copy.

The formal tasks use the following magnitude thresholds:

| Region | `Mw` thresholds |
|---|---|
| Southern California | 4.0, 5.0 |
| Japan | 4.0, 5.0, 6.0 |
| New Zealand | 4.0, 5.0 |
| Greece | 4.0, 5.0 |
| Indonesia | 5.0, 6.0 |
| Iran | 5.0, 6.0 |
| Italy | 4.0, 5.0 |
| Mexico | 4.0, 5.0, 6.0 |
| Taiwan | 4.0, 5.0 |
| Central Chile | 4.0, 5.0, 6.0 |

## Figure generation

After the RF pipeline is complete, run:

```bash
python code/figures/fig_1b_wsm_japan.py
python code/figures/fig_1c_regions_wsm.py
python code/figures/fig_1d_wsm_counts.py
python code/figures/fig_2_3_random_forest.py
```

With the monthly USGS files in place, Figure 1(a) and Figure 1(e) are generated
with:

```bash
python code/figures/fig_1a_usgs_japan.py
python code/figures/fig_1e_monthly_counts.py
```

The Figure 1 subpanels were combined manually in Adobe Photoshop. The retained
composite is `results/figures/Fig1.png`.

## ConvLSTM workflow

Generate the seven-channel dataset:

```bash
python code/convlstm/01_build_dataset.py
```

Select a model and repeated-run number for training and evaluation:

```bash
python code/convlstm/02_train_convlstm.py --model M1 --run 1
python code/convlstm/03_evaluate_convlstm.py --model M1 --run 1
```

After all five models and ten runs have been evaluated, generate the retained
nine-run summaries and robustness source tables:

```bash
python code/convlstm/04_summarize_convlstm_results.py
python code/convlstm/05_evaluate_robustness.py
python code/convlstm/06_export_prediction_frequency.py
```

The repeated-run exclusions are M1 run 9, M2 run 6, M3 run 10, M4 run 9, and
M5 run 9. Preview a selected command sequence with:

```bash
python code/convlstm/run_convlstm_pipeline.py --dry-run
```

The fixed loss-weight NPZ is distributed with the Zenodo derived-data archive;
its generation code is not included.

## Main outputs

- `results/random_forest/summaries/japan_rf_summary.csv`: Table 1
- `results/random_forest/summaries/rf_accuracy_by_region_threshold.csv`:
  Figure 2
- `results/random_forest/summaries/rf_importance_by_region_threshold.csv`:
  Figure 3
- `results/figures/`: final figures and figure-source CSV files
- `results/convlstm/summary_metrics.csv`: Table 2 and Figure 4 source data
- `results/convlstm/area_summary.csv`: Table 3 source data
- `results/convlstm/figure5_source/`: Figure 5 source grids
- `results/convlstm/robustness/`: Figures 6-7 source data

Further output descriptions are provided in
[`results/README.md`](results/README.md).

## License

The source code is available under the [MIT License](LICENSE). WSM2025 and USGS
data follow the terms of their original providers.
