# Results

## Random-forest outputs

Each formal region-threshold task is stored under:

```text
results/random_forest/runs/<region>/mw_<threshold>/
```

Each run directory contains:

- `scores_per_repeat.csv`: test accuracy for every repeated experiment
- `feature_importances_per_repeat.csv`: nine feature importances per repeat
- `metrics_summary.csv`: mean and standard deviation of accuracy
- `feature_importance_summary.csv`: per-task feature-importance summary
- `run_metadata.json`: task definition, sample counts, and RF settings

Manuscript-ready summaries are stored in
`results/random_forest/summaries/`:

- `japan_rf_summary.csv`: source table for manuscript Table 1
- `rf_accuracy_by_region_threshold.csv`: source data for Figure 2
- `rf_importance_by_region_threshold.csv`: task-level source data for Figure 3
- `rf_importance_by_threshold.csv`: threshold-level Figure 3 summaries
- `rf_importance_all_thresholds.csv`: summary across all formal tasks

## Figure outputs

`results/figures/` contains the regenerated Figure 1 subpanels, Figures 2-3,
and the corresponding source CSV files. `Fig1.png` is the retained final
composite assembled manually in Adobe Photoshop. Publication-quality TIFF
exports remain local and are excluded from Git.

## ConvLSTM outputs

- `convlstm/run_metrics.csv`: all ten runs and the retained-run flag
- `convlstm/summary_metrics.csv`: nine-run mean and sample standard deviation
- `convlstm/area_summary.csv`: predicted and target area fractions
- `convlstm/figure5_source/`: prediction-frequency and target-mask grids
- `convlstm/robustness/`: ranked CSI and threshold/distance sensitivity tables

Model checkpoints are written under `convlstm/checkpoints/` and are excluded
from Git. They can be archived separately with the derived dataset.
