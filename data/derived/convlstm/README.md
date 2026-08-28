# ConvLSTM derived data

`01_build_dataset.py` writes the seven-channel Japan dataset to `dataset/`.
The generated arrays are not stored in Git because of their size and can be
regenerated from the monthly USGS catalog files and WSM2025.

Training additionally requires the fixed loss-weight file:

```text
data/derived/convlstm/loss_weight_map_mw4.npz
```

The file will be distributed with the derived-data archive on Zenodo. Its NPZ
key is `sequence_array`. The code used to generate this weight map is not part
of the repository.

`region_meta.json` records the study bounds, grid dimensions, channel order,
100-km neighborhood setting, and minimum local WSM record count.

