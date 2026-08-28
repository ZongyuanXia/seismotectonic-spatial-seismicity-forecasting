# Data

## Raw inputs

Raw third-party files are kept locally under `data/raw/` and are excluded from
Git by `.gitignore`.

### WSM2025

Download the CSV edition of the World Stress Map Database Release 2025 and
place it at:

```text
data/raw/WSM_Database_2025.csv
```

Official sources:

- <https://doi.org/10.5880/WSM.2025.001>
- <https://www.world-stress-map.org/download>

The RF preprocessing code expects the WSM2025 fields used for event location,
date, depth, stress orientation, quality, stress regime, indicator type, and
earthquake magnitude.

### USGS monthly catalog files

Download monthly CSV files from the USGS Earthquake Catalog and place them in:

```text
data/raw/usgs_csv/
```

Use the naming convention `usgsYYYY_MM.csv`, for example
`usgs1973_01.csv`. Figure 1(a) and Figure 1(e) require January 1964 through May
2021. These files are not required for the RF association analysis.

Official catalog: <https://earthquake.usgs.gov/earthquakes/search/>

The ConvLSTM dataset builder uses January 1973 through May 2021 and reads the
same monthly naming convention.

## Generated data

The RF pipeline creates the following local analysis products. They are not
tracked in Git because they can be regenerated from WSM2025:

```text
data/derived/wsm2025_filtered_records.csv
data/derived/region_record_counts.csv
data/derived/regional_records/<region>.csv
data/derived/event_features/<region>_features.csv
data/derived/labeled_events/<region>_labeled_events.csv
data/derived/sample_availability_by_region_threshold.csv
```

- `regional_records/` contains the 10 spatial study-region subsets.
- `event_features/` contains the nine event-level seismotectonic features.
- `labeled_events/` adds approximate `Mw` and the 4.0, 5.0, and 6.0 binary
  labels used to assess sample availability and run the selected RF tasks.

Do not place credentials, local absolute paths, or restricted data in this
directory.

## ConvLSTM derived data

The seven-channel arrays and `region_meta.json` are generated under
`data/derived/convlstm/` and are not tracked in Git. The fixed loss-weight file
is downloaded from the associated Zenodo archive and placed at
`data/derived/convlstm/loss_weight_map_mw4.npz`. See
`data/derived/convlstm/README.md` for the expected file layout.
