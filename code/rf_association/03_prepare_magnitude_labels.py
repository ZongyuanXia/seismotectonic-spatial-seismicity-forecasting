"""Convert WSM event magnitudes to approximate Mw and create RF labels."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from config import (
    FEATURE_COLUMNS,
    FEATURES_DIR,
    LABELS_DIR,
    MAGNITUDE_THRESHOLDS,
    REGIONS,
    SAMPLE_AVAILABILITY_PATH,
    ensure_output_directories,
    region_keys,
)


def energy_from_magnitude(magnitude: float, magnitude_type: str) -> float:
    magnitude_type = str(magnitude_type).lower().strip()
    magnitude = float(magnitude)
    if magnitude_type in {"mww", "mwc", "mwb", "mwr", "mi", "mwp", "ms", "ms_20", "ml", "mlr", "mlv"}:
        coefficient, intercept = 1.5, 4.8
    elif magnitude_type in {"mb", "mb_lg", "mblg", "mlg", "lg", "mfa", "fa"}:
        coefficient, intercept = 2.4, -1.2
    elif magnitude_type in {"md", "mdl", "mc"}:
        coefficient, intercept = 1.404, 4.56
    elif magnitude_type == "me":
        coefficient, intercept = 1.5, 4.4
    else:
        coefficient, intercept = 1.5, 4.8
    return float(10.0 ** (coefficient * magnitude + intercept))


def approximate_mw(magnitude: pd.Series, magnitude_type: pd.Series) -> pd.Series:
    numeric_magnitude = pd.to_numeric(magnitude, errors="coerce")
    values = [
        (np.log10(energy_from_magnitude(value, mag_type)) - 4.8) / 1.5 if np.isfinite(value) else np.nan
        for value, mag_type in zip(numeric_magnitude, magnitude_type)
    ]
    return pd.Series(values, index=magnitude.index, dtype=float)


def label_column(threshold: float) -> str:
    return f"LABEL_MW{int(threshold)}"


def analysis_ready_events(df: pd.DataFrame) -> pd.DataFrame:
    required = ["MAG_MW", *FEATURE_COLUMNS]
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"Input is missing required column(s): {', '.join(missing)}")
    eligible = df.copy()
    for column in required:
        eligible[column] = pd.to_numeric(eligible[column], errors="coerce")
    return eligible.dropna(subset=required).copy()


def availability_rows(region_key: str, eligible: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    for threshold in MAGNITUDE_THRESHOLDS:
        labels = eligible["MAG_MW"] >= threshold
        rows.append({
            "region": region_key,
            "region_name": REGIONS[region_key].name,
            "threshold_mw": threshold,
            "n_events": int(len(eligible)),
            "n_positive": int(labels.sum()),
            "n_negative": int((~labels).sum()),
        })
    return rows


def update_availability_table(rows: list[dict], processed_regions: list[str]) -> None:
    new_rows = pd.DataFrame(rows)
    if SAMPLE_AVAILABILITY_PATH.exists():
        existing = pd.read_csv(SAMPLE_AVAILABILITY_PATH, encoding="utf-8")
        existing = existing.loc[~existing["region"].isin(processed_regions)].copy()
        summary = pd.concat((existing, new_rows), ignore_index=True)
    else:
        summary = new_rows
    region_order = {key: index for index, key in enumerate(REGIONS)}
    summary["_region_order"] = summary["region"].map(region_order)
    summary = summary.sort_values(["_region_order", "threshold_mw"]).drop(columns="_region_order")
    summary.to_csv(SAMPLE_AVAILABILITY_PATH, index=False, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regions", nargs="+", default=["japan"], help="Region keys, or all.")
    args = parser.parse_args()

    ensure_output_directories()
    selected_regions = region_keys(args.regions)
    summary_rows: list[dict] = []
    for key in selected_regions:
        input_path = FEATURES_DIR / f"{key}_features.csv"
        if not input_path.exists():
            raise FileNotFoundError(f"Run 02_compute_seismotectonic_features.py first: {input_path}")
        df = pd.read_csv(input_path, encoding="utf-8", low_memory=False)
        missing = {"EQ_MAG", "MAG_TYPE"} - set(df.columns)
        if missing:
            raise ValueError(f"{input_path} is missing column(s): {', '.join(sorted(missing))}")

        df["MAG_MW"] = approximate_mw(df["EQ_MAG"], df["MAG_TYPE"])
        df = df.loc[df["MAG_MW"].notna()].copy()
        for threshold in MAGNITUDE_THRESHOLDS:
            df[label_column(threshold)] = (df["MAG_MW"] >= threshold).astype(int)

        output_path = LABELS_DIR / f"{key}_labeled_events.csv"
        df.to_csv(output_path, index=False, encoding="utf-8")
        eligible = analysis_ready_events(df)
        summary_rows.extend(availability_rows(key, eligible))
        print(f"Saved {output_path} ({len(df)} events; {len(eligible)} have all nine RF features.)")

    update_availability_table(summary_rows, selected_regions)
    print(f"Saved {SAMPLE_AVAILABILITY_PATH}")


if __name__ == "__main__":
    main()
