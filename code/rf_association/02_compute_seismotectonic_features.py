"""Compute the nine event-level seismotectonic features used by the RF analysis."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    FEATURES_DIR,
    FILTERED_WSM_PATH,
    GRID_STEP_DEG,
    KM_PER_DEG_LAT,
    LOCAL_DIRECTION_STD_MAX_DEG,
    MIN_ORIENTATION_RECORDS,
    RADIUS_KM,
    REGIONAL_RECORDS_DIR,
    ensure_output_directories,
    region_keys,
)


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8", low_memory=False)


def local_distance_km(lat0: float, lon0: float, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Local planar distance, matching the deep-learning data generator."""
    cos_lat = max(0.15, abs(math.cos(math.radians(float(lat0)))))
    dlat = (np.asarray(lat, dtype=float) - float(lat0)) * KM_PER_DEG_LAT
    dlon = (np.asarray(lon, dtype=float) - float(lon0)) * KM_PER_DEG_LAT * cos_lat
    return np.sqrt(dlat**2 + dlon**2)


def distance_weight(distance_km: np.ndarray) -> np.ndarray:
    distance_km = np.asarray(distance_km, dtype=float)
    weight = np.zeros_like(distance_km, dtype=float)
    near = distance_km <= 0.1 * RADIUS_KM
    middle = (distance_km > 0.1 * RADIUS_KM) & (distance_km <= RADIUS_KM)
    weight[near] = 1.0
    weight[middle] = 1.0 - (distance_km[middle] - 0.1 * RADIUS_KM) / (0.9 * RADIUS_KM)
    return weight


def quality_weight(values: pd.Series) -> np.ndarray:
    return values.astype(str).str.upper().map({"A": 1.0, "B": 0.8, "C": 0.6}).fillna(0.0).to_numpy(float)


def method_weight(values: pd.Series) -> np.ndarray:
    mapping = {"FMS": 1.0, "FMF": 0.9, "BO": 0.8, "DIF": 0.8, "HF": 0.7, "GFI": 0.6, "GF": 0.6, "GV": 0.6, "OC": 0.5}
    return values.astype(str).str.upper().map(mapping).fillna(0.7).to_numpy(float)


def axial_mean_and_resultant(azimuth_deg: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    azimuth_deg = np.asarray(azimuth_deg, dtype=float) % 180.0
    weights = np.asarray(weights, dtype=float)
    valid = np.isfinite(azimuth_deg) & np.isfinite(weights) & (weights > 0)
    if not np.any(valid):
        return np.nan, np.nan
    radians = np.deg2rad(2.0 * azimuth_deg[valid])
    local_weights = weights[valid]
    sin_component = np.sum(local_weights * np.sin(radians))
    cos_component = np.sum(local_weights * np.cos(radians))
    total_weight = np.sum(local_weights)
    mean_azimuth = 0.5 * np.degrees(np.arctan2(sin_component, cos_component))
    if mean_azimuth < 0:
        mean_azimuth += 180.0
    resultant = np.hypot(sin_component, cos_component) / total_weight
    return float(mean_azimuth), float(np.clip(resultant, 1e-12, 1.0))


def axial_difference(first: float, second: float) -> float:
    difference = abs(first - second) % 180.0
    return float(min(difference, 180.0 - difference))


def maximum_axial_spread(azimuths: list[float] | np.ndarray) -> float:
    values = np.asarray(azimuths, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return np.nan
    return float(max(axial_difference(values[i], values[j]) for i in range(len(values)) for j in range(i + 1, len(values))))


def orientation_std_deg(resultant: float) -> float:
    if not np.isfinite(resultant) or resultant <= 0:
        return np.nan
    return float(np.degrees(np.sqrt(-np.log(np.clip(resultant, 1e-12, 1.0)) / 2.0)))


def valid_local_direction_cells(event_lat: float, event_lon: float, historical: pd.DataFrame) -> pd.DataFrame:
    """Construct the local direction field used by the two spatial orientation metrics."""
    if len(historical) < MIN_ORIENTATION_RECORDS:
        return pd.DataFrame(columns=("lat", "lon", "mean_azimuth"))

    cos_lat = max(0.15, abs(math.cos(math.radians(event_lat))))
    lat_margin = RADIUS_KM / KM_PER_DEG_LAT
    lon_margin = RADIUS_KM / (KM_PER_DEG_LAT * cos_lat)
    lat_grid = np.arange(event_lat - lat_margin, event_lat + lat_margin + 1e-9, GRID_STEP_DEG)
    lon_grid = np.arange(event_lon - lon_margin, event_lon + lon_margin + 1e-9, GRID_STEP_DEG)

    latitudes = historical["LAT"].to_numpy(float)
    longitudes = historical["LON"].to_numpy(float)
    azimuths = historical["AZI"].to_numpy(float)
    q_weights = quality_weight(historical["QUALITY"])
    m_weights = method_weight(historical["TYPE"])
    rows: list[dict[str, float]] = []

    for grid_lat in lat_grid:
        for grid_lon in lon_grid:
            distances = local_distance_km(grid_lat, grid_lon, latitudes, longitudes)
            nearby = distances <= RADIUS_KM
            if int(nearby.sum()) < MIN_ORIENTATION_RECORDS:
                continue
            weights = distance_weight(distances[nearby]) * q_weights[nearby] * m_weights[nearby]
            mean_azimuth, resultant = axial_mean_and_resultant(azimuths[nearby], weights)
            standard_deviation = orientation_std_deg(resultant)
            if not np.isfinite(mean_azimuth) or not np.isfinite(standard_deviation):
                continue
            if standard_deviation <= LOCAL_DIRECTION_STD_MAX_DEG:
                rows.append({"lat": grid_lat, "lon": grid_lon, "mean_azimuth": mean_azimuth})
    return pd.DataFrame(rows)


def depth_metrics(historical: pd.DataFrame) -> tuple[float, float, float]:
    depth = pd.to_numeric(historical["DEPTH"], errors="coerce").dropna()
    if depth.empty:
        return np.nan, np.nan, np.nan
    return float(depth.mean()), float(depth.median()), float(depth.std(ddof=0))


def regime_metrics(historical: pd.DataFrame) -> tuple[float, float]:
    regimes = historical["REGIME"].dropna().astype(str).str.strip()
    regimes = regimes[regimes.ne("")]
    if regimes.empty:
        return np.nan, np.nan
    probabilities = regimes.value_counts(normalize=True).to_numpy(float)
    entropy = -np.sum(probabilities * np.log(probabilities))
    gini_simpson = 1.0 - np.sum(probabilities**2)
    return float(entropy), float(gini_simpson)


def build_event_features(events: pd.DataFrame, wsm: pd.DataFrame) -> pd.DataFrame:
    wsm_dates = pd.to_datetime(wsm["DATE"], errors="coerce", utc=True)
    wsm_latitudes = wsm["LAT"].to_numpy(float)
    wsm_longitudes = wsm["LON"].to_numpy(float)
    rows: list[dict] = []

    for index, event in events.iterrows():
        event_date = pd.to_datetime(event["DATE"], errors="coerce", utc=True)
        distances = local_distance_km(event["LAT"], event["LON"], wsm_latitudes, wsm_longitudes)
        historical = wsm.loc[(distances <= RADIUS_KM) & (wsm_dates < event_date)].copy()

        mean_depth, median_depth, depth_std = depth_metrics(historical)
        entropy, gini_simpson = regime_metrics(historical)
        standard_deviation = resultant = maximum_spread = rotation = np.nan

        if len(historical) >= MIN_ORIENTATION_RECORDS:
            local_distances = distances[(distances <= RADIUS_KM) & (wsm_dates < event_date)]
            weights = distance_weight(local_distances) * quality_weight(historical["QUALITY"]) * method_weight(historical["TYPE"])
            _, resultant = axial_mean_and_resultant(historical["AZI"].to_numpy(float), weights)
            standard_deviation = orientation_std_deg(resultant)

            direction_cells = valid_local_direction_cells(event["LAT"], event["LON"], historical)
            if len(direction_cells) >= 2:
                maximum_spread = maximum_axial_spread(direction_cells["mean_azimuth"].to_numpy(float))
            if not direction_cells.empty:
                center_distances = local_distance_km(event["LAT"], event["LON"], direction_cells["lat"], direction_cells["lon"])
                event_cell_azimuth = float(direction_cells.iloc[int(np.argmin(center_distances))]["mean_azimuth"])
                background_azimuth, _ = axial_mean_and_resultant(direction_cells["mean_azimuth"].to_numpy(float), np.ones(len(direction_cells)))
                rotation = axial_difference(background_azimuth, event_cell_azimuth)

        row = event.to_dict()
        row.update({
            "MEAN_LOCAL_DEPTH": mean_depth,
            "MEDIAN_LOCAL_DEPTH": median_depth,
            "STD_LOCAL_DEPTH": depth_std,
            "STD_LOCAL": standard_deviation,
            "CONSISTENCY_R_MEAN": resultant,
            "MAX_SPREAD_100KM_DEG": maximum_spread,
            "MAX_ROTATION_DEG": rotation,
            "REGIME_ENTROPY": entropy,
            "REGIME_GINI": gini_simpson,
        })
        rows.append(row)
        if (index + 1) % 100 == 0 or index + 1 == len(events):
            print(f"Processed {index + 1}/{len(events)} events.")
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regions", nargs="+", default=["japan"], help="Region keys, or all.")
    parser.add_argument("--wsm", type=Path, default=FILTERED_WSM_PATH, help="Filtered WSM2025 records from step 01.")
    args = parser.parse_args()

    if not args.wsm.exists():
        raise FileNotFoundError(f"Run 01_prepare_wsm_data.py first: {args.wsm}")
    ensure_output_directories()
    wsm = read_csv(args.wsm)
    for key in region_keys(args.regions):
        event_path = REGIONAL_RECORDS_DIR / f"{key}.csv"
        if not event_path.exists():
            raise FileNotFoundError(f"Run step 01 for {key}: {event_path}")
        features = build_event_features(read_csv(event_path), wsm)
        output_path = FEATURES_DIR / f"{key}_features.csv"
        features.to_csv(output_path, index=False, encoding="utf-8")
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
