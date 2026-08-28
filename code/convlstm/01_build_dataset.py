# -*- coding: utf-8 -*-
"""
Build the seven-channel Japan dataset used by the ConvLSTM experiments.

Purpose
-------
Generate a 7-channel earthquake forecasting dataset:
    0 MAG
    1 DEPTH_LOCAL_MEAN
    2 DEPTH_LOCAL_STD
    3 MAX_SPREAD_100KM_DEG
    4 STD_LOCAL
    5 REGIME_ENTROPY
    6 REGIME_GINI

Core constraints
----------------
1) The original MAG generation logic is kept unchanged as far as possible:
   USGS monthly CSV -> magnitude-type energy conversion -> approximate Mw -> start_mag filtering
   -> [start_value, 255] mapping -> same-pixel max value per frame.

2) The target Y is unchanged:
   Y = last frame, channel 0 only, shape (B, 1, 1, H, W).

3) The six extra WSM channels are computed only on grids where the current MAG frame has
   valid USGS events. They do not create independent dense background maps.

4) WSM local neighborhood is changed from fixed d-cell expansion to a true 100 km spatial
   neighborhood. The code first uses a row-dependent candidate rectangle and then filters
   candidates by actual local-distance approximation in km.

5) Step2 is refactored to cache unique frame-level sparse features. This avoids recomputing
   WSM cumulative/history statistics repeatedly for every overlapping sample.

6) DEPTH_LOCAL_STD uses the same WSM time range and the same true-100-km spatial neighborhood
   as DEPTH_LOCAL_MEAN. It is mapped from km to [100, 255] with a configurable upper bound.

Notes
-----
- This file supports globally anchored regional crop. If config.crop_x_min/crop_x_max/
  crop_y_min/crop_y_max exist, they are used first. Otherwise the crop box is inferred from
  lat_min/lat_max/lon_min/lon_max.
- WSM records are indexed by relative crop-grid buckets, and each bucket is sorted by day;
  time filtering uses np.searchsorted.
- The 100 km distance is evaluated by an equirectangular local approximation, sufficient
  for the local-neighborhood scale here and much faster than repeated haversine calls.
"""

import argparse
import os
import re
import gc
import time
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import CONVLSTM_DATA_DIR, RAW_DATA_DIR, config

rotate_sequence = None


# ==========================================================
# 0. Channel definition
# ==========================================================
CHANNEL_ORDER = [
    "MAG",
    "DEPTH_LOCAL_MEAN",
    "DEPTH_LOCAL_STD",
    "MAX_SPREAD_100KM_DEG",
    "STD_LOCAL",
    "REGIME_ENTROPY",
    "REGIME_GINI",
]
EXPECTED_CHANNELS = len(CHANNEL_ORDER)
EARTH_RADIUS_KM = 6371.0088
KM_PER_DEG_LAT = 111.32


# ==========================================================
# 1. Time and grid utilities
# ==========================================================
def _parse_time_ym(s):
    s = str(s).strip().replace('/', '-').replace('_', '-')
    y, m = s.split('-')
    return int(y), int(m)


def _month_range():
    y0, m0 = _parse_time_ym(getattr(config, "time_start", "1973_01"))
    y1, m1 = _parse_time_ym(getattr(config, "time_end", "2021_05"))
    months = []
    y, m = y0, m0
    while (y < y1) or (y == y1 and m <= m1):
        months.append((y, m))
        m += 1
        if m == 13:
            m = 1
            y += 1
    return months


def _date_meta():
    y0, m0 = _parse_time_ym(getattr(config, "time_start", "1973_01"))
    y1, m1 = _parse_time_ym(getattr(config, "time_end", "2021_05"))
    start = pd.Timestamp(f"{y0:04d}-{m0:02d}-01", tz="UTC")
    end = pd.Timestamp(f"{y1:04d}-{m1:02d}-01", tz="UTC") + pd.offsets.MonthEnd(0)
    n_days = (end.normalize() - start.normalize()).days + 1
    return start, n_days


def _latlon_to_xy(lat, lon, H, W):
    """Original global-grid mapping. y=0 corresponds to -90 deg; y increases northward."""
    lat = np.asarray(lat, dtype=np.float32)
    lon = np.asarray(lon, dtype=np.float32)
    x = ((lon + 180.0) / 360.0 * W).astype(np.int32)
    y = ((lat + 90.0) / 180.0 * H).astype(np.int32)
    x = x.clip(0, W - 1)
    y = y.clip(0, H - 1)
    return x, y


def _latlon_to_xy_raw(lat, lon, H, W):
    """Global-grid raw integer coordinates without clipping."""
    lat = np.asarray(lat, dtype=np.float32)
    lon = np.asarray(lon, dtype=np.float32)
    x = np.floor((lon + 180.0) / 360.0 * W).astype(np.int32)
    y = np.floor((lat + 90.0) / 180.0 * H).astype(np.int32)
    return x, y


def _xy_center_to_latlon_global(x_global, y_global, H_global, W_global):
    lat = ((np.asarray(y_global, dtype=np.float64) + 0.5) / float(H_global)) * 180.0 - 90.0
    lon = ((np.asarray(x_global, dtype=np.float64) + 0.5) / float(W_global)) * 360.0 - 180.0
    return lat, lon


def _cfg_first(names, default=None):
    for name in names:
        if hasattr(config, name):
            return getattr(config, name)
    return default


def _get_region_bounds_from_config():
    lat_min = _cfg_first(["lat_min", "LAT_MIN", "region_lat_min", "REGION_LAT_MIN"], None)
    lat_max = _cfg_first(["lat_max", "LAT_MAX", "region_lat_max", "REGION_LAT_MAX"], None)
    lon_min = _cfg_first(["lon_min", "LON_MIN", "region_lon_min", "REGION_LON_MIN"], None)
    lon_max = _cfg_first(["lon_max", "LON_MAX", "region_lon_max", "REGION_LON_MAX"], None)
    if any(v is None for v in [lat_min, lat_max, lon_min, lon_max]):
        return None
    lat_min, lat_max = sorted([float(lat_min), float(lat_max)])
    lon_min, lon_max = sorted([float(lon_min), float(lon_max)])
    if not (-90.0 <= lat_min < lat_max <= 90.0):
        raise ValueError(f"Illegal latitude range: lat_min={lat_min}, lat_max={lat_max}")
    if not (-180.0 <= lon_min < lon_max <= 180.0):
        raise ValueError("This version does not support regions crossing the ±180° dateline.")
    return lat_min, lat_max, lon_min, lon_max


def _region_pixel_box(bounds, H_global, W_global):
    """Convert lat/lon bounds to a closed pixel crop box on the global mother grid."""
    lat_min, lat_max, lon_min, lon_max = bounds
    x_min = int(np.floor((lon_min + 180.0) / 360.0 * W_global))
    x_max_exclusive = int(np.ceil((lon_max + 180.0) / 360.0 * W_global))
    y_min = int(np.floor((lat_min + 90.0) / 180.0 * H_global))
    y_max_exclusive = int(np.ceil((lat_max + 90.0) / 180.0 * H_global))
    x_min = max(0, min(W_global - 1, x_min))
    y_min = max(0, min(H_global - 1, y_min))
    x_max = max(0, min(W_global - 1, x_max_exclusive - 1))
    y_max = max(0, min(H_global - 1, y_max_exclusive - 1))
    if x_max < x_min or y_max < y_min:
        raise ValueError(f"Empty crop box: bounds={bounds}, crop={(x_min, x_max, y_min, y_max)}")
    return x_min, x_max, y_min, y_max


def _effective_bounds_from_pixel_box(pixel_box, H_global, W_global):
    """Back-calculate the exact lat/lon bounds covered by a closed pixel box."""
    x_min, x_max, y_min, y_max = pixel_box
    lon_min_eff = x_min / float(W_global) * 360.0 - 180.0
    lon_max_eff = (x_max + 1) / float(W_global) * 360.0 - 180.0
    lat_min_eff = y_min / float(H_global) * 180.0 - 90.0
    lat_max_eff = (y_max + 1) / float(H_global) * 180.0 - 90.0
    return lat_min_eff, lat_max_eff, lon_min_eff, lon_max_eff


def try_read_csv(path):
    for enc in ["utf-8-sig", "utf-8", "gbk", "cp936", "latin1"]:
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except Exception:
            continue
    raise RuntimeError(f"CSV read failed: {path}")


# ==========================================================
# 2. USGS magnitude conversion and mapping
# ==========================================================
def _energy_from_mag_type(mag, mag_type):
    t = str(mag_type).lower()
    m = float(mag)
    if t in ["mww", "mwc", "mwb", "mwr", "mi", "mwp", "ms", "ms_20", "ml", "mlr", "mlv"]:
        a, b = 1.5, 4.8
    elif t in ["mb", "mb_lg", "mblg", "mlg", "lg", "mfa", "fa"]:
        a, b = 2.4, -1.2
    elif t in ["md", "mdl", "mc"]:
        a, b = 1.404, 4.56
    elif t in ["me"]:
        a, b = 1.5, 4.4
    else:
        a, b = 1.5, 4.8
    return np.power(10.0, a * m + b)


def _mag_from_energy(E):
    return (np.log10(np.clip(E, 1e-12, None)) - 4.8) / 1.5


def mag_to_float_100_255_from_Mw(Mw, start_mag, start_value):
    Mw = np.asarray(Mw, dtype=np.float32)
    v = np.log10(np.clip(Mw - (start_mag - 0.5), 1e-12, None))
    vmin = np.log10(max(1e-12, 0.5))
    vmax = np.log10(max(1e-12, 10.0 - (start_mag - 0.5)))
    frac = np.clip((v - vmin) / max(1e-12, (vmax - vmin)), 0.0, 1.0)
    return (start_value + frac * (255.0 - start_value)).astype(np.float32)


# ==========================================================
# 3. Value mapping and grouping utilities
# ==========================================================
def _map_linear_valid_to_100_255(vals, valid_mask, vmin, vmax):
    vals = np.asarray(vals, dtype=np.float32)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    out = np.zeros_like(vals, dtype=np.float32)
    if not np.any(valid_mask):
        return out
    frac = (vals[valid_mask] - vmin) / max(1e-12, (vmax - vmin))
    frac = np.clip(frac, 0.0, 1.0)
    out[valid_mask] = 100.0 + frac * (255.0 - 100.0)
    return out.astype(np.float32)


def depth_to_float_100_255(depth_km, valid_mask, depth_min=0.0, depth_max=700.0):
    return _map_linear_valid_to_100_255(depth_km, valid_mask, depth_min, depth_max)


def depth_std_to_float_100_255(depth_std_km, valid_mask, depth_std_max=100.0):
    return _map_linear_valid_to_100_255(depth_std_km, valid_mask, 0.0, depth_std_max)


def std_local_to_float_100_255(std_deg, valid_mask, std_max=45.0):
    return _map_linear_valid_to_100_255(std_deg, valid_mask, 0.0, std_max)


def max_spread_to_float_100_255(spread_deg, valid_mask):
    return _map_linear_valid_to_100_255(spread_deg, valid_mask, 0.0, 90.0)


def entropy_to_float_100_255(entropy_vals, valid_mask, k_regime):
    max_entropy = float(np.log(max(2, int(k_regime))))
    return _map_linear_valid_to_100_255(entropy_vals, valid_mask, 0.0, max_entropy)


def gini_to_float_100_255(gini_vals, valid_mask, k_regime):
    k_regime = max(2, int(k_regime))
    max_gini = 1.0 - 1.0 / k_regime
    return _map_linear_valid_to_100_255(gini_vals, valid_mask, 0.0, max_gini)


def _group_max_by_key(keys_u32, vals_1d):
    if keys_u32.size == 0:
        return np.empty((0,), dtype=np.uint32), np.empty((0,), dtype=vals_1d.dtype)
    order = np.argsort(keys_u32, kind="mergesort")
    k = keys_u32[order]
    v = vals_1d[order]
    diff = np.ones_like(k, dtype=bool)
    diff[1:] = k[1:] != k[:-1]
    idx = np.flatnonzero(diff)
    out_k, out_v = [], []
    for i, j in zip(idx, list(idx[1:]) + [len(k)]):
        out_k.append(k[i])
        out_v.append(v[i:j].max(axis=0))
    return np.asarray(out_k, dtype=np.uint32), np.asarray(out_v, dtype=vals_1d.dtype)


# ==========================================================
# 4. File lookup utilities
# ==========================================================
def _find_usgs_month_file(usgs_dir, year, month):
    m2 = f"{int(month):02d}"
    y = int(year)
    candidates = [
        os.path.join(usgs_dir, f"{y}_{m2}.csv"),
        os.path.join(usgs_dir, f"{y}-{m2}.csv"),
        os.path.join(usgs_dir, f"usgs{y}_{m2}.csv"),
        os.path.join(usgs_dir, f"usgs_{y}_{m2}.csv"),
        os.path.join(usgs_dir, f"USGS{y}_{m2}.csv"),
        os.path.join(usgs_dir, f"USGS_{y}_{m2}.csv"),
        os.path.join(usgs_dir, f"usgs-{y}-{m2}.csv"),
        os.path.join(usgs_dir, f"USGS-{y}-{m2}.csv"),
    ]
    for fp in candidates:
        if os.path.exists(fp):
            return fp
    rx = re.compile(rf"(?i)^(?:usgs[-_]?)*{y}[-_]?{m2}\.csv$")
    try:
        for name in os.listdir(usgs_dir):
            if rx.match(name):
                return os.path.join(usgs_dir, name)
    except FileNotFoundError:
        return None
    return None


def _find_wsm_month_file(wsm_dir, year, month):
    m2 = f"{int(month):02d}"
    y = int(year)
    candidates = [
        os.path.join(wsm_dir, f"{y}_{m2}.csv"),
        os.path.join(wsm_dir, f"{y}-{m2}.csv"),
        os.path.join(wsm_dir, f"wsm{y}_{m2}.csv"),
        os.path.join(wsm_dir, f"wsm_{y}_{m2}.csv"),
        os.path.join(wsm_dir, f"WSM{y}_{m2}.csv"),
        os.path.join(wsm_dir, f"WSM_{y}_{m2}.csv"),
        os.path.join(wsm_dir, f"wsm-{y}-{m2}.csv"),
        os.path.join(wsm_dir, f"WSM-{y}-{m2}.csv"),
    ]
    for fp in candidates:
        if os.path.exists(fp):
            return fp
    rx = re.compile(rf"(?i)^(?:wsm[-_]?)*{y}[-_]?{m2}\.csv$")
    try:
        for name in os.listdir(wsm_dir):
            if rx.match(name):
                return os.path.join(wsm_dir, name)
    except FileNotFoundError:
        return None
    return None


# ==========================================================
# 5. WSM directional/statistical utilities
# ==========================================================
def axial_mean_R(az_deg, weights):
    az = np.asarray(az_deg, dtype=float) % 180.0
    w = np.asarray(weights, dtype=float)
    mask = np.isfinite(az) & np.isfinite(w) & (w > 0)
    if mask.sum() == 0:
        return np.nan, np.nan
    az = az[mask]
    w = w[mask]
    rad2 = np.deg2rad(2.0 * az)
    sin2 = np.sum(w * np.sin(rad2))
    cos2 = np.sum(w * np.cos(rad2))
    W = np.sum(w)
    mean = 0.5 * np.degrees(np.arctan2(sin2, cos2))
    if mean < 0:
        mean += 180.0
    R = np.sqrt(sin2 ** 2 + cos2 ** 2) / W
    return float(mean), float(np.clip(R, 1e-12, 1.0))


def angular_diff(a, b):
    d = abs(a - b)
    if d > 180.0:
        d = 360.0 - d
    return min(d, 180.0 - d)


def max_spread(az_list):
    az = np.asarray(az_list, dtype=float)
    az = az[np.isfinite(az)]
    if len(az) < 2:
        return np.nan
    m = 0.0
    for i in range(len(az)):
        for j in range(i + 1, len(az)):
            m = max(m, angular_diff(az[i], az[j]))
    return float(m)


def quality_map(q):
    q = str(q).upper()
    return {"A": 1.0, "B": 0.8, "C": 0.6, "D": 0.4}.get(q, 0.0)


def method_map(m):
    m = str(m).upper()
    return {
        "FMS": 1.0, "FMF": 0.9, "BO": 0.8, "DIF": 0.8,
        "HF": 0.7, "GF": 0.6, "GV": 0.6, "OC": 0.5,
    }.get(m, 0.7)


def dist_weight_km(dist_km, radius_km):
    if dist_km <= 0.1 * radius_km:
        return 1.0
    if dist_km <= radius_km:
        return 1.0 - (dist_km - 0.1 * radius_km) / max(1e-12, 0.9 * radius_km)
    return 0.0


# ==========================================================
# 6. Main class
# ==========================================================
class MultiProcessDataset:
    def __init__(self):
        self.global_H = int(getattr(config, "height", 3840))
        self.global_W = int(getattr(config, "width", 3840))
        self.pic = int(getattr(config, "pic", 9))
        self.dtday = int(getattr(config, "dtday", 1))
        self.percent = float(getattr(config, "percent", 0.7))

        self.region_bounds = _get_region_bounds_from_config()
        mode_raw = _cfg_first(["region_mode", "REGION_MODE"], None)
        if mode_raw is None:
            self.region_mode = "anchored_crop" if self.region_bounds is not None else "global"
        else:
            self.region_mode = str(mode_raw).lower().strip()
        if self.region_mode in ["none", "all", "global", "false", "0"]:
            self.region_mode = "global"
        elif self.region_mode in ["anchored_crop", "crop", "regional", "region", "local"]:
            self.region_mode = "anchored_crop"
        else:
            raise ValueError("config.region_mode only supports 'global' or 'anchored_crop'.")

        self.crop_box = None
        self.effective_bounds = None
        if self.region_mode == "anchored_crop":
            explicit_crop = all(hasattr(config, k) for k in ["crop_x_min", "crop_x_max", "crop_y_min", "crop_y_max"])
            if explicit_crop:
                x_min = int(getattr(config, "crop_x_min"))
                x_max = int(getattr(config, "crop_x_max"))
                y_min = int(getattr(config, "crop_y_min"))
                y_max = int(getattr(config, "crop_y_max"))
                if not (0 <= x_min <= x_max < self.global_W and 0 <= y_min <= y_max < self.global_H):
                    raise ValueError(f"Invalid explicit crop box: {(x_min, x_max, y_min, y_max)}")
                self.crop_box = (x_min, x_max, y_min, y_max)
            else:
                if self.region_bounds is None:
                    raise ValueError("anchored_crop requires either crop_x/y config or lat/lon bounds.")
                self.crop_box = _region_pixel_box(self.region_bounds, self.global_H, self.global_W)
            self.effective_bounds = _effective_bounds_from_pixel_box(self.crop_box, self.global_H, self.global_W)
            x_min, x_max, y_min, y_max = self.crop_box
            self.H = y_max - y_min + 1
            self.W = x_max - x_min + 1
        else:
            self.H = self.global_H
            self.W = self.global_W

        self.depth_min = float(getattr(config, "depth_min", 0.0))
        self.depth_max = float(getattr(config, "depth_max", 700.0))

        # WSM 100-km neighborhood configuration.
        self.wsm_neighbor_km = float(getattr(config, "wsm_neighbor_km", 100.0))
        self.wsm_min_n_local = int(getattr(config, "wsm_min_n_local", 5))
        self.wsm_std_threshold = float(getattr(config, "wsm_std_threshold", 25.0))
        self.wsm_time_mode = str(getattr(config, "wsm_time_mode", "cumulative")).lower()
        self.std_local_map_max = float(getattr(config, "std_local_map_max", 45.0))
        self.depth_std_map_max = float(getattr(config, "depth_std_map_max", 100.0))
        self.local_mean_spread_cache = bool(getattr(config, "wsm_local_mean_cache", True))

        # Exact runtime optimization for cumulative WSM features:
        # if no new WSM record appears between two frame_end days, the cumulative
        # WSM set is identical. We therefore map raw end_day to the latest actual
        # WSM observation day <= end_day and reuse center/local-direction metrics.
        self.wsm_effective_snapshot_cache = bool(getattr(config, "wsm_effective_snapshot_cache", True))
        self.wsm_center_metric_cache = bool(getattr(config, "wsm_center_metric_cache", True))
        self.wsm_global_local_mean_cache = bool(getattr(config, "wsm_global_local_mean_cache", True))
        # Solution B direction-range optimization:
        # For direction-related channels, first build the first 100-km grid coverage
        # C1 of all MAG-active centers, then precompute an orientation field on
        # C0 ∪ C1. STD_LOCAL is read from this field at C0, and
        # MAX_SPREAD_100KM_DEG is computed from valid field values inside C1(c).
        # MAG generation is not changed. Depth/regime channels still use the
        # original center-level 100-km WSM query.
        self.wsm_precompute_direction_field = bool(getattr(config, "wsm_precompute_direction_field", True))
        self.wsm_direction_use_kdtree = bool(getattr(config, "wsm_direction_use_kdtree", True))
        self.wsm_direction_c2_radius_factor = float(getattr(config, "wsm_direction_c2_radius_factor", 2.05))
        self.wsm_direction_tree_query_factor = float(getattr(config, "wsm_direction_tree_query_factor", 1.35))
        self._latest_wsm_day = None
        self._center_metric_cache = {}
        self._local_mean_global_cache = {}
        self._grid_radius_cache = {}

        if self.wsm_time_mode not in ["frame", "cumulative"]:
            raise ValueError("config.wsm_time_mode only supports 'frame' or 'cumulative'.")

        # Degree resolution of the global mother grid.
        self.dlat_deg = 180.0 / float(self.global_H)
        self.dlon_deg = 360.0 / float(self.global_W)

        self.base_dir = None
        self.stage3_dir = None
        self.stage4_dir = None

        # Built in Step2.
        self._wsm_bucket = None
        self._regime2code = None
        self._global_k_regime = 2

    # ------------------------------------------------------
    # Geometry utilities
    # ------------------------------------------------------
    def _local_xy_to_latlon(self, x_local, y_local):
        if self.region_mode == "anchored_crop":
            x_min, _, y_min, _ = self.crop_box
            x_global = int(x_local) + x_min
            y_global = int(y_local) + y_min
        else:
            x_global = int(x_local)
            y_global = int(y_local)
        lat, lon = _xy_center_to_latlon_global(x_global, y_global, self.global_H, self.global_W)
        return float(lat), float(lon)

    def _candidate_cell_radius(self, lat0):
        lat0_rad = math.radians(float(lat0))
        cos_lat = max(0.15, abs(math.cos(lat0_rad)))
        dy_cells = int(math.ceil(self.wsm_neighbor_km / max(1e-12, KM_PER_DEG_LAT * self.dlat_deg)))
        dx_cells = int(math.ceil(self.wsm_neighbor_km / max(1e-12, KM_PER_DEG_LAT * cos_lat * self.dlon_deg)))
        return dx_cells, dy_cells

    @staticmethod
    def _distance_km_local(lat0, lon0, lat_arr, lon_arr):
        lat_arr = np.asarray(lat_arr, dtype=np.float64)
        lon_arr = np.asarray(lon_arr, dtype=np.float64)
        cos_lat = max(0.15, abs(math.cos(math.radians(float(lat0)))))
        dlat_km = (lat_arr - float(lat0)) * KM_PER_DEG_LAT
        dlon_km = (lon_arr - float(lon0)) * KM_PER_DEG_LAT * cos_lat
        return np.sqrt(dlat_km ** 2 + dlon_km ** 2).astype(np.float32)

    def _write_region_meta(self):
        if self.base_dir is None:
            return
        meta = {
            "region_mode": self.region_mode,
            "global_H": self.global_H,
            "global_W": self.global_W,
            "output_H": self.H,
            "output_W": self.W,
            "dlat_deg": self.dlat_deg,
            "dlon_deg": self.dlon_deg,
            "configured_bounds": None if self.region_bounds is None else {
                "lat_min": self.region_bounds[0],
                "lat_max": self.region_bounds[1],
                "lon_min": self.region_bounds[2],
                "lon_max": self.region_bounds[3],
            },
            "crop_box_closed": None if self.crop_box is None else {
                "x_min": self.crop_box[0], "x_max": self.crop_box[1],
                "y_min": self.crop_box[2], "y_max": self.crop_box[3],
            },
            "effective_bounds": None if self.effective_bounds is None else {
                "lat_min": self.effective_bounds[0],
                "lat_max": self.effective_bounds[1],
                "lon_min": self.effective_bounds[2],
                "lon_max": self.effective_bounds[3],
            },
            "channel_order": CHANNEL_ORDER,
            "channel_count": EXPECTED_CHANNELS,
            "target_y": "last frame, channel 0, MAG only",
            "wsm_neighbor_km": self.wsm_neighbor_km,
            "wsm_min_n_local": self.wsm_min_n_local,
            "wsm_std_threshold": self.wsm_std_threshold,
            "depth_std_map_max": self.depth_std_map_max,
            "wsm_time_mode": self.wsm_time_mode,
            "wsm_spatial_rule": "row-dependent candidate cell rectangle + true <= wsm_neighbor_km distance filter",
            "wsm_precompute_direction_field": self.wsm_precompute_direction_field,
            "step2_storage": "sparse frame cache + sample frame references",
        }
        fp = os.path.join(self.base_dir, "region_meta.json")
        with open(fp, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print(f"[REGION-META] written: {fp}")

    # ------------------------------------------------------
    # Step 1A: USGS daily packing
    # ------------------------------------------------------
    def _step1_usgs(self, usgs_path):
        H_global, W_global = self.global_H, self.global_W
        start_utc, n_days = _date_meta()
        events_by_day = np.empty((n_days,), dtype=object)
        months = _month_range()

        st_mag = float(getattr(config, "start_mag", 4.0))
        st_val = float(getattr(config, "start_value", 100.0))

        found_months = 0
        read_ok = 0
        kept_events = 0
        nonempty_days = 0
        by_dayidx = defaultdict(list)
        drop_magtypes = {"mint", "unknown", "ma", "mun", "no", "uk", "fa", "lg", "mdl", "mlv"}

        print("Step1 USGS month CSV:", end=" ")
        for y, m in tqdm(months, unit="mon"):
            fname = _find_usgs_month_file(usgs_path, y, m)
            if fname is None:
                continue
            found_months += 1
            try:
                df = pd.read_csv(fname)
            except Exception:
                continue
            read_ok += 1

            required_cols_lower = ["time", "latitude", "longitude", "mag"]
            # Compatible with USGS lower-case and some exported title-case columns.
            colmap = {c.lower(): c for c in df.columns}
            if not all(c in colmap for c in required_cols_lower):
                continue
            use_cols = [colmap[c] for c in required_cols_lower]
            optional = []
            for c in ["magtype", "type"]:
                if c in colmap:
                    optional.append(colmap[c])
            df = df[use_cols + optional].copy()
            df = df.rename(columns={colmap["time"]: "time", colmap["latitude"]: "latitude", colmap["longitude"]: "longitude", colmap["mag"]: "mag"})
            if "magtype" in colmap:
                df = df.rename(columns={colmap["magtype"]: "magType"})
            if "type" in colmap:
                df = df.rename(columns={colmap["type"]: "type"})

            if "type" in df.columns:
                df = df[df["type"] == "earthquake"]
            if "magType" in df.columns:
                mt = df["magType"].astype(str).str.lower()
                df = df[~mt.isin(drop_magtypes)]

            df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
            df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
            df["mag"] = pd.to_numeric(df["mag"], errors="coerce")
            df = df.dropna(subset=["latitude", "longitude", "mag"])
            if df.empty:
                continue

            E = df.apply(lambda r: _energy_from_mag_type(float(r["mag"]), str(r.get("magType", ""))), axis=1).to_numpy(np.float32)
            Mw = _mag_from_energy(E)
            keep = Mw >= st_mag
            if not np.any(keep):
                continue

            df = df.loc[keep].copy()
            Mw = Mw[keep]
            mag_f32 = mag_to_float_100_255_from_Mw(Mw, start_mag=st_mag, start_value=st_val)

            dt = pd.to_datetime(df["time"], errors="coerce", utc=True).dt.normalize()
            valid_time_mask = ~dt.isna()
            if not np.any(valid_time_mask):
                continue
            df = df.loc[valid_time_mask].copy()
            mag_f32 = mag_f32[valid_time_mask.to_numpy()]
            dt = dt[valid_time_mask]
            day_idx = (dt - start_utc).dt.days.to_numpy()

            lat = df["latitude"].to_numpy(np.float32)
            lon = df["longitude"].to_numpy(np.float32)
            x_global, y_global = _latlon_to_xy(lat, lon, H_global, W_global)
            mask = (day_idx >= 0) & (day_idx < n_days)

            if self.region_mode == "anchored_crop":
                x_min, x_max, y_min, y_max = self.crop_box
                lat_min_eff, lat_max_eff, lon_min_eff, lon_max_eff = self.effective_bounds
                coord_closed_mask = (
                    (lat >= lat_min_eff) & (lat <= lat_max_eff) &
                    (lon >= lon_min_eff) & (lon <= lon_max_eff)
                )
                mask = mask & coord_closed_mask
                x = np.clip(x_global, x_min, x_max) - x_min
                y_pix = np.clip(y_global, y_min, y_max) - y_min
            else:
                x = x_global
                y_pix = y_global

            for di, xx, yy, mg in zip(day_idx[mask], x[mask], y_pix[mask], mag_f32[mask]):
                by_dayidx[int(di)].append({"x": int(xx), "y": int(yy), "mag": float(mg)})
                kept_events += 1

        for i in range(n_days):
            arr = by_dayidx.get(i, [])
            events_by_day[i] = np.array(arr, dtype=object) if len(arr) > 0 else np.zeros((0,), dtype=object)
            if len(arr) > 0:
                nonempty_days += 1

        out_fp = os.path.join(self.base_dir, "usgs_by_day.npz")
        np.savez_compressed(out_fp, events_by_day=events_by_day)
        print(f"[Step1-USGS] written {out_fp} days={n_days}")
        print(f"[USGS] months_found={found_months}, months_read_ok={read_ok}, kept_events={kept_events}, nonempty_days={nonempty_days}")
        if kept_events == 0 or nonempty_days == 0:
            raise RuntimeError("USGS data is empty. Check usgs_path/file names/time range/start_mag.")

    # ------------------------------------------------------
    # Step 1B: WSM daily packing
    # ------------------------------------------------------
    def _normalize_wsm_dataframe(self, df):
        cols_upper = {c.upper(): c for c in df.columns}
        required = ["LAT", "LON", "DATE", "AZI", "QUALITY", "TYPE"]
        missing = [c for c in required if c not in cols_upper]
        if missing:
            raise RuntimeError(f"WSM missing required columns: {missing}")

        lat_col = cols_upper["LAT"]
        lon_col = cols_upper["LON"]
        date_col = cols_upper["DATE"]
        azi_col = cols_upper["AZI"]
        quality_col = cols_upper["QUALITY"]
        type_col = cols_upper["TYPE"]
        regime_col = cols_upper["REGIME"] if "REGIME" in cols_upper else None
        depth_col = cols_upper["DEPTH"] if "DEPTH" in cols_upper else None

        use_cols = [lat_col, lon_col, date_col, azi_col, quality_col, type_col]
        if regime_col is not None:
            use_cols.append(regime_col)
        if depth_col is not None:
            use_cols.append(depth_col)
        out = df[use_cols].copy()
        rename = {lat_col: "LAT", lon_col: "LON", date_col: "DATE", azi_col: "AZI", quality_col: "QUALITY", type_col: "TYPE"}
        if regime_col is not None:
            rename[regime_col] = "REGIME"
        if depth_col is not None:
            rename[depth_col] = "DEPTH"
        out = out.rename(columns=rename)

        out["LAT"] = pd.to_numeric(out["LAT"], errors="coerce")
        out["LON"] = pd.to_numeric(out["LON"], errors="coerce")
        out["AZI"] = pd.to_numeric(out["AZI"], errors="coerce")
        if "DEPTH" in out.columns:
            out["DEPTH"] = pd.to_numeric(out["DEPTH"], errors="coerce")
        out["DATE"] = pd.to_datetime(out["DATE"], errors="coerce", utc=True).dt.normalize()
        out["QUALITY"] = out["QUALITY"].astype(str).str.upper().str.strip()
        out["TYPE"] = out["TYPE"].astype(str).str.upper().str.strip()
        if "REGIME" in out.columns:
            out["REGIME"] = out["REGIME"].astype(str).str.upper().str.strip()
            out.loc[out["REGIME"].isin(["", "NAN", "NONE"]), "REGIME"] = np.nan

        out = out.dropna(subset=["LAT", "LON", "DATE", "AZI", "QUALITY", "TYPE"])
        out = out[out["QUALITY"].isin(["A", "B", "C"])].copy()

        start_utc, n_days = _date_meta()
        end_utc = start_utc + pd.Timedelta(days=n_days - 1)
        out = out[(out["DATE"] >= start_utc) & (out["DATE"] <= end_utc)].copy()

        # Buffer by true km, not by fixed cell count.
        if self.region_mode == "anchored_crop":
            lat_min_eff, lat_max_eff, lon_min_eff, lon_max_eff = self.effective_bounds
        else:
            lat_min_eff, lat_max_eff, lon_min_eff, lon_max_eff = -90.0, 90.0, -180.0, 180.0

        lat_abs_max = max(abs(lat_min_eff), abs(lat_max_eff))
        cos_min = max(0.15, math.cos(math.radians(lat_abs_max)))
        buffer_lat = self.wsm_neighbor_km / KM_PER_DEG_LAT
        buffer_lon = self.wsm_neighbor_km / (KM_PER_DEG_LAT * cos_min)

        out = out[
            (out["LAT"] >= lat_min_eff - buffer_lat) & (out["LAT"] <= lat_max_eff + buffer_lat) &
            (out["LON"] >= lon_min_eff - buffer_lon) & (out["LON"] <= lon_max_eff + buffer_lon)
        ].copy()
        return out

    def _step1_wsm(self, wsm_path):
        start_utc, n_days = _date_meta()
        events_by_day = np.empty((n_days,), dtype=object)
        by_dayidx = defaultdict(list)

        found_units = 0
        read_ok = 0
        kept_events = 0
        nonempty_days = 0

        print("Step1 WSM CSV:", end=" ")

        def _consume_dataframe(df):
            nonlocal kept_events
            if df is None or df.empty:
                return
            try:
                df_norm = self._normalize_wsm_dataframe(df)
            except Exception as e:
                print(f"\n[WARN] WSM skipped: {e}")
                return
            if df_norm.empty:
                return

            day_idx = (df_norm["DATE"] - start_utc).dt.days.to_numpy()
            lat = df_norm["LAT"].to_numpy(np.float32)
            lon = df_norm["LON"].to_numpy(np.float32)
            x_global_raw, y_global_raw = _latlon_to_xy_raw(lat, lon, self.global_H, self.global_W)
            if self.region_mode == "anchored_crop":
                x_min, _, y_min, _ = self.crop_box
                x_raw = x_global_raw - x_min
                y_raw = y_global_raw - y_min
            else:
                x_raw = x_global_raw
                y_raw = y_global_raw

            in_region = (x_raw >= 0) & (x_raw < self.W) & (y_raw >= 0) & (y_raw < self.H)
            azi = df_norm["AZI"].to_numpy(np.float32) % 180.0
            q_w = df_norm["QUALITY"].map(quality_map).to_numpy(np.float32)
            m_w = df_norm["TYPE"].map(method_map).to_numpy(np.float32)
            regime = df_norm["REGIME"].astype(object).to_numpy() if "REGIME" in df_norm.columns else np.array([None] * len(df_norm), dtype=object)
            depth = df_norm["DEPTH"].to_numpy(np.float32) if "DEPTH" in df_norm.columns else np.full((len(df_norm),), np.nan, dtype=np.float32)

            mask = (day_idx >= 0) & (day_idx < n_days)
            for di, xx, yy, ir, la, lo, az, qw, mw, dep, rg in zip(
                day_idx[mask], x_raw[mask], y_raw[mask], in_region[mask],
                lat[mask], lon[mask], azi[mask], q_w[mask], m_w[mask], depth[mask], regime[mask]
            ):
                by_dayidx[int(di)].append({
                    "x_raw": int(xx),
                    "y_raw": int(yy),
                    "in_region": bool(ir),
                    "lat": float(la),
                    "lon": float(lo),
                    "azi": float(az),
                    "q_w": float(qw),
                    "m_w": float(mw),
                    "depth": None if not np.isfinite(dep) else float(dep),
                    "regime": None if pd.isna(rg) else str(rg),
                })
                kept_events += 1

        if os.path.isfile(wsm_path):
            found_units += 1
            try:
                df = try_read_csv(wsm_path)
                read_ok += 1
                _consume_dataframe(df)
            except Exception as e:
                print(f"\n[WARN] failed to read WSM file: {e}")
        else:
            months = _month_range()
            for y, m in tqdm(months, unit="mon"):
                fname = _find_wsm_month_file(wsm_path, y, m)
                if fname is None:
                    continue
                found_units += 1
                try:
                    df = pd.read_csv(fname)
                    read_ok += 1
                    _consume_dataframe(df)
                except Exception:
                    continue

        for i in range(n_days):
            arr = by_dayidx.get(i, [])
            events_by_day[i] = np.array(arr, dtype=object) if len(arr) > 0 else np.zeros((0,), dtype=object)
            if len(arr) > 0:
                nonempty_days += 1

        out_fp = os.path.join(self.base_dir, "wsm_by_day.npz")
        np.savez_compressed(out_fp, events_by_day=events_by_day)
        print(f"[Step1-WSM] written {out_fp} days={n_days}")
        print(f"[WSM] units_found={found_units}, units_read_ok={read_ok}, kept_events={kept_events}, nonempty_days={nonempty_days}")
        if kept_events == 0 or nonempty_days == 0:
            raise RuntimeError("WSM data is empty. Check wsm_path/columns/time range/spatial range.")

    # ------------------------------------------------------
    # WSM index and query
    # ------------------------------------------------------
    def _build_wsm_index(self, wsm_by_day):
        regime_set = set()
        for arr in wsm_by_day:
            if arr is None or len(arr) == 0:
                continue
            for item in arr:
                rg = item.get("regime", None)
                if rg is not None and str(rg).strip() != "":
                    regime_set.add(str(rg).upper().strip())
        self._global_k_regime = max(2, len(regime_set))
        self._regime2code = {rg: i for i, rg in enumerate(sorted(regime_set))}

        buckets = defaultdict(lambda: defaultdict(list))
        for day, arr in enumerate(wsm_by_day):
            if arr is None or len(arr) == 0:
                continue
            for item in arr:
                x = int(item["x_raw"])
                y = int(item["y_raw"])
                rg = item.get("regime", None)
                rg_code = -1
                if rg is not None:
                    rg_norm = str(rg).upper().strip()
                    rg_code = self._regime2code.get(rg_norm, -1)
                b = buckets[(x, y)]
                b["day"].append(day)
                b["lat"].append(float(item.get("lat", np.nan)))
                b["lon"].append(float(item.get("lon", np.nan)))
                b["azi"].append(float(item["azi"]))
                b["q_w"].append(float(item["q_w"]))
                b["m_w"].append(float(item["m_w"]))
                b["depth"].append(float(item["depth"]) if item.get("depth", None) is not None else np.nan)
                b["regime"].append(int(rg_code))

        index = {}
        for key, b in buckets.items():
            day_arr = np.asarray(b["day"], dtype=np.int32)
            order = np.argsort(day_arr, kind="mergesort")
            index[key] = {
                "day": day_arr[order],
                "lat": np.asarray(b["lat"], dtype=np.float32)[order],
                "lon": np.asarray(b["lon"], dtype=np.float32)[order],
                "azi": np.asarray(b["azi"], dtype=np.float32)[order],
                "q_w": np.asarray(b["q_w"], dtype=np.float32)[order],
                "m_w": np.asarray(b["m_w"], dtype=np.float32)[order],
                "depth": np.asarray(b["depth"], dtype=np.float32)[order],
                "regime": np.asarray(b["regime"], dtype=np.int32)[order],
                "x": np.full((len(order),), key[0], dtype=np.int32),
                "y": np.full((len(order),), key[1], dtype=np.int32),
            }
        # latest_wsm_day[d] = latest day <= d that has at least one WSM record.
        # For cumulative windows [0, end_day], replacing end_day by latest_wsm_day[end_day]
        # is mathematically identical and enables cache reuse across many daily frames.
        latest = np.full((len(wsm_by_day),), -1, dtype=np.int32)
        last = -1
        nonempty = 0
        for day, arr in enumerate(wsm_by_day):
            if arr is not None and len(arr) > 0:
                last = int(day)
                nonempty += 1
            latest[day] = last
        self._latest_wsm_day = latest

        self._wsm_bucket = index
        print(
            f"[WSM-INDEX] buckets={len(index)}, regime_classes={self._global_k_regime}, "
            f"nonempty_wsm_days={nonempty}"
        )

    def _effective_wsm_window(self, start_day, end_day):
        start_day = int(start_day)
        end_day = int(end_day)
        if (
            self.wsm_effective_snapshot_cache
            and self.wsm_time_mode == "cumulative"
            and start_day == 0
            and self._latest_wsm_day is not None
        ):
            if end_day < 0:
                return 0, -1
            idx = min(max(end_day, 0), len(self._latest_wsm_day) - 1)
            return 0, int(self._latest_wsm_day[idx])
        return start_day, end_day

    def _query_wsm_neighbors(self, center_x, center_y, start_day, end_day):
        start_day, end_day = self._effective_wsm_window(start_day, end_day)
        if end_day < start_day or end_day < 0:
            return None
        if self._wsm_bucket is None or len(self._wsm_bucket) == 0:
            return None
        lat0, lon0 = self._local_xy_to_latlon(center_x, center_y)
        dx_cells, dy_cells = self._candidate_cell_radius(lat0)

        arrays = defaultdict(list)
        for yy in range(int(center_y) - dy_cells, int(center_y) + dy_cells + 1):
            for xx in range(int(center_x) - dx_cells, int(center_x) + dx_cells + 1):
                b = self._wsm_bucket.get((xx, yy), None)
                if b is None:
                    continue
                lo = np.searchsorted(b["day"], int(start_day), side="left")
                hi = np.searchsorted(b["day"], int(end_day), side="right")
                if hi <= lo:
                    continue
                for name in ["lat", "lon", "azi", "q_w", "m_w", "depth", "regime", "x", "y"]:
                    arrays[name].append(b[name][lo:hi])

        if len(arrays["lat"]) == 0:
            return None

        out = {name: np.concatenate(arrays[name]) for name in arrays}
        dist = self._distance_km_local(lat0, lon0, out["lat"], out["lon"])
        mask = np.isfinite(dist) & (dist <= self.wsm_neighbor_km)
        if mask.sum() == 0:
            return None
        for name in list(out.keys()):
            out[name] = out[name][mask]
        out["dist_km"] = dist[mask]
        return out

    def _compute_local_mean_azi_for_cell(self, x, y, start_day, end_day, local_cache):
        start_day, end_day = self._effective_wsm_window(start_day, end_day)
        cache_key = (int(x), int(y), int(start_day), int(end_day))
        if self.local_mean_spread_cache and cache_key in local_cache:
            return local_cache[cache_key]

        q = self._query_wsm_neighbors(int(x), int(y), int(start_day), int(end_day))
        if q is None or q["azi"].size < self.wsm_min_n_local:
            ans = (np.nan, np.nan)
            if self.local_mean_spread_cache:
                local_cache[cache_key] = ans
            return ans

        weights = q["q_w"] * q["m_w"] * np.asarray([dist_weight_km(d, self.wsm_neighbor_km) for d in q["dist_km"]], dtype=np.float32)
        mean_azi, R = axial_mean_R(q["azi"], weights)
        if not np.isfinite(mean_azi) or not np.isfinite(R):
            ans = (np.nan, np.nan)
        else:
            std_deg = float(np.degrees(np.sqrt(-np.log(R) / 2.0)))
            if std_deg > self.wsm_std_threshold:
                ans = (np.nan, std_deg)
            else:
                ans = (mean_azi, std_deg)
        if self.local_mean_spread_cache:
            local_cache[cache_key] = ans
        return ans


    def _xy_array_within_radius(self, center_x, center_y, radius_km, clip_to_output=True):
        """Return relative-grid (x,y) cells whose cell centers are within radius_km.

        For C1, clip_to_output=True because feature maps are only written inside the
        output crop. For C2 prefiltering, clip_to_output=False allows buffered WSM
        buckets outside the crop to be included near region boundaries.
        """
        radius_km = float(radius_km)
        cache_key = (int(center_x), int(center_y), round(radius_km, 3), bool(clip_to_output))
        if cache_key in self._grid_radius_cache:
            return self._grid_radius_cache[cache_key]

        lat0, lon0 = self._local_xy_to_latlon(int(center_x), int(center_y))
        lat0_rad = math.radians(float(lat0))
        cos_lat = max(0.15, abs(math.cos(lat0_rad)))
        dy_cells = int(math.ceil(radius_km / max(1e-12, KM_PER_DEG_LAT * self.dlat_deg)))
        dx_cells = int(math.ceil(radius_km / max(1e-12, KM_PER_DEG_LAT * cos_lat * self.dlon_deg)))

        if self.region_mode == "anchored_crop":
            x_min_global, _, y_min_global, _ = self.crop_box
            x_lo_allowed = -x_min_global
            x_hi_allowed = self.global_W - 1 - x_min_global
            y_lo_allowed = -y_min_global
            y_hi_allowed = self.global_H - 1 - y_min_global
        else:
            x_min_global = 0
            y_min_global = 0
            x_lo_allowed = 0
            x_hi_allowed = self.W - 1
            y_lo_allowed = 0
            y_hi_allowed = self.H - 1

        if clip_to_output:
            x_lo_allowed = 0
            x_hi_allowed = self.W - 1
            y_lo_allowed = 0
            y_hi_allowed = self.H - 1

        x0 = int(center_x)
        y0 = int(center_y)
        x_lo = max(x_lo_allowed, x0 - dx_cells)
        x_hi = min(x_hi_allowed, x0 + dx_cells)
        y_lo = max(y_lo_allowed, y0 - dy_cells)
        y_hi = min(y_hi_allowed, y0 + dy_cells)
        if x_hi < x_lo or y_hi < y_lo:
            arr = np.empty((0, 2), dtype=np.int32)
            self._grid_radius_cache[cache_key] = arr
            return arr

        xs = np.arange(x_lo, x_hi + 1, dtype=np.int32)
        ys = np.arange(y_lo, y_hi + 1, dtype=np.int32)
        xx, yy = np.meshgrid(xs, ys)
        x_flat = xx.ravel()
        y_flat = yy.ravel()
        if self.region_mode == "anchored_crop":
            x_global = x_flat + x_min_global
            y_global = y_flat + y_min_global
        else:
            x_global = x_flat
            y_global = y_flat
        lat_arr, lon_arr = _xy_center_to_latlon_global(x_global, y_global, self.global_H, self.global_W)
        dist = self._distance_km_local(lat0, lon0, lat_arr, lon_arr)
        keep = np.isfinite(dist) & (dist <= radius_km)
        arr = np.stack([x_flat[keep], y_flat[keep]], axis=1).astype(np.int32)
        self._grid_radius_cache[cache_key] = arr
        return arr

    def _xy_to_key_u32(self, xy):
        xy = np.asarray(xy, dtype=np.int32)
        if xy.size == 0:
            return np.empty((0,), dtype=np.uint32)
        return (xy[:, 1].astype(np.int64) * int(self.W) + xy[:, 0].astype(np.int64)).astype(np.uint32)

    def _key_to_xy_tuple(self, key):
        key = int(key)
        return int(key % self.W), int(key // self.W)

    def _collect_direction_ranges(self, center_keys):
        """Build C1 maps and the C2 spatial prefilter range for direction metrics.

        C0: MAG-active centers.
        C1(c): all output-grid cells within true 100 km of c.
        C2: conservative second expansion, implemented as about 2*100 km from C0.
            C2 is used only to prefilter WSM buckets; exact <=100 km filtering is
            still applied when computing each grid cell's orientation.
        """
        center_keys = np.asarray(center_keys, dtype=np.uint32)
        c1_map = {}
        c1_union_keys = set()
        c2_xy_set = set()
        c2_radius = self.wsm_neighbor_km * float(self.wsm_direction_c2_radius_factor)

        for key in center_keys:
            cx, cy = self._key_to_xy_tuple(int(key))
            c1_xy = self._xy_array_within_radius(cx, cy, self.wsm_neighbor_km, clip_to_output=True)
            c1_keys = self._xy_to_key_u32(c1_xy)
            c1_map[int(key)] = c1_keys
            if c1_keys.size > 0:
                c1_union_keys.update(int(k) for k in c1_keys.tolist())

            # Conservative C2: two-step direction range used for WSM spatial prefilter.
            c2_xy = self._xy_array_within_radius(cx, cy, c2_radius, clip_to_output=False)
            for xx, yy in c2_xy:
                c2_xy_set.add((int(xx), int(yy)))

        d_query_keys = sorted(set(int(k) for k in center_keys.tolist()) | c1_union_keys)
        d_query_xy = np.asarray([(int(k % self.W), int(k // self.W)) for k in d_query_keys], dtype=np.int32)
        c2_xy = np.asarray(sorted(c2_xy_set), dtype=np.int32) if c2_xy_set else np.empty((0, 2), dtype=np.int32)
        return c1_map, np.asarray(d_query_keys, dtype=np.uint32), d_query_xy, c2_xy

    def _collect_wsm_records_in_xy_cells(self, xy_cells, start_day, end_day):
        """Collect WSM records in prefiltered relative-grid cells and time window."""
        start_day, end_day = self._effective_wsm_window(start_day, end_day)
        if end_day < start_day or end_day < 0:
            return None
        if self._wsm_bucket is None or len(self._wsm_bucket) == 0:
            return None
        if xy_cells is None or len(xy_cells) == 0:
            return None

        arrays = defaultdict(list)
        seen = set()
        for xx, yy in np.asarray(xy_cells, dtype=np.int32):
            key = (int(xx), int(yy))
            if key in seen:
                continue
            seen.add(key)
            b = self._wsm_bucket.get(key, None)
            if b is None:
                continue
            lo = np.searchsorted(b["day"], int(start_day), side="left")
            hi = np.searchsorted(b["day"], int(end_day), side="right")
            if hi <= lo:
                continue
            for name in ["lat", "lon", "azi", "q_w", "m_w", "depth", "regime", "x", "y"]:
                arrays[name].append(b[name][lo:hi])

        if len(arrays["lat"]) == 0:
            return None
        return {name: np.concatenate(arrays[name]) for name in arrays}

    def _compute_direction_field_for_grid(self, d_query_xy, wsm_records):
        """Compute local orientation field on D_query = C0 ∪ C1.

        The input WSM records have already been filtered by the two-step C2 spatial
        range and by the frame's WSM time window. This method still applies exact
        <=100 km filtering for each queried grid cell before computing mean_azi and
        STD_LOCAL, preserving the local 100-km definition of the direction field.
        """
        d_query_xy = np.asarray(d_query_xy, dtype=np.int32)
        field = {}
        if d_query_xy.size == 0 or wsm_records is None or len(wsm_records.get("lat", [])) == 0:
            return field

        rec_lat = np.asarray(wsm_records["lat"], dtype=np.float64)
        rec_lon = np.asarray(wsm_records["lon"], dtype=np.float64)
        rec_azi = np.asarray(wsm_records["azi"], dtype=np.float32)
        rec_qw = np.asarray(wsm_records["q_w"], dtype=np.float32)
        rec_mw = np.asarray(wsm_records["m_w"], dtype=np.float32)
        n_rec = rec_lat.size
        if n_rec == 0:
            return field

        use_tree = False
        tree = None
        rec_xy_km = None
        ref_lat = 0.0
        if self.wsm_direction_use_kdtree:
            try:
                from scipy.spatial import cKDTree
                # A fixed projection is used only to obtain generous candidates.
                # Exact local-distance filtering is still performed afterwards.
                q_lats = []
                q_lons = []
                for xx, yy in d_query_xy:
                    la, lo = self._local_xy_to_latlon(int(xx), int(yy))
                    q_lats.append(la)
                    q_lons.append(lo)
                ref_lat = float(np.nanmean(q_lats)) if len(q_lats) else float(np.nanmean(rec_lat))
                cos_ref = max(0.15, abs(math.cos(math.radians(ref_lat))))
                rec_xy_km = np.column_stack([rec_lon * KM_PER_DEG_LAT * cos_ref, rec_lat * KM_PER_DEG_LAT])
                tree = cKDTree(rec_xy_km)
                use_tree = True
            except Exception:
                use_tree = False
                tree = None
                rec_xy_km = None

        for xx, yy in d_query_xy:
            xx = int(xx)
            yy = int(yy)
            lat0, lon0 = self._local_xy_to_latlon(xx, yy)
            if use_tree:
                cos_ref = max(0.15, abs(math.cos(math.radians(ref_lat))))
                q_pt = np.array([lon0 * KM_PER_DEG_LAT * cos_ref, lat0 * KM_PER_DEG_LAT], dtype=np.float64)
                idx = tree.query_ball_point(q_pt, r=self.wsm_neighbor_km * self.wsm_direction_tree_query_factor)
                if len(idx) == 0:
                    continue
                idx = np.asarray(idx, dtype=np.int64)
            else:
                idx = np.arange(n_rec, dtype=np.int64)

            dist = self._distance_km_local(lat0, lon0, rec_lat[idx], rec_lon[idx])
            keep = np.isfinite(dist) & (dist <= self.wsm_neighbor_km)
            if int(np.sum(keep)) < self.wsm_min_n_local:
                continue
            idx2 = idx[keep]
            dist2 = dist[keep]
            dist_w = np.asarray([dist_weight_km(float(d), self.wsm_neighbor_km) for d in dist2], dtype=np.float32)
            weights = rec_qw[idx2] * rec_mw[idx2] * dist_w
            mean_azi, R = axial_mean_R(rec_azi[idx2], weights)
            if not np.isfinite(mean_azi) or not np.isfinite(R):
                continue
            std_deg = float(np.degrees(np.sqrt(-np.log(R) / 2.0)))
            field[(xx, yy)] = {
                "mean": float(mean_azi),
                "std": float(std_deg),
                "std_valid": True,
                "spread_valid": bool(std_deg <= self.wsm_std_threshold),
                "n": int(idx2.size),
            }
        return field

    def _compute_wsm_metrics_for_centers(self, center_keys, start_day, end_day):
        """
        Compute the six non-MAG WSM channels for MAG-active centers.

        Solution B direction-range implementation
        -----------------------------------------
        MAG is not changed. Depth/regime metrics still use the original center-level
        100-km WSM query. Direction metrics are changed to the intended grid-field
        execution:
            C0 = MAG-active centers;
            C1(c) = all output-grid cells within true 100 km of center c;
            D_query = C0 ∪ C1;
            C2 = conservative two-step spatial range used to prefilter WSM records;
            direction_field is computed once on D_query;
            STD_LOCAL(c) = direction_field[c].std;
            MAX_SPREAD(c) = max axial spread of valid direction_field[g], g ∈ C1(c).

        This removes the expensive nested pattern:
            center -> WSM 100-km query -> occupied cells -> another WSM 100-km query.
        """
        start_day, end_day = self._effective_wsm_window(start_day, end_day)
        center_keys = np.asarray(center_keys, dtype=np.uint32)
        U = center_keys.size
        depth_vals = np.zeros((U,), dtype=np.float32)
        depth_std_vals = np.zeros((U,), dtype=np.float32)
        spread_vals = np.zeros((U,), dtype=np.float32)
        std_vals = np.zeros((U,), dtype=np.float32)
        ent_vals = np.zeros((U,), dtype=np.float32)
        gini_vals = np.zeros((U,), dtype=np.float32)

        depth_valid = np.zeros((U,), dtype=bool)
        depth_std_valid = np.zeros((U,), dtype=bool)
        spread_valid = np.zeros((U,), dtype=bool)
        std_valid = np.zeros((U,), dtype=bool)
        ent_valid = np.zeros((U,), dtype=bool)
        gini_valid = np.zeros((U,), dtype=bool)

        center_metric_cache = self._center_metric_cache if self.wsm_center_metric_cache else None

        # Pending centers are those not already resolved by center-level cache.
        pending_indices = []
        pending_keys = []
        for i, key in enumerate(center_keys):
            metric_cache_key = (int(key), int(start_day), int(end_day))
            if center_metric_cache is not None and metric_cache_key in center_metric_cache:
                cached = center_metric_cache[metric_cache_key]
                (
                    depth_vals[i], depth_std_vals[i], spread_vals[i], std_vals[i], ent_vals[i], gini_vals[i],
                    depth_valid[i], depth_std_valid[i], spread_valid[i], std_valid[i], ent_valid[i], gini_valid[i],
                ) = cached
            else:
                pending_indices.append(i)
                pending_keys.append(int(key))

        if len(pending_indices) == 0:
            return {
                "depth_raw": depth_vals,
                "depth_valid": depth_valid,
                "depth_std_raw": depth_std_vals,
                "depth_std_valid": depth_std_valid,
                "spread_raw": spread_vals,
                "spread_valid": spread_valid,
                "std_raw": std_vals,
                "std_valid": std_valid,
                "entropy_raw": ent_vals,
                "entropy_valid": ent_valid,
                "gini_raw": gini_vals,
                "gini_valid": gini_valid,
            }

        pending_keys_arr = np.asarray(pending_keys, dtype=np.uint32)

        # --------------------------------------------------
        # Direction metrics: C0 -> C1 -> C2 -> direction field.
        # --------------------------------------------------
        c1_map = {}
        direction_field = {}
        if self.wsm_precompute_direction_field:
            c1_map, _, d_query_xy, c2_xy = self._collect_direction_ranges(pending_keys_arr)
            wsm_for_direction = self._collect_wsm_records_in_xy_cells(c2_xy, start_day, end_day)
            direction_field = self._compute_direction_field_for_grid(d_query_xy, wsm_for_direction)

        # --------------------------------------------------
        # Center-level non-direction metrics keep the original query path.
        # STD_LOCAL is not recomputed here when direction field is enabled;
        # it is read from direction_field at the center grid.
        # --------------------------------------------------
        for out_i, key in zip(pending_indices, pending_keys):
            cx = int(key % self.W)
            cy = int(key // self.W)

            # Direction channel 4: STD_LOCAL(c) = direction field std at center c.
            if self.wsm_precompute_direction_field:
                dfc = direction_field.get((cx, cy), None)
                if dfc is not None and bool(dfc.get("std_valid", False)):
                    std_vals[out_i] = float(dfc["std"])
                    std_valid[out_i] = True

                # Direction channel 3: MAX_SPREAD from all valid direction-field cells in C1(c).
                local_azis = []
                for gkey in c1_map.get(int(key), np.empty((0,), dtype=np.uint32)):
                    gx = int(int(gkey) % self.W)
                    gy = int(int(gkey) // self.W)
                    dfg = direction_field.get((gx, gy), None)
                    if dfg is not None and bool(dfg.get("spread_valid", False)):
                        ma = float(dfg.get("mean", np.nan))
                        if np.isfinite(ma):
                            local_azis.append(ma)
                if len(local_azis) >= 2:
                    spread_vals[out_i] = float(max_spread(local_azis))
                    spread_valid[out_i] = True

            # Depth and regime channels: unchanged original center-level 100-km WSM query.
            q = self._query_wsm_neighbors(cx, cy, start_day, end_day)
            if q is not None and q["azi"].size >= self.wsm_min_n_local:
                dep_good = np.isfinite(q["depth"])
                if np.any(dep_good):
                    dep_local = q["depth"][dep_good].astype(np.float32)
                    depth_vals[out_i] = float(np.mean(dep_local))
                    depth_std_vals[out_i] = float(np.std(dep_local, ddof=0))
                    depth_valid[out_i] = True
                    depth_std_valid[out_i] = True

                reg_valid = q["regime"][q["regime"] >= 0]
                if reg_valid.size > 0:
                    _, cnt = np.unique(reg_valid, return_counts=True)
                    p = cnt.astype(np.float64) / cnt.sum()
                    ent_vals[out_i] = float(-np.sum(p * np.log(p)))
                    gini_vals[out_i] = float(1.0 - np.sum(p ** 2))
                    ent_valid[out_i] = True
                    gini_valid[out_i] = True

                # Fallback only if direction-field mode is disabled.
                if not self.wsm_precompute_direction_field:
                    weights = q["q_w"] * q["m_w"] * np.asarray(
                        [dist_weight_km(d, self.wsm_neighbor_km) for d in q["dist_km"]],
                        dtype=np.float32,
                    )
                    _, R = axial_mean_R(q["azi"], weights)
                    if np.isfinite(R):
                        std_vals[out_i] = float(np.degrees(np.sqrt(-np.log(R) / 2.0)))
                        std_valid[out_i] = True
                    local_xy = np.unique(np.stack([q["x"], q["y"]], axis=1), axis=0)
                    local_azis = []
                    local_mean_cache = self._local_mean_global_cache if self.wsm_global_local_mean_cache else {}
                    for lcx, lcy in local_xy:
                        mean_azi, _ = self._compute_local_mean_azi_for_cell(
                            int(lcx), int(lcy), start_day, end_day, local_mean_cache
                        )
                        if np.isfinite(mean_azi):
                            local_azis.append(mean_azi)
                    if len(local_azis) >= 2:
                        spread_vals[out_i] = float(max_spread(local_azis))
                        spread_valid[out_i] = True

            if center_metric_cache is not None:
                metric_cache_key = (int(key), int(start_day), int(end_day))
                center_metric_cache[metric_cache_key] = (
                    float(depth_vals[out_i]), float(depth_std_vals[out_i]), float(spread_vals[out_i]),
                    float(std_vals[out_i]), float(ent_vals[out_i]), float(gini_vals[out_i]),
                    bool(depth_valid[out_i]), bool(depth_std_valid[out_i]), bool(spread_valid[out_i]),
                    bool(std_valid[out_i]), bool(ent_valid[out_i]), bool(gini_valid[out_i]),
                )

        return {
            "depth_raw": depth_vals,
            "depth_valid": depth_valid,
            "depth_std_raw": depth_std_vals,
            "depth_std_valid": depth_std_valid,
            "spread_raw": spread_vals,
            "spread_valid": spread_valid,
            "std_raw": std_vals,
            "std_valid": std_valid,
            "entropy_raw": ent_vals,
            "entropy_valid": ent_valid,
            "gini_raw": gini_vals,
            "gini_valid": gini_valid,
        }

    # ------------------------------------------------------
    # Step 2: frame-first sparse cache + sample references
    # ------------------------------------------------------
    def step2_build_samples(
        self,
        build_tags=("test",),
        resume=True,
        overwrite_stage2=False,
        frame_cache_subdir="stage2_frame_cache_7c_100km",
    ):
        """
        Build Step2 with disk-level frame checkpoint/resume.

        Main use case for the current experiment
        ----------------------------------------
        - stage2_train.npz has already been verified as correct.
        - Only stage2_test.npz needs to be completed.
        - The job walltime is limited, so every computed frame must be written to disk
          immediately and reused after interruption.

        Parameters
        ----------
        build_tags : tuple/list/str
            Tags to build. Recommended now: ("test",). Use ("train", "test") only if
            both stage2 files must be regenerated.
        resume : bool
            If True, existing valid frame cache files are skipped.
        overwrite_stage2 : bool
            If False and stage2_<tag>.npz already exists, that tag is skipped.
            For the current case, keep False so stage2_train.npz is never overwritten.
        frame_cache_subdir : str
            Directory under base_dir used to store per-frame sparse features.
        """
        H, W = self.H, self.W
        pic, dtday = self.pic, self.dtday
        span = (pic + 1) * dtday

        if isinstance(build_tags, str):
            build_tags = [t.strip().lower() for t in build_tags.split(",") if t.strip()]
        else:
            build_tags = [str(t).strip().lower() for t in build_tags]
        build_tags = [t for t in build_tags if t in {"train", "test"}]
        if len(build_tags) == 0:
            raise ValueError("build_tags must contain 'train' and/or 'test'.")

        usgs_fp = os.path.join(self.base_dir, "usgs_by_day.npz")
        wsm_fp = os.path.join(self.base_dir, "wsm_by_day.npz")
        if not os.path.exists(usgs_fp):
            raise FileNotFoundError(f"Missing {usgs_fp}. Run Step1 first or check base_dir.")
        if not os.path.exists(wsm_fp):
            raise FileNotFoundError(f"Missing {wsm_fp}. Run Step1 first or check base_dir.")

        usgs = np.load(usgs_fp, allow_pickle=True)["events_by_day"]
        wsm = np.load(wsm_fp, allow_pickle=True)["events_by_day"]
        n_days = len(usgs)
        assert n_days == len(wsm), "USGS and WSM day counts are inconsistent."

        split = int(n_days * self.percent)
        n_train = max(0, split - span + 1)
        n_test = max(0, n_days - split - span + 1)

        if not any(len(a) > 0 for a in usgs):
            raise RuntimeError("step2_build_samples: usgs_by_day is empty.")
        if not any(len(a) > 0 for a in wsm):
            raise RuntimeError("step2_build_samples: wsm_by_day is empty.")

        def _usgs_kv(d):
            arr = usgs[d]
            if arr is None or len(arr) == 0:
                return None
            x = np.array([int(item["x"]) for item in arr], dtype=np.int32)
            y = np.array([int(item["y"]) for item in arr], dtype=np.int32)
            keys = (y * W + x).astype(np.uint32)
            mag_vals = np.array([float(item["mag"]) for item in arr], dtype=np.float32)
            return {"keys": keys, "mag": mag_vals}

        usgs_kv = [_usgs_kv(i) for i in range(n_days)]
        daily_count = np.array([0 if kv is None else int(kv["keys"].size) for kv in usgs_kv], dtype=np.int32)
        prefix_count = np.concatenate([[0], np.cumsum(daily_count)])

        self._build_wsm_index(wsm)

        frame_cache_dir = os.path.join(self.base_dir, frame_cache_subdir)
        os.makedirs(frame_cache_dir, exist_ok=True)

        required_frame_keys = [
            "keys_union", "mag", "depth_local", "depth_std",
            "spread", "std", "entropy", "gini",
        ]

        def _frame_fp(frame_start):
            return os.path.join(frame_cache_dir, f"frame_{int(frame_start):08d}.npz")

        def _empty_frame():
            return {
                "keys_union": np.empty((0,), dtype=np.uint32),
                "mag": np.empty((0,), dtype=np.float32),
                "depth_local": np.empty((0,), dtype=np.float32),
                "depth_std": np.empty((0,), dtype=np.float32),
                "spread": np.empty((0,), dtype=np.float32),
                "std": np.empty((0,), dtype=np.float32),
                "entropy": np.empty((0,), dtype=np.float32),
                "gini": np.empty((0,), dtype=np.float32),
            }

        def _frame_file_is_valid(frame_start):
            fp = _frame_fp(frame_start)
            if not os.path.exists(fp):
                return False
            try:
                with np.load(fp, allow_pickle=True) as z:
                    if not all(k in z.files for k in required_frame_keys):
                        return False
                    if "channel_order" in z.files:
                        ch = [str(x) for x in z["channel_order"].tolist()]
                        if ch != CHANNEL_ORDER:
                            return False
                    # Minimal structural check.
                    n = len(z["keys_union"])
                    for k in required_frame_keys[1:]:
                        if len(z[k]) != n:
                            return False
                return True
            except Exception:
                return False

        def _save_frame(frame_start, frame):
            fp = _frame_fp(frame_start)
            tmp_fp = fp + ".tmp.npz"
            np.savez_compressed(
                tmp_fp,
                keys_union=np.asarray(frame["keys_union"], dtype=np.uint32),
                mag=np.asarray(frame["mag"], dtype=np.float32),
                depth_local=np.asarray(frame["depth_local"], dtype=np.float32),
                depth_std=np.asarray(frame["depth_std"], dtype=np.float32),
                spread=np.asarray(frame["spread"], dtype=np.float32),
                std=np.asarray(frame["std"], dtype=np.float32),
                entropy=np.asarray(frame["entropy"], dtype=np.float32),
                gini=np.asarray(frame["gini"], dtype=np.float32),
                channel_order=np.asarray(CHANNEL_ORDER, dtype=object),
                frame_start=np.asarray([int(frame_start)], dtype=np.int32),
            )
            os.replace(tmp_fp, fp)

        def _load_frame(frame_start):
            fp = _frame_fp(frame_start)
            with np.load(fp, allow_pickle=True) as z:
                return {k: z[k].copy() for k in required_frame_keys}

        def _sample_has_usgs(L, R):
            L = max(0, int(L))
            R = min(n_days - 1, int(R))
            if R < L:
                return False
            return (prefix_count[R + 1] - prefix_count[L]) > 0

        def _collect_refs(a0, a1, tag):
            sample_refs = []
            frame_set = set()
            for d0 in tqdm(range(a0, a1), desc=f"Step2 collect refs [{a0},{a1}) {tag}", unit="sample"):
                L = int(d0)
                R = int(d0 + span - 1)
                if not _sample_has_usgs(L, R):
                    continue
                starts = [int(d0 + j * dtday) for j in range(pic + 1)]
                sample_refs.append(starts)
                frame_set.update(starts)

            sample_refs = np.asarray(sample_refs, dtype=np.int32) if len(sample_refs) > 0 else np.empty((0, pic + 1), dtype=np.int32)
            frame_starts = np.asarray(sorted(frame_set), dtype=np.int32)
            print(f"[Step2-{tag}] collected samples={len(sample_refs)}, unique_frames={len(frame_starts)}")
            return sample_refs, frame_starts

        def _build_frame(frame_start):
            a = int(frame_start)
            b = min(n_days - 1, a + dtday - 1)
            if a > b:
                return _empty_frame()

            K_mag, V_mag = [], []
            for d in range(a, b + 1):
                kv = usgs_kv[d]
                if kv is None or kv["keys"].size == 0:
                    continue
                K_mag.append(kv["keys"])
                V_mag.append(kv["mag"])

            if len(K_mag) == 0:
                return _empty_frame()

            all_keys = np.concatenate(K_mag)
            all_mag = np.concatenate(V_mag)
            mag_keys, mag_vals = _group_max_by_key(all_keys, all_mag)
            keys_union = np.unique(all_keys).astype(np.uint32)
            U = keys_union.size

            mag_arr = np.zeros((U,), dtype=np.float32)
            if mag_keys.size > 0:
                pos = np.searchsorted(keys_union, mag_keys)
                mag_arr[pos] = mag_vals.astype(np.float32)

            if self.wsm_time_mode == "cumulative":
                wa, wb = 0, b
            else:
                wa, wb = a, b
            wa_raw, wb_raw = wa, wb
            wa, wb = self._effective_wsm_window(wa, wb)

            t0 = time.time()
            metrics = self._compute_wsm_metrics_for_centers(keys_union, wa, wb)
            frame_elapsed = time.time() - t0

            depth_local_mapped = depth_to_float_100_255(
                metrics["depth_raw"], metrics["depth_valid"],
                depth_min=self.depth_min, depth_max=self.depth_max,
            )
            depth_std_mapped = depth_std_to_float_100_255(
                metrics["depth_std_raw"], metrics["depth_std_valid"],
                depth_std_max=self.depth_std_map_max,
            )
            spread_mapped = max_spread_to_float_100_255(metrics["spread_raw"], metrics["spread_valid"])
            std_mapped = std_local_to_float_100_255(metrics["std_raw"], metrics["std_valid"], std_max=self.std_local_map_max)
            ent_mapped = entropy_to_float_100_255(metrics["entropy_raw"], metrics["entropy_valid"], k_regime=self._global_k_regime)
            gini_mapped = gini_to_float_100_255(metrics["gini_raw"], metrics["gini_valid"], k_regime=self._global_k_regime)

            slow_sec = float(getattr(config, "step2_slow_frame_log_sec", 60.0))
            if frame_elapsed >= slow_sec:
                print(
                    f"\n[Step2][SLOW-FRAME] frame_start={a}, frame_end={b}, "
                    f"centers={U}, wsm_window_raw=({wa_raw},{wb_raw}), "
                    f"wsm_window_eff=({wa},{wb}), elapsed={frame_elapsed:.1f}s"
                )

            return {
                "keys_union": keys_union,
                "mag": mag_arr.astype(np.float32),
                "depth_local": depth_local_mapped.astype(np.float32),
                "depth_std": depth_std_mapped.astype(np.float32),
                "spread": spread_mapped.astype(np.float32),
                "std": std_mapped.astype(np.float32),
                "entropy": ent_mapped.astype(np.float32),
                "gini": gini_mapped.astype(np.float32),
            }

        def _compute_or_resume_frames(frame_starts):
            frame_starts = np.asarray(sorted(set(int(x) for x in frame_starts)), dtype=np.int32)
            if len(frame_starts) == 0:
                return
            done_before = sum(1 for fs in frame_starts if _frame_file_is_valid(int(fs)))
            print(
                f"[Step2][FRAME-CACHE] dir={frame_cache_dir}\n"
                f"[Step2][FRAME-CACHE] required={len(frame_starts)}, already_valid={done_before}, missing={len(frame_starts)-done_before}"
            )
            pbar = tqdm(frame_starts, desc="Step2 compute/resume frames", unit="frame")
            for fs in pbar:
                fs = int(fs)
                if resume and _frame_file_is_valid(fs):
                    pbar.set_postfix_str(f"skip fs={fs}")
                    continue
                fr = _build_frame(fs)
                _save_frame(fs, fr)
                pbar.set_postfix_str(f"saved fs={fs}")
                del fr
                gc.collect()
            pbar.close()

        def _write_stage2_from_cache(tag, sample_refs, tag_frame_starts):
            tag_frame_starts = np.asarray(sorted(set(int(x) for x in tag_frame_starts)), dtype=np.int32)
            missing = [int(fs) for fs in tag_frame_starts if not _frame_file_is_valid(int(fs))]
            if missing:
                raise RuntimeError(
                    f"Cannot write stage2_{tag}.npz because {len(missing)} frame files are missing/corrupt. "
                    f"First examples: {missing[:10]}"
                )
            frames = np.empty((len(tag_frame_starts),), dtype=object)
            for i, fs in enumerate(tqdm(tag_frame_starts, desc=f"Step2 load cached frames for {tag}", unit="frame")):
                frames[i] = _load_frame(int(fs))
            out_fp = os.path.join(self.base_dir, f"stage2_{tag}.npz")
            tmp_fp = out_fp + ".tmp.npz"
            np.savez_compressed(
                tmp_fp,
                sample_frame_starts=sample_refs,
                frame_starts=tag_frame_starts,
                frames=frames,
                channel_order=np.asarray(CHANNEL_ORDER, dtype=object),
            )
            os.replace(tmp_fp, out_fp)
            print(f"[Step2-{tag}] written samples={len(sample_refs)}, unique_frames={len(tag_frame_starts)}, file={out_fp}")
            del frames
            gc.collect()

        tag_ranges = {
            "train": (0, n_train),
            "test": (split, split + n_test),
        }

        print(
            f"[Step2] requested build_tags={build_tags}, resume={resume}, overwrite_stage2={overwrite_stage2}, "
            f"wsm_time_mode={self.wsm_time_mode}, wsm_neighbor_km={self.wsm_neighbor_km}"
        )
        print(f"[Step2] n_days={n_days}, split={split}, n_train={n_train}, n_test={n_test}, span={span}")

        collected = {}
        all_required_frames = set()
        for tag in build_tags:
            out_fp = os.path.join(self.base_dir, f"stage2_{tag}.npz")
            if os.path.exists(out_fp) and not overwrite_stage2:
                print(f"[Step2-{tag}] exists and overwrite_stage2=False; skip writing: {out_fp}")
                continue
            a0, a1 = tag_ranges[tag]
            refs, frame_starts = _collect_refs(a0, a1, tag)
            collected[tag] = (refs, frame_starts)
            all_required_frames.update(map(int, frame_starts.tolist()))

        if len(collected) == 0:
            print("[Step2] Nothing to build. All requested stage2 files already exist or build_tags is empty.")
            return

        all_required_frames = np.asarray(sorted(all_required_frames), dtype=np.int32)
        print(f"[Step2] total required unique frames for requested tags={len(all_required_frames)}")
        if len(all_required_frames) > 0:
            print(f"[Step2] required frame range=({int(all_required_frames.min())},{int(all_required_frames.max())})")

        _compute_or_resume_frames(all_required_frames)

        for tag, (refs, frame_starts) in collected.items():
            _write_stage2_from_cache(tag, refs, frame_starts)

        print(
            f"[Step2] Done for tags={list(collected.keys())}. "
            f"channel_order={CHANNEL_ORDER}, regime_class_count={self._global_k_regime}, "
            f"wsm_time_mode={self.wsm_time_mode}, wsm_neighbor_km={self.wsm_neighbor_km}"
        )

    # ------------------------------------------------------
    # Step 3: render sparse frames to dense (T,7,H,W)
    # ------------------------------------------------------
    def genpic(self, sample_frames):
        H, W = self.H, self.W
        T = len(sample_frames)
        out = np.zeros((T, EXPECTED_CHANNELS, H, W), dtype=np.float32)
        for t, fr in enumerate(sample_frames):
            keys = fr.get("keys_union", None)
            if keys is None or keys.size == 0:
                continue
            y = (keys // W).astype(np.int64)
            x = (keys % W).astype(np.int64)
            idx = (y * W + x).astype(np.int64)
            np.maximum.at(out[t, 0].ravel(), idx, np.asarray(fr.get("mag", 0), dtype=np.float32))
            np.maximum.at(out[t, 1].ravel(), idx, np.asarray(fr.get("depth_local", 0), dtype=np.float32))
            np.maximum.at(out[t, 2].ravel(), idx, np.asarray(fr.get("depth_std", 0), dtype=np.float32))
            np.maximum.at(out[t, 3].ravel(), idx, np.asarray(fr.get("spread", 0), dtype=np.float32))
            np.maximum.at(out[t, 4].ravel(), idx, np.asarray(fr.get("std", 0), dtype=np.float32))
            np.maximum.at(out[t, 5].ravel(), idx, np.asarray(fr.get("entropy", 0), dtype=np.float32))
            np.maximum.at(out[t, 6].ravel(), idx, np.asarray(fr.get("gini", 0), dtype=np.float32))
        return out

    def expic(self, out_name=None, pathpre=None, part_size_step3=1, workers_render=0):
        assert self.base_dir and self.stage3_dir
        os.makedirs(self.stage3_dir, exist_ok=True)

        def _load_stage2(tag):
            fp = os.path.join(self.base_dir, f"stage2_{tag}.npz")
            if not os.path.exists(fp):
                return np.empty((0, self.pic + 1), dtype=np.int32), {}, fp
            z = np.load(fp, allow_pickle=True)
            if "sample_frame_starts" in z.files:
                sample_refs = z["sample_frame_starts"]
                frame_starts = z["frame_starts"]
                frames = z["frames"]
                frame_dict = {int(fs): frames[i] for i, fs in enumerate(frame_starts)}
                return sample_refs, frame_dict, fp
            # Backward compatibility with old stage2 object format.
            samples = z["samples"]
            frame_dict = {i: samples[i] for i in range(len(samples))}
            refs = np.arange(len(samples), dtype=np.int32)[:, None]
            return refs, frame_dict, fp

        def _render_and_save(tag):
            sample_refs, frame_dict, fp = _load_stage2(tag)
            print(f"[Step3] load {tag}: samples={len(sample_refs)}, frames={len(frame_dict)}, source={fp}")
            if len(sample_refs) == 0:
                return 0
            buf = []
            part_id = 0
            saved = 0
            pbar = tqdm(total=len(sample_refs), desc=f"render {tag}", unit="sample")
            for i in range(len(sample_refs)):
                refs = sample_refs[i]
                if refs.ndim == 0 or len(refs) == 1 and int(refs[0]) in frame_dict and isinstance(frame_dict[int(refs[0])], (list, np.ndarray)):
                    # old compatibility path, rarely used
                    sample_frames = frame_dict[int(refs[0])]
                else:
                    sample_frames = [frame_dict[int(fs)] for fs in refs]
                arr = self.genpic(sample_frames)  # (T,7,H,W)
                buf.append(arr[np.newaxis, ...])
                if len(buf) >= part_size_step3:
                    out = np.concatenate(buf, axis=0)
                    np.savez_compressed(os.path.join(self.stage3_dir, f"origin_{tag}_part{part_id}.npz"), **{f"origin_{tag}": out})
                    saved += out.shape[0]
                    buf.clear()
                    part_id += 1
                pbar.update(1)
            pbar.close()
            if len(buf) > 0:
                out = np.concatenate(buf, axis=0)
                np.savez_compressed(os.path.join(self.stage3_dir, f"origin_{tag}_part{part_id}.npz"), **{f"origin_{tag}": out})
                saved += out.shape[0]
                buf.clear()
                part_id += 1
            print(f"[Step3] {tag} saved samples={saved}, parts={part_id}")
            return saved

        _render_and_save("train")
        _render_and_save("test")

    # ------------------------------------------------------
    # Step 4: split dense stage3 to X/Y
    # ------------------------------------------------------
    def genfinalFor4(self, path_out, dt=1000, st=0, workers=None, case=0, pathpre="./"):
        if case != 0:
            raise ValueError("The archived manuscript dataset uses case=0 without rotation augmentation.")
        case_str = ["_n", "_r"]
        os.makedirs(self.stage4_dir, exist_ok=True)

        def _list_parts(tag):
            files = [f for f in os.listdir(self.stage3_dir) if f.startswith(f"origin_{tag}_part") and f.endswith(".npz")]
            files.sort(key=lambda s: int(re.search(r"part(\d+)\.npz", s).group(1)))
            return [os.path.join(self.stage3_dir, f) for f in files]

        def _split_tag(tag):
            files = _list_parts(tag)
            gid = 0
            print(f"[Step4] splitting {tag} ...")
            pbar = tqdm(desc=f"split {tag}", unit="sample")
            for fp in files:
                with np.load(fp, allow_pickle=True) as z:
                    arr = z[f"origin_{tag}"]  # (B,T,7,H,W), 0..255
                if arr.ndim != 5:
                    raise ValueError(f"stage3 dimension error: {fp}, shape={arr.shape}")
                if arr.shape[2] == 1:
                    print(f"[Step4][WARN] {os.path.basename(fp)} has C=1; WSM channels are padded with zeros.")
                    pad = np.zeros((arr.shape[0], arr.shape[1], EXPECTED_CHANNELS - 1, arr.shape[3], arr.shape[4]), dtype=arr.dtype)
                    arr = np.concatenate([arr, pad], axis=2)
                elif arr.shape[2] != EXPECTED_CHANNELS:
                    raise ValueError(f"stage3 channel mismatch: {fp}, C={arr.shape[2]}, expected={EXPECTED_CHANNELS}")

                arr = arr.astype(np.float32) / 255.0
                if case == 1:
                    if rotate_sequence is None:
                        raise RuntimeError("case=1 requires rotate_sequence, but rotate import failed.")
                    arr = rotate_sequence(arr)

                X = arr[:, :-1, :, :, :]      # (B,pic,7,H,W)
                Y = arr[:, -1:, 0:1, :, :]    # (B,1,1,H,W), MAG only
                B = arr.shape[0]
                for b in range(B):
                    prefix = os.path.join(self.stage4_dir, f"second_half_{gid}")
                    np.savez_compressed(f"{prefix}{case_str[case]}_{tag}_x.npz", sequence_array=X[b:b + 1])
                    np.savez_compressed(f"{prefix}{case_str[case]}_{tag}_y.npz", sequence_array=Y[b:b + 1])
                    gid += 1
                    pbar.update(1)
                del arr, X, Y
                gc.collect()
            pbar.close()

        _split_tag("train")
        _split_tag("test")
        print("[Step4] done.")

    # ------------------------------------------------------
    # Scheduler
    # ------------------------------------------------------
    def mainGenOriginData(
        self,
        steps=(1, 2, 3, 4),
        usgs_path="./usgs_csv",
        wsm_path="./WSM_Database_2025.csv",
        out_name="data_mag_wsm7c_100km_jp",
        out_root=None,
        part_size_step3=1,
        workers=None,
        dt=1000,
        case=0,
        step2_build_tags=("test",),
        step2_resume=True,
        step2_overwrite_stage2=False,
        step2_frame_cache_subdir="stage2_frame_cache_7c_100km_solutionB_directionrange",
    ):
        if out_root is None:
            out_root = str(CONVLSTM_DATA_DIR)
        self.base_dir = os.path.join(out_root, out_name)
        self.stage3_dir = os.path.join(self.base_dir, "stage3")
        self.stage4_dir = os.path.join(self.base_dir, "stage4")
        os.makedirs(self.base_dir, exist_ok=True)
        self._write_region_meta()

        print("[MAG+WSM7C-100KM-DIRECTIONRANGE] MAG unchanged; direction metrics use C0/C1/C2 grid-field optimization.")
        print("[CHANNEL_ORDER]", CHANNEL_ORDER)
        print("[PATH] usgs_path =", usgs_path)
        print("[PATH] wsm_path  =", wsm_path)
        print("[GRID] global_H global_W =", self.global_H, self.global_W)
        print("[GRID] output_H output_W pic dtday percent =", self.H, self.W, self.pic, self.dtday, self.percent)
        print("[REGION] mode =", self.region_mode, "configured_bounds =", self.region_bounds)
        print("[REGION] crop_box_closed =", self.crop_box, "effective_bounds =", self.effective_bounds)
        print("[WSM] neighbor_km =", self.wsm_neighbor_km, "min_n_local =", self.wsm_min_n_local,
              "std_threshold =", self.wsm_std_threshold, "depth_std_map_max =", self.depth_std_map_max,
              "time_mode =", self.wsm_time_mode)
        print("[WSM-CACHE] effective_snapshot =", self.wsm_effective_snapshot_cache,
              "center_metric_cache =", self.wsm_center_metric_cache,
              "global_local_mean_cache =", self.wsm_global_local_mean_cache,
              "precompute_direction_field =", self.wsm_precompute_direction_field)
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

        if isinstance(steps, int):
            steps = {steps}
        elif isinstance(steps, str):
            steps = set(int(s) for s in re.findall(r"\d+", steps))
        else:
            steps = set(steps)

        if 1 in steps:
            self._step1_usgs(usgs_path)
            self._step1_wsm(wsm_path)
            print("Step 1 is finished! [USGS mag + WSM]")
        if 2 in steps:
            self.step2_build_samples(
                build_tags=step2_build_tags,
                resume=step2_resume,
                overwrite_stage2=step2_overwrite_stage2,
                frame_cache_subdir=step2_frame_cache_subdir,
            )
            print("Step 2 is finished! [disk frame checkpoint + sample refs]")
        if 3 in steps:
            self.expic(out_name=out_name, pathpre=out_root, part_size_step3=part_size_step3, workers_render=0)
            print("Step 3 is finished! [(B,T,7,H,W)]")
        if 4 in steps:
            self.genfinalFor4(path_out=out_name, dt=dt, case=case, pathpre=out_root)
            print("Step 4 is finished! [X=(B,pic,7,H,W), Y=(B,1,1,H,W)]")


def parse_args():
    parser = argparse.ArgumentParser(description="Build the seven-channel ConvLSTM dataset.")
    parser.add_argument("--steps", default="1,2,3,4", help="Comma-separated stages to run.")
    parser.add_argument("--usgs-dir", type=Path, default=RAW_DATA_DIR / "usgs_csv")
    parser.add_argument("--wsm-file", type=Path, default=RAW_DATA_DIR / "WSM_Database_2025.csv")
    parser.add_argument("--output-root", type=Path, default=CONVLSTM_DATA_DIR)
    parser.add_argument("--output-name", default="dataset")
    parser.add_argument("--overwrite-stage2", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    steps = tuple(int(value.strip()) for value in args.steps.split(",") if value.strip())
    generator = MultiProcessDataset()
    generator.mainGenOriginData(
        steps=steps,
        usgs_path=str(args.usgs_dir),
        wsm_path=str(args.wsm_file),
        out_name=args.output_name,
        out_root=str(args.output_root),
        part_size_step3=1,
        dt=1000,
        case=0,
        step2_build_tags=("train", "test"),
        step2_resume=not args.no_resume,
        step2_overwrite_stage2=args.overwrite_stage2,
        step2_frame_cache_subdir="stage2_frame_cache_7c_100km_solutionB_directionrange",
    )


if __name__ == "__main__":
    main()
