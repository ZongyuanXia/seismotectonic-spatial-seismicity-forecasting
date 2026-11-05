# -*- coding: utf-8 -*-
"""
02_get_charactrastic.py
事件级动态 Stress2Grid（震前 + 100 km 范围）
用于提取与震级相关的局部构造环境指标（非预测用途）

当前输出指标包括：
    MEAN_LOCAL_AZI            # 震中平均应力场方向（基于震前局部历史事件加权轴向平均）
    STD_LOCAL                 # 区域应力场方向标准差（由一致性系数 R 换算）
    CONSISTENCY_R_MEAN        # 区域应力场方向一致性系数
    MAX_SPREAD_100KM_DEG      # 区域应力场方向偏转角最值（局部有效网格间最大轴向差，<=90°）
    MAX_ROTATION_DEG          # 震中应力场背景旋转角（区域有效网格平均方向 vs 震中所在网格方向）
    REGIONAL_REGIME           # 区域主导机制
    REGIME_ENTROPY            # 区域应力机制香农熵
    REGIME_GINI               # 区域应力机制基尼指数

新增输出指标：
    MEAN_LOCAL_DEPTH          # 震前100 km范围内历史记录平均深度
    MEDIAN_LOCAL_DEPTH        # 震前100 km范围内历史记录中位深度
    STD_LOCAL_DEPTH           # 震前100 km范围内历史记录深度标准差
    N_LOCAL                   # 震前3年内、100 km范围内历史事件数
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'shared'))
from geodesic import haversine_km, angular_bounds_deg

# --------------------------------
# 输入文件
# --------------------------------
EQ_CSV = r"D:\World_force\Project_stress\idea_final\01_regional_data\earthquake_from_wsm_us_southern_california.csv"
WSM_CSV = r"D:\World_force\Project_stress\WSM_Database_2025.csv"

OUT_CSV = r"D:\World_force\Project_stress\idea_final\02_characteristic\sc\events_metrics_sc.csv"

# --------------------------------
# 参数
# --------------------------------
RADIUS_KM = 100.0
MIN_N = 5
STD_THRESHOLD = 25.0   # 局部网格有效性的标准差阈值
GRID_STEP_DEG = 0.1    # 局部网格步长

# ===== 新增 =====
ACTIVITY_TIME_YEARS = 3.0   # 活动性指标时间窗（年）
DEPTH_COL = "DEPTH"         # WSM 中的深度列名


# ========= 基础函数 =========
def try_read_csv(path):
    for enc in ["utf-8-sig", "utf-8", "gbk", "cp936", "latin1"]:
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except Exception:
            continue
    raise RuntimeError(f"CSV 读取失败：{path}")


def haversine(lat1, lon1, lat2, lon2):
    """计算两点间大圆距离（km）。支持标量与数组混合输入。"""
    return haversine_km(lat1, lon1, lat2, lon2)


def axial_mean_R(az_deg, weights):
    """
    轴向圆统计（2θ 变换），适用于 SHmax 这类 0~180° 无向轴。

    输入：
        az_deg  : 一组方位角（度）
        weights : 对应权重

    返回：
        mean    : 轴向平均方向（0~180°）
        R       : 一致性系数 / 合成向量长度（0~1）
    """
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
    """
    轴向方向差（<=90°）。
    适用于 SHmax 这类无向轴。
    """
    d = abs(a - b)
    if d > 180.0:
        d = 360.0 - d
    return min(d, 180.0 - d)


def max_spread(az_list):
    """
    多个轴向角之间的最大轴向差（<=90°）。
    """
    az = np.asarray(az_list, dtype=float)
    az = az[np.isfinite(az)]
    if len(az) < 2:
        return np.nan

    m = 0.0
    for i in range(len(az)):
        for j in range(i + 1, len(az)):
            d = angular_diff(az[i], az[j])
            m = max(m, d)
    return float(m)


def quality_map(q):
    """
    WSM 质量等级映射为权重。
    仅 A/B/C 会在主流程中保留；此处保留 D 映射仅为兼容函数定义。
    """
    q = str(q).upper()
    return {"A": 1.0, "B": 0.8, "C": 0.6, "D": 0.4}.get(q, 0.0)


def method_map(m):
    """
    不同应力/机制类型映射为方法权重。
    """
    m = str(m).upper()
    return {
        "FMS": 1.0, "FMF": 0.9, "BO": 0.8, "DIF": 0.8,
        "HF": 0.7, "GF": 0.6, "GV": 0.6, "OC": 0.5
    }.get(m, 0.7)


def dist_weight(d):
    """
    距离权重：
    在 0.1R 内权重为 1，之后线性下降，在 R 处为 0。
    """
    R = RADIUS_KM
    if d <= 0.1 * R:
        return 1.0
    if d <= R:
        return 1.0 - (d - 0.1 * R) / (0.9 * R)
    return 0.0


def compute_local_grid_metrics(
    ev_lat, ev_lon, sub,
    radius_km=RADIUS_KM,
    grid_step_deg=GRID_STEP_DEG,
    min_n_cell=MIN_N,
    std_threshold=STD_THRESHOLD
):
    """
    在目标事件周围构建局部动态 Stress2Grid 网格。

    对每个局部网格点：
        1) 找到该网格点 radius_km 范围内的历史事件
        2) 对这些事件的 AZI 做距离/质量/方法加权轴向平均
        3) 计算 mean_azi、R、std_deg
        4) 仅保留样本数足够且 std_deg <= std_threshold 的有效网格点

    返回：
        grid_df，列包括：
            glat, glon, mean_azi, R, std_deg, n_cell
    """
    if len(sub) < min_n_cell:
        return pd.DataFrame(columns=["glat", "glon", "mean_azi", "R", "std_deg", "n_cell"])

    lat_bound, lon_bound = angular_bounds_deg(ev_lat, radius_km)
    # Retain the old event-relative grid spacing, but never omit a candidate
    # centre in the exact spherical circle.
    lat_margin = max(radius_km / 111.0, lat_bound)
    lon_margin = max(radius_km / (111.0 * np.cos(np.radians(ev_lat)) + 1e-6), lon_bound)

    lat_min = ev_lat - lat_margin
    lat_max = ev_lat + lat_margin
    lon_min = ev_lon - lon_margin
    lon_max = ev_lon + lon_margin

    lat_grid = np.arange(lat_min, lat_max + 1e-6, grid_step_deg)
    lon_grid = np.arange(lon_min, lon_max + 1e-6, grid_step_deg)

    sub_lat = sub["LAT"].values
    sub_lon = sub["LON"].values
    sub_azi = sub["AZI"].values
    sub_q = sub["QUALITY"].map(quality_map).values
    sub_m = sub["TYPE"].map(method_map).values

    rows = []

    for glat in lat_grid:
        for glon in lon_grid:
            # C1: candidate grid centres must be inside the event-centred circle.
            if haversine(ev_lat, ev_lon, glat, glon) > radius_km:
                continue
            # C2: each retained grid centre uses its own 100-km WSM circle.
            dist = haversine(glat, glon, sub_lat, sub_lon)
            mask = dist <= radius_km

            if mask.sum() < min_n_cell:
                continue

            azi_loc = sub_azi[mask]
            dist_loc = dist[mask]
            w_q = sub_q[mask]
            w_m = sub_m[mask]
            w_d = np.array([dist_weight(d) for d in dist_loc])
            w = w_q * w_m * w_d

            mean_azi, R = axial_mean_R(azi_loc, w)
            if not np.isfinite(mean_azi) or not np.isfinite(R):
                continue

            std_deg = np.degrees(np.sqrt(-np.log(R) / 2.0))
            if std_deg > std_threshold:
                continue

            rows.append({
                "glat": glat,
                "glon": glon,
                "mean_azi": mean_azi,
                "R": R,
                "std_deg": std_deg,
                "n_cell": int(mask.sum())
            })

    return pd.DataFrame(rows)


def get_event_cell_azi(ev_lat, ev_lon, grid_df):
    """
    获取震中所在网格的 mean_azi。
    做法：
        在所有有效网格中，找到距震中最近的网格中心。
    """
    if grid_df is None or len(grid_df) == 0:
        return np.nan

    d = haversine(
        ev_lat, ev_lon,
        grid_df["glat"].values,
        grid_df["glon"].values
    )
    idx = np.argmin(d)
    return float(grid_df.iloc[idx]["mean_azi"])


def grid_mean_azi(grid_df):
    """
    计算所有有效局部网格方向的轴向平均值（等权）。
    该量用于定义局部网格背景场平均方向。
    """
    if grid_df is None or len(grid_df) == 0:
        return np.nan

    az = grid_df["mean_azi"].values
    w = np.ones(len(az), dtype=float)
    mean_azi, _ = axial_mean_R(az, w)
    return mean_azi


# ===== 新增 =====
def compute_depth_stats(sub, depth_col=DEPTH_COL):
    """
    基于震前局部历史记录，计算深度统计量。
    """
    if depth_col not in sub.columns:
        return np.nan, np.nan, np.nan

    depth = pd.to_numeric(sub[depth_col], errors="coerce")
    depth = depth[np.isfinite(depth)]

    if len(depth) == 0:
        return np.nan, np.nan, np.nan

    mean_depth = float(depth.mean())
    median_depth = float(depth.median())

    # 与 pandas 默认行为一致：样本标准差 ddof=1
    std_depth = float(depth.std(ddof=1)) if len(depth) >= 2 else np.nan

    return mean_depth, median_depth, std_depth


# ===== 新增 =====
def count_local_events_in_time_window(sub, ev_date, years=ACTIVITY_TIME_YEARS):
    """
    统计：
        距目标事件 100 km 范围内，
        且位于目标事件发生前 years 年时间窗内的历史事件数量。
    """
    if sub is None or len(sub) == 0:
        return 0

    start_date = ev_date - pd.Timedelta(days=365.25 * years)
    mask = (sub["DATE"] >= start_date) & (sub["DATE"] < ev_date)
    return int(mask.sum())


# ========= 主流程 =========
def main():
    print("[LOAD] EQ catalog ...")
    eq = try_read_csv(EQ_CSV)
    eq["DATE"] = pd.to_datetime(eq["DATE"], errors="coerce", utc=True)

    print("[LOAD] WSM raw data ...")
    wsm = try_read_csv(WSM_CSV)
    wsm["DATE"] = pd.to_datetime(wsm["DATE"], errors="coerce", utc=True)

    # ===== 新增：确保深度列为数值 =====
    if DEPTH_COL in wsm.columns:
        wsm[DEPTH_COL] = pd.to_numeric(wsm[DEPTH_COL], errors="coerce")

    # 仅保留 A/B/C 质量条目
    wsm = wsm[wsm["QUALITY"].isin(["A", "B", "C"])].copy()

    results = []

    for _, evt in eq.iterrows():
        ev_lat = evt["LAT"]
        ev_lon = evt["LON"]
        ev_date = evt["DATE"]

        # -----------------------------
        # 震前 100 km 范围内全部历史记录
        # -----------------------------
        wsm["dist"] = haversine(ev_lat, ev_lon, wsm["LAT"], wsm["LON"])
        history_mask = wsm["DATE"] < ev_date
        sub = wsm[history_mask & (wsm["dist"] <= RADIUS_KM)].copy()
        # Triangle inequality: all records within 100 km of any C1 grid
        # centre are within 200 km of the target event. This is only a
        # candidate pool; every grid uses its own exact 100-km filter.
        grid_sub = wsm[history_mask & (wsm["dist"] <= 2.0 * RADIUS_KM)].copy()

        # -----------------------------
        # 初始化输出指标
        # -----------------------------
        mean_local_azi = np.nan
        std_local = np.nan
        consistency_r_mean = np.nan
        max_spread_100km_deg = np.nan
        max_rotation_deg = np.nan
        regional_regime = np.nan
        regime_entropy = np.nan
        regime_gini = np.nan

        # ===== 新增：深度类指标 =====
        mean_local_depth = np.nan
        median_local_depth = np.nan
        std_local_depth = np.nan

        # ===== 新增：活动性指标 =====
        n_local = 0

        # -----------------------------
        # 深度类指标（基于震前100 km范围内全部历史记录）
        # -----------------------------
        mean_local_depth, median_local_depth, std_local_depth = compute_depth_stats(
            sub=sub,
            depth_col=DEPTH_COL
        )

        # -----------------------------
        # 活动性指标（基于震前3年内、100 km范围内历史记录）
        # -----------------------------
        n_local = count_local_events_in_time_window(
            sub=sub,
            ev_date=ev_date,
            years=ACTIVITY_TIME_YEARS
        )

        # -----------------------------
        # 方向类指标（基于震前历史事件集合）
        # -----------------------------
        if len(sub) >= MIN_N:
            azi = sub["AZI"].values
            dist = sub["dist"].values
            w_q = sub["QUALITY"].map(quality_map).values
            w_m = sub["TYPE"].map(method_map).values
            w_d = np.array([dist_weight(d) for d in dist])
            w = w_d * w_q * w_m

            # 震中平均应力场方向 + 区域方向一致性
            mean_local_azi, consistency_r_mean = axial_mean_R(azi, w)

            # 区域应力场方向标准差
            std_local = np.degrees(np.sqrt(-np.log(consistency_r_mean) / 2.0))

            # -----------------------------
            # 局部动态网格场（用于网格型方向指标）
            # -----------------------------
            grid_df = compute_local_grid_metrics(
                ev_lat=ev_lat,
                ev_lon=ev_lon,
                sub=grid_sub,
                radius_km=RADIUS_KM,
                grid_step_deg=GRID_STEP_DEG,
                min_n_cell=MIN_N,
                std_threshold=STD_THRESHOLD
            )

            # 区域应力场方向偏转角最值：有效网格间最大轴向差
            if len(grid_df) >= 2:
                max_spread_100km_deg = max_spread(grid_df["mean_azi"].values)

            # 震中应力场背景旋转角：
            # 局部有效网格场平均方向 vs 震中所在网格方向
            if len(grid_df) >= 1:
                regional_grid_mean_azi = grid_mean_azi(grid_df)
                event_cell_azi = get_event_cell_azi(ev_lat, ev_lon, grid_df)

                if np.isfinite(regional_grid_mean_azi) and np.isfinite(event_cell_azi):
                    max_rotation_deg = angular_diff(regional_grid_mean_azi, event_cell_azi)

        # -----------------------------
        # 机制类指标（基于震前历史事件集合）
        # -----------------------------
        if len(sub) >= MIN_N and "REGIME" in sub.columns:
            cnt = sub["REGIME"].value_counts()
            if len(cnt) > 0:
                regional_regime = cnt.idxmax()

            total = cnt.sum()
            p = cnt / total

            # 区域应力机制香农熵
            regime_entropy = -np.sum(p * np.log(p))

            # 区域应力机制基尼指数 / Gini-Simpson
            regime_gini = 1.0 - np.sum(p ** 2)

        # -----------------------------
        # 汇总输出
        # -----------------------------
        row = evt.to_dict()
        row.update({
            "MEAN_LOCAL_AZI": mean_local_azi,
            "STD_LOCAL": std_local,
            "CONSISTENCY_R_MEAN": consistency_r_mean,
            "MAX_SPREAD_100KM_DEG": max_spread_100km_deg,
            "MAX_ROTATION_DEG": max_rotation_deg,
            "REGIONAL_REGIME": regional_regime,
            "REGIME_ENTROPY": regime_entropy,
            "REGIME_GINI": regime_gini,

            # ===== 新增 =====
            "MEAN_LOCAL_DEPTH": mean_local_depth,
            "MEDIAN_LOCAL_DEPTH": median_local_depth,
            "STD_LOCAL_DEPTH": std_local_depth,
            "N_LOCAL": n_local,
        })

        results.append(row)

    out = pd.DataFrame(results)
    out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print("[SAVE]", OUT_CSV)
    print("✓ Done.")


if __name__ == "__main__":
    main()