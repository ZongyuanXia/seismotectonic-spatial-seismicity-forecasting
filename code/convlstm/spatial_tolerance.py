"""Latitude-aware spatial expansion used by the formal evaluation.

Core logic
----------
1. 输入数组为 (B, 1, H, W) 或 (B, 1, 1, H, W)。
2. H / row / x_ 表示纬度方向。
3. W / col / y_ 表示经度方向。
4. 为与 Zhang 文献中的评价拓展方向保持一致：
   - 纬度方向采用较大拓展半径：lat_radius = 2 * d；
   - 经度方向采用较小拓展半径：lon_radius = d；
   - 在高纬度带，经度方向像素实际距离变短，因此 lon_radius 随纬度带适当增加。
5. 若 config 中存在 crop_y_min / crop_y_max，则将全球 layerAccess 规则裁剪到当前 anchored crop 区域。
6. 若不存在 crop_y_min / crop_y_max，则使用完整全球图逻辑。
"""

import numpy as np
from config import config


def build_local_rules(height, d=3):
    """
    将全球 layerAccess 拓展规则裁剪到当前 anchored crop 区域。

    依赖:
        config.crop_y_min
        config.crop_y_max
        config.layerAccess

    返回:
        local_rules = [(local_start, local_end, lat_radius, lon_radius), ...]

    注意:
        1. crop_y_min / crop_y_max 是全球母网格中的闭区间行号；
        2. local_start / local_end 是当前区域图中的半开区间；
        3. lat_radius 作用在数组 row 方向，即纬度方向；
        4. lon_radius 作用在数组 col 方向，即经度方向；
        5. 如果 config 中没有 crop_y_min / crop_y_max，则返回 None，表示使用完整全球图逻辑。
    """
    crop_y_min = getattr(config, "crop_y_min", None)
    crop_y_max = getattr(config, "crop_y_max", None)

    if crop_y_min is None or crop_y_max is None:
        return None

    crop_y_min = int(crop_y_min)
    crop_y_max = int(crop_y_max)

    expected_h = crop_y_max - crop_y_min + 1
    if expected_h != height:
        raise ValueError(
            f"spacelayer regional crop height mismatch: "
            f"height={height}, but crop_y_max-crop_y_min+1={expected_h}. "
            f"Please check config.crop_y_min/crop_y_max or dataset H."
        )

    ss, ns = config.layerAccess

    # ============================================================
    # Zhang-style expansion:
    #   row / latitude direction    -> larger radius = 2*d
    #   col / longitude direction  -> base radius = d
    #   longitude radius increases at higher latitudes.
    # ============================================================
    lat_radius = 2 * d

    global_rules = []

    # 中心纬度带：纬度方向 ±2d，经度方向 ±d
    global_rules.append((ss[-1], ns[0], lat_radius, d))

    # 南北两侧纬度带：
    # 纬度方向保持 ±2d；
    # 经度方向随纬度升高逐步增加，以补偿经度方向单像素实际距离变短。
    lon_radii = [d + _ for _ in range(1, len(ss))]

    for i, lon_radius in enumerate(lon_radii):
        global_rules.append((ss[-i - 2], ss[-i - 1], lat_radius, lon_radius))
        global_rules.append((ns[i], ns[i + 1], lat_radius, lon_radius))

    # 当前 crop 的全球半开区间
    crop_start = crop_y_min
    crop_end = crop_y_max + 1

    local_rules = []

    for g_start, g_end, lat_radius, lon_radius in global_rules:
        inter_start = max(crop_start, int(g_start))
        inter_end = min(crop_end, int(g_end))

        if inter_start >= inter_end:
            continue

        local_start = inter_start - crop_start
        local_end = inter_end - crop_start

        local_start = max(0, local_start)
        local_end = min(height, local_end)

        if local_start < local_end:
            local_rules.append((local_start, local_end, lat_radius, lon_radius))

    if len(local_rules) == 0:
        raise ValueError(
            f"No valid local spacelayer rules. "
            f"crop_y_min={crop_y_min}, crop_y_max={crop_y_max}, "
            f"height={height}, layerAccess={config.layerAccess}"
        )

    return local_rules


def spacelayer(x, d=3):
    """
    对预测图或真实图进行空间拓展。

    输入:
        x: (B, 1, H, W) 或 (B, 1, 1, H, W)

    输出:
        xnew: (B, 1, H, W)

    说明:
        - 该函数对连续值图进行拓展；
        - 后续评价时再执行 value01 * 255 >= threshold；
        - 不应先二值化再调用本函数，否则与 evaluate 代码口径不一致。
    """
    x = np.asarray(x)

    if x.ndim == 5 and x.shape[2] == 1:
        x = x[:, :, 0, :, :]
    elif x.ndim != 4:
        raise ValueError(f"spacelayer expects (B,1,H,W) or (B,1,1,H,W); got {x.shape}")

    number, _0, height, width = x.shape
    x = x.reshape((-1, height, width))
    xnew = x.copy()

    # ============================================================
    # 区域 anchored crop 分支
    # ============================================================
    local_rules = build_local_rules(height, d=d)

    if local_rules is not None:
        for idxmin, idxmax, lat_radius, lon_radius in local_rules:
            xnew = spacemini(
                x=x,
                idxmin=idxmin,
                idxmax=idxmax,
                xnew=xnew,
                lat_radius=lat_radius,
                lon_radius=lon_radius,
            )

    else:
        # ========================================================
        # 完整全球图逻辑
        # ========================================================
        ss, ns = config.layerAccess

        lat_radius = 2 * d

        # 中心纬度带
        xnew = spacemini(
            x=x,
            idxmin=ss[-1],
            idxmax=ns[0],
            xnew=xnew,
            lat_radius=lat_radius,
            lon_radius=d,
        )

        # 南北两侧纬度带
        lon_radii = [d + _ for _ in range(1, len(ss))]

        for i, lon_radius in enumerate(lon_radii):
            xnew = spacemini(
                x=x,
                idxmin=ss[-i - 2],
                idxmax=ss[-i - 1],
                xnew=xnew,
                lat_radius=lat_radius,
                lon_radius=lon_radius,
            )
            xnew = spacemini(
                x=x,
                idxmin=ns[i],
                idxmax=ns[i + 1],
                xnew=xnew,
                lat_radius=lat_radius,
                lon_radius=lon_radius,
            )

    xnew = xnew.reshape((number, _0, height, width))
    return xnew


def spacemini(x, idxmin, idxmax, xnew, lat_radius, lon_radius):
    """
    对指定纬度带内的非零点进行局部矩形拓展。

    参数:
        x:       (N, H, W)
        idxmin: 目标纬度带起始 row
        idxmax: 目标纬度带结束 row，半开区间
        xnew:   拓展后的输出
        lat_radius: row / 纬度方向拓展半径
        lon_radius: col / 经度方向拓展半径

    注意:
        np.argwhere(x1 > 0.0) 返回的索引为:
            [_n, row, col]
        其中 row 是纬度方向，col 是经度方向。
    """
    _01, height, width = x.shape

    idxmin = int(idxmin)
    idxmax = int(idxmax)
    lat_radius = int(lat_radius)
    lon_radius = int(lon_radius)

    # 防止区域图或无效纬度带造成空切片
    if idxmax <= 0 or idxmin >= height or idxmin >= idxmax:
        return xnew

    idxmin = max(0, idxmin)
    idxmax = min(height, idxmax)

    x1 = np.zeros_like(x)
    x1[:, idxmin:idxmax, :] = x[:, idxmin:idxmax, :]

    indexs = np.argwhere(x1 > 0.0)

    # 保持旧逻辑：小值先写，大值后写；重叠区域保留较大值对应的覆盖结果。
    indexs = sorted(indexs, key=lambda _1: x1[_1[0], _1[1], _1[2]])

    for _1 in indexs:
        _n, row, col = int(_1[0]), int(_1[1]), int(_1[2])

        top = row - lat_radius if row - lat_radius >= idxmin else idxmin
        bottom = row + lat_radius + 1 if row + lat_radius + 1 <= height else height

        left = col - lon_radius if col - lon_radius >= 0 else 0
        right = col + lon_radius + 1 if col + lon_radius + 1 <= width else width

        xnew[_n, top:bottom, left:right] = x[_n, row, col]

    return xnew

