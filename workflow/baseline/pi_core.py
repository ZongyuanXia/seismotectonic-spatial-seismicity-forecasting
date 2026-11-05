"""PI map from sparse daily local earthquake counts.

For each base day b, z-score the rates on [b,t1) and [b,t2) separately,
subtract them, average over b, square, subtract the spatial mean, and keep
positive values. The two-pass accumulation is algebraically the same as
materializing every full-grid z-score map; only floating-point summation order
can differ. Every spatial mean/variance includes all H*W cells.
"""
from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class PIResult:
    image: np.ndarray       # float32, shape (H,W), dimensionless hotspot score
    default_value: np.float32
    base_days: int


def _update(vector, pixel_index, counts, total, square_total, sign):
    """Add or remove one sparse day and update sum/count-squared exactly."""
    if len(pixel_index) == 0:
        return total, square_total
    indices = pixel_index.astype(np.intp, copy=False)
    delta = counts.astype(np.int64, copy=False) * sign
    old = vector[indices]
    new = old + delta
    if np.any(new < 0):
        raise ValueError("Negative interval event count; daily-count index is inconsistent.")
    vector[indices] = new
    total += float(delta.sum(dtype=np.int64))
    square_total += float(np.sum(delta.astype(np.float64) * (old + new), dtype=np.float64))
    return total, square_total


def _normalizer(total, square_total, pixels, days, epsilon):
    """Return 1/(std(count)+days*epsilon) and mean(count).

    z-score(count/days, epsilon) = (count-mean(count)) /
    (std(count) + days*epsilon). Constant maps have zero z-score.
    """
    mean = total / pixels
    variance = max(0.0, square_total / pixels - mean * mean)
    if variance == 0.0:
        return 0.0, mean
    return 1.0 / (math.sqrt(variance) + days * epsilon), mean


def compute_pi(daily, day0, input_frames, days_per_frame, epsilon=1e-12):
    height = int(daily.metadata["height"])
    width = int(daily.metadata["width"])
    pixels = height * width
    t0 = int(day0)
    t2 = t0 + input_frames * days_per_frame
    t1 = t2 - days_per_frame
    last_base = t1 - days_per_frame
    if not (0 <= t0 <= last_base < t1 < t2 <= int(daily.metadata["n_days"])):
        raise ValueError("PI time boundaries fall outside the daily catalog.")
    if epsilon < 0:
        raise ValueError("epsilon must be nonnegative.")
    k = last_base - t0 + 1

    # The only dense counts held in memory: two current interval states.
    first = np.zeros(pixels, dtype=np.int64)
    second = np.zeros(pixels, dtype=np.int64)
    sum_first = square_first = sum_second = square_second = 0.0
    for day in range(t0, t2):
        indices, counts = daily.get(day)
        sum_second, square_second = _update(
            second, indices, counts, sum_second, square_second, +1)
        if day < t1:
            sum_first, square_first = _update(
                first, indices, counts, sum_first, square_first, +1)

    weight_first = np.empty(k, dtype=np.float64)
    weight_second = np.empty(k, dtype=np.float64)
    constant = 0.0
    for step in range(k):
        base_day = t0 + step
        wf, mf = _normalizer(sum_first, square_first, pixels, t1 - base_day, epsilon)
        ws, ms = _normalizer(sum_second, square_second, pixels, t2 - base_day, epsilon)
        weight_first[step] = wf
        weight_second[step] = ws
        constant += mf * wf - ms * ws
        if step + 1 < k:
            indices, counts = daily.get(base_day)
            sum_first, square_first = _update(
                first, indices, counts, sum_first, square_first, -1)
            sum_second, square_second = _update(
                second, indices, counts, sum_second, square_second, -1)

    del first, second
    prefix_first = np.cumsum(weight_first)
    prefix_second = np.cumsum(weight_second)
    signed_change = np.zeros(pixels, dtype=np.float64)
    for day in range(t0, t2):
        indices, counts = daily.get(day)
        if len(indices) == 0:
            continue
        if day < t1:
            base_index = min(day - t0, k - 1)
            weight = prefix_second[base_index] - prefix_first[base_index]
        else:
            weight = prefix_second[-1]
        signed_change[indices.astype(np.intp, copy=False)] += counts * (weight / k)

    default_signed = constant / k
    signed_change += default_signed
    np.square(signed_change, out=signed_change)
    background = float(signed_change.mean(dtype=np.float64))
    signed_change -= background
    np.maximum(signed_change, 0.0, out=signed_change)
    default_value = np.float32(max(default_signed * default_signed - background, 0.0))
    image = signed_change.reshape(height, width).astype(np.float32)
    return PIResult(image=image, default_value=default_value, base_days=k)
