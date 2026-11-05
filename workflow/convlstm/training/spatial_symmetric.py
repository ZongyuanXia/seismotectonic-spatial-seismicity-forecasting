"""Symmetric spatial matching on the frozen Japan grid, with NumPy only.

All radii are grid cells, NOT kilometres. Within-band widths retain the old
definition. Across a band boundary, two points match if either point's old
unclipped rectangle contains the other. This is the smallest symmetric
superset of those rectangles; it is not a constant-distance geodesic circle.
"""

from __future__ import annotations

import numpy as np

EVALUATION_VERSION = "symmetric_band_union_v1"
MATCHING_RULE = (
    "A matches B iff B is in A's unclipped latitude-band rectangle OR A is in "
    "B's unclipped rectangle; all overlapping values are combined by maximum"
)


def japan_rules(d: int = 3):
    """Half-open row bands; values are (first, last, row_radius, col_radius)."""
    if type(d) is not int or d < 0:
        raise ValueError("d must be a nonnegative integer in grid cells")
    # With d=0 there is no expansion, including at the northern band.
    return ((0, 145, 2 * d, d), (145, 340, 2 * d, d + 1 if d else 0))


def _validate_rules(rules, shape):
    coverage = np.zeros(shape[0], dtype=np.int16)
    parsed = []
    for rule in rules:
        if len(rule) != 4 or any(type(x) is not int for x in rule):
            raise ValueError("Each rule must contain four integers")
        first, last, lat_radius, lon_radius = rule
        if not 0 <= first < last <= shape[0] or min(lat_radius, lon_radius) < 0:
            raise ValueError("Invalid row band or negative expansion radius")
        coverage[first:last] += 1
        parsed.append(rule)
    if not np.all(coverage == 1):
        raise ValueError("Latitude bands must partition all image rows exactly once")
    return tuple(parsed)


def _maximum_shifted(source, radius, axis):
    """Maximum over [-radius,+radius], clipped only by the image edge."""
    result = np.zeros_like(source)
    length = source.shape[axis]
    for offset in range(-min(radius, length - 1), min(radius, length - 1) + 1):
        lo, hi = max(0, -offset), min(length, length - offset)
        src = [slice(None), slice(None)]
        dst = [slice(None), slice(None)]
        src[axis] = slice(lo, hi)
        dst[axis] = slice(lo + offset, hi + offset)
        np.maximum(result[tuple(dst)], source[tuple(src)], out=result[tuple(dst)])
    return result


def _box_maximum(source, lat_radius, lon_radius):
    return _maximum_shifted(_maximum_shifted(source, lat_radius, 0), lon_radius, 1)


def symmetric_expand(image: np.ndarray, d: int = 3, rules=None) -> np.ndarray:
    """Return the maximum nonnegative input value over symmetric neighbours.

    The output is independent of rule order. Adding subthreshold values cannot
    erase a positive hit. At a positive threshold, expanding continuous values
    and thresholding equals thresholding first and expanding the binary mask.
    Custom rules support small reproducible tests; production uses 340x200.
    """
    image = np.asarray(image)
    if (image.ndim != 2 or min(image.shape) < 1 or
            image.dtype.kind not in "bifu" or not np.isfinite(image).all()):
        raise ValueError("Expected a finite, nonempty 2-D numeric image")
    if rules is None:
        if image.shape != (340, 200):
            raise ValueError("The default Japan rules require a 340x200 image")
        rules = japan_rules(d)
    rules = _validate_rules(tuple(rules), image.shape)
    source = np.maximum(image, 0)
    result = source.copy()
    destination_cache = {}
    for first, last, lat_radius, lon_radius in rules:
        # Directed source rectangles, with no artificial band-edge clipping.
        band = np.zeros_like(source)
        band[first:last] = source[first:last]
        if np.any(band):
            np.maximum(result, _box_maximum(band, lat_radius, lon_radius), out=result)
        # Add the transpose relation: destinations also use their own radius.
        radius = lat_radius, lon_radius
        if radius not in destination_cache:
            destination_cache[radius] = _box_maximum(source, *radius)
        np.maximum(result[first:last], destination_cache[radius][first:last],
                   out=result[first:last])
    return result
