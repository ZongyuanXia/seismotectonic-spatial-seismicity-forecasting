"""Geographic distances and conservative search bounds for WSM features.

All inputs are geographic latitude/longitude in degrees. Distances are
two-dimensional surface great-circle distances in kilometres.
"""

from __future__ import annotations

import numpy as np


EARTH_RADIUS_KM = 6371.0088


def haversine_km(lat1, lon1, lat2, lon2):
    """Return great-circle distance in km; supports NumPy broadcasting."""
    p1 = np.deg2rad(np.asarray(lat1, dtype=np.float64))
    l1 = np.deg2rad(np.asarray(lon1, dtype=np.float64))
    p2 = np.deg2rad(np.asarray(lat2, dtype=np.float64))
    l2 = np.deg2rad(np.asarray(lon2, dtype=np.float64))
    a = np.sin((p2 - p1) / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin((l2 - l1) / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def angular_bounds_deg(latitude, radius_km):
    """Conservative latitude/longitude offsets containing a spherical cap.

    Longitude is 180 degrees when the cap reaches a pole. The offsets are
    for candidate retrieval only; haversine_km <= radius_km decides membership.
    """
    lat = float(latitude)
    radius = float(radius_km)
    if not (-90.0 <= lat <= 90.0) or not (0.0 <= radius <= np.pi * EARTH_RADIUS_KM):
        raise ValueError("Invalid latitude or radius")
    delta = radius / EARTH_RADIUS_KM
    lat_delta = np.rad2deg(delta)
    if np.deg2rad(abs(lat)) + delta >= np.pi / 2:
        lon_delta = 180.0
    else:
        lon_delta = np.rad2deg(np.arcsin(min(1.0, np.sin(delta) / np.cos(np.deg2rad(lat)))))
    return float(lat_delta), float(lon_delta)


def unit_sphere_xyz(lat, lon):
    """Map geographic degrees to Earth-centred unit vectors for cKDTree."""
    phi = np.deg2rad(np.asarray(lat, dtype=np.float64))
    lam = np.deg2rad(np.asarray(lon, dtype=np.float64))
    return np.stack((np.cos(phi) * np.cos(lam), np.cos(phi) * np.sin(lam), np.sin(phi)), axis=-1)


def chord_radius(radius_km):
    """Exact unit-sphere chord radius for a great-circle search radius."""
    return 2.0 * np.sin(float(radius_km) / (2.0 * EARTH_RADIUS_KM))


def grid_cell_diagonal_km(dlat_deg, dlon_deg):
    """Conservative global upper bound for a grid cell's surface diagonal."""
    return EARTH_RADIUS_KM * np.deg2rad(np.hypot(float(dlat_deg), float(dlon_deg)))
