"""
cesm_hawc.orbit
================
Analytical orbit ground-track generation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_EARTH_RADIUS_M = 6_371_000.0
_GM_EARTH = 3.986004418e14        # m^3 s^-2
_EARTH_ROTATION_RATE = 7.2921150e-5  # rad/s


def generate_sso_ground_track(
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    cadence_s: float,
    altitude_km: float = 600.0,
    inclination_deg: float = 98.0,
    start_lon_deg: float = 0.0,
) -> pd.DataFrame:
    """
    Generate an analytical sun-synchronous orbit ground track.

    Uses circular Keplerian mechanics with Earth's rotation. Good enough for
    a representative HAWCSat orbit if no orbit files available.

    Returns
    -------
    DataFrame with columns: time (pd.Timestamp), lat (degrees), lon (degrees)
    """
    a = _EARTH_RADIUS_M + altitude_km * 1e3
    period_s = 2.0 * np.pi * np.sqrt(a ** 3 / _GM_EARTH)

    omega_orb = 2.0 * np.pi / period_s
    inc = np.radians(inclination_deg)
    raan = np.radians(start_lon_deg)
    total_s = (end_time - start_time).total_seconds()

    n_steps = int(total_s // cadence_s) + 1
    t = np.arange(n_steps) * cadence_s
    theta = omega_orb * t
    lat = np.degrees(np.arcsin(np.sin(inc) * np.sin(theta)))
    u = np.arctan2(np.cos(inc) * np.sin(theta), np.cos(theta))
    lon = np.degrees(raan + u - _EARTH_ROTATION_RATE * t)
    lon = ((lon + 180.0) % 360.0) - 180.0

    return pd.DataFrame({
        "time": [start_time + pd.Timedelta(seconds=float(ti)) for ti in t],
        "lat": lat,
        "lon": lon,
    })
