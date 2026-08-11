from __future__ import annotations

import pandas as pd

from cesm_hawc.orbit import generate_sso_ground_track


def test_ground_track_bounds_and_cadence():
    start = pd.Timestamp("2035-01-01T00:00:00Z")
    end = pd.Timestamp("2035-01-01T02:00:00Z")
    track = generate_sso_ground_track(start, end, cadence_s=60.0, inclination_deg=98.0)

    assert len(track) == int((end - start).total_seconds() // 60.0) + 1
    assert track["time"].iloc[0] == start
    assert track["lat"].between(-98.0, 98.0).all()
    assert track["lon"].between(-180.0, 180.0).all()


def test_ground_track_altitude_changes_period():
    start = pd.Timestamp("2035-01-01T00:00:00Z")
    end = pd.Timestamp("2035-01-01T06:00:00Z")
    low = generate_sso_ground_track(start, end, cadence_s=300.0, altitude_km=400.0)
    high = generate_sso_ground_track(start, end, cadence_s=300.0, altitude_km=800.0)
    # a higher orbit has a longer period -> fewer oscillations in the same
    # window -> latitude at a fixed later time differs between the two
    assert not low["lat"].equals(high["lat"])
