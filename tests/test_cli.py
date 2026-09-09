from __future__ import annotations

import pandas as pd
import pytest

import cesm_hawc.file_index as file_index_module
import cesm_hawc.orbit_files as orbit_files_module
from cesm_hawc.cli import _build_orbit_track_subdaily_jobs, build_parser, main
from cesm_hawc.config import OrbitConfig


def test_build_parser_requires_mode():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["save-inputs", "--config", "config.toml"])  # missing --mode


def test_save_inputs_dry_run_single(tmp_path, capsys):
    config = tmp_path / "config.toml"
    config.write_text(
        '[single]\n'
        'waccm_background = "/nonexistent/background.nc"\n'
        'waccm_injection  = ""\n'
        'time_idx = 0\n'
        'obs_time = "2035-01-01T00:00:00Z"\n'
        f'out_dir = "{tmp_path}"\n'
        '\n'
        '[geometry]\n'
        'tangent_lat = 30.6\n'
        'tangent_lon = 180.0\n'
        '\n'
        '[instrument]\n'
        'wavelengths_nm = [470.0, 745.0, 1020.0]\n'
    )
    # Should resolve config and report a job count without touching any
    # (nonexistent) WACCM file.
    main(["save-inputs", "--config", str(config), "--mode", "single", "--dry-run"])


def test_require_sim_deps_exits_with_install_instructions(monkeypatch):
    import builtins

    from cesm_hawc.cli import _require_sim_deps

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name in ("hawcsimulator", "sasktran2"):
            raise ImportError(f"no module named {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(SystemExit) as exc_info:
        _require_sim_deps()
    assert "cesm-hawc[sim]" in str(exc_info.value)


def test_build_orbit_track_subdaily_jobs_nearest_snapshot(tmp_path, monkeypatch):
    """Verifies the core correctness claim of h2_cadence='subdaily': every
    observation is assigned to whichever h2 snapshot is nearest to it in
    real elapsed time, including across a calendar-date boundary (an
    observation near midnight closer to the *next* date's snapshot than to
    the current date's own) -- not a fixed clock-time split, and not
    silently collapsed to one file per day the way index_by_date would."""
    snapshots = {
        pd.Timestamp("2030-01-07 00:00:00"): "/fake/2030-01-07-00000.nc",
        pd.Timestamp("2030-01-07 12:00:00"): "/fake/2030-01-07-43200.nc",
        pd.Timestamp("2030-01-08 00:00:00"): "/fake/2030-01-08-00000.nc",
        pd.Timestamp("2030-01-08 12:00:00"): "/fake/2030-01-08-43200.nc",
    }

    def fake_extract_observations(orbit_paths, sim_date, cadence_s, center_pixel, epoch):  # noqa: ARG001
        base = pd.Timestamp(sim_date)
        # +1h  -> clearly nearest this date's own 00:00 snapshot
        # +11h -> clearly nearest this date's own 12:00 snapshot
        # +23h -> nearest the *next* date's 00:00 snapshot if one exists
        #         (only 1h away vs. 11h to this date's own 12:00), else
        #         falls back to this date's own 12:00 (last available)
        return [
            {"time": base + pd.Timedelta(hours=1), "lat": 0.0, "lon": 0.0,
             "observer_lat": 0.0, "observer_lon": 0.0, "observer_alt": 0.0},
            {"time": base + pd.Timedelta(hours=11), "lat": 0.0, "lon": 0.0,
             "observer_lat": 0.0, "observer_lon": 0.0, "observer_alt": 0.0},
            {"time": base + pd.Timedelta(hours=23), "lat": 0.0, "lon": 0.0,
             "observer_lat": 0.0, "observer_lon": 0.0, "observer_alt": 0.0},
        ]

    monkeypatch.setattr(orbit_files_module, "load_orbit_files", lambda *a, **k: ["/fake/orbit.nc"])
    monkeypatch.setattr(orbit_files_module, "build_orbit_day_index", lambda *a, **k: {0: ["/fake/orbit.nc"]})
    monkeypatch.setattr(orbit_files_module, "extract_observations", fake_extract_observations)
    monkeypatch.setattr(file_index_module, "index_by_timestamp", lambda *a, **k: dict(snapshots))

    o = OrbitConfig(
        out_dir=str(tmp_path), n_workers=1, orbit_dir="/fake", waccm_data_dir="/fake",
        case_name="test_case", h2_cadence="subdaily", obs_cadence_s=720,
    )

    # extract_observations is called once per case date; case dates come from
    # the mocked snapshots' own dates, so both 2030-01-07 and 2030-01-08 get
    # their +1h/+11h/+23h observations generated above.
    jobs = _build_orbit_track_subdaily_jobs(o, str(tmp_path), alt_grid_m=None, run_l2=False,
                                             case_name="test_case")

    by_label = {label: obs for label, obs, *_ in jobs}
    assert set(by_label.keys()) == {
        "2030-01-07-00000", "2030-01-07-43200", "2030-01-08-00000", "2030-01-08-43200",
    }

    def times(label):
        return sorted(pd.Timestamp(o["time"]) for o in by_label[label])

    # 07 00:00 bucket: only 07's own +1h (01:00) -- +23h from 07 (23:00) moved to 08 00:00.
    assert times("2030-01-07-00000") == [pd.Timestamp("2030-01-07 01:00:00")]
    # 07 12:00 bucket: 07's own +11h (11:00) only.
    assert times("2030-01-07-43200") == [pd.Timestamp("2030-01-07 11:00:00")]
    # 08 00:00 bucket: 08's own +1h (01:00) AND 07's +23h (07 23:00, nearer to 08 00:00 than to 07 12:00).
    assert times("2030-01-08-00000") == [
        pd.Timestamp("2030-01-07 23:00:00"), pd.Timestamp("2030-01-08 01:00:00"),
    ]
    # 08 12:00 bucket: 08's own +11h (11:00) AND 08's own +23h (08 23:00 -- no 09th
    # snapshot exists, so it falls back to the last available, 08 12:00).
    assert times("2030-01-08-43200") == [
        pd.Timestamp("2030-01-08 11:00:00"), pd.Timestamp("2030-01-08 23:00:00"),
    ]
