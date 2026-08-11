from __future__ import annotations

from pathlib import Path

import pytest

from cesm_hawc.config import ConfigError, load_config

REPO_ROOT = Path(__file__).parent.parent


def test_load_example_config():
    cfg = load_config(REPO_ROOT / "config.example.toml")
    assert cfg.single is not None
    assert cfg.batch is not None
    assert cfg.orbit is not None
    assert cfg.orbit_real is not None
    assert cfg.geometry is not None
    assert cfg.instrument.wavelengths_nm

    alt_grid = cfg.instrument.altitude_grid_m()
    assert alt_grid[0] == cfg.instrument.alt_grid_start_m
    assert alt_grid[-1] == cfg.instrument.alt_grid_stop_m


def test_missing_file_raises():
    with pytest.raises(ConfigError):
        load_config("/nonexistent/path/config.toml")


def test_missing_required_key_raises(tmp_path):
    path = tmp_path / "bad_config.toml"
    path.write_text('[single]\nwaccm_background = "x.nc"\n')  # missing obs_time, out_dir
    with pytest.raises(ConfigError):
        load_config(path)


def test_optional_tables_are_none_when_absent(tmp_path):
    path = tmp_path / "minimal_config.toml"
    path.write_text('[instrument]\nwavelengths_nm = [470.0, 745.0]\n')
    cfg = load_config(path)
    assert cfg.single is None
    assert cfg.batch is None
    assert cfg.orbit is None
    assert cfg.orbit_real is None
    assert cfg.geometry is None
    assert cfg.instrument.wavelengths_nm == [470.0, 745.0]


def test_orbit_track_source_validation(tmp_path):
    path = tmp_path / "bad_orbit.toml"
    path.write_text('[orbit]\ntrack_source = "bogus"\nout_dir = "~/out"\n')
    with pytest.raises(ConfigError):
        load_config(path)
