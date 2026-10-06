from __future__ import annotations

from pathlib import Path

import pytest

from cesm_hawc.config import ConfigError, load_config

REPO_ROOT = Path(__file__).parent.parent


def test_load_example_config():
    cfg = load_config(REPO_ROOT / "config.example.toml")
    assert cfg.case.name
    assert cfg.fixed is not None
    assert cfg.orbit is not None
    assert cfg.instrument.wavelengths_nm

    alt_grid = cfg.instrument.altitude_grid_m()
    assert alt_grid[0] == cfg.instrument.alt_grid_start_m
    assert alt_grid[-1] == cfg.instrument.alt_grid_stop_m


def test_missing_file_raises():
    with pytest.raises(ConfigError):
        load_config("/nonexistent/path/config.toml")


def test_missing_required_key_raises(tmp_path):
    path = tmp_path / "bad_config.toml"
    path.write_text('[case]\nname = "x"\n')  # missing waccm_dir, pattern, out_dir
    with pytest.raises(ConfigError):
        load_config(path)


_MINIMAL_CASE = '[case]\nname = "c"\nwaccm_dir = "/d/{name}/hist"\npattern = "*.nc"\nout_dir = "~/out"\n'


def test_optional_tables_are_none_when_absent(tmp_path):
    path = tmp_path / "minimal_config.toml"
    path.write_text(_MINIMAL_CASE)
    cfg = load_config(path)
    assert cfg.fixed is None
    assert cfg.orbit is None
    assert cfg.instrument.wavelengths_nm == [470.0, 745.0, 1020.0]
    assert not cfg.case.run_l2


def test_case_table_is_required(tmp_path):
    path = tmp_path / "no_case.toml"
    path.write_text('[instrument]\nwavelengths_nm = [470.0, 745.0]\n')
    with pytest.raises(ConfigError, match=r"\[case\]"):
        load_config(path)


def test_waccm_dir_name_placeholder(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(_MINIMAL_CASE)
    cfg = load_config(path)
    assert cfg.case.waccm_dir_for("other_case") == "/d/other_case/hist"
    assert not cfg.case.out_dir.startswith("~")


def test_orbit_missing_required_key_raises(tmp_path):
    path = tmp_path / "bad_orbit.toml"
    path.write_text(_MINIMAL_CASE + '[orbit]\ncenter_pixel = 256\n')  # missing orbit_dir
    with pytest.raises(ConfigError):
        load_config(path)


def test_bad_h2_cadence_raises_config_error(tmp_path):
    path = tmp_path / "bad_cadence.toml"
    path.write_text(_MINIMAL_CASE + '[orbit]\norbit_dir = "/o"\nh2_cadence = "hourly"\n')
    with pytest.raises(ConfigError):
        load_config(path)
