from __future__ import annotations

import pytest

from cesm_hawc.cli import build_parser, main


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
    assert "conda-forge" in str(exc_info.value)
