"""
cesm_hawc.config
=================
Typed config.toml schema.

Copy ``config.example.toml`` to ``config.toml`` at the project root and fill
in your paths, then::

    from cesm_hawc.config import load_config
    cfg = load_config("config.toml")
    cfg.case.name

A config has one required table, ``[case]`` (what to read and where to
write), one table per run mode (``[fixed]``, ``[orbit]``) and an optional
``[instrument]`` table. Every path is ``os.path.expanduser``'d at load
time, so call sites never need to do it themselves.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ConfigError(Exception):
    """Raised for a missing or malformed config.toml key."""


def _require(table: dict, key: str, section: str) -> Any:
    if key not in table:
        raise ConfigError(f"config.toml [{section}] is missing required key '{key}'")
    return table[key]


def _expand(path: str | None) -> str | None:
    return os.path.expanduser(path) if path else path


@dataclass(frozen=True)
class CaseConfig:
    """``[case]`` — one model case: which history files to read, the label
    its output is written under, and options shared by every run mode.

    ``waccm_dir`` may contain a ``{name}`` placeholder, replaced by the
    case name (including a ``--case-name`` override), so one config can
    drive several cases stored side by side, e.g.
    ``"/archive/{name}/atm/hist"``.
    """
    name: str
    waccm_dir: str
    pattern: str
    out_dir: str
    n_workers: int = 1
    time_index: int = 0
    start_date: str | None = None
    end_date: str | None = None
    run_l2: bool = False
    strip_ozone: bool = False

    @classmethod
    def from_toml_dict(cls, d: dict) -> "CaseConfig":
        return cls(
            name=str(_require(d, "name", "case")),
            waccm_dir=_expand(_require(d, "waccm_dir", "case")),
            pattern=_require(d, "pattern", "case"),
            out_dir=_expand(_require(d, "out_dir", "case")),
            n_workers=int(d.get("n_workers", 1)),
            time_index=int(d.get("time_index", 0)),
            start_date=d.get("start_date") or None,
            end_date=d.get("end_date") or None,
            run_l2=bool(d.get("run_l2", False)),
            strip_ozone=bool(d.get("strip_ozone", False)),
        )

    def waccm_dir_for(self, name: str) -> str:
        """``waccm_dir`` with any ``{name}`` placeholder filled in."""
        return self.waccm_dir.replace("{name}", name)


@dataclass(frozen=True)
class FixedConfig:
    """``[fixed]`` — one column at a fixed tangent point and solar geometry,
    simulated once per history file matched by ``[case]``.

    ``obs_time`` sets the observation time for every file. If omitted, it is
    taken from each file's name: ``YYYY-MM`` (monthly files) becomes the
    15th at 12:00 UTC, ``YYYY-MM-DD[-SSSSS]`` that date and second of day.
    """
    tangent_lat: float
    tangent_lon: float
    sza_deg: float = 60.0
    saa_deg: float = 0.0
    obs_time: str | None = None

    @classmethod
    def from_toml_dict(cls, d: dict) -> "FixedConfig":
        return cls(
            tangent_lat=float(_require(d, "tangent_lat", "fixed")),
            tangent_lon=float(_require(d, "tangent_lon", "fixed")),
            sza_deg=float(d.get("sza_deg", 60.0)),
            saa_deg=float(d.get("saa_deg", 0.0)),
            obs_time=d.get("obs_time") or None,
        )


@dataclass(frozen=True)
class OrbitConfig:
    """``[orbit]`` — observations sampled along a real HAWC orbit ground
    track and moved onto the model case's dates.

    The *n*-th model date (sorted) uses orbit day *n* mod *D*, where *D* is
    the number of days the orbit set spans from ``orbit_epoch``.

    ``h2_cadence``: ``"daily"`` (default) assumes one history file per
    calendar date. ``"subdaily"`` is for output written more than once per
    day (e.g. 12-hourly); one job is dispatched per file rather than per
    day, and each observation is assigned to whichever snapshot is nearest
    to it in time, including across midnight. Do not use ``"daily"`` with
    more than one file per date: only one of them would be kept.
    """
    orbit_dir: str
    orbit_pattern: str = "orbit_*.nc"
    orbit_epoch: str = "2019-08-01"
    center_pixel: int = 256
    h2_cadence: str = "daily"
    obs_cadence_s: float = 60.0

    @classmethod
    def from_toml_dict(cls, d: dict) -> "OrbitConfig":
        h2_cadence = d.get("h2_cadence", "daily")
        if h2_cadence not in ("daily", "subdaily"):
            raise ConfigError(
                f"config.toml [orbit] h2_cadence must be 'daily' or 'subdaily', got {h2_cadence!r}"
            )
        return cls(
            orbit_dir=_expand(_require(d, "orbit_dir", "orbit")),
            orbit_pattern=d.get("orbit_pattern", "orbit_*.nc"),
            orbit_epoch=d.get("orbit_epoch", "2019-08-01"),
            center_pixel=int(d.get("center_pixel", 256)),
            h2_cadence=h2_cadence,
            obs_cadence_s=float(d.get("obs_cadence_s", 60.0)),
        )


@dataclass(frozen=True)
class InstrumentConfig:
    """``[instrument]`` — ALI wavelengths and the shared altitude grid.

    There is no ``noise_straylight_fraction`` key: the noise model's
    straylight fraction is always hardcoded to 0.0 (see ``cesm_hawc.noise``)
    and is not user-configurable.
    """
    wavelengths_nm: list[float]
    alt_grid_start_m: float
    alt_grid_stop_m: float
    alt_grid_step_m: float

    @classmethod
    def from_toml_dict(cls, d: dict) -> "InstrumentConfig":
        return cls(
            wavelengths_nm=list(d.get("wavelengths_nm", [470.0, 745.0, 1020.0])),
            alt_grid_start_m=float(d.get("alt_grid_start_m", 0.0)),
            alt_grid_stop_m=float(d.get("alt_grid_stop_m", 65000.0)),
            alt_grid_step_m=float(d.get("alt_grid_step_m", 1000.0)),
        )

    def altitude_grid_m(self):
        import numpy as np
        return np.arange(
            self.alt_grid_start_m,
            self.alt_grid_stop_m + self.alt_grid_step_m,
            self.alt_grid_step_m,
        )


@dataclass(frozen=True)
class CesmHawcConfig:
    """A loaded config.toml: ``case`` is always present; ``fixed`` and
    ``orbit`` are ``None`` when their table is absent."""
    case: CaseConfig
    fixed: FixedConfig | None
    orbit: OrbitConfig | None
    instrument: InstrumentConfig


def load_config(path: str | Path) -> CesmHawcConfig:
    """Load and validate config.toml.

    ``[case]`` is required. ``[fixed]`` and ``[orbit]`` are needed only by
    their own mode, and ``[instrument]`` falls back to its defaults. Raises
    ``ConfigError`` for a missing file, table or required key.
    """
    path = Path(path)
    if not path.exists():
        raise ConfigError(
            f"config.toml not found at {path}\n"
            "Copy config.example.toml -> config.toml and fill in your paths."
        )
    with open(path, "rb") as f:
        raw = tomllib.load(f)

    if "case" not in raw:
        raise ConfigError("config.toml is missing the required [case] table")

    return CesmHawcConfig(
        case=CaseConfig.from_toml_dict(raw["case"]),
        fixed=FixedConfig.from_toml_dict(raw["fixed"]) if "fixed" in raw else None,
        orbit=OrbitConfig.from_toml_dict(raw["orbit"]) if "orbit" in raw else None,
        instrument=InstrumentConfig.from_toml_dict(raw.get("instrument", {})),
    )
