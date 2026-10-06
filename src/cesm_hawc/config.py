"""Read and validate the ``config.toml`` file used by the CLI.

A config has one required table, ``[case]`` (what to read and where to
write), one table per run mode (``[fixed]``, ``[orbit]``) and an optional
``[instrument]`` table. Paths are expanded with ``os.path.expanduser`` when
loaded.

Examples
--------
>>> from cesm_hawc.config import load_config
>>> cfg = load_config("config.toml")
>>> cfg.case.name
'my_case'
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ConfigError(Exception):
    """A config file, table or required key is missing or invalid."""


def _require(table: dict, key: str, section: str) -> Any:
    """``table[key]``, or ``ConfigError`` naming the table if it's missing."""
    if key not in table:
        raise ConfigError(f"config.toml [{section}] is missing required key '{key}'")
    return table[key]


def _expand(path: str | None) -> str | None:
    """``os.path.expanduser(path)``, passing empty values through."""
    return os.path.expanduser(path) if path else path


@dataclass(frozen=True)
class CaseConfig:
    """The ``[case]`` table: the model case to process and shared options.

    Attributes
    ----------
    name : str
        Free-form label for the case; output is written to
        ``out_dir/<name>/``.
    waccm_dir : str
        Directory of CAM history files. May contain a ``{name}`` placeholder
        (see :meth:`waccm_dir_for`), so one config can drive several cases
        stored side by side, e.g. ``"/archive/{name}/atm/hist"``.
    pattern : str
        Glob pattern selecting the history files, e.g. ``"*.cam.h0.*.nc"``.
    out_dir : str
        Output root.
    n_workers : int
        Worker processes. Default 1.
    time_index : int
        Time slice read from each history file. Default 0.
    start_date, end_date : str or None
        Optional inclusive date range (``"YYYY-MM-DD"``), compared with the
        date in each file name.
    run_l2 : bool
        Also run the L2 retrieval. Default False.
    strip_ozone : bool
        Zero WACCM ozone before simulating; output goes to
        ``out_dir/<name>_no_ozone/``. Default False.
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
        """Build from the parsed TOML table ``d``, applying defaults.

        Raises
        ------
        ConfigError
            If a required key is missing or a value is invalid.
        """
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
        """Return ``waccm_dir`` with ``{name}`` replaced.

        Parameters
        ----------
        name : str
            Case name to substitute, e.g. a ``--case-name`` override.

        Returns
        -------
        str
            The resolved directory.
        """
        return self.waccm_dir.replace("{name}", name)


@dataclass(frozen=True)
class FixedConfig:
    """The ``[fixed]`` table: one column at a fixed point per history file.

    Attributes
    ----------
    tangent_lat, tangent_lon : float
        Tangent point [degrees]; longitude may be −180–180 or 0–360.
    sza_deg, saa_deg : float
        Solar zenith and azimuth angles at the tangent point [degrees].
        Defaults 60 and 0.
    obs_time : str or None
        Observation time for every file. If None, it is taken from each
        file name (see :func:`cesm_hawc.file_index.filename_time`).
    """

    tangent_lat: float
    tangent_lon: float
    sza_deg: float = 60.0
    saa_deg: float = 0.0
    obs_time: str | None = None

    @classmethod
    def from_toml_dict(cls, d: dict) -> "FixedConfig":
        """Build from the parsed TOML table ``d``, applying defaults.

        Raises
        ------
        ConfigError
            If a required key is missing or a value is invalid.
        """
        return cls(
            tangent_lat=float(_require(d, "tangent_lat", "fixed")),
            tangent_lon=float(_require(d, "tangent_lon", "fixed")),
            sza_deg=float(d.get("sza_deg", 60.0)),
            saa_deg=float(d.get("saa_deg", 0.0)),
            obs_time=d.get("obs_time") or None,
        )


@dataclass(frozen=True)
class OrbitConfig:
    """The ``[orbit]`` table: observations along a real HAWC orbit track.

    The *n*-th model date (sorted) uses orbit day *n* mod *D*, where *D* is
    the number of days the orbit set spans from ``orbit_epoch``.

    Attributes
    ----------
    orbit_dir : str
        Directory of orbit files.
    orbit_pattern : str
        Glob pattern for orbit files. Default ``"orbit_*.nc"``.
    orbit_epoch : str
        Time origin of the orbit file set; orbit ``time`` values are
        seconds since this date. Default ``"2019-08-01"``.
    center_pixel : int
        Across-track pixel used as the tangent point. Default 256.
    h2_cadence : {"daily", "subdaily"}
        ``"daily"`` (default) assumes one history file per date and runs one
        job per date. ``"subdaily"`` is for several files per date (e.g.
        12-hourly): one job runs per file, and each observation is assigned
        to the nearest snapshot in time, including across midnight. With
        ``"daily"``, only one file per date would be kept.
    obs_cadence_s : float
        Spacing between sampled observations [s]. Default 60.
    """

    orbit_dir: str
    orbit_pattern: str = "orbit_*.nc"
    orbit_epoch: str = "2019-08-01"
    center_pixel: int = 256
    h2_cadence: str = "daily"
    obs_cadence_s: float = 60.0

    @classmethod
    def from_toml_dict(cls, d: dict) -> "OrbitConfig":
        """Build from the parsed TOML table ``d``, applying defaults.

        Raises
        ------
        ConfigError
            If a required key is missing or a value is invalid.
        """
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
    """The ``[instrument]`` table: simulated wavelengths and altitude grid.

    The noise model is not configurable; see
    :func:`cesm_hawc.noise.default_noise_model`.

    Attributes
    ----------
    wavelengths_nm : list of float
        Simulated wavelengths [nm]. Default ``[470.0, 745.0, 1020.0]``.
    alt_grid_start_m, alt_grid_stop_m, alt_grid_step_m : float
        Altitude grid start, inclusive end and spacing [m]. Defaults 0,
        65000 and 1000.
    """

    wavelengths_nm: list[float]
    alt_grid_start_m: float
    alt_grid_stop_m: float
    alt_grid_step_m: float

    @classmethod
    def from_toml_dict(cls, d: dict) -> "InstrumentConfig":
        """Build from the parsed TOML table ``d``, applying defaults."""
        return cls(
            wavelengths_nm=list(d.get("wavelengths_nm", [470.0, 745.0, 1020.0])),
            alt_grid_start_m=float(d.get("alt_grid_start_m", 0.0)),
            alt_grid_stop_m=float(d.get("alt_grid_stop_m", 65000.0)),
            alt_grid_step_m=float(d.get("alt_grid_step_m", 1000.0)),
        )

    def altitude_grid_m(self):
        """Return the altitude grid.

        Returns
        -------
        numpy.ndarray
            Altitudes [m] from ``alt_grid_start_m`` to ``alt_grid_stop_m``
            inclusive, every ``alt_grid_step_m``.
        """
        import numpy as np
        return np.arange(
            self.alt_grid_start_m,
            self.alt_grid_stop_m + self.alt_grid_step_m,
            self.alt_grid_step_m,
        )


@dataclass(frozen=True)
class CesmHawcConfig:
    """A loaded config file, as returned by :func:`load_config`.

    Attributes
    ----------
    case : CaseConfig
        The ``[case]`` table.
    fixed : FixedConfig or None
        The ``[fixed]`` table, or None if absent.
    orbit : OrbitConfig or None
        The ``[orbit]`` table, or None if absent.
    instrument : InstrumentConfig
        The ``[instrument]`` table, with defaults if absent.
    """

    case: CaseConfig
    fixed: FixedConfig | None
    orbit: OrbitConfig | None
    instrument: InstrumentConfig


def load_config(path: str | Path) -> CesmHawcConfig:
    """Load and validate a config file.

    Parameters
    ----------
    path : str or pathlib.Path
        The TOML file.

    Returns
    -------
    CesmHawcConfig
        The parsed config.

    Raises
    ------
    ConfigError
        If the file, the ``[case]`` table or a required key is missing, or a
        value is invalid. ``[fixed]`` and ``[orbit]`` are optional here; the
        CLI reports a missing one when its mode is run.
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
