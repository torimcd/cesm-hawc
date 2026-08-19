"""
cesm_hawc.config
=================
Typed config.toml schema, replacing raw ``tomllib.load()`` + dict indexing.

Copy ``config.example.toml`` to ``config.toml`` at the project root and fill
in your paths, then::

    from cesm_hawc.config import load_config
    cfg = load_config("config.toml")
    cfg.geometry.tangent_lat

Every ``out_dir``-style field is ``os.path.expanduser``'d at load time, so
call sites never need to remember to do it themselves.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ImportError:
        sys.exit("Python < 3.11 requires tomli: pip install tomli")


class ConfigError(Exception):
    """Raised for a missing or malformed config.toml key."""


def _require(table: dict, key: str, section: str) -> Any:
    if key not in table:
        raise ConfigError(f"config.toml [{section}] is missing required key '{key}'")
    return table[key]


def _expand(path: str | None) -> str | None:
    return os.path.expanduser(path) if path else path


@dataclass(frozen=True)
class SingleConfig:
    """``[single]`` — one WACCM h0/h2 file, one column, end to end."""
    waccm_background: str
    waccm_injection: str | None
    time_idx: int
    obs_time: str
    out_dir: str

    @classmethod
    def from_toml_dict(cls, d: dict) -> "SingleConfig":
        return cls(
            waccm_background=_require(d, "waccm_background", "single"),
            waccm_injection=d.get("waccm_injection") or None,
            time_idx=int(d.get("time_idx", 0)),
            obs_time=_require(d, "obs_time", "single"),
            out_dir=_expand(_require(d, "out_dir", "single")),
        )


@dataclass(frozen=True)
class BatchConfig:
    """``[batch]`` — a directory of monthly h0 files, one column per file."""
    waccm_background_dir: str
    waccm_injection_dir: str | None
    h0_pattern: str
    month_filter: list[str]
    out_dir: str
    n_workers: int

    @classmethod
    def from_toml_dict(cls, d: dict) -> "BatchConfig":
        return cls(
            waccm_background_dir=_require(d, "waccm_background_dir", "batch"),
            waccm_injection_dir=d.get("waccm_injection_dir") or None,
            h0_pattern=d.get("h0_pattern", "*.cam.h0.*.nc"),
            month_filter=list(d.get("month_filter", [])),
            out_dir=_expand(_require(d, "out_dir", "batch")),
            n_workers=int(d.get("n_workers", 1)),
        )


@dataclass(frozen=True)
class OrbitConfig:
    """``[orbit]`` — orbit-track runs (``cesm-hawc run --mode orbit-track``):
    a real HAWC orbit-track file set matched to one CESM case's daily h2
    files by day-of-year offset from ``orbit_epoch``, optionally with full
    L2 retrieval. One case per run (``case_name``) -- run it once per case
    (background or injection) you need output for.
    """
    out_dir: str
    n_workers: int
    orbit_dir: str
    waccm_data_dir: str
    case_name: str

    orbit_pattern: str = "orbit_*.nc"
    orbit_epoch: str = "2019-08-01"
    center_pixel: int = 256
    h2_pattern: str = "*.cam.h2.*.nc"
    obs_cadence_s: float = 60.0
    run_start_date: str | None = None
    run_end_date: str | None = None
    run_l2: bool = False
    strip_ozone: bool = False

    @classmethod
    def from_toml_dict(cls, d: dict) -> "OrbitConfig":
        return cls(
            out_dir=_expand(_require(d, "out_dir", "orbit")),
            n_workers=int(d.get("n_workers", 1)),
            orbit_dir=_expand(_require(d, "orbit_dir", "orbit")),
            waccm_data_dir=_expand(_require(d, "waccm_data_dir", "orbit")),
            case_name=_require(d, "case_name", "orbit"),
            orbit_pattern=d.get("orbit_pattern", "orbit_*.nc"),
            orbit_epoch=d.get("orbit_epoch", "2019-08-01"),
            center_pixel=int(d.get("center_pixel", 256)),
            h2_pattern=d.get("h2_pattern", "*.cam.h2.*.nc"),
            obs_cadence_s=float(d.get("obs_cadence_s", 60.0)),
            run_start_date=d.get("run_start_date") or None,
            run_end_date=d.get("run_end_date") or None,
            run_l2=bool(d.get("run_l2", False)),
            strip_ozone=bool(d.get("strip_ozone", False)),
        )


@dataclass(frozen=True)
class OrbitRealConfig:
    """``[orbit_real]`` — per-orbit-file, per-pixel runs
    (``cesm-hawc run --mode orbit-file``)."""
    orbit_dir: str
    orbit_pattern: str
    waccm_background_dir: str
    waccm_injection_dir: str | None
    h2_pattern: str
    out_dir: str
    n_workers: int
    across_indices: list[int]
    time_stride: int

    @classmethod
    def from_toml_dict(cls, d: dict) -> "OrbitRealConfig":
        return cls(
            orbit_dir=_expand(_require(d, "orbit_dir", "orbit_real")),
            orbit_pattern=d.get("orbit_pattern", "orbit_*.nc"),
            waccm_background_dir=_expand(_require(d, "waccm_background_dir", "orbit_real")),
            waccm_injection_dir=_expand(d.get("waccm_injection_dir")) or None,
            h2_pattern=d.get("h2_pattern", "*.cam.h2.*.nc"),
            out_dir=_expand(_require(d, "out_dir", "orbit_real")),
            n_workers=int(d.get("n_workers", 1)),
            across_indices=list(d.get("across_indices", [])),
            time_stride=int(d.get("time_stride", 1)),
        )


@dataclass(frozen=True)
class GeometryConfig:
    """``[geometry]`` — fixed tangent point, shared by ``single``/``batch``."""
    tangent_lat: float
    tangent_lon: float
    sza_deg: float
    saa_deg: float

    @classmethod
    def from_toml_dict(cls, d: dict) -> "GeometryConfig":
        return cls(
            tangent_lat=float(_require(d, "tangent_lat", "geometry")),
            tangent_lon=float(_require(d, "tangent_lon", "geometry")),
            sza_deg=float(d.get("sza_deg", 60.0)),
            saa_deg=float(d.get("saa_deg", 0.0)),
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
    single: SingleConfig | None
    batch: BatchConfig | None
    orbit: OrbitConfig | None
    orbit_real: OrbitRealConfig | None
    geometry: GeometryConfig | None
    instrument: InstrumentConfig


def load_config(path: str | Path) -> CesmHawcConfig:
    """Load and validate config.toml. Each top-level table is optional
    except ``[instrument]``; a mode that needs a missing table raises
    ``ConfigError`` when the CLI tries to use it, not here, so a config
    file only needs to define the tables its intended use case requires.
    """
    path = Path(path)
    if not path.exists():
        raise ConfigError(
            f"config.toml not found at {path}\n"
            "Copy config.example.toml -> config.toml and fill in your paths."
        )
    with open(path, "rb") as f:
        raw = tomllib.load(f)

    return CesmHawcConfig(
        single=SingleConfig.from_toml_dict(raw["single"]) if "single" in raw else None,
        batch=BatchConfig.from_toml_dict(raw["batch"]) if "batch" in raw else None,
        orbit=OrbitConfig.from_toml_dict(raw["orbit"]) if "orbit" in raw else None,
        orbit_real=OrbitRealConfig.from_toml_dict(raw["orbit_real"]) if "orbit_real" in raw else None,
        geometry=GeometryConfig.from_toml_dict(raw["geometry"]) if "geometry" in raw else None,
        instrument=InstrumentConfig.from_toml_dict(raw.get("instrument", {})),
    )
