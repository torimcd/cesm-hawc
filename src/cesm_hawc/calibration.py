"""Workarounds for cache races when many simulations run in parallel.

``hawcsimulator`` and ``aliprocessing`` build cached files the first time a
simulator is constructed. When many worker processes start at once, they
race to write those files. The functions here pre-build the caches once in
the main process and stop ``hawcsimulator`` rewriting its calibration file
on every call. They do nothing if the ``[sim]`` extra isn't installed.
"""

from __future__ import annotations

import logging
import os

log = logging.getLogger(__name__)

_patched = False


def _cache_file_path(name: str, version: str) -> str:
    """Path of ``hawcsimulator``'s cached calibration file for ``(name, version)``."""
    cache_dir = os.path.expanduser("~/.local/share/hawc-simulator/ali/calibration")
    return os.path.join(cache_dir, f"{name}_{version}.nc")


def patch_calibration_database_race() -> None:
    """Stop ``hawcsimulator`` rewriting its calibration cache on every call.

    ``hawcsimulator``'s ``calibration_database()`` rewrites its cached NetCDF
    file every time it is called, including whenever an
    ``IdealALISimulator`` is constructed. With many worker processes using
    the same (often network-mounted) file, this causes ``PermissionError``
    and ``KeyError`` races. After patching, an existing cache file is reused
    instead. This is safe because the file's contents depend only on its
    ``(name, version)``.

    Safe to call more than once; does nothing if ``hawcsimulator`` isn't
    installed. :func:`cesm_hawc.configure_environment` calls it for you.
    """
    global _patched
    if _patched:
        return
    try:
        from hawcsimulator.ali import calibration as _cal_mod
        from hawcsimulator.ali.configurations import (
            ideal_dolp_imager as _ideal_dolp_imager_mod,
        )
    except ImportError:
        return

    _orig_calibration_database = _cal_mod.calibration_database

    def _safe_calibration_database(name: str, version: str):
        cache_file = _cache_file_path(name, version)
        if os.path.exists(cache_file):
            return cache_file
        return _orig_calibration_database(name, version)

    _cal_mod.calibration_database = _safe_calibration_database
    _ideal_dolp_imager_mod.calibration_database = _safe_calibration_database
    _patched = True


def warm_calibration_database(name: str = "ideal_spectrograph", version: str = "v1") -> None:
    """Build ``hawcsimulator``'s calibration cache file once.

    Call this in the main process before starting worker processes, so they
    don't all race to create the file. Failures are logged as warnings, not
    raised.

    Parameters
    ----------
    name, version : str, optional
        Calibration dataset to build. The defaults are the dataset the
        ``ideal_dolp_imager`` simulator uses.
    """
    try:
        from hawcsimulator.ali.calibration import calibration_database
    except ImportError:
        return
    try:
        calibration_database(name, version)
    except Exception as e:
        log.warning("Could not pre-warm calibration database: %s", e)


def warm_retrieval_optical_database() -> None:
    """Build the retrieval's Mie database cache once.

    Every ``IdealALISimulator`` builds ``aliprocessing``'s
    ``aerosol_median_radius_db()`` when constructed. Call this in the main
    process before starting worker processes, so they don't race to write
    it (a worker reading a half-written file fails with an xarray "did not
    find a match in any of xarray's currently installed IO backends"
    error). Failures are logged as warnings, not raised.
    """
    try:
        from aliprocessing.l2.optical import aerosol_median_radius_db
    except ImportError:
        return
    try:
        aerosol_median_radius_db()
    except Exception as e:
        log.warning("Could not pre-warm retrieval optical database: %s", e)
