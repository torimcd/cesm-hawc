"""
cesm_hawc.calibration
======================
Workarounds for ``hawcsimulator``'s calibration-database cache, needed only
when running many simulations in parallel or in a batch. Requires
``hawcsimulator`` ([sim] extra).
"""

from __future__ import annotations

import logging
import os

log = logging.getLogger(__name__)

_patched = False


def _cache_file_path(name: str, version: str) -> str:
    cache_dir = os.path.expanduser("~/.local/share/hawc-simulator/ali/calibration")
    return os.path.join(cache_dir, f"{name}_{version}.nc")


def patch_calibration_database_race() -> None:
    """
    ``hawcsimulator``'s ``calibration_database()`` unconditionally rewrites
    its cached .nc file (``clobber=True``) every time it's called, including
    internally whenever an ``IdealALISimulator`` is constructed. Under many
    worker processes hitting the same (often NFS-mounted) cache file
    concurrently, this produces ``PermissionError``/``KeyError`` races.

    This patches it to be idempotent: if the cache file already exists on
    disk, skip the rewrite and trust it. Safe because the cache content only
    depends on the ``(name, version)`` pair, which is fixed per run.

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
    """
    Pre-build the calibration database once, serially, in the main process
    before dispatching worker processes. If ``n_workers > 1`` and all
    workers start simultaneously, they otherwise race to create the cache
    file, causing ``PermissionError`` on shared filesystems for all but the
    first.
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
    """
    Pre-build ``aliprocessing``'s own retrieval-side Mie database
    (``aerosol_median_radius_db()``) once, serially, in the main process
    before dispatching worker processes.
    """
    try:
        from aliprocessing.l2.optical import aerosol_median_radius_db
    except ImportError:
        return
    try:
        aerosol_median_radius_db()
    except Exception as e:
        log.warning("Could not pre-warm retrieval optical database: %s", e)
