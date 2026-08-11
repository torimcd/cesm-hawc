"""
cesm_hawc.env
=============
Explicit, opt-in setup for process-wide state needed by the simulation
([sim] extra) code paths. Nothing in ``cesm_hawc`` mutates global state on
``import`` — call ``configure_environment()`` yourself before running your
own orbit/batch scripts, or use the ``cesm-hawc`` CLI, which calls it once
at startup automatically.

Idempotent: safe to call more than once (e.g. once per worker process).
"""

from __future__ import annotations

import logging
import sys

_configured = False


def configure_environment() -> None:
    """Apply the process-wide tweaks the [sim] code paths rely on:

    - Disable astropy's IERS Earth-orientation auto-download. Solar-geometry
      (SZA/SAA) calculations don't need arcsecond-level precision, and
      compute nodes often have no internet access. Also disables
      ``auto_max_age`` since simulation dates can be years past astropy's
      predictive IERS window (millisecond-scale UT1-UTC drift is irrelevant
      here).
    - Silence the ``hamilton`` logger. Hamilton (the DAG framework
      ``hawcsimulator`` is built on) logs a per-node error box on every node
      exception via ``logger.error()``, including ones deliberately caught
      and handled elsewhere (e.g. a night-side SZA skip) — this can fire
      dozens of times per observation. Genuine failures are still logged by
      the caller.
    - Filter a known-benign ``sys.unraisablehook`` noise source: sasktran2's
      Rust-backed objects being garbage-collected on a different thread than
      they were created on (inside a threaded scheduler) raises an
      "unsendable ... _core_rust" RuntimeError via ``__del__``/GC, which
      cannot be caught with try/except since it never goes through the
      normal call stack. It doesn't affect the correctness of completed
      results, so this hook filters just that message and re-raises
      anything else via the original hook.
    - Patch the calibration-database race condition (see
      ``cesm_hawc.calibration.patch_calibration_database_race``), if
      ``hawcsimulator`` is installed.
    """
    global _configured
    if _configured:
        return

    try:
        from astropy.utils import iers

        iers.conf.auto_download = False
        iers.conf.auto_max_age = None
    except ImportError:
        pass  # astropy is only pulled in by the [sim] extra

    logging.getLogger("hamilton").setLevel(logging.CRITICAL)

    original_unraisablehook = sys.unraisablehook

    def _filtered_unraisablehook(unraisable):
        msg = str(unraisable.exc_value) if unraisable.exc_value else ""
        if "unsendable" in msg and "_core_rust" in msg:
            return
        original_unraisablehook(unraisable)

    sys.unraisablehook = _filtered_unraisablehook

    from cesm_hawc.calibration import patch_calibration_database_race

    patch_calibration_database_race()

    _configured = True
