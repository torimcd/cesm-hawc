"""Process-wide setup needed by the simulation (``[sim]``) code paths.

Nothing in ``cesm_hawc`` changes global state on import. Call
:func:`configure_environment` yourself before running your own scripts
against the simulator, or use the ``cesm-hawc`` CLI, which calls it once at
startup.
"""

from __future__ import annotations

import logging
import sys

_configured = False


def configure_environment() -> None:
    """Apply the process-wide settings the ``[sim]`` code paths rely on.

    Safe to call more than once (e.g. once in the main process and again in
    each worker process); calls after the first do nothing.

    Notes
    -----
    This function:

    - Disables astropy's IERS Earth-orientation auto-download and its
      ``auto_max_age`` check. Solar-geometry calculations don't need
      arcsecond precision, compute nodes often have no internet access, and
      simulation dates can be years past astropy's predictive IERS window.
    - Silences the ``hamilton`` logger. Hamilton, the DAG framework
      ``hawcsimulator`` is built on, logs an error box for every node
      exception, including ones handled by the caller (such as a night-side
      observation being skipped). Genuine failures are still logged by
      cesm-hawc.
    - Filters one known-benign message from ``sys.unraisablehook``:
      ``sasktran2``'s Rust-backed objects raise an "unsendable ...
      _core_rust" ``RuntimeError`` when garbage-collected on a different
      thread from the one that created them. It cannot be caught with
      ``try``/``except`` and does not affect results. Anything else is
      passed to the original hook.
    - Patches the ``hawcsimulator`` calibration-database race condition, if
      ``hawcsimulator`` is installed (see
      :func:`cesm_hawc.calibration.patch_calibration_database_race`).
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
