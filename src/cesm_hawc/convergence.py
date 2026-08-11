"""
cesm_hawc.convergence
======================
L2 retrieval convergence diagnostics.

skretrieval's l2 ``xr.Dataset`` carries no reliable convergence flag in its
``.attrs``. The trustworthy source is ``scipy.optimize.least_squares``'
``verbose=2`` stdout — confirmed against real production output, where a
genuine non-convergence was caught this way that a naive attrs-based check
missed entirely. Capture stdout during the retrieval call (e.g. via
``contextlib.redirect_stdout``) and parse it with
``parse_scipy_convergence()``.
"""

from __future__ import annotations

import re

_CONVERGED_PATTERNS = [
    (re.compile(r"`ftol` termination condition is satisfied"), "ftol"),
    (re.compile(r"`xtol` termination condition is satisfied"), "xtol"),
    (re.compile(r"`gtol` termination condition is satisfied"), "gtol"),
]
_NOT_CONVERGED_PATTERNS = [
    (re.compile(r"maximum number of function evaluations is exceeded", re.IGNORECASE), "max_nfev"),
    (re.compile(r"maximum number of iterations is exceeded", re.IGNORECASE), "max_iter"),
]
_NFEV_PATTERN = re.compile(r"Function evaluations (\d+)")


def parse_scipy_convergence(captured_stdout: str) -> dict:
    """Parse ``scipy.optimize.least_squares``' ``verbose=2`` stdout for the
    real convergence status and function-evaluation count. Returns
    ``{converged, termination_reason, n_function_evaluations}``, with
    ``None`` values if no recognized message was found."""
    result = {"converged": None, "termination_reason": None, "n_function_evaluations": None}

    for pattern, reason in _CONVERGED_PATTERNS:
        if pattern.search(captured_stdout):
            result["converged"] = True
            result["termination_reason"] = reason
            break
    else:
        for pattern, reason in _NOT_CONVERGED_PATTERNS:
            if pattern.search(captured_stdout):
                result["converged"] = False
                result["termination_reason"] = reason
                break

    m = _NFEV_PATTERN.search(captured_stdout)
    if m:
        result["n_function_evaluations"] = int(m.group(1))

    return result


def extract_l2_native_diagnostics(l2_obj) -> dict:
    """Pull ``num_iterations``/``cost`` directly off the l2 Dataset, as a
    cross-check against the stdout-parsed convergence info."""
    if l2_obj is None:
        return {"l2_num_iterations": None, "l2_final_cost": None}
    try:
        n_iter = int(l2_obj["num_iterations"].values) if "num_iterations" in l2_obj else None
    except Exception:
        n_iter = None
    try:
        cost = float(l2_obj["cost"].values) if "cost" in l2_obj else None
    except Exception:
        cost = None
    return {"l2_num_iterations": n_iter, "l2_final_cost": cost}
