"""Convergence diagnostics for the L2 retrieval.

The retrieval's output dataset carries no reliable convergence flag. The
dependable source is the ``verbose=2`` output that
``scipy.optimize.least_squares`` prints during the retrieval: capture
stdout around the call (e.g. with ``contextlib.redirect_stdout``) and pass
it to :func:`parse_scipy_convergence`.
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
    """Read the convergence status from ``least_squares`` verbose output.

    Parameters
    ----------
    captured_stdout : str
        Everything ``scipy.optimize.least_squares(..., verbose=2)`` printed
        during one retrieval.

    Returns
    -------
    dict
        ``converged`` (bool), ``termination_reason`` (``"ftol"``,
        ``"xtol"``, ``"gtol"``, ``"max_nfev"`` or ``"max_iter"``) and
        ``n_function_evaluations`` (int). Each value is ``None`` if no
        recognized message was found.
    """
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
    """Read the iteration count and final cost from an L2 dataset.

    Use these as a cross-check on :func:`parse_scipy_convergence`.

    Parameters
    ----------
    l2_obj : xarray.Dataset or None
        The ``"l2"`` product from ``simulator.run()``.

    Returns
    -------
    dict
        ``l2_num_iterations`` (int) and ``l2_final_cost`` (float), each
        ``None`` if missing or unreadable.
    """
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
