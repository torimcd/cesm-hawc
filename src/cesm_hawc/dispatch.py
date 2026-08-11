"""
cesm_hawc.dispatch
===================
Generic resumable worker-pool dispatch, shared by every batch/orbit CLI
mode. Runs serially when ``n_workers <= 1`` (useful for debugging), else
via ``ProcessPoolExecutor``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool

log = logging.getLogger(__name__)


def run_jobs(
    fn: Callable,
    jobs: list[tuple],
    n_workers: int,
    max_tasks_per_child: int | None = None,
    on_result: Callable[[object], None] | None = None,
) -> list:
    """
    Run ``fn(*job)`` for each ``job`` in ``jobs``, serially if
    ``n_workers <= 1`` else across a ``ProcessPoolExecutor`` with up to
    ``min(n_workers, len(jobs))`` workers.

    ``max_tasks_per_child`` recycles each worker process after that many
    jobs, bounding per-worker memory growth from state that isn't released
    between jobs (e.g. an unclosed xarray file handle) — set it to ``1`` for
    jobs known to leak significant memory per call.

    ``on_result`` (optional) is called with each result as it completes, for
    progress logging. Every result is still collected and returned in
    completion order (not necessarily job order) when running in parallel.

    Callers are expected to follow the ``"OK ..."``/``"FAIL ..."`` status
    string convention (or return whatever their own result type is) — this
    function is agnostic to the result shape.

    If a worker is killed abruptly (e.g. OOM), the whole
    ``ProcessPoolExecutor`` is poisoned and every other in-progress job is
    abandoned too, even ones that would have succeeded. Anything already
    written to disk by completed jobs is unaffected; re-running the same
    job list with resumable per-job logic picks up where it left off.
    """
    results: list = []

    if n_workers <= 1:
        for job in jobs:
            result = fn(*job)
            results.append(result)
            if on_result is not None:
                on_result(result)
        return results

    workers = min(n_workers, len(jobs))
    pool_kwargs = {"max_workers": workers}
    if max_tasks_per_child is not None:
        pool_kwargs["max_tasks_per_child"] = max_tasks_per_child

    try:
        with ProcessPoolExecutor(**pool_kwargs) as pool:
            futures = {pool.submit(fn, *job): job for job in jobs}
            for fut in as_completed(futures):
                result = fut.result()
                results.append(result)
                if on_result is not None:
                    on_result(result)
    except BrokenProcessPool:
        log.error(
            "Process pool broke, likely from a worker being killed abruptly "
            "(e.g. OOM). Work already completed and written to disk is "
            "safe; re-submitting this exact job list will skip work done "
            "by resumable job functions and continue from where this run "
            "stopped."
        )
        raise

    return results
