"""Run independent jobs serially or across a pool of worker processes."""

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
    """Run ``fn(*job)`` for every job, serially or in parallel.

    Parameters
    ----------
    fn : callable
        Job function. Must be defined at module level so it can be pickled
        for worker processes.
    jobs : list of tuple
        Positional arguments for each call of ``fn``.
    n_workers : int
        Number of worker processes. ``1`` or less runs the jobs serially in
        this process, which is easiest to debug; otherwise up to
        ``min(n_workers, len(jobs))`` processes are used.
    max_tasks_per_child : int, optional
        Replace each worker process after this many jobs, which limits memory
        growth from state not released between jobs. Use ``1`` for jobs known
        to leak memory. Default: workers are never replaced.
    on_result : callable, optional
        Called with each result as it completes, e.g. for progress logging.

    Returns
    -------
    list
        The result of every job. In parallel runs they are in completion
        order, not job order.

    Raises
    ------
    concurrent.futures.process.BrokenProcessPool
        If a worker process is killed abruptly (e.g. out of memory). The
        whole pool is lost, including jobs still in progress; output already
        written by finished jobs is unaffected, so resumable job functions
        can continue when the same job list is re-run.

    Notes
    -----
    cesm-hawc's job functions return a status string starting with
    ``"OK"`` or ``"FAIL"``, but this function doesn't depend on the result
    type.
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
