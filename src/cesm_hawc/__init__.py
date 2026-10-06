"""Simulate HAWC ALI observations and retrievals from CESM2/WACCM output.

The package has two tiers. The base install extracts WACCM columns and
saves simulator-ready inputs (:mod:`cesm_hawc.waccm`,
:mod:`cesm_hawc.save_inputs`). The ``[sim]`` extra adds the ALI forward
model and L2 retrieval (:mod:`cesm_hawc.simulation`,
:mod:`cesm_hawc.constituents`). Call :func:`configure_environment` once
before using the ``[sim]`` tier from your own code.
"""

from cesm_hawc.env import configure_environment

__all__ = ["configure_environment"]

__version__ = "0.2.0"
