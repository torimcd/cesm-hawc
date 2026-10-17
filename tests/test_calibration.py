from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("hawcsimulator")

from cesm_hawc.calibration import _cache_file_path  # noqa: E402


def _unpatched_calibration_module():
    """A fresh copy of hawcsimulator.ali.calibration, unaffected by
    cesm_hawc.calibration.patch_calibration_database_race."""
    spec = importlib.util.find_spec("hawcsimulator.ali.calibration")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cache_file_path_matches_hawcsimulator():
    """The patch must look for the calibration file where hawcsimulator
    writes it, on every platform."""
    upstream = _unpatched_calibration_module().calibration_database("ideal_spectrograph", "v1")
    assert Path(_cache_file_path("ideal_spectrograph", "v1")) == Path(upstream)
