"""
cesm_hawc.noise
================
The ALI simulator's noise model, standardized to one construction site.

``straylight_fraction`` is always hardcoded to 0.0 here — it is not a
config.toml option and should not be constructed ad hoc elsewhere. Every
caller in this package uses ``default_noise_model()`` instead of
instantiating ``ALINoiseModel`` directly.
"""

from __future__ import annotations


def default_noise_model():
    """Return the project's standard ``ALINoiseModel``.

    ``straylight_fraction=0.0`` is a fixed constant, not a tunable
    parameter — do not read it from config or expose it as a CLI flag.
    """
    from hawcsimulator.noise import ALINoiseModel

    return ALINoiseModel(straylight_fraction=0.0)
