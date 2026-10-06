"""The ALI instrument noise model used throughout cesm-hawc.

Every caller in this package uses :func:`default_noise_model` rather than
constructing ``hawcsimulator.noise.ALINoiseModel`` directly, so the noise
settings are defined in one place.
"""

from __future__ import annotations


def default_noise_model():
    """Return the ``ALINoiseModel`` used for every cesm-hawc simulation.

    Returns
    -------
    hawcsimulator.noise.ALINoiseModel
        Noise model with ``straylight_fraction=0.0`` and the
        ``hawcsimulator`` defaults for every other setting.

    Notes
    -----
    ``straylight_fraction`` is fixed at 0.0. It is deliberately not a
    config option or CLI flag.

    The model is not seeded, so repeated runs draw different noise.
    """
    from hawcsimulator.noise import ALINoiseModel

    return ALINoiseModel(straylight_fraction=0.0)
