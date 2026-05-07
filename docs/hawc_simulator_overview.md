# ALI Simulator Architecture

The simulator is built on a Hamilton dataflow framework — a directed graph of processing steps defined as Python functions. The base Simulator class (simulator.py) orchestrates this by loading modules and executing a Hamilton Driver.

## Common Pipeline (all variants)
Every simulator runs the same four-stage pipeline:

Atmosphere → Limb Observation Geometry → Radiative Transfer (FER) → Instrument Model → (optional: POR + L2)

1. Atmosphere — builds atmospheric constituents (Rayleigh, O3, aerosol, solar irradiance, albedo)
2. Limb Observation — sets up viewing geometry with tangent altitudes + 450 km observer altitude
3. FER (Front-End Radiance) — runs SASKTRAN2 radiative transfer to produce Stokes I, Q, U
4. Instrument Model — converts FER → L1B measurements (this is where the variants diverge)

## The Three Simulator Variants
Aspect	IdealALISimulator (Spectrograph)	IdealALISimulator (Imager)	ALIPhase0Simulator (Full)
File	ideal_spectrograph.py	ideal_dolp_imager.py	full_inst.py
Instrument step	ideal_inst	ideal_imager	full_inst
Measurement model	Direct Stokes I, Q, U with added noise	Mueller matrix from 3 polarization angles	Realistic L1A→L1B via ali_l1 package
Min tangent alt	10 km	-500 m	-500 m
Wavelengths	400–800 nm (continuous)	400–800 nm (continuous)	11 specific bands: 610–1560 nm
Error modeling	Fixed DOLP/AOLP/intensity errors	Noise propagated through Mueller inversion + systematic errors	Realistic detector noise/gains
External dep	None	None	ali_l1 package required
Spectrograph (ideal_inst)
inst_model.py / steps/ideal_inst.py

The simplest model. It directly measures Stokes parameters and adds configurable Gaussian errors:

intensity_error: 1% (default)
dolp_error: 0.003
aolp_error: 0.2°
Can output any combination of I, dolp, aolp, q. This is the "ideal" in the sense that the instrument is assumed to directly observe polarization state without optical complexity.

Imager (ideal_imager)
inst_model_imager.py / steps/ideal_imager.py

More physically realistic polarimetric model. It simulates a filter-based imaging polarimeter that takes three measurements at polarization angles (-60°, 0°, 60°), then reconstructs Stokes I, Q, U via Mueller matrix inversion. This means:

Noise is propagated through the matrix inversion (not just added at the end)
Supports pointing errors (pointing_error_1sigma parameter)
Tangent altitude range extends down to -500 m (ground level)
Phase 0 Full Instrument (full_inst)
configurations/full_inst.py / steps/full_inst.py

The highest-fidelity model, representing the actual planned ALI instrument. It uses the external ali_l1 package to simulate realistic L1A→L1B detector processing (gains, detector response, specific band filter set). Only operates at 11 specific wavelength bands (610–1560 nm) rather than a continuous spectrum.

## Where to Look

The configuration classes are in src/hawcsimulator/ali/configurations/ and the step implementations are in src/hawcsimulator/ali/steps/. The notebook notebooks/ali/examples/quickstart.ipynb shows how to use these in practice.

