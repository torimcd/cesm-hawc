# cesm-hawc

This package feeds atmosphere model output from the Community Earth System Model v. 2 (CESM2) Whole Atmosphere Community Climate Model (WACCM) into the [High-altitude Aerosols, Water vapor, and Clouds satellite (HAWCsat)  simulator](https://github.com/usask-arg/hawc-simulator), to
simulate what the Aerosol Limb Imager (ALI) on the HAWCsat satellite would
observe and retrieve for a given model atmosphere. 

## What it does

```{mermaid}
flowchart TB
  subgraph prep ["cesm-hawc"]
    direction LR
    A["CESM2/WACCM<br/>history files"] --> B["Column extraction<br/>T, P, H₂O, O₃, NO₂,<br/>MAM4 sulfate"]
    B --> C["Mode-matched Mie<br/>extinction profiles"]
  end
  subgraph sim ["hawcsimulator + SASKTRAN2"]
    direction LR
    D["ALI forward<br/>model"] --> E["L1b radiance<br/>+ DoLP"]
    E --> F["L2 retrieval<br/>extinction +<br/>median radius"]
  end
  prep --> sim
```

1. Extracts one or many atmospheric columns from WACCM output, at fixed
   locations or along a real HAWC orbit ground track.
2. Converts the MAM4 accumulation- and coarse-mode sulfate into extinction
   profiles using Mie databases matched to each mode's width.
3. Runs the ALI forward model (`IdealALISimulator`) and, optionally, the L2
   retrieval.
4. Writes retrieved aerosol extinction and median radius alongside the
   model "truth" for comparison.

## Two install tiers

| Tier | Install | What you can do |
|------|---------|-----------------|
| Base | `pip install cesm-hawc` | Extract WACCM columns and save simulator-ready input files (`cesm-hawc save-inputs`). Needs only numpy/xarray/scipy/pandas. |
| Simulation | `pip install cesm-hawc[sim]` | Run the forward model and L2 retrieval end to end (`cesm-hawc run`), via `hawcsimulator` and `sasktran2`. |

## Where to go next

- New users: [Installation](installation.md), then the [Quickstart](quickstart.md).
- Preparing a CESM run: [WACCM output variables](waccm_variables.md).
- Choosing how to sample the model: [Run modes](user-guide/run-modes.md).
- How the conversion works: [Methods](background/methods.md).

## Acronyms

| Term | Meaning |
|------|---------|
| ALI | Aerosol Limb Imager, the HAWCsat instrument this package simulates |
| HAWC | High-altitude Aerosols, Water vapour and Clouds mission |
| CESM2 / WACCM | Community Earth System Model v2 / Whole Atmosphere Community Climate Model |
| MAM4 | Four-mode Modal Aerosol Module used by CESM2 |
| SAI | Stratospheric aerosol injection |
| h0 / h2 | CAM history file streams (here: monthly means / daily or sub-daily output) |
| L1b / L2 | Calibrated radiances / retrieved geophysical profiles |
| DoLP | Degree of linear polarization |
