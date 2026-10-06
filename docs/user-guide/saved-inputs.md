# Using saved inputs

`cesm-hawc save-inputs` writes each column as a small NetCDF file. These files
can drive the simulator directly, with native `sasktran2` and
`hawcsimulator` calls and no `cesm_hawc` import.

## File contents

Every file contains the extracted WACCM profile on `altitude_m` [m]:

| Variable | Units | Description |
|----------|-------|-------------|
| `pressure_pa` | Pa | Pressure |
| `temperature_k` | K | Temperature |
| `specific_humidity` | kg/kg | Specific humidity |
| `vmr_o3`, `vmr_no2`, `vmr_so2` | mol/mol | Gas volume mixing ratios |
| `vmr_h2o` | mol/mol | Water vapour, blended from `Q` and chemistry `H2O` |
| `n_air_cm3` | cm⁻³ | Air number density |
| `sulfate_a1_N_cm3`, `sulfate_a3_N_cm3` | cm⁻³ | Accumulation / coarse mode number concentration |
| `sulfate_a1_r_um`, `sulfate_a3_r_um` | μm | Accumulation / coarse mode median radius |

with attributes `latitude`, `longitude`, `time_index`, `sigma_a1` and
`sigma_a3`.

When `sasktran2` was installed at save time (and `--profiles-only` was not
used), the file also holds, for each mode `aerosol_accum` and
`aerosol_coarse`:

| Variable | Dims | Description |
|----------|------|-------------|
| `{mode}_extinction_per_m` | `wavelength_nm`, `altitude_m` | Truth extinction at the configured wavelengths [m⁻¹] |
| `{mode}_reference_extinction_per_m` | `altitude_m` | Extinction at the 745 nm reference wavelength [m⁻¹] |
| `{mode}_median_radius_nm` | `altitude_m` | Median radius, clipped to the Mie database range [nm] |

and the attributes needed to rebuild the Mie databases:
`extinction_reference_wavelength_nm`, `mode_width_accum`,
`mode_width_coarse`, `mie_refractive_index`, `mie_wavelength_grid_nm` and
`mie_median_radius_grid_nm`.

The `includes_constituents` attribute (0 or 1) says which kind of file it is.

## Running the simulator from a saved file

The Mie databases themselves can't be stored in a file, so they are rebuilt
from the saved parameters; everything else is read directly.

```python
import xarray as xr
import sasktran2 as sk
from hawcsimulator.ali.configurations.ideal_dolp_imager import IdealALISimulator
from hawcsimulator.noise import ALINoiseModel

ds = xr.open_dataset("my_case.cam.h0.2035-02.nc")   # a file written by save-inputs
assert ds.attrs["includes_constituents"], "file was saved with --profiles-only"
alt_m = ds["altitude_m"].values


def mode_constituent(mode: str) -> sk.constituent.ExtinctionScatterer:
    width_key = "mode_width_accum" if mode == "aerosol_accum" else "mode_width_coarse"
    mode_db = sk.database.MieDatabase(
        sk.mie.distribution.LogNormalDistribution().freeze(mode_width=ds.attrs[width_key]),
        sk.mie.refractive.H2SO4(),          # ds.attrs["mie_refractive_index"]
        ds.attrs["mie_wavelength_grid_nm"],
        median_radius=ds.attrs["mie_median_radius_grid_nm"],
    )
    return sk.constituent.ExtinctionScatterer(
        mode_db,
        altitudes_m=alt_m,
        extinction_per_m=ds[f"{mode}_reference_extinction_per_m"].values,
        extinction_wavelength_nm=ds.attrs["extinction_reference_wavelength_nm"],
        median_radius=ds[f"{mode}_median_radius_nm"].values,
    )


constituents = {
    "o3": sk.constituent.VMRAltitudeAbsorber(sk.optical.O3DBM(), altitudes_m=alt_m, vmr=ds["vmr_o3"].values),
    "no2": sk.constituent.VMRAltitudeAbsorber(sk.optical.NO2Vandaele(), altitudes_m=alt_m, vmr=ds["vmr_no2"].values),
    "aerosol_accum": mode_constituent("aerosol_accum"),
    "aerosol_coarse": mode_constituent("aerosol_coarse"),
}

sim_input = {
    "tangent_latitude": ds.attrs["latitude"],
    "tangent_longitude": ds.attrs["longitude"],
    "tangent_solar_zenith_angle": 60.0,
    "tangent_solar_azimuth_angle": 0.0,
    "altitude_grid": alt_m,
    "polarization_states": ["I", "dolp"],
    "sample_wavelengths": [470.0, 745.0, 1020.0],
    "time": "2035-02-01T12:00:00Z",      # your own observation time
    "l1b_cfg": {"noise_model": ALINoiseModel(straylight_fraction=0.0)},
    "constituents": constituents,
}
data = IdealALISimulator().run(["l2", "front_end_radiance", "l1b"], sim_input)
```

This matches what `cesm-hawc run` does: the `ideal_dolp_imager` instrument
model with the same noise model.

```{note}
The Mie databases are rebuilt from scratch the first time this runs, which
can take a few minutes. `sasktran2` caches them on disk afterwards.
```
