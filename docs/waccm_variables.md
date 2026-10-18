# WACCM output variables

## Required variables

Add any missing variables to `fincl` in `user_nl_cam` before running WACCM.

| Variable | Units | Description | If missing |
|----------|-------|-------------|------------|
| `T` | K | Temperature on hybrid levels | Error |
| `Q` | kg/kg | Specific humidity | Error |
| `PS` | Pa | Surface pressure | Error |
| `hyam`, `hybm` | — | Hybrid level coefficients | Error |
| `P0` | Pa | Reference pressure | Warning; 100 000 Pa assumed |
| `O3` | mol/mol | Ozone, used in the simulation | Warning; treated as zero |
| `NO2` | mol/mol | NO₂, used in the simulation | Warning; treated as zero |
| `so4_a1`, `so4_a3` | kg/kg | Accumulation / coarse mode sulfate mass mixing ratio, used in the simulation | Warning; the whole mode is left out of the simulation |
| `num_a1`, `num_a3` | #/kg | Accumulation / coarse mode number mixing ratio, used in the simulation | Warning; the whole mode is left out of the simulation |
| `H2O` | mol/mol | Chemistry water vapour, saved with the column only | Warning; `Q` is used throughout |
| `SO2` | mol/mol | Gas-phase SO₂, saved with the column only | Warning; treated as zero |

`H2O` and `SO2` are extracted and saved by `save-inputs`, but the simulator
doesn't use them.

## Example fincl addition in user_nl_cam

```
fincl1 = 'O3', 'NO2', 'H2O', 'SO2', 'so4_a1', 'so4_a3', 'num_a1', 'num_a3'
```

## MAM4 mode parameters (WACCM/BWSSP245)

| Mode | Suffix | σ_g | Description |
|------|--------|-----|-------------|
| Accumulation | `_a1` | 1.6 | Accumulation-mode sulfate |
| Aitken | `_a2` | 1.6 | Minor contribution (not used) |
| Coarse | `_a3` | 1.2 | Coarse-mode sulfate |

> **Note:** σ_g = 1.6 for the accumulation mode and σ_g = 1.2 for the coarse
> mode are both WACCM-specific for stratospheric sulfate (Mills et al. 2016).

## VBS-SOA naming (BWSSP245)

Secondary organic aerosol tracers are named `soa1_a1` through `soa5_a1`
(not `soa_a1`). These are not used by the ALI simulation.

## H₂O blending

WACCM has two H₂O fields:
- `Q` (specific humidity): well-constrained in the troposphere
- `H2O` (chemistry tracer): well-constrained in the stratosphere

cesm-hawc combines them into one water vapour profile, using `Q` below the
tropopause and `H2O` above it. The switch is centred at 100 hPa and spread
over a factor of 10 in pressure (about 316 to 32 hPa), with a smooth cosine
weighting so each field contributes half at 100 hPa.

The combined profile is saved as `vmr_h2o` in the files `save-inputs`
writes, but the simulator doesn't use it.

## References

- Mills et al. (2016), *J. Geophys. Res. Atmos.*, 121, 2332–2348
- Liu et al. (2016), *Geosci. Model Dev.*, 9, 505–522
- Richter et al. (2022), ARISE-SAI protocol, *Geosci. Model Dev.*, 15, 8221–8243
