# WACCM output variables for cesm-hawc-ali

## Required variables

Add any missing variables to `fincl` in `user_nl_cam` before running WACCM.

| Variable | Units | Description | Critical? |
|----------|-------|-------------|-----------|
| `T` | K | Temperature on hybrid levels | Required |
| `Q` | kg/kg | Specific humidity (dynamics) | Required |
| `PS` | Pa | Surface pressure | Required |
| `hyam`, `hybm` | — | Hybrid sigma coefficients | Required |
| `O3` | mol/mol | Ozone VMR | **KEY** |
| `NO2` | mol/mol | NO₂ VMR | KEY (zeros if absent) |
| `H2O` | mol/mol | Chemistry H₂O (blended with Q above 100 hPa) | KEY |
| `SO2` | mol/mol | Gas-phase SO₂ (precursor diagnostic) | KEY |
| `so4_a1` | kg/kg | Accumulation mode sulfate mass mixing ratio | **KEY** |
| `so4_a3` | kg/kg | Coarse mode sulfate mass mixing ratio | **KEY** |
| `num_a1` | #/kg | Accumulation mode number mixing ratio | **KEY** |
| `num_a3` | #/kg | Coarse mode number mixing ratio | **KEY** |

## Example fincl addition in user_nl_cam

```
fincl1 = 'O3', 'NO2', 'H2O', 'SO2', 'so4_a1', 'so4_a3', 'num_a1', 'num_a3'
```

## MAM4 mode parameters (WACCM/BWSSP245)

| Mode | Suffix | σ_g | Description |
|------|--------|-----|-------------|
| Accumulation | `_a1` | 1.8 | Fresh SO₂ injection signal |
| Aitken | `_a2` | 1.6 | Minor contribution (not used) |
| **Coarse** | **`_a3`** | **1.2** | **Aged sulfate — dominates ALI signal after ~2 weeks** |

> **Note:** σ_g = 1.2 for the coarse mode is WACCM-specific for stratospheric
> sulfate (Mills et al. 2016). Generic MAM4 uses 1.6 for this mode.

## VBS-SOA naming (BWSSP245)

Secondary organic aerosol tracers are named `soa1_a1` through `soa5_a1`
(not `soa_a1`). These are not used by the ALI simulation.

## H₂O blending

WACCM has two H₂O fields:
- `Q` (specific humidity): well-constrained in the troposphere
- `H2O` (chemistry tracer): well-constrained in the stratosphere

The module blends them with a cosine taper centred at 100 hPa (default),
with a transition width of one decade in log-pressure.

## References

- Mills et al. (2016), *J. Geophys. Res. Atmos.*, 121, 2332–2348
- Liu et al. (2016), *Geosci. Model Dev.*, 9, 505–522
- Richter et al. (2022), ARISE-SAI protocol, *Geosci. Model Dev.*, 15, 8221–8243
