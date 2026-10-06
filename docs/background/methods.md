# Methods

This page describes how cesm-hawc turns a WACCM column into a simulated ALI
observation. Module names refer to the `cesm_hawc` package.

## Column extraction

`waccm.WACCMAtmosphere` selects the model column nearest the requested
latitude and longitude. Longitudes are normalized to 0–360° first, matching
the CAM grid, so −180–180 inputs select the correct column.

### Pressure and altitude

Pressure on model levels comes from the hybrid coefficients,

$$
p_k = a_k p_0 + b_k p_s ,
$$

with $p_0$ from `P0` (100 000 Pa if absent). Heights come from hydrostatic
integration of the hypsometric equation using virtual temperature
$T_v = T\,[1 + (R_v/R_d - 1)\,q]$:

$$
\Delta z_k = \frac{R_d T_{v,k}}{g_0} \ln\frac{p_{k-1/2}}{p_{k+1/2}} .
$$

Interface pressures are the geometric means of adjacent levels, with the
surface pressure at the bottom and half the top-level pressure at the model
top. Each level sits at the midpoint of its layer.

```{important}
Integration uses constant standard gravity $g_0$, so heights are
geopotential heights. At 30 km this is about 140 m below geometric altitude,
and about 320 m at 45 km. Heights are also measured from the local surface
(`z_surface`, default 0 m): surface elevation (`PHIS`) is not added, so over
high terrain the whole column is placed lower than its true altitude.
```

### Interpolation to the output grid

Profiles are interpolated linearly in altitude onto the configured grid
(default 0–65 km every 1 km). Pressure, gas mixing ratios, air density and
aerosol number are interpolated in log space. Values beyond the model's
highest or lowest level are held at the end values.

### Water vapour

WACCM carries tropospheric humidity `Q` and a stratospheric chemistry tracer
`H2O`. They are blended with a cosine taper in log pressure, centred at
100 hPa and one decade wide: `Q` (converted to a mixing ratio) below,
`H2O` above.

## Sulfate aerosol

Two MAM4 modes are used: accumulation (`_a1`) and coarse (`_a3`). The
Aitken mode (`_a2`) is not used.

| Mode | Constituent | Geometric standard deviation $\sigma_g$ |
|------|-------------|---------------------------|
| Accumulation, `_a1` | `aerosol_accum` | 1.6 |
| Coarse, `_a3` | `aerosol_coarse` | 1.2 (Mills et al., 2016) |

### Number and median radius

For each level, number concentration $N$ and lognormal median radius $r_m$
follow from the mode's sulfate mass mixing ratio and number mixing ratio,
using the lognormal mass moment

$$
M = N \cdot \tfrac{4}{3}\pi \rho \, r_m^3 \exp\!\left(\tfrac{9}{2}\ln^2\sigma_g\right),
$$

with particle density $\rho = 1600$ kg m⁻³. Air density is computed from the
level's pressure and temperature. $N$ is floored at 1 m⁻³ and $r_m$ at 1 nm.

% TODO: give the source for rho = 1600 kg m-3 and state whether so4_a* mass
% is treated as dry sulfate.

### Extinction

Each mode gets its own Mie database (`sasktran2`), built for a lognormal size
distribution with that mode's $\sigma_g$ and the H₂SO₄ refractive index:

| Parameter | Value |
|-----------|-------|
| Wavelengths | 470, 525, 745, 869, 1020, 1230, 1450, 1500, 1560, 1750, 2000, 2250, 2500 nm |
| Median radii | 10–590 nm in 10 nm steps |
| Single-scattering albedo | clamped below 1 (to 0.99999), as in `aliprocessing` |

Extinction is $N \times \sigma_\text{ext}(r_m, \lambda)$, where
$\sigma_\text{ext}$ is the distribution-weighted total cross-section from
that database. Median radii are clipped to 10–590 nm. Where $r_m$ is below
10 nm the extinction is set to zero. Truth extinction at other wavelengths is
taken at the nearest database wavelength, so configured wavelengths should be
on the list above.

Each mode enters the simulator as a `sasktran2` `ExtinctionScatterer` defined
by its 745 nm extinction and median radius. The radiative transfer model then
uses the same mode-matched database for extinction and phase function at
every wavelength.

## Simulated atmosphere

`hawcsimulator`'s default atmosphere contains Rayleigh scattering, MIPAS
standard ozone, solar irradiance and a Lambertian surface with albedo 0.3.
cesm-hawc replaces the MIPAS ozone with WACCM ozone and adds WACCM NO₂ and
the two sulfate modes.

The L2 retrieval does not see these assumptions. It retrieves a single
lognormal aerosol mode with $\sigma_g = 1.6$ (`aliprocessing`), with ozone
constrained closely to the MIPAS climatology.

## Instrument model

cesm-hawc uses the `ideal_dolp_imager` configuration of `hawcsimulator`'s
`IdealALISimulator`: a filter-based imaging polarimeter that measures three
polarization angles and recovers radiance and degree of linear polarization,
with tangent altitudes from −0.5 to 50 km.

% TODO: explain why ideal_dolp_imager was chosen over ideal_spectrograph.

### Instrument noise

Noise comes from `hawcsimulator.noise.ALINoiseModel` with
`straylight_fraction = 0` (the upstream default is 0.02). The model is not
seeded, so repeated runs draw different noise.

% TODO: state the rationale for straylight_fraction = 0.

## Orbit sampling

In `orbit` mode, observations come from real orbit files but are
placed on the simulated model dates:

1. The dates of the case's history files are sorted, and the *n*-th date
   uses orbit day *n* mod *D*, where *D* is the number of days the orbit set
   spans. Gaps in the history files shift this pairing.
2. From that orbit day's files, observations are taken every
   `obs_cadence_s` seconds at `center_pixel`. The starting offset is
   rotated from file to file, so that successive orbits sample different
   points along the track instead of repeating the same latitudes.
3. Each observation keeps its real time of day, tangent point and satellite
   position, but its date is replaced with the simulated date. Solar angles
   are computed from the resulting time and position, and night-side
   observations are skipped.

## Sulfate burden

`WACCMAtmosphere.sulfate_column_burden` integrates the sulfate mass of both
modes between 15 and 35 km, giving the burden in mg m⁻². `fixed` mode
reports it in each file's `summary.txt`.

## References

- Liu, X., et al. (2016). Description and evaluation of a new four-mode
  version of the Modal Aerosol Module (MAM4) within version 5.3 of the
  Community Atmosphere Model. *Geosci. Model Dev.*, 9, 505–522.
- Mills, M. J., et al. (2016). Global volcanic aerosol properties derived from
  emissions, 1990–2014, using CESM1(WACCM). *J. Geophys. Res. Atmos.*, 121,
  2332–2348.
