# Citing cesm-hawc

If you use cesm-hawc in published work, please cite the software:

> McDonald, V.: cesm-hawc, Zenodo, <https://doi.org/10.5281/zenodo.23630940>.

This DOI always points to the latest release; Zenodo also gives each release
its own DOI, listed on that page, for citing the exact version you used. The
repository's `CITATION.cff` file has the full citation details; on GitHub,
use "Cite this repository".

Please also cite the packages that do the radiative transfer and retrieval,
giving the versions you used (`pip show hawcsimulator aliprocessing sasktran2`):

- Zawada, D.: hawcsimulator (version 2026.6.0), University of Saskatchewan,
  <https://github.com/usask-arg/hawc-simulator>.
- Rieger, L. and Zawada, D.: aliprocessing (version 2026.10.0), University of
  Saskatchewan, <https://github.com/usask-arg/ali-processing>.
- SASKTRAN2, <https://github.com/usask-arg/sasktran2>. Its authors ask that
  these papers be acknowledged:
  - Zawada, D. J., Dueck, S. R., Rieger, L. A., Bourassa, A. E., Lloyd, N. D.,
    and Degenstein, D. A. (2015). High-resolution and Monte Carlo additions to
    the SASKTRAN radiative transfer model. *Atmos. Meas. Tech.*, 8, 2609–2623.
    <https://doi.org/10.5194/amt-8-2609-2015>
  - Bourassa, A. E., Degenstein, D. A., and Llewellyn, E. J. (2008). SASKTRAN:
    A spherical geometry radiative transfer code for efficient estimation of
    limb scattered sunlight. *J. Quant. Spectrosc. Radiat. Transf.*, 109(1),
    52–73. <https://doi.org/10.1016/j.jqsrt.2007.07.007>

```{note}
The SASKTRAN2 authors ask users to contact them before publishing results
that use SASKTRAN2. SASKTRAN2 is licensed for academic and educational use;
for commercial use, contact its authors.
```

## Related references

- Liu, X., et al. (2016). Description and evaluation of a new four-mode
  version of the Modal Aerosol Module (MAM4) within version 5.3 of the
  Community Atmosphere Model. *Geosci. Model Dev.*, 9, 505–522.
- Mills, M. J., et al. (2016). Global volcanic aerosol properties derived from
  emissions, 1990–2014, using CESM1(WACCM). *J. Geophys. Res. Atmos.*, 121,
  2332–2348.
