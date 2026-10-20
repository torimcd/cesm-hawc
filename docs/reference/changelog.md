# Changelog

All notable changes to cesm-hawc are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## 1.0.0 — unreleased

First public release. Changes since 0.2.0:

### Changed

- Python 3.11 or later is now required for both install tiers.
- The `[sim]` extra now installs `aliprocessing` 2026.10.0 or later from PyPI,
  instead of from a pinned git commit.
- **Breaking:** the run modes are now `fixed` and `orbit`, and every run
  processes one model case. The config file has a required `[case]` table
  (`name`, `waccm_dir`, `pattern`, `out_dir`, ...), plus `[fixed]` and
  `[orbit]` for each mode's settings.
  - `single` and `batch` are replaced by `fixed`, which runs one column per
    history file matched by `[case]`. `[single]`, `[batch]` and `[geometry]`
    are removed.
  - `orbit-track` is renamed `orbit`. `waccm_data_dir`/`case_name` are
    replaced by `[case] waccm_dir` (with a `{name}` placeholder) and
    `[case] name`; `run_start_date`/`run_end_date` by
    `[case] start_date`/`end_date`. Output layout is unchanged.
  - `run_l2` and `strip_ozone` now apply to both modes.
- **Breaking:** `run_ali_simulation` takes one history file and returns the
  simulator output, truth extinction and sulfate burden.
- **Changes results:** sulfate median radii now come from CESM's wet median
  diameters (`dgnumwet1`, `dgnumwet3`) when the history file has them,
  instead of being derived from dry sulfate mass with an assumed density. Files without
  wet diameters fall back to the old method with a warning.
- The sulfate burden is computed directly from the dry sulfate mass mixing
  ratios, and saved columns include the dry mass concentration of each mode.
- `save-inputs --mode orbit` now records each observation's time and
  satellite position in the saved file, and supports sub-daily output.

### Removed

- The `orbit-file` mode and `[orbit_real]` table.
- Background/injection pairing and anomaly diagnostics.
- `examples/quickstart.py`. The [Tutorial](../tutorial.ipynb) walks through
  one column, and `cesm-hawc run --mode orbit` covers orbit geometry.

### Added

- This documentation site.
- `default_noise_model(seed=...)` for reproducible instrument noise.
- A step-by-step tutorial notebook.
- Package metadata for PyPI: authors, links and classifiers.
- Continuous integration for the tests and the package build.
- Runs with the L2 retrieval reject simulated wavelengths outside the
  retrieval's wavelength range, instead of retrieving with no aerosol optical
  properties at those wavelengths.

### Fixed

- The base install now depends on `netCDF4` and `dask`, which are needed to
  open history files.
- The calibration-cache workaround now finds `hawcsimulator`'s cache on macOS
  and Windows as well as Linux.
