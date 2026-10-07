# Changelog

All notable changes to cesm-hawc are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## Unreleased

### Changed

- Python 3.11 or later is now required for both install tiers.
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
- `save-inputs --mode orbit` now records each observation's time and
  satellite position in the saved file, and supports sub-daily output.

### Removed

- The `orbit-file` mode and `[orbit_real]` table.
- Background/injection pairing and anomaly diagnostics. Compare cases in
  your own analysis code instead.

### Added

- This documentation site.
- `default_noise_model(seed=...)` for reproducible instrument noise.

### Fixed

- `examples/quickstart.py` uses the `ideal_dolp_imager` instrument model and
  passes a noise model, matching the CLI.

% TODO: add entries for 0.2.0 and earlier, including the longitude fix in
% WACCMAtmosphere.get_column_profiles. Before that fix, requested longitudes
% in -180..180 that were negative all selected the lon = 0 column, which
% affects results from any earlier version.
