# Orbit files

The `orbit-track` and `orbit-file` modes take their viewing geometry from a
set of HAWC orbit NetCDF files: one file per orbit (roughly one revolution),
giving the satellite position and the tangent-point locations of the ALI
image over time.

% TODO: say where users can obtain orbit files, who produces them, and under
% what terms; or ship a small sample file with the package.

## Required contents

| Name | Kind | Dimensions | Description |
|------|------|------------|-------------|
| `time` | variable | `time` | Seconds since `orbit_epoch` (`orbit-track`), or a CF-decodable time (`orbit-file`) |
| `latitude` | variable | `time`, `along`, `across` | Tangent-point latitude [degrees] |
| `longitude` | variable | `time`, `along`, `across` | Tangent-point longitude [degrees], −180–180 or 0–360 |
| `observer_latitude` | variable | `time` | Satellite latitude [degrees] |
| `observer_longitude` | variable | `time` | Satellite longitude [degrees] |
| `observer_altitude` | variable | `time` | Satellite altitude [m] |
| `start_time` | global attribute | — | Start of the orbit, parseable by pandas |

Only `along = 0` is used. The tangent point for `orbit-track` is the single
across-track pixel `center_pixel` (default 256, the centre of a 512-pixel
row); `orbit-file` uses the pixels listed in `across_indices`.

## How files are grouped by day

Each file is assigned to the calendar date of its `start_time` attribute
(falling back to a `YYYY-MM-DD` date in the file name). A date can therefore
map to several files, and an orbit that crosses midnight belongs to the date
it started on.

For `orbit-track`, files are indexed by days since `orbit_epoch`. Reading
every file's `start_time` can take minutes for a large set, so the index is
cached in `out_dir/.orbit_day_index_cache.json` and rebuilt automatically
when files are added, removed or modified.

## File names

The default pattern is `orbit_*.nc`; change it with `orbit_pattern`.
