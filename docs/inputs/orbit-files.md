# Orbit files

The `orbit` mode takes its viewing geometry from a
set of HAWC orbit NetCDF files: one file per orbit (roughly one revolution),
giving the satellite position and the tangent-point locations of the ALI
image over time.

% TODO: say where users can obtain orbit files, who produces them, and under
% what terms; or ship a small sample file with the package.

## Required contents

| Name | Kind | Dimensions | Description |
|------|------|------------|-------------|
| `time` | variable | `time` | Seconds since `orbit_epoch` |
| `latitude` | variable | `time`, `along`, `across` | Tangent-point latitude [degrees] |
| `longitude` | variable | `time`, `along`, `across` | Tangent-point longitude [degrees], −180–180 or 0–360 |
| `observer_latitude` | variable | `time` | Satellite latitude [degrees] |
| `observer_longitude` | variable | `time` | Satellite longitude [degrees] |
| `observer_altitude` | variable | `time` | Satellite altitude [m] |
| `start_time` | global attribute | — | Start of the orbit, parseable by pandas |

Only `along = 0` is used. The tangent point is the single across-track
pixel `center_pixel` (default 256, the centre of a 512-pixel row).

## How files are grouped by day

Each file is assigned to the day (counted from `orbit_epoch`) of its
`start_time` attribute. A day can therefore map to several files, and an
orbit that crosses midnight belongs to the day it started on. Reading
every file's `start_time` can take minutes for a large set, so the index is
cached in `out_dir/.orbit_day_index_cache.json` and rebuilt automatically
when files are added, removed or modified.

## File names

The default pattern is `orbit_*.nc`; change it with `orbit_pattern`.
