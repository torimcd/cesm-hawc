# Changelog

All notable changes to cesm-hawc are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## Unreleased

### Changed

- Python 3.11 or later is now required for both install tiers.

### Added

- This documentation site.

% TODO: add entries for 0.2.0 and earlier, including the longitude fix in
% WACCMAtmosphere.get_column_profiles. Before that fix, requested longitudes
% in -180..180 that were negative all selected the lon = 0 column, which
% affects results from any earlier version.
