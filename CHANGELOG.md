# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [PEP 440](https://www.python.org/dev/peps/pep-0440/)
and uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [1.1.0]

### Added
- The [hydrosar.ipynb](notebooks/hydrosar.ipynb) provides a walkthrough of generating HYDRO30 products from RTC30 products locally using the `hydrosar` package 
- The [hydrosar-on-demand.ipynb](notebooks/hydrosar-on-demand.ipynb) provides a walkthrough of requesting the RTC30 and HYDRO30 products from the <https://hyp3-watermap.asf.alaska.edu/> deployment
- [OSL_README.md](docs/OSL_README.md) describing how to setup a development environment in ASF's OpenScienceLab.

### Changed
- When calculating the perennial water threshold, using 0.68 of the reverse CDF (1-sigma) instead of 0.95 (2-sigma) provides more reasonable results.

### Fixed
- `asf_tools>=0.6` dependency is now correctly specified in the `pyproject.toml`

### Removed
- The `water-extent-map.ipynb` and the `water-extent-map-on-demand.ipynb` notebooks carried over `asf_tools`.

## [1.0.0]

### Added
- Release of the HydroSAR package, as used in the [hyp3-watermap](https://hyp3-watermap.asf.alaska.edu) production environment
  - The HydroSAR codes were initially added to the [asf_tools](https://github.com/ASFHyP3/asf-tools) package for convenience but has been migrated here to foster better co-development and better credit the HydroSAR project. See [#8](https://github.com/fjmeyer/HydroSAR/pull/8) for more details
