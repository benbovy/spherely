# ![spherely](docs/_static/spherely_logo.svg)

![Tests](https://github.com/benbovy/spherely/actions/workflows/run-tests.yaml/badge.svg)
[![Docs](https://readthedocs.org/projects/spherely/badge/?version=latest)](https://spherely.readthedocs.io)
[![Coverage](https://codecov.io/gh/benbovy/spherely/branch/main/graph/badge.svg)](https://app.codecov.io/gh/benbovy/spherely?branch=main)

*Python library for manipulation and analysis of geometric objects on the sphere.*

Spherely is the counterpart of [Shapely](https://github.com/shapely/shapely)
(2.0+) for manipulation and analysis of spherical geometric objects. It is using
the widely deployed open-source geometry library
[s2geometry](https://github.com/google/s2geometry) via the library
[s2geography](https://github.com/paleolimbot/s2geography) which provides a
[GEOS](https://libgeos.org) compatibility layer on top of s2geometry.

**This library is at an early stage of development.**

## Installation

The easiest way to install Spherely is via its binary packages available for
Linux, MacOS, and Windows platforms on [conda-forge](https://conda-forge.org/)
and [PyPI](https://pypi.org/project/spherely/).

Install the binary wheel using [pip](https://pip.pypa.io/):

``` sh
$ pip install spherely
```

Install the conda-forge package using
[conda](https://docs.conda.io/projects/conda/en/stable/):

``` sh
$ conda install spherely --channel conda-forge
```

To compile and install Spherely from source, see detailed instructions in the
[documentation](https://spherely.readthedocs.io/en/latest/install.html).

## Documentation

https://spherely.readthedocs.io

## License

Spherely is licensed under BSD 3-Clause license. See the LICENSE file for more
details.

## Acknowledgment

The development of this project has been supported by two
[NumFOCUS](https://numfocus.org) Small Development Grants (GeoPandas 2022 round
1 and GeoPandas 2023 round 3).
