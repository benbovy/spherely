# ![spherely](docs/_static/spherely_logo.svg)

![Tests](https://github.com/benbovy/spherely/actions/workflows/run-tests.yaml/badge.svg)
[![Docs](https://readthedocs.org/projects/spherely/badge/?version=latest)](https://spherely.readthedocs.io)

*Python library for manipulation and analysis of geometric objects on the sphere.*

Spherely is the counterpart of [Shapely](https://github.com/shapely/shapely)
(2.0+) for manipulation and analysis of spherical geometric objects. It is using
the widely deployed open-source geometry library
[s2geometry](https://github.com/google/s2geometry) via the library
[s2geography](https://github.com/paleolimbot/s2geography) which provides a
[GEOS](https://libgeos.org) compatibility layer on top of s2geometry.

This library is at an early stage of development.

## Requirements

- Python
- Numpy
- s2geography
- s2geometry

## Installation

There is no pre-compiled package available at the moment. See the section below
for instructions on how to setup a development environment and build / install
spherely from source.

## Setting up a development environment using conda

After cloning this repo, create a conda environment using the ci/environment.yml
file with the required dependencies:

```
$ conda env create -f spherely/ci/environment.yml
$ conda activate spherely-dev
```

Build and install `s2spherely`:

```
$ cd spherely
$ python -m pip install . -v --no-build-isolation
```

Run the tests:

```
$ pytest . -v
```
