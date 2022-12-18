# Spherely

![Tests](https://github.com/benbovy/spherely/actions/workflows/run-tests.yaml/badge.svg)
[![Docs](https://readthedocs.org/projects/spherely/badge/?version=latest)](https://spherely.readthedocs.io)

Manipulation and analysis of geometric objects on the sphere.

Spherely is the counterpart of [Shapely](https://github.com/shapely/shapely)
(2.0+) for manipulation and analysis of spherical geometric objects. It is using
the widely deployed open-source geometry library
[s2geometry](https://github.com/google/s2geometry) via the library
[s2geography](https://github.com/paleolimbot/s2geography) which provides a
[GEOS](https://libgeos.org) compatibility layer on top of s2geometry.

Not much to see here for the moment.

## Requirements

- Python
- Numpy
- s2geography
- s2geometry

## Installation (from source)

Clone this repository and run the following command from it's root directory:

```
$ python -m pip install .
```

## Setting up a development environment using conda

After cloning this repo, create a conda environment using the
ci/environment.yml file:

```
$ conda env create -f spherely/ci/environment.yml
$ conda activate spherely-dev
```

Clone and install `s2geography` (https://github.com/paleolimbot/s2geography):

```
$ git clone https://github.com/paleolimbot/s2geography
$ cmake -S s2geography -B s2geography/build -DCMAKE_CXX_STANDARD=17 -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX
$ cmake --build s2geography/build
$ cmake --install s2geography/build
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
