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

This library is at an early stage of development.

## Requirements

- Python
- Numpy
- [s2geography](https://github.com/paleolimbot/s2geography)
- [s2geometry](https://github.com/google/s2geometry)

Additional requirements when building spherely from source:

- C++ compiler supporting C++17 standard
- CMake
- [scikit-build-core](https://github.com/scikit-build/scikit-build-core)

(Note: C++11 or C++14 should work too but we have no plan to maintain
compatibility with those older standards)

## Installation

There is no pre-compiled package available at the moment. See the section below
for instructions on how to setup a development environment and build / install
spherely from source.

## Setting up a development environment using conda

After cloning this repo, create a conda environment using the `ci/environment.yml`
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

Note that you can specify a build directory in order to avoid rebuilding the
whole library from scratch each time after editing the code (requires
scikit-build-core v0.2.0+):

```
$ python -m pip install . -v --no-build-isolation --config-settings build-dir=build/skbuild
```

Run the tests:

```
$ pytest . -v
```

Spherely also uses [pre-commit](https://pre-commit.com/) for code
auto-formatting and linting at every commit. After installing it, you can enable
pre-commit hooks with the following command:

```
$ pre-commit install
```

(Note: you can skip the pre-commit checks with `git commit --no-verify`)

## Using the latest s2geography version

If you want to compile spherely against the latest version of s2geography, use:

 ```
 $ git clone https://github.com/paleolimbot/s2geography
 $ cmake -S s2geography -B s2geography/build -DCMAKE_CXX_STANDARD=17 -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX
 $ cmake --build s2geography/build
 $ cmake --install s2geography/build
 ```
