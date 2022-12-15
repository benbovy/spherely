# Spherely

![Tests](https://github.com/benbovy/spherely/actions/workflows/run-tests.yaml/badge.svg)

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
