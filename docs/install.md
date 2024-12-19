(install)=

# Installation

## Built distributions

The easiest way to install Spherely is via its binary packages available for
Linux, MacOS, and Windows platforms on [conda-forge](https://conda-forge.org/)
and [PyPI](https://pypi.org/project/spherely/).

### Installation of Python binary wheels (PyPI)

Install the last released binary wheel, e.g., using [pip](https://pip.pypa.io/):

``` sh
$ pip install spherely
```

### Installation of Conda packages (conda-forge)

Install the last released conda-forge package using
[conda](https://docs.conda.io/projects/conda/en/stable/):

``` sh
$ conda install spherely --channel conda-forge
```

## Installation from source

Compiling and installing Spherely from source may be useful for development
purpose and/or for building it against a specific version of S2Geography and/or
S2Geometry.

### Requirements

- Python
- Numpy
- [s2geography](https://github.com/paleolimbot/s2geography) v0.2.0 or higher
- [s2geometry](https://github.com/google/s2geometry) v0.11.1 or higher

Additional build dependencies:

- C++ compiler supporting C++17 standard
- CMake
- [scikit-build-core](https://github.com/scikit-build/scikit-build-core)

### Setting up a development environment using conda

All the requirements listed above are available via conda-forge.

After cloning Spherely's [source
repository](https://github.com/benbovy/spherely), create a conda environment
with the required (and development) dependencies using the
`ci/environment-dev.yml` file:

```sh
$ conda env create -f spherely/ci/environment-dev.yml
$ conda activate spherely-dev
```

Build and install Spherely:

```sh
$ cd spherely
$ python -m pip install . -v --no-build-isolation
```

Note that you can specify a build directory in order to avoid rebuilding the
whole library from scratch each time after editing the code:

```sh
$ python -m pip install . -v --no-build-isolation --config-settings build-dir=build/skbuild
```

Run the tests:

```sh
$ pytest . -v
```

Spherely also uses [pre-commit](https://pre-commit.com/) for code
auto-formatting and linting at every commit. After installing it, you can enable
pre-commit hooks with the following command:

```sh
$ pre-commit install
```

(Note: you can skip the pre-commit checks with `git commit --no-verify`)

### Using the latest S2Geography version

If you want to compile Spherely against the latest version of S2Geography, use:

 ```sh
 $ git clone https://github.com/paleolimbot/s2geography
 $ cmake \
 $     -S s2geography \
 $     -B s2geography/build \
 $     -DCMAKE_CXX_STANDARD=17 \
 $     -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX
 $ cmake --build s2geography/build
 $ cmake --install s2geography/build
 ```
