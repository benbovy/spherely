#!/bin/bash

# Build and install absl, s2 and s2geography on posix systems.
#
# This script requires environment variables to be set
#  - DEPENDENCIES_DIR=/path/to/cached/prefix -- to build or use as cache
#  - ABSL_VERSION
#  - S2GEOMETRY_VERSION
#  - S2GEOGRAPHY_VERSION
#  - CXX_STANDARD
#
# This script assumes that library sources have been downloaded or copied in
# DEPENDENCIES_DIR (e.g., $DEPENDENCIES_DIR/absl-src-$ABSL_VERSION).
pushd .

set -e

if [ -z "$DEPENDENCIES_DIR" ]; then
    echo "DEPENDENCIES_DIR must be set"
    exit 1
elif [ -z "$ABSL_VERSION" ]; then
    echo "ABSL_VERSION must be set"
    exit 1
elif [ -z "$S2GEOMETRY_VERSION" ]; then
    echo "S2GEOMETRY_VERSION must be set"
    exit 1
elif [ -z "$S2GEOGRAPHY_VERSION" ]; then
    echo "S2GEOGRAPHY_VERSION must be set"
    exit 1
elif [ -z "$CXX_STANDARD" ]; then
    echo "CXX_STANDARD must be set"
    exit 1
fi

SRC_DIR=$DEPENDENCIES_DIR/src
BUILD_DIR=$DEPENDENCIES_DIR/build
INSTALL_DIR=$DEPENDENCIES_DIR/dist

mkdir -p $SRC_DIR
mkdir -p $BUILD_DIR
mkdir -p $INSTALL_DIR

ABSL_SRC_DIR=$DEPENDENCIES_DIR/absl-src-$ABSL_VERSION
S2GEOMETRY_SRC_DIR=$DEPENDENCIES_DIR/s2geometry-src-$S2GEOMETRY_VERSION
S2GEOGRAPHY_SRC_DIR=$DEPENDENCIES_DIR/s2geography-src-$S2GEOGRAPHY_VERSION

ABSL_BUILD_DIR=$BUILD_DIR/absl-src-$ABSL_VERSION
S2GEOMETRY_BUILD_DIR=$BUILD_DIR/s2geometry-src-$S2GEOMETRY_VERSION
S2GEOGRAPHY_BUILD_DIR=$BUILD_DIR/s2geography-src-$S2GEOGRAPHY_VERSION

build_install_dependencies(){
    echo "----- Installing cmake"
    pip install cmake

    echo "------ Clean build and install directories"

    rm -rf $BUILD_DIR/*
    rm -rf $INSTALL_DIR/*

    echo "----- Downloading, building and installing absl-$ABSL_VERSION"

    cd $DEPENDENCIES_DIR
    curl -o absl.tar.gz -L https://github.com/abseil/abseil-cpp/archive/refs/tags/$ABSL_VERSION.tar.gz
    tar -xf absl.tar.gz -C $SRC_DIR
    rm -f absl.tar.gz

    cmake -S $SRC_DIR/abseil-cpp-$ABSL_VERSION -B $ABSL_BUILD_DIR \
        -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -DCMAKE_CXX_STANDARD=$CXX_STANDARD \
        -DCMAKE_BUILD_TYPE=Release \
        -DABSL_ENABLE_INSTALL=ON

    cmake --build $ABSL_BUILD_DIR
    cmake --install $ABSL_BUILD_DIR

    echo "----- Downloading, building and installing s2geometry-$S2GEOMETRY_VERSION"

    cd $DEPENDENCIES_DIR
    curl -o s2geometry.tar.gz -L https://github.com/google/s2geometry/archive/refs/tags/v$S2GEOMETRY_VERSION.tar.gz
    tar -xf s2geometry.tar.gz -C $SRC_DIR
    rm -f s2geometry.tar.gz

    cmake -S $SRC_DIR/s2geometry-$S2GEOMETRY_VERSION -B $S2GEOMETRY_BUILD_DIR \
        -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR \
        -DBUILD_TESTS=OFF \
        -DBUILD_EXAMPLES=OFF \
        -UGOOGLETEST_ROOT \
        -DCMAKE_CXX_STANDARD=$CXX_STANDARD \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=ON

    cmake --build $S2GEOMETRY_BUILD_DIR
    cmake --install $S2GEOMETRY_BUILD_DIR

    echo "----- Downloading, building and installing s2geography-$S2GEOGRAPHY_VERSION"

    cd $DEPENDENCIES_DIR
    curl -o s2geography.tar.gz -L https://github.com/paleolimbot/s2geography/archive/refs/tags/$S2GEOGRAPHY_VERSION.tar.gz
    tar -xf s2geography.tar.gz -C $SRC_DIR
    rm -f s2geography.tar.gz

    # TODO: remove when fixed in s2geography
    cd $SRC_DIR/s2geography-$S2GEOGRAPHY_VERSION
    if [ "$(uname)" == "Darwin" ]; then
        patch -p1 < $PROJECT_DIR/ci/s2geography-add-openssl-as-requirement.patch
    else
        patch -p1 < /project/ci/s2geography-add-openssl-as-requirement.patch
    fi

    cmake -S $SRC_DIR/s2geography-$S2GEOGRAPHY_VERSION -B $S2GEOGRAPHY_BUILD_DIR \
        -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR \
        -DS2GEOGRAPHY_BUILD_TESTS=OFF \
        -DS2GEOGRAPHY_S2_SOURCE=AUTO \
        -DS2GEOGRAPHY_BUILD_EXAMPLES=OFF \
        -DCMAKE_CXX_STANDARD=$CXX_STANDARD \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=ON

    cmake --build $S2GEOGRAPHY_BUILD_DIR
    cmake --install $S2GEOGRAPHY_BUILD_DIR
}


echo "----- Installing OpenSSL in Linux container"

if [ "$(uname)" != "Darwin" ]; then
    # assume manylinux2014 https://cibuildwheel.pypa.io/en/stable/faq/
    # TODO: this is done outside of build_install_dependencies so it can
    # work with a cached install directory, but it doesn't prevent
    # installing an openssl version greater than the one used to build
    # libraries in build_install_dependencies (shoudn't be likely, though).
    yum install -y openssl-devel
fi


if [ -d "$INSTALL_DIR/include/s2geography" ]; then
    echo "----- Using cached install directory $INSTALL_DIR"
else
    build_install_dependencies
fi

popd
