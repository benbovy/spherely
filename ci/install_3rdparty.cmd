:: Build and install absl, s2 and s2geography on posix systems.
::
:: This script requires environment variables to be set
::  - DEPENDENCIES_DIR=/path/to/cached/prefix -- to build or use as cache
::  - ABSL_VERSION
::  - S2GEOMETRY_VERSION
::  - S2GEOGRAPHY_VERSION
::  - CXX_STANDARD
::
:: This script assumes that library sources have been downloaded or copied in
:: DEPENDENCIES_DIR (e.g., %DEPENDENCIES_DIR%/absl-src-%ABSL_VERSION%).

set SRC_DIR=%DEPENDENCIES_DIR%\src
set BUILD_DIR=%DEPENDENCIES_DIR%\build
set INSTALL_DIR=%DEPENDENCIES_DIR%\dist

if exist %INSTALL_DIR%\include\s2geography (
  echo Using cached install directory %INSTALL_DIR%
  exit /B 0
)

mkdir %SRC_DIR%
mkdir %BUILD_DIR%
mkdir %INSTALL_DIR%

set ABSL_SRC_DIR=%DEPENDENCIES_DIR%\absl-src-%ABSL_VERSION%
set S2GEOMETRY_SRC_DIR=%DEPENDENCIES_DIR%\s2geometry-src-%S2GEOMETRY_VERSION%
set S2GEOGRAPHY_SRC_DIR=%DEPENDENCIES_DIR%\s2geography-src-%S2GEOGRAPHY_VERSION%

set ABSL_BUILD_DIR=%BUILD_DIR%\absl-src-%ABSL_VERSION%
set S2GEOMETRY_BUILD_DIR=%BUILD_DIR%\s2geometry-src-%S2GEOMETRY_VERSION%
set S2GEOGRAPHY_BUILD_DIR=%BUILD_DIR%\s2geography-src-%S2GEOGRAPHY_VERSION%

echo "----- Installing cmake"
pip install ninja cmake

echo "----- Downloading, building and installing absl-%ABSL_VERSION%"

cd %DEPENDENCIES_DIR%
curl -o absl.tar.gz -L https://github.com/abseil/abseil-cpp/archive/refs/tags/%ABSL_VERSION%.tar.gz
tar -xf absl.tar.gz -C %SRC_DIR%

cmake -GNinja ^
    -S %SRC_DIR%/abseil-cpp-%ABSL_VERSION% ^
    -B %ABSL_BUILD_DIR% ^
    -DCMAKE_INSTALL_PREFIX=%INSTALL_DIR% ^
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON ^
    -DCMAKE_CXX_STANDARD=%CXX_STANDARD% ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DABSL_ENABLE_INSTALL=ON

IF %ERRORLEVEL% NEQ 0 exit /B 1
cmake --build %ABSL_BUILD_DIR%
IF %ERRORLEVEL% NEQ 0 exit /B 2
cmake --install %ABSL_BUILD_DIR%

echo "----- Downloading, building and installing s2geometry-%S2GEOMETRY_VERSION%"

cd %DEPENDENCIES_DIR%
curl -o s2geometry.tar.gz -L https://github.com/google/s2geometry/archive/refs/tags/v%S2GEOMETRY_VERSION%.tar.gz
tar -xf s2geometry.tar.gz -C %SRC_DIR%

cmake -GNinja ^
    -S %SRC_DIR%/s2geometry-%S2GEOMETRY_VERSION% ^
    -B %S2GEOMETRY_BUILD_DIR% ^
    -DCMAKE_INSTALL_PREFIX=%INSTALL_DIR% ^
    -DOPENSSL_ROOT_DIR=%OPENSSL_ROOT_DIR% ^
    -DBUILD_TESTS=OFF ^
    -DBUILD_EXAMPLES=OFF ^
    -UGOOGLETEST_ROOT ^
    -DCMAKE_CXX_STANDARD=%CXX_STANDARD% ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DCMAKE_WINDOWS_EXPORT_ALL_SYMBOLS=TRUE ^
    -DBUILD_SHARED_LIBS=ON

IF %ERRORLEVEL% NEQ 0 exit /B 3
cmake --build %S2GEOMETRY_BUILD_DIR%
IF %ERRORLEVEL% NEQ 0 exit /B 4
cmake --install %S2GEOMETRY_BUILD_DIR%

echo "----- Downloading, building and installing s2geography-%S2GEOGRAPHY_VERSION%"

cd %DEPENDENCIES_DIR%
curl -o s2geography.tar.gz -L https://github.com/paleolimbot/s2geography/archive/refs/tags/%S2GEOGRAPHY_VERSION%.tar.gz
tar -xf s2geography.tar.gz -C %SRC_DIR%

# TODO: remove when fixed in s2geography
cd %SRC_DIR%/s2geography-%S2GEOGRAPHY_VERSION%
patch -p1 < %PROJECT_DIR%\ci\s2geography-add-openssl-as-requirement.patch

cmake -GNinja ^
    -S %SRC_DIR%/s2geography-%S2GEOGRAPHY_VERSION% ^
    -B %S2GEOGRAPHY_BUILD_DIR% ^
    -DCMAKE_INSTALL_PREFIX=%INSTALL_DIR% ^
    -DOPENSSL_ROOT_DIR=%OPENSSL_ROOT_DIR% ^
    -DS2GEOGRAPHY_BUILD_TESTS=OFF ^
    -DS2GEOGRAPHY_S2_SOURCE=AUTO ^
    -DS2GEOGRAPHY_BUILD_EXAMPLES=OFF ^
    -DCMAKE_CXX_STANDARD=%CXX_STANDARD% ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DCMAKE_WINDOWS_EXPORT_ALL_SYMBOLS=TRUE ^
    -DBUILD_SHARED_LIBS=ON

IF %ERRORLEVEL% NEQ 0 exit /B 5
cmake --build %S2GEOGRAPHY_BUILD_DIR%
IF %ERRORLEVEL% NEQ 0 exit /B 6
cmake --install %S2GEOGRAPHY_BUILD_DIR%
