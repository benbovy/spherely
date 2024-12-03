:: Build and install absl, s2 and s2geography on Windows.
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

rem set ABSL_SRC_DIR=%DEPENDENCIES_DIR%\absl-src-%ABSL_VERSION%
rem set S2GEOMETRY_SRC_DIR=%DEPENDENCIES_DIR%\s2geometry-src-%S2GEOMETRY_VERSION%
set S2GEOGRAPHY_SRC_DIR=%DEPENDENCIES_DIR%\s2geography-src-%S2GEOGRAPHY_VERSION%

rem set ABSL_BUILD_DIR=%BUILD_DIR%\absl-src-%ABSL_VERSION%
rem set S2GEOMETRY_BUILD_DIR=%BUILD_DIR%\s2geometry-src-%S2GEOMETRY_VERSION%
set S2GEOGRAPHY_BUILD_DIR=%BUILD_DIR%\s2geography-src-%S2GEOGRAPHY_VERSION%

echo %CMAKE_PREFIX_PATH%

echo "----- Installing cmake"
pip install ninja cmake

rem echo "----- Downloading, building and installing absl-%ABSL_VERSION%"

rem cd %DEPENDENCIES_DIR%
rem curl -o absl.tar.gz -L https://github.com/abseil/abseil-cpp/archive/refs/tags/%ABSL_VERSION%.tar.gz
rem tar -xf absl.tar.gz -C %SRC_DIR%

rem cmake -GNinja ^
rem     -S %SRC_DIR%/abseil-cpp-%ABSL_VERSION% ^
rem     -B %ABSL_BUILD_DIR% ^
rem     -DCMAKE_INSTALL_PREFIX=%INSTALL_DIR% ^
rem     -DCMAKE_POSITION_INDEPENDENT_CODE=ON ^
rem     -DCMAKE_CXX_STANDARD=%CXX_STANDARD% ^
rem     -DCMAKE_BUILD_TYPE=Release ^
rem     -DABSL_ENABLE_INSTALL=ON

rem IF %ERRORLEVEL% NEQ 0 exit /B 1
rem cmake --build %ABSL_BUILD_DIR%
rem IF %ERRORLEVEL% NEQ 0 exit /B 2
rem cmake --install %ABSL_BUILD_DIR%

rem echo "----- Downloading, building and installing s2geometry-%S2GEOMETRY_VERSION%"

rem echo %OPENSSL_ROOT_DIR%

rem cd %DEPENDENCIES_DIR%
rem curl -o s2geometry.tar.gz -L https://github.com/google/s2geometry/archive/refs/tags/v%S2GEOMETRY_VERSION%.tar.gz
rem tar -xf s2geometry.tar.gz -C %SRC_DIR%

rem cmake -GNinja ^
rem     -S %SRC_DIR%/s2geometry-%S2GEOMETRY_VERSION% ^
rem     -B %S2GEOMETRY_BUILD_DIR% ^
rem     -DCMAKE_INSTALL_PREFIX=%INSTALL_DIR% ^
rem     -DOPENSSL_ROOT_DIR=%OPENSSL_ROOT_DIR% ^
rem     -DBUILD_TESTS=OFF ^
rem     -DBUILD_EXAMPLES=OFF ^
rem     -UGOOGLETEST_ROOT ^
rem     -DCMAKE_CXX_STANDARD=%CXX_STANDARD% ^
rem     -DCMAKE_BUILD_TYPE=Release ^
rem     -DCMAKE_WINDOWS_EXPORT_ALL_SYMBOLS=TRUE ^
rem     -DBUILD_SHARED_LIBS=ON

rem IF %ERRORLEVEL% NEQ 0 exit /B 3
rem cmake --build %S2GEOMETRY_BUILD_DIR%
rem IF %ERRORLEVEL% NEQ 0 exit /B 4
rem cmake --install %S2GEOMETRY_BUILD_DIR%

echo "----- Downloading, building and installing s2geography-%S2GEOGRAPHY_VERSION%"

cd %DEPENDENCIES_DIR%
curl -o s2geography.tar.gz -L https://github.com/paleolimbot/s2geography/archive/refs/tags/%S2GEOGRAPHY_VERSION%.tar.gz
tar -xf s2geography.tar.gz -C %SRC_DIR%

rem TODO: remove when fixed in s2geography
rem (https://github.com/paleolimbot/s2geography/pull/53)
cd %SRC_DIR%/s2geography-%S2GEOGRAPHY_VERSION%
patch -i %PROJECT_DIR%\ci\s2geography-add-openssl-as-requirement.patch

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
