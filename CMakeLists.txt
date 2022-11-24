cmake_minimum_required(VERSION 3.15...3.25)
project(
  "${SKBUILD_PROJECT_NAME}"
  LANGUAGES CXX
  VERSION "${SKBUILD_PROJECT_VERSION}")

include(FetchContent)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "The C++ standard to build with")
set(CMAKE_CXX_STANDARD_REQUIRED ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/third_party/cmake")

find_package(
  Python
  COMPONENTS Interpreter Development.Module
  REQUIRED)

find_package(pybind11 CONFIG REQUIRED)

# TODO: make it more flexible (non-conda)
set(S2GEOGRAPHY_ROOT_DIR "$ENV{CONDA_PREFIX}")

find_package(s2geography REQUIRED)

add_library(sksphere MODULE src/main.cpp)

target_compile_definitions(sksphere
                           PRIVATE VERSION_INFO=${PROJECT_VERSION})

target_link_libraries(sksphere
  PRIVATE pybind11::module pybind11::lto
  PUBLIC s2geography
  )

pybind11_extension(sksphere)
if(NOT MSVC AND NOT ${CMAKE_BUILD_TYPE} MATCHES Debug|RelWithDebInfo)
    # Strip unnecessary sections of the binary on Linux/macOS
    pybind11_strip(sksphere)
endif()

install(TARGETS sksphere LIBRARY DESTINATION .)
