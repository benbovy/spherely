cmake_minimum_required(VERSION 3.15...3.25)
project(
  "${SKBUILD_PROJECT_NAME}"
  LANGUAGES CXX
  VERSION "${SKBUILD_PROJECT_VERSION}")

set(CMAKE_CXX_STANDARD 17 CACHE STRING "The C++ standard to build with")
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(SPHERELY_CODE_COVERAGE "Enable coverage reporting" OFF)

# Dependencies

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/third_party/cmake")

find_package(
  Python
  COMPONENTS Interpreter Development.Module
  REQUIRED)

find_package(pybind11 CONFIG REQUIRED)

find_package(s2 CONFIG REQUIRED)
if(s2_FOUND)
  get_target_property(s2_INCLUDE_DIRS s2::s2 INTERFACE_INCLUDE_DIRECTORIES)
  message(STATUS "Found s2: ${s2_INCLUDE_DIRS}")
else()
  message(FATAL_ERROR "Couldn't find s2")
endif()

# this is needed so that openssl headers included from s2geometry headers are found.
find_package(OpenSSL REQUIRED)
target_include_directories(s2::s2 INTERFACE ${OPENSSL_INCLUDE_DIR})

find_package(s2geography CONFIG REQUIRED)
if(${s2geography_FOUND})
  get_target_property(s2geography_INCLUDE_DIRS s2geography INTERFACE_INCLUDE_DIRECTORIES)
  message(STATUS "Found s2geography v${s2geography_VERSION}: ${s2geography_INCLUDE_DIRS}")
else()
  message(FATAL_ERROR "Couldn't find s2geography")
endif()

if(SPHERELY_CODE_COVERAGE)
  message(STATUS "Building spherely with coverage enabled")
  add_library(coverage_config INTERFACE)
endif()

# Compile definitions and flags

# How src/s2_tmp_memory_budget.cpp declares
# FLAGS_s2shape_index_tmp_memory_budget. A shared s2 exports it as *data*, and
# an import library defines only __imp_<symbol> for data, so the reference has
# to be routed through that entry with __declspec(dllimport); a static s2
# defines the bare symbol and wants the plain extern that s2's own header
# already provides. Both are in use -- conda-forge ships a DLL, vcpkg builds
# s2geometry static on Windows -- so take the answer from the imported target.
#
# The fallback is the dllimport form because the two mistakes are not
# symmetric: a plain extern against a DLL cannot link at all (LNK2001), while
# dllimport against a static s2 links as long as the object defining the flag
# is in the image -- which it is here, since it also defines
# MutableS2ShapeIndex -- at the cost of warning LNK4217.
if(WIN32)
  get_target_property(_s2_type s2::s2 TYPE)
  if(_s2_type STREQUAL "STATIC_LIBRARY")
    set(SPHERELY_S2_DLLIMPORT 0)
  else()
    set(SPHERELY_S2_DLLIMPORT 1)
  endif()
  message(STATUS "s2::s2 is ${_s2_type}: SPHERELY_S2_DLLIMPORT=${SPHERELY_S2_DLLIMPORT}")
else()
  set(SPHERELY_S2_DLLIMPORT 0)
endif()

if (MSVC)
  # used in s2geometry's CMakeLists.txt but not defined in target
  # TODO: move this in FindS2.cmake?
  target_compile_definitions(s2::s2 INTERFACE _USE_MATH_DEFINES)
  target_compile_definitions(s2::s2 INTERFACE NOMINMAX)
  target_compile_options(s2::s2 INTERFACE /J)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR
    CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR
    (CMAKE_CXX_COMPILER_ID MATCHES "Intel" AND NOT WIN32))
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra -Wreorder")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wunused-variable -Wunused-parameter")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wconversion -Wold-style-cast -Wsign-conversion")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
  add_definitions(-D_CRT_SECURE_NO_WARNINGS)
  add_definitions(-D_SILENCE_TR1_NAMESPACE_DEPRECATION_WARNING)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc /MP /bigobj /J")
  set(CMAKE_EXE_LINKER_FLAGS /MANIFEST:NO)
endif()

# Build

set(CPP_SOURCES
  src/accessors-geog.cpp
  src/boolean-operations.cpp
  src/creation.cpp
  src/geography.cpp
  src/io.cpp
  src/predicates.cpp
  src/spherely.cpp)

if(${s2geography_VERSION} VERSION_GREATER_EQUAL "0.2.0")
    set(CPP_SOURCES ${CPP_SOURCES}
      src/geoarrow.cpp
      src/predicates_common.cpp
      src/projections.cpp
      src/s2_tmp_memory_budget.cpp
      src/spatial_index.cpp)
endif()

add_library(spherely MODULE ${CPP_SOURCES})

target_compile_definitions(
  spherely
  PRIVATE
  VERSION_INFO=${PROJECT_VERSION}
  S2GEOGRAPHY_VERSION=${s2geography_VERSION}
  S2GEOGRAPHY_VERSION_MAJOR=${s2geography_VERSION_MAJOR}
  S2GEOGRAPHY_VERSION_MINOR=${s2geography_VERSION_MINOR}
  SPHERELY_S2_DLLIMPORT=${SPHERELY_S2_DLLIMPORT})

target_link_libraries(spherely
  PRIVATE pybind11::module pybind11::lto pybind11::windows_extras
  PUBLIC s2::s2 s2geography
  )

pybind11_extension(spherely)
if(NOT MSVC AND NOT ${CMAKE_BUILD_TYPE} MATCHES Debug|RelWithDebInfo)
    # Strip unnecessary sections of the binary on Linux/macOS
    pybind11_strip(spherely)
endif()

set_target_properties(spherely PROPERTIES CXX_VISIBILITY_PRESET "hidden")

if (SPHERELY_CODE_COVERAGE)
  target_compile_options(coverage_config INTERFACE -O0 -g --coverage)
  target_link_options(coverage_config INTERFACE --coverage)
  target_link_libraries(spherely PUBLIC coverage_config)
endif()

# Install

install(TARGETS spherely LIBRARY DESTINATION .)

# install type annotations
install(FILES src/spherely.pyi DESTINATION .)
