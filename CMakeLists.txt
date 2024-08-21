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
  # not yet provided by s2geography
  # get_target_property(s2geography_INCLUDE_DIRS s2geography INTERFACE_INCLUDE_DIRECTORIES)
  message(STATUS "Found s2geography v${s2geography_VERSION}")
else()
  message(FATAL_ERROR "Couldn't find s2geography")
endif()

if(SPHERELY_CODE_COVERAGE)
  message(STATUS "Building spherely with coverage enabled")
  add_library(coverage_config INTERFACE)
endif()

# Compile definitions and flags

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

add_library(spherely MODULE
  src/geography.cpp
  src/accessors-geog.cpp
  src/predicates.cpp
  src/spherely.cpp
  )

target_compile_definitions(spherely
                           PRIVATE VERSION_INFO=${PROJECT_VERSION})

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
