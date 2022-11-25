# The MIT License (MIT)
#
# Copyright (c) 2020 Benoit Bovy
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

#
# FindS2GEOGRAPHY
# ---------------
#
# Find s2geography include directories and libraries.
#
# This module will set the following variables:
#
#  S2GEOGRAPHY_FOUND          - System has S2
#  S2GEOGRAPHY_INCLUDE_DIRS   - The S2 include directories
#  S2GEOGRAPHY_LIBRARIES      - The libraries needed to use S2
#

include(FindPackageHandleStandardArgs)

find_path(s2geography_INCLUDE_DIR s2geography.h
  HINTS
    ENV S2GEOGRAPHY_ROOT
    ENV S2GEOGRAPHY_DIR
    ${S2GEOGRAPHY_ROOT_DIR}
  PATH_SUFFIXES
    include
    include/s2geography
  )

find_library(s2geography_LIBRARY
  NAMES s2geography
  HINTS
    ENV S2GEOGRAPHY_ROOT
    ENV S2GEOGRAPHY_DIR
    ${S2GEOGRAPHY_ROOT_DIR}
  PATH_SUFFIXES
    lib
    libs
    Library
  )

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(s2geography
  REQUIRED_VARS s2geography_INCLUDE_DIR s2geography_LIBRARY
  )

if(S2GEOGRAPHY_FOUND)
  set(s2geography_INCLUDE_DIRS ${s2geography_INCLUDE_DIR})
  set(s2geography_LIBRARIES ${s2geography_LIBRARY})

  add_library(s2geography SHARED IMPORTED)
  set_target_properties(s2geography PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES ${s2geography_INCLUDE_DIRS}
    IMPORTED_LOCATION ${s2geography_LIBRARIES}
    IMPORTED_IMPLIB ${s2geography_LIBRARIES}
    )

  mark_as_advanced(s2geography_INCLUDE_DIRS s2geography_LIBRARIES)
endif()
