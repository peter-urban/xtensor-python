############################################################################
# Copyright (c) Wolf Vollprecht, Johan Mabille and Sylvain Corlay          #
# Copyright (c) QuantStack                                                 #
#                                                                          #
# Distributed under the terms of the BSD 3-Clause License.                 #
#                                                                          #
# The full license is in the file LICENSE, distributed with this software. #
############################################################################

cmake_minimum_required(VERSION 3.29)
project(xtensor-python)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})
set(XTENSOR_PYTHON_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Versioning
# ==========

set(XTENSOR_PYTHON_CONFIG_FILE
    "${XTENSOR_PYTHON_INCLUDE_DIR}/xtensor-python/xtensor_python_config.hpp")
file(STRINGS ${XTENSOR_PYTHON_CONFIG_FILE} xtensor_python_version_defines
    REGEX "#define XTENSOR_PYTHON_VERSION_(MAJOR|MINOR|PATCH)")
foreach(ver ${xtensor_python_version_defines})
    if(ver MATCHES "#define XTENSOR_PYTHON_VERSION_(MAJOR|MINOR|PATCH) +([^ ]+)$")
        set(XTENSOR_PYTHON_VERSION_${CMAKE_MATCH_1} "${CMAKE_MATCH_2}" CACHE INTERNAL "")
    endif()
endforeach()
set(${PROJECT_NAME}_VERSION
    ${XTENSOR_PYTHON_VERSION_MAJOR}.${XTENSOR_PYTHON_VERSION_MINOR}.${XTENSOR_PYTHON_VERSION_PATCH})
message(STATUS "xtensor-python v${${PROJECT_NAME}_VERSION}")

# Dependencies
# ============

set(xtensor_REQUIRED_VERSION 0.27.0)
if(TARGET xtensor)
    set(xtensor_VERSION ${XTENSOR_VERSION_MAJOR}.${XTENSOR_VERSION_MINOR}.${XTENSOR_VERSION_PATCH})
    # Note: This is not SEMVER compatible comparison
    if( NOT ${xtensor_VERSION} VERSION_GREATER_EQUAL ${xtensor_REQUIRED_VERSION})
        message(ERROR "Mismatch xtensor versions. Found '${xtensor_VERSION}' but requires: '${xtensor_REQUIRED_VERSION}'")
    else()
        message(STATUS "Found xtensor v${xtensor_VERSION}")
    endif()
else()
    find_package(xtensor ${xtensor_REQUIRED_VERSION} REQUIRED)
    message(STATUS "Found xtensor: ${xtensor_INCLUDE_DIRS}/xtensor")
endif()

find_package(Python COMPONENTS Interpreter Development REQUIRED)

# Backend detection
# =================
# Options: AUTO (default), ON (required), OFF (disabled)
set(XTENSOR_PYTHON_USE_PYBIND11 "AUTO" CACHE STRING "Enable pybind11 backend (AUTO/ON/OFF)")
set(XTENSOR_PYTHON_USE_NANOBIND "AUTO" CACHE STRING "Enable nanobind backend (AUTO/ON/OFF)")

set(pybind11_REQUIRED_VERSION 3.0.0)
set(XTENSOR_PYTHON_HAS_PYBIND11 OFF)

if(NOT XTENSOR_PYTHON_USE_PYBIND11 STREQUAL "OFF")
    if (NOT TARGET pybind11::headers)
        # Defaults to ON for cmake >= 3.18
        set(PYBIND11_FINDPYTHON OFF)
        find_package(pybind11 ${pybind11_REQUIRED_VERSION} QUIET)
    endif()
    
    if(TARGET pybind11::headers OR pybind11_FOUND)
        set(XTENSOR_PYTHON_HAS_PYBIND11 ON)
        message(STATUS "Found pybind11 v${pybind11_VERSION}: enabling pybind11 backend")
    elseif(XTENSOR_PYTHON_USE_PYBIND11 STREQUAL "ON")
        message(FATAL_ERROR "pybind11 backend explicitly requested but pybind11 >= ${pybind11_REQUIRED_VERSION} not found")
    else()
        message(STATUS "pybind11 not found: pybind11 backend disabled")
    endif()
endif()

set(XTENSOR_PYTHON_HAS_NANOBIND OFF)

if(NOT XTENSOR_PYTHON_USE_NANOBIND STREQUAL "OFF")
    # Use official nanobind CMake integration
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
        OUTPUT_STRIP_TRAILING_WHITESPACE 
        OUTPUT_VARIABLE nanobind_ROOT
        ERROR_QUIET
        RESULT_VARIABLE NANOBIND_RESULT
    )
    
    if(NANOBIND_RESULT EQUAL 0 AND EXISTS "${nanobind_ROOT}")
        find_package(nanobind QUIET CONFIG)
    endif()
    
    if(nanobind_FOUND)
        set(XTENSOR_PYTHON_HAS_NANOBIND ON)
        message(STATUS "Found nanobind: enabling nanobind backend")
    elseif(XTENSOR_PYTHON_USE_NANOBIND STREQUAL "ON")
        message(FATAL_ERROR "nanobind backend explicitly requested but nanobind not found")
    else()
        message(STATUS "nanobind not found: nanobind backend disabled")
    endif()
endif()

# Require at least one backend
if(NOT XTENSOR_PYTHON_HAS_PYBIND11 AND NOT XTENSOR_PYTHON_HAS_NANOBIND)
    message(FATAL_ERROR "No binding backend found. Install pybind11 (>= ${pybind11_REQUIRED_VERSION}) or nanobind, or set XTENSOR_PYTHON_USE_PYBIND11=ON or XTENSOR_PYTHON_USE_NANOBIND=ON to make it required.")
endif()

# Look for NumPy headers, except if NUMPY_INCLUDE_DIRS is passed,
# which is required under some circumstances (such as wasm, where
# there is no real python executable)
if(NOT NUMPY_INCLUDE_DIRS)
    find_package(NumPy REQUIRED)
endif()
message(STATUS "Found numpy: ${NUMPY_INCLUDE_DIRS}")

# Build
# =====

set(XTENSOR_PYTHON_HEADERS
        ${XTENSOR_PYTHON_INCLUDE_DIR}/xtensor-python/detail/pyarray_backstrides.hpp
        ${XTENSOR_PYTHON_INCLUDE_DIR}/xtensor-python/detail/pystrides_adaptor.hpp
        ${XTENSOR_PYTHON_INCLUDE_DIR}/xtensor-python/xtensor_python_config.hpp
        )

# Add pybind11 backend headers
if(XTENSOR_PYTHON_HAS_PYBIND11)
    list(APPEND XTENSOR_PYTHON_HEADERS
        ${XTENSOR_PYTHON_INCLUDE_DIR}/xtensor-python/pybind11/pyarray.hpp
        ${XTENSOR_PYTHON_INCLUDE_DIR}/xtensor-python/pybind11/pycontainer.hpp
        ${XTENSOR_PYTHON_INCLUDE_DIR}/xtensor-python/pybind11/pynative_casters.hpp
        ${XTENSOR_PYTHON_INCLUDE_DIR}/xtensor-python/pybind11/pytensor.hpp
        ${XTENSOR_PYTHON_INCLUDE_DIR}/xtensor-python/pybind11/pyvectorize.hpp
        ${XTENSOR_PYTHON_INCLUDE_DIR}/xtensor-python/pybind11/xtensor_type_caster_base.hpp
    )
endif()

# Add nanobind backend headers
if(XTENSOR_PYTHON_HAS_NANOBIND)
    list(APPEND XTENSOR_PYTHON_HEADERS
        ${XTENSOR_PYTHON_INCLUDE_DIR}/xtensor-python/nanobind/pytensor.hpp
    )
endif()

add_library(xtensor-python INTERFACE)
target_include_directories(xtensor-python INTERFACE
    "$<BUILD_INTERFACE:${XTENSOR_PYTHON_INCLUDE_DIR};${NUMPY_INCLUDE_DIRS}>"
    $<INSTALL_INTERFACE:include>)
target_link_libraries(xtensor-python INTERFACE xtensor)

# Add backend-specific include directories
if(XTENSOR_PYTHON_HAS_PYBIND11)
    target_include_directories(xtensor-python INTERFACE
        "$<BUILD_INTERFACE:${pybind11_INCLUDE_DIRS}>")
endif()
if(XTENSOR_PYTHON_HAS_NANOBIND)
    target_include_directories(xtensor-python INTERFACE
        "$<BUILD_INTERFACE:${nanobind_INCLUDE_DIRS}>")
endif()

get_target_property(inc_dir xtensor-python INTERFACE_INCLUDE_DIRECTORIES)

OPTION(BUILD_TESTS "xtensor test suite" OFF)
OPTION(DOWNLOAD_GTEST "build gtest from downloaded sources" OFF)

if(DOWNLOAD_GTEST OR GTEST_SRC_DIR)
    set(BUILD_TESTS ON)
endif()

if(BUILD_TESTS)
    if(MSVC)
        set(PYTHON_MODULE_EXTENSION ".pyd")
    else()
        set(PYTHON_MODULE_EXTENSION ".so")
    endif()

    add_subdirectory(test)
    add_subdirectory(benchmark)
endif()

# Installation
# ============

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(TARGETS xtensor-python
        EXPORT ${PROJECT_NAME}-targets)

# Makes the project importable from the build directory
export(EXPORT ${PROJECT_NAME}-targets
       FILE "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Targets.cmake")

install(FILES ${XTENSOR_PYTHON_HEADERS}
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/xtensor-python)


configure_file(${PROJECT_NAME}.pc.in
               "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}.pc"
                @ONLY)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}.pc"
        DESTINATION "${CMAKE_INSTALL_DATADIR}/pkgconfig/")

set(XTENSOR_PYTHON_CMAKECONFIG_INSTALL_DIR "${CMAKE_INSTALL_DATADIR}/cmake/${PROJECT_NAME}" CACHE
    STRING "install path for xtensor-pythonConfig.cmake")

configure_package_config_file(${PROJECT_NAME}Config.cmake.in
                              "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
                              INSTALL_DESTINATION ${XTENSOR_PYTHON_CMAKECONFIG_INSTALL_DIR})

# xtensor-python is header-only and does not depend on the architecture.
# Remove CMAKE_SIZEOF_VOID_P from xtensor-pythonConfigVersion.cmake so that an xtensor-pythonConfig.cmake
# generated for a 64 bit target can be used for 32 bit targets and vice versa.
set(_XTENSOR_CMAKE_SIZEOF_VOID_P ${CMAKE_SIZEOF_VOID_P})
unset(CMAKE_SIZEOF_VOID_P)
write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
                                 VERSION ${${PROJECT_NAME}_VERSION}
                                 COMPATIBILITY AnyNewerVersion)
set(CMAKE_SIZEOF_VOID_P ${_XTENSOR_CMAKE_SIZEOF_VOID_P})
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
              ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
        DESTINATION ${XTENSOR_PYTHON_CMAKECONFIG_INSTALL_DIR})
install(EXPORT ${PROJECT_NAME}-targets
        FILE ${PROJECT_NAME}Targets.cmake
        DESTINATION ${XTENSOR_PYTHON_CMAKECONFIG_INSTALL_DIR})

