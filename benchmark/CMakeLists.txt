############################################################################
# Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     #
#                                                                          #
# Distributed under the terms of the BSD 3-Clause License.                 #
#                                                                          #
# The full license is in the file LICENSE, distributed with this software. #
############################################################################

if (CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    project(xtensor-python-benchmark)

    find_package(xtensor-python REQUIRED CONFIG)
    set(XTENSOR_PYTHON_INCLUDE_DIR ${xtensor-python_INCLUDE_DIRS})
endif ()

message(STATUS "Forcing tests build type to Release")
set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)

include(CheckCXXCompilerFlag)

string(TOUPPER "${CMAKE_BUILD_TYPE}" U_CMAKE_BUILD_TYPE)

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Intel")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -Wunused-parameter -Wextra -Wreorder -Wconversion")
    CHECK_CXX_COMPILER_FLAG("-std=c++17" HAS_CPP17_FLAG)

    if (HAS_CPP17_FLAG)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
    else()
        message(FATAL_ERROR "Unsupported compiler -- xtensor requires C++17 support!")
    endif()

    # Enable link time optimization and set the default symbol
    # visibility to hidden (very important to obtain small binaries)
    if (NOT ${U_CMAKE_BUILD_TYPE} MATCHES DEBUG)
        # Default symbol visibility
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden")

        # Check for Link Time Optimization support
        # (GCC/Clang)
        CHECK_CXX_COMPILER_FLAG("-flto" HAS_LTO_FLAG)
        if (HAS_LTO_FLAG)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -flto")
        endif()

        # Intel equivalent to LTO is called IPO
        if (CMAKE_CXX_COMPILER_ID MATCHES "Intel")
            CHECK_CXX_COMPILER_FLAG("-ipo" HAS_IPO_FLAG)
            if (HAS_IPO_FLAG)
                set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ipo")
            endif()
        endif()
    endif()
endif()

if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc /MP /bigobj")
    set(CMAKE_EXE_LINKER_FLAGS /MANIFEST:NO)
    foreach(flag_var
            CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
            CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
        string(REPLACE "/MD" "-MT" ${flag_var} "${${flag_var}}")
    endforeach()
endif()

# ============================================================================
# Require pybind11
# ============================================================================
if (NOT TARGET pybind11::headers)
    set(PYBIND11_FINDPYTHON OFF)
    find_package(pybind11 REQUIRED)
endif()
message(STATUS "Benchmark: Found pybind11 v${pybind11_VERSION}")

# ============================================================================
# Require nanobind
# ============================================================================
# nanobind requires find_package(Python) with Development component
find_package(Python COMPONENTS Interpreter Development REQUIRED)

execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE nanobind_ROOT
    RESULT_VARIABLE NANOBIND_RESULT
)

if(NOT NANOBIND_RESULT EQUAL 0)
    message(FATAL_ERROR "nanobind not found! Install nanobind (e.g. 'conda install nanobind' or 'pip install nanobind').")
endif()

find_package(nanobind CONFIG REQUIRED)
message(STATUS "Benchmark: Found nanobind v${nanobind_VERSION}")

# ============================================================================
# pybind11 benchmark module
# ============================================================================
set(XTENSOR_PYTHON_BENCHMARK
    main.cpp
)

set(XTENSOR_PYTHON_BENCHMARK_TARGET benchmark_xtensor_python)
add_library(${XTENSOR_PYTHON_BENCHMARK_TARGET} MODULE EXCLUDE_FROM_ALL
            ${XTENSOR_PYTHON_BENCHMARK} ${XTENSOR_PYTHON_HEADERS})

set_target_properties(${XTENSOR_PYTHON_BENCHMARK_TARGET} PROPERTIES PREFIX "")
set_target_properties(${XTENSOR_PYTHON_BENCHMARK_TARGET} PROPERTIES SUFFIX "${PYTHON_MODULE_EXTENSION}")

if (APPLE)
    target_link_libraries(${XTENSOR_PYTHON_BENCHMARK_TARGET} PRIVATE "-undefined dynamic_lookup")
elseif (MSVC)
    target_link_libraries(${XTENSOR_PYTHON_BENCHMARK_TARGET} ${PYTHON_LIBRARIES})
else ()
    target_link_libraries(${XTENSOR_PYTHON_BENCHMARK_TARGET} PRIVATE xtensor-python)
endif()

# ============================================================================
# nanobind benchmark module
# ============================================================================
nanobind_add_module(benchmark_xtensor_nanobind EXCLUDE_FROM_ALL
    main_nanobind.cpp
)

target_include_directories(benchmark_xtensor_nanobind PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${NUMPY_INCLUDE_DIRS}
)

target_link_libraries(benchmark_xtensor_nanobind PRIVATE xtensor)

# ============================================================================
# Copy benchmark scripts
# ============================================================================
configure_file(benchmark_pyarray.py benchmark_pyarray.py COPYONLY)
configure_file(benchmark_pytensor.py benchmark_pytensor.py COPYONLY)
configure_file(benchmark_pybind_array.py benchmark_pybind_array.py COPYONLY)
configure_file(benchmark_pyvectorize.py benchmark_pyvectorize.py COPYONLY)
configure_file(benchmark_pybind_vectorize.py benchmark_pybind_vectorize.py COPYONLY)
configure_file(run_benchmarks.py run_benchmarks.py COPYONLY)

# ============================================================================
# Benchmark targets
# ============================================================================
add_custom_target(xbenchmark
        COMMAND "${PYTHON_EXECUTABLE}" "benchmark_pyarray.py"
        COMMAND "${PYTHON_EXECUTABLE}" "benchmark_pytensor.py"
        COMMAND "${PYTHON_EXECUTABLE}" "benchmark_pybind_array.py"
        COMMAND "${PYTHON_EXECUTABLE}" "benchmark_pyvectorize.py"
        COMMAND "${PYTHON_EXECUTABLE}" "benchmark_pybind_vectorize.py"
        DEPENDS ${XTENSOR_PYTHON_BENCHMARK_TARGET})

add_custom_target(xbenchmark_full
        COMMAND "${PYTHON_EXECUTABLE}" "run_benchmarks.py"
        DEPENDS ${XTENSOR_PYTHON_BENCHMARK_TARGET} benchmark_xtensor_nanobind
        COMMENT "Running full benchmark suite (pybind11 vs nanobind comparison)")
