############################################################################
# Copyright (c) Wolf Vollprecht, Johan Mabille and Sylvain Corlay          #
# Copyright (c) QuantStack                                                 #
#                                                                          #
# Distributed under the terms of the BSD 3-Clause License.                 #
#                                                                          #
# The full license is in the file LICENSE, distributed with this software. #
############################################################################

cmake_minimum_required(VERSION 3.29)

if (CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    project(xtensor-python-test)

    find_package(xtensor REQUIRED CONFIG)
    set(XTENSOR_INCLUDE_DIR ${xtensor_INCLUDE_DIRS})

    find_package(xtensor-python REQUIRED CONFIG)
    set(XTENSOR_PYTHON_INCLUDE_DIR ${xtensor-python_INCLUDE_DIRS})
endif ()

message(STATUS "Forcing tests build type to Release")
set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)

include(CheckCXXCompilerFlag)

string(TOUPPER "${CMAKE_BUILD_TYPE}" U_CMAKE_BUILD_TYPE)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")

if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc /MP /bigobj")
    set(CMAKE_EXE_LINKER_FLAGS /MANIFEST:NO)
endif()

if (DOWNLOAD_GTEST OR GTEST_SRC_DIR)
    if(DOWNLOAD_GTEST)
        # Download and unpack googletest at configure time
        configure_file(downloadGTest.cmake.in googletest-download/CMakeLists.txt)
    else()
        # Copy local source of googletest at configure time
        configure_file(copyGTest.cmake.in googletest-download/CMakeLists.txt)
    endif()
    execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
                    RESULT_VARIABLE result
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download )
    if(result)
        message(FATAL_ERROR "CMake step for googletest failed: ${result}")
    endif()
    execute_process(COMMAND ${CMAKE_COMMAND} --build .
                    RESULT_VARIABLE result
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download )
    if(result)
        message(FATAL_ERROR "Build step for googletest failed: ${result}")
    endif()

    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

    # Add googletest directly to our build. This defines
    # the gtest and gtest_main targets.
    add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/googletest-src
                     ${CMAKE_CURRENT_BINARY_DIR}/googletest-build EXCLUDE_FROM_ALL)

    set(GTEST_INCLUDE_DIRS "${gtest_SOURCE_DIR}/include")
    set(GTEST_BOTH_LIBRARIES gtest_main gtest)
else()
    find_package(GTest REQUIRED)
endif()

find_package(Threads)

include_directories(${GTEST_INCLUDE_DIRS})

# Common test dependencies
set(COMMON_TEST_LIBS ${GTEST_BOTH_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${PYTHON_LIBRARIES})

# ============================================================================
# pybind11 backend tests
# ============================================================================
if(XTENSOR_PYTHON_HAS_PYBIND11)
    message(STATUS "Building pybind11 backend tests")
    
    set(XTENSOR_PYTHON_PYBIND11_TESTS
        main.cpp
        test_pyarray.cpp
        test_pyarray_traits.cpp
        test_pytensor.cpp
        test_pyvectorize.cpp
        test_sfinae.cpp
    )

    add_executable(test_xtensor_python_pybind11 ${XTENSOR_PYTHON_PYBIND11_TESTS} ${XTENSOR_PYTHON_HEADERS})
    target_compile_definitions(test_xtensor_python_pybind11 PRIVATE XTENSOR_PYTHON_BACKEND_PYBIND11)
    target_link_libraries(test_xtensor_python_pybind11 xtensor-python ${COMMON_TEST_LIBS})
    target_include_directories(test_xtensor_python_pybind11 PRIVATE ${pybind11_INCLUDE_DIRS})
    
    if(DOWNLOAD_GTEST OR GTEST_SRC_DIR)
        add_dependencies(test_xtensor_python_pybind11 gtest_main)
    endif()
endif()

# ============================================================================
# nanobind backend tests
# ============================================================================
if(XTENSOR_PYTHON_HAS_NANOBIND)
    message(STATUS "Building nanobind backend tests")
    
    # Build nanobind-static library (provides include directories)
    nanobind_build_library(nanobind-static)
    
    # nanobind tests - pytensor and pyarray
    set(XTENSOR_PYTHON_NANOBIND_TESTS
        main.cpp
        test_pytensor.cpp
        test_pyarray.cpp
    )

    add_executable(test_xtensor_python_nanobind ${XTENSOR_PYTHON_NANOBIND_TESTS} ${XTENSOR_PYTHON_HEADERS})
    target_compile_definitions(test_xtensor_python_nanobind PRIVATE XTENSOR_PYTHON_BACKEND_NANOBIND)
    target_link_libraries(test_xtensor_python_nanobind xtensor-python ${COMMON_TEST_LIBS} nanobind-static)
    
    if(DOWNLOAD_GTEST OR GTEST_SRC_DIR)
        add_dependencies(test_xtensor_python_nanobind gtest_main)
    endif()
endif()

# ============================================================================
# Combined test target
# ============================================================================
add_custom_target(xtest)

if(XTENSOR_PYTHON_HAS_PYBIND11)
    add_custom_target(xtest_pybind11 
        COMMAND ./test_xtensor_python_pybind11 
        DEPENDS test_xtensor_python_pybind11)
    add_dependencies(xtest xtest_pybind11)
endif()

if(XTENSOR_PYTHON_HAS_NANOBIND)
    add_custom_target(xtest_nanobind 
        COMMAND ./test_xtensor_python_nanobind 
        DEPENDS test_xtensor_python_nanobind)
    add_dependencies(xtest xtest_nanobind)
endif()

# Backward compatibility: alias to pybind11 test if available
if(XTENSOR_PYTHON_HAS_PYBIND11)
    add_custom_target(test_xtensor_python DEPENDS test_xtensor_python_pybind11)
endif()

