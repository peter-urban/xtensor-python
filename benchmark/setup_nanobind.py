############################################################################
# Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     #
#                                                                          #
# Distributed under the terms of the BSD 3-Clause License.                 #
#                                                                          #
# The full license is in the file LICENSE, distributed with this software. #
############################################################################

"""
Build script for benchmark_xtensor_nanobind extension using CMake.

Usage:
    python setup_nanobind.py build_ext --inplace
"""

import sys
import os
import subprocess
import shutil
import glob

# Check if nanobind is available
try:
    import nanobind
except ImportError:
    print("nanobind not installed, skipping nanobind extension build")
    sys.exit(1)


def build_with_cmake():
    """Build the extension using CMake."""
    here = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.join(here, 'build_nanobind')
    include_dir = os.path.abspath(os.path.join(here, '..', 'include'))
    # Convert backslashes to forward slashes for CMake compatibility on Windows
    include_dir = include_dir.replace('\\', '/')

    # Create build directory
    os.makedirs(build_dir, exist_ok=True)

    # Write CMakeLists.txt for nanobind build
    cmake_content = f"""
cmake_minimum_required(VERSION 3.18)
project(benchmark_xtensor_nanobind)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimize for release builds
set(CMAKE_BUILD_TYPE Release)

# Find Python
find_package(Python 3.8 REQUIRED COMPONENTS Interpreter Development.Module NumPy)

# Find nanobind
execute_process(
    COMMAND "${{Python_EXECUTABLE}}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

# Find xtensor
find_package(xtensor REQUIRED)

# Build the extension
nanobind_add_module(benchmark_xtensor_nanobind main_nanobind.cpp)

target_include_directories(benchmark_xtensor_nanobind PRIVATE
    {include_dir}
    ${{Python_NumPy_INCLUDE_DIRS}}
)

target_link_libraries(benchmark_xtensor_nanobind PRIVATE xtensor)

# Enable optimizations
target_compile_options(benchmark_xtensor_nanobind PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-O3 -march=native -fvisibility=hidden>
    $<$<CXX_COMPILER_ID:MSVC>:/O2>
)
"""
    cmake_file = os.path.join(build_dir, 'CMakeLists.txt')
    with open(cmake_file, 'w') as f:
        f.write(cmake_content)

    # Symlink the source file
    src_file = os.path.join(here, 'main_nanobind.cpp')
    dst_file = os.path.join(build_dir, 'main_nanobind.cpp')
    if os.path.exists(dst_file):
        os.remove(dst_file)
    os.symlink(src_file, dst_file)

    # Configure
    cmake_args = [
        'cmake',
        '.',
        f'-DCMAKE_BUILD_TYPE=Release',
    ]

    print(f"Configuring with: {' '.join(cmake_args)}")
    subprocess.check_call(cmake_args, cwd=build_dir)

    # Build
    build_args = ['cmake', '--build', '.', '--config', 'Release', '-j4']
    print(f"Building with: {' '.join(build_args)}")
    subprocess.check_call(build_args, cwd=build_dir)

    # Copy the built extension to the source directory
    so_files = glob.glob(os.path.join(build_dir, 'benchmark_xtensor_nanobind*.so'))
    if not so_files:
        so_files = glob.glob(os.path.join(build_dir, 'benchmark_xtensor_nanobind*.pyd'))
    for so_file in so_files:
        dest = os.path.join(here, os.path.basename(so_file))
        print(f"Copying {so_file} to {dest}")
        shutil.copy2(so_file, dest)

    print("Build successful!")


if __name__ == '__main__':
    if len(sys.argv) > 1 and 'build_ext' in sys.argv:
        build_with_cmake()
    else:
        print("Usage: python setup_nanobind.py build_ext --inplace")
        sys.exit(1)
