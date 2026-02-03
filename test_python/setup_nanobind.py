############################################################################
# Copyright (c) Wolf Vollprecht, Johan Mabille and Sylvain Corlay          #
# Copyright (c) QuantStack                                                 #
#                                                                          #
# Distributed under the terms of the BSD 3-Clause License.                 #
#                                                                          #
# The full license is in the file LICENSE, distributed with this software. #
############################################################################

"""
Build script for xtensor_nanobind_test extension using CMake.

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
    
    # Create build directory
    os.makedirs(build_dir, exist_ok=True)
    
    # Configure
    cmake_args = [
        'cmake',
        here,
        f'-DCMAKE_BUILD_TYPE=Release',
    ]
    
    print(f"Configuring with: {' '.join(cmake_args)}")
    subprocess.check_call(cmake_args, cwd=build_dir)
    
    # Build
    build_args = ['cmake', '--build', '.', '--config', 'Release', '-j4']
    print(f"Building with: {' '.join(build_args)}")
    subprocess.check_call(build_args, cwd=build_dir)
    
    # Copy the built extension to the source directory
    import glob
    so_files = glob.glob(os.path.join(build_dir, 'xtensor_nanobind_test*.so'))
    for so_file in so_files:
        dest = os.path.join(here, os.path.basename(so_file))
        print(f"Copying {so_file} to {dest}")
        import shutil
        shutil.copy2(so_file, dest)
    
    print("Build successful!")


if __name__ == '__main__':
    if len(sys.argv) > 1 and 'build_ext' in sys.argv:
        build_with_cmake()
    else:
        print("Usage: python setup_nanobind.py build_ext --inplace")
        sys.exit(1)
