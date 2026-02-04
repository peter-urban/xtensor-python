# xtensor-python meson build file
# Copyright (c) Wolf Vollprecht, Johan Mabille and Sylvain Corlay
# Copyright (c) QuantStack
# Distributed under the terms of the BSD 3-Clause License.

project(
    'xtensor-python',
    'cpp',
    version: '0.29.0',
    license: 'BSD-3-Clause',
    default_options: [
        'cpp_std=c++20',
        'warning_level=2',
    ],
    meson_version: '>=1.0.0',
)

# ============================================================================
# Configuration
# ============================================================================

cpp = meson.get_compiler('cpp')
include_dir = include_directories('include')

# ============================================================================
# Dependencies
# ============================================================================

# Required: xtensor
xtensor_dep = dependency('xtensor', version: '>=0.27.0', required: true)

# Required: Python
python = import('python').find_installation(pure: false)
python_dep = python.dependency()

# Required: NumPy
numpy_inc = run_command(
    python, '-c', 'import numpy; print(numpy.get_include())',
    check: true,
).stdout().strip()
numpy_dep = declare_dependency(include_directories: include_directories(numpy_inc))

# Optional backends (from WrapDB)
use_pybind11 = get_option('use_pybind11')
use_nanobind = get_option('use_nanobind')

# pybind11 backend
pybind11_dep = disabler()
has_pybind11 = false

if use_pybind11 != 'disabled'
    pybind11_dep = dependency('pybind11', required: use_pybind11 == 'enabled')
    has_pybind11 = pybind11_dep.found()
endif

# nanobind backend
nanobind_dep = disabler()
has_nanobind = false

if use_nanobind != 'disabled'
    nanobind_dep = dependency('nanobind', required: use_nanobind == 'enabled')
    has_nanobind = nanobind_dep.found()
endif

# Require at least one backend
if not has_pybind11 and not has_nanobind
    error('No binding backend found. Install pybind11 or nanobind.')
endif

# ============================================================================
# Header-only library
# ============================================================================

xtensor_python_headers = files(
    'include/xtensor-python/detail/pyarray_backstrides.hpp',
    'include/xtensor-python/detail/pystrides_adaptor.hpp',
    'include/xtensor-python/xtensor_python_config.hpp',
)

if has_pybind11
    xtensor_python_headers += files(
        'include/xtensor-python/pybind11/pyarray.hpp',
        'include/xtensor-python/pybind11/pycontainer.hpp',
        'include/xtensor-python/pybind11/pynative_casters.hpp',
        'include/xtensor-python/pybind11/pytensor.hpp',
        'include/xtensor-python/pybind11/pyvectorize.hpp',
        'include/xtensor-python/pybind11/xtensor_type_caster_base.hpp',
    )
endif

if has_nanobind
    xtensor_python_headers += files(
        'include/xtensor-python/nanobind/pycontainer.hpp',
        'include/xtensor-python/nanobind/pytensor.hpp',
    )
endif

# Create the header-only library dependency
xtensor_python_dep = declare_dependency(
    include_directories: include_dir,
    dependencies: [xtensor_dep, numpy_dep],
)

# ============================================================================
# Tests
# ============================================================================

if get_option('tests')
    subdir('test')
endif

# ============================================================================
# Installation
# ============================================================================

install_headers(
    xtensor_python_headers,
    subdir: 'xtensor-python',
    preserve_path: true,
)

pkg = import('pkgconfig')
pkg.generate(
    name: 'xtensor-python',
    description: 'Python bindings for xtensor',
    version: meson.project_version(),
    requires: ['xtensor'],
)

# ============================================================================
# Summary
# ============================================================================

summary({
    'Version': meson.project_version(),
    'pybind11 backend': has_pybind11,
    'nanobind backend': has_nanobind,
    'Build tests': get_option('tests'),
}, section: 'Configuration')
