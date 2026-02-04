# xtensor-python tests meson build file

# ============================================================================
# Dependencies
# ============================================================================

gtest_dep = dependency('gtest', required: true)
gtest_main_dep = dependency('gtest_main', required: true)

# For embedding Python in test executables
python_embed_dep = python.dependency(embed: true)

common_test_deps = [
    xtensor_python_dep,
    python_embed_dep,
    gtest_dep,
    gtest_main_dep,
]

# ============================================================================
# pybind11 backend tests
# ============================================================================

if has_pybind11
    pybind11_test_sources = files(
        'main.cpp',
        'test_pyarray.cpp',
        'test_pyarray_traits.cpp',
        'test_pytensor.cpp',
        'test_pyvectorize.cpp',
        'test_sfinae.cpp',
    )
    
    test_pybind11 = executable(
        'test_xtensor_python_pybind11',
        pybind11_test_sources,
        dependencies: common_test_deps + [pybind11_dep],
        cpp_args: ['-DXTENSOR_PYTHON_BACKEND_PYBIND11'],
    )
    
    test('pybind11_tests', test_pybind11)
endif

# ============================================================================
# nanobind backend tests
# ============================================================================

if has_nanobind
    nanobind_test_sources = files(
        'main.cpp',
        'test_pytensor.cpp',
    )
    
    test_nanobind = executable(
        'test_xtensor_python_nanobind',
        nanobind_test_sources,
        dependencies: common_test_deps + [nanobind_dep],
        cpp_args: ['-DXTENSOR_PYTHON_BACKEND_NANOBIND'],
    )
    
    test('nanobind_tests', test_nanobind)
endif
