/***************************************************************************
* Copyright (c) Wolf Vollprecht, Johan Mabille and Sylvain Corlay          *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "gtest/gtest.h"
#include "test_common.hpp"

// ============================================================================
// Backend selection via compile-time define
// ============================================================================
#if defined(XTENSOR_PYTHON_BACKEND_PYBIND11)
    #include "xtensor-python/pybind11/pytensor.hpp"
    #include "xtensor-python/pybind11/pyarray.hpp"
    #include "xtensor-python/pybind11/pyvectorize.hpp"
    #include "pybind11/pybind11.h"
    #include "pybind11/numpy.h"
    #define BACKEND_NAME pybind11
#elif defined(XTENSOR_PYTHON_BACKEND_NANOBIND)
    #include "xtensor-python/nanobind/pytensor.hpp"
    #include "xtensor-python/nanobind/pyarray.hpp"
    #include "xtensor-python/nanobind/pyvectorize.hpp"
    #include <nanobind/nanobind.h>
    #include <nanobind/ndarray.h>
    #define BACKEND_NAME nanobind
#else
    // Default to pybind11 for backwards compatibility
    #include "xtensor-python/pybind11/pytensor.hpp"
    #include "xtensor-python/pybind11/pyarray.hpp"
    #include "xtensor-python/pybind11/pyvectorize.hpp"
    #include "pybind11/pybind11.h"
    #include "pybind11/numpy.h"
    #define BACKEND_NAME pybind11
#endif

// Create test suite name from backend
#define CONCAT_IMPL(a, b) a##_##b
#define CONCAT(a, b) CONCAT_IMPL(a, b)
#define TEST_SUITE_NAME CONCAT(pyvectorize, BACKEND_NAME)

namespace xt
{
    // Bring backend-specific types into xt namespace for tests
    using BACKEND_NAME::pyarray;
    using BACKEND_NAME::pytensor;
    using BACKEND_NAME::pyvectorize;

    double f1(double a, double b)
    {
        return a + b;
    }

    using shape_type = std::vector<std::size_t>;

    TEST(TEST_SUITE_NAME, function)
    {
        auto vecf1 = pyvectorize(f1);
        shape_type shape = { 3, 2 };
        pyarray<double> a(shape, 1.5);
        pyarray<double> b(shape, 2.3);
        pyarray<double> c = vecf1(a, b);
        EXPECT_EQ(a(0, 0) + b(0, 0), c(0, 0));
    }

    TEST(TEST_SUITE_NAME, lambda)
    {
        auto vecf1 = pyvectorize([](double a, double b) { return a + b; });
        shape_type shape = { 3, 2 };
        pyarray<double> a(shape, 1.5);
        pyarray<double> b(shape, 2.3);
        pyarray<double> c = vecf1(a, b);
        EXPECT_EQ(a(0, 0) + b(0, 0), c(0, 0));
    }

    TEST(TEST_SUITE_NAME, complex)
    {
        using complex_t = std::complex<double>;
        shape_type shape = { 3, 2 };
        pyarray<complex_t> a(shape, complex_t(1.2, 2.5));
        auto f = pyvectorize([](complex_t x) { return std::abs(x); });
        auto res = f(a);
        double exp = std::abs(a(1, 1));
        EXPECT_EQ(exp, res(1, 1));
    }
}
