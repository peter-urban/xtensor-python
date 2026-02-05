/***************************************************************************
* Copyright (c) Wolf Vollprecht, Johan Mabille and Sylvain Corlay          *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include <limits>

#include "gtest/gtest.h"

// ============================================================================
// Backend selection via compile-time define
// ============================================================================
#if defined(XTENSOR_PYTHON_BACKEND_PYBIND11)
    #include "xtensor-python/pybind11/pytensor.hpp"
    #include "xtensor-python/pybind11/pyarray.hpp"
    #define BACKEND_NAME pybind11
#elif defined(XTENSOR_PYTHON_BACKEND_NANOBIND)
    #include "xtensor-python/nanobind/pytensor.hpp"
    #include "xtensor-python/nanobind/pyarray.hpp"
    #define BACKEND_NAME nanobind
#else
    // Default to pybind11 for backwards compatibility
    #include "xtensor-python/pybind11/pytensor.hpp"
    #include "xtensor-python/pybind11/pyarray.hpp"
    #define BACKEND_NAME pybind11
#endif

#include "xtensor/containers/xarray.hpp"
#include "xtensor/containers/xtensor.hpp"

// Create test suite name from backend
#define CONCAT_IMPL(a, b) a##_##b
#define CONCAT(a, b) CONCAT_IMPL(a, b)
#define TEST_SUITE_NAME CONCAT(sfinae, BACKEND_NAME)

namespace xt
{
    // Bring backend-specific types into xt namespace for tests
    using BACKEND_NAME::pyarray;
    using BACKEND_NAME::pytensor;
    template <class E, std::enable_if_t<!xt::has_fixed_rank_t<E>::value, int> = 0>
    inline bool sfinae_has_fixed_rank(E&&)
    {
        return false;
    }

    template <class E, std::enable_if_t<xt::has_fixed_rank_t<E>::value, int> = 0>
    inline bool sfinae_has_fixed_rank(E&&)
    {
        return true;
    }

    TEST(TEST_SUITE_NAME, fixed_rank)
    {
        xt::pyarray<size_t> a = {{9, 9, 9}, {9, 9, 9}};
        xt::pytensor<size_t, 1> b = {9, 9};
        xt::pytensor<size_t, 2> c = {{9, 9}, {9, 9}};

        EXPECT_TRUE(sfinae_has_fixed_rank(a) == false);
        EXPECT_TRUE(sfinae_has_fixed_rank(b) == true);
        EXPECT_TRUE(sfinae_has_fixed_rank(c) == true);
    }

    TEST(TEST_SUITE_NAME, get_rank)
    {
        xt::pytensor<double, 1> A = xt::zeros<double>({2});
        xt::pytensor<double, 2> B = xt::zeros<double>({2, 2});
        xt::pyarray<double> C = xt::zeros<double>({2, 2});

        EXPECT_TRUE(xt::get_rank<decltype(A)>::value == 1ul);
        EXPECT_TRUE(xt::get_rank<decltype(B)>::value == 2ul);
        EXPECT_TRUE(xt::get_rank<decltype(C)>::value == SIZE_MAX);
    }
}
