/***************************************************************************
* Copyright (c) Wolf Vollprecht, Johan Mabille and Sylvain Corlay          *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include <numeric>

#include "xtensor/core/xmath.hpp"
#include "xtensor/containers/xarray.hpp"
#include "xtensor/containers/xtensor.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "xtensor-python/nanobind/pytensor.hpp"

namespace nb = nanobind;

// Use nanobind pytensor
using xt::nanobind::pytensor;

// pyarray equivalent for nanobind - use dynamic rank pytensor or xarray with adapter
// For now, we'll use pytensor with specific ranks

// Examples

double example1(pytensor<double, 1>& m)
{
    return m(0);
}

pytensor<double, 2> example2(pytensor<double, 2>& m)
{
    return m + 2;
}

// Broadcast operations using pytensor
pytensor<double, 2> array_addition(pytensor<double, 2>& a, pytensor<double, 2>& b)
{
    return a + b;
}

pytensor<double, 2> array_subtraction(pytensor<double, 2>& a, pytensor<double, 2>& b)
{
    return a - b;
}

pytensor<double, 2> array_multiplication(pytensor<double, 2>& a, pytensor<double, 2>& b)
{
    return a * b;
}

pytensor<double, 2> array_division(pytensor<double, 2>& a, pytensor<double, 2>& b)
{
    return a / b;
}

// Readme example
double readme_example1(pytensor<double, 2>& m)
{
    auto sines = xt::sin(m);
    return std::accumulate(sines.cbegin(), sines.cend(), 0.0);
}

// Shape comparison
bool compare_shapes(pytensor<double, 2>& a, pytensor<double, 2>& b)
{
    return a.shape() == b.shape();
}

// Test row major
template <class T>
using ndarray = pytensor<T, 1>;

void test_rm(ndarray<int> const& x)
{
    ndarray<int> y = x;
    // Create a zeros tensor
    pytensor<int, 1> z = xt::zeros<int>({10});
}

NB_MODULE(xtensor_nanobind_test, m)
{
    m.doc() = "Test module for xtensor nanobind bindings";

    m.def("example1", &example1);
    m.def("example2", &example2);

    m.def("array_addition", &array_addition);
    m.def("array_subtraction", &array_subtraction);
    m.def("array_multiplication", &array_multiplication);
    m.def("array_division", &array_division);

    m.def("readme_example1", &readme_example1);

    m.def("compare_shapes", &compare_shapes);

    m.def("test_rm", &test_rm);
}
