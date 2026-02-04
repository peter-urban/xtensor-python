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
#include "xtensor/containers/xfixed.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "xtensor-python/nanobind/pytensor.hpp"
#include "xtensor-python/nanobind/pynative_casters.hpp"

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

// Example3 functions using native casters (xt::xarray, xt::xtensor, xt::xtensor_fixed)

xt::xarray<int> example3_xarray(const xt::xarray<int>& m)
{
    return xt::transpose(m) + 2;
}

xt::xarray<int, xt::layout_type::column_major> example3_xarray_colmajor(
    const xt::xarray<int, xt::layout_type::column_major>& m)
{
    return xt::transpose(m) + 2;
}

xt::xtensor<int, 3> example3_xtensor3(const xt::xtensor<int, 3>& m)
{
    return xt::transpose(m) + 2;
}

xt::xtensor<int, 2> example3_xtensor2(const xt::xtensor<int, 2>& m)
{
    return xt::transpose(m) + 2;
}

xt::xtensor<int, 2, xt::layout_type::column_major> example3_xtensor2_colmajor(
    const xt::xtensor<int, 2, xt::layout_type::column_major>& m)
{
    return xt::transpose(m) + 2;
}

xt::xtensor_fixed<int, xt::xshape<4, 3, 2>> example3_xfixed3(const xt::xtensor_fixed<int, xt::xshape<2, 3, 4>>& m)
{
    return xt::transpose(m) + 2;
}

xt::xtensor_fixed<int, xt::xshape<3, 2>> example3_xfixed2(const xt::xtensor_fixed<int, xt::xshape<2, 3>>& m)
{
    return xt::transpose(m) + 2;
}

xt::xtensor_fixed<int, xt::xshape<3, 2>, xt::layout_type::column_major> example3_xfixed2_colmajor(
    const xt::xtensor_fixed<int, xt::xshape<2, 3>, xt::layout_type::column_major>& m)
{
    return xt::transpose(m) + 2;
}

// Row major / Column major layout-specific functions
void row_major_tensor(pytensor<double, 3, xt::layout_type::row_major>& arg)
{
    if (!std::is_same<decltype(arg.begin()), double*>::value)
    {
        throw std::runtime_error("TEST FAILED");
    }
}

// Note: pyarray equivalent not available for nanobind, using pytensor with dynamic layout
void col_major_array(pytensor<double, 2, xt::layout_type::column_major>& arg)
{
    if (!std::is_same<decltype(arg.template begin<xt::layout_type::column_major>()), double*>::value)
    {
        throw std::runtime_error("TEST FAILED");
    }
}

// Scalar (0-dimensional) pytensor
pytensor<int, 0> xscalar(const pytensor<int, 1>& arg)
{
    return xt::sum(arg);
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

    // Example3 functions using native casters
    m.def("example3_xarray", &example3_xarray);
    m.def("example3_xarray_colmajor", &example3_xarray_colmajor);
    m.def("example3_xtensor3", &example3_xtensor3);
    m.def("example3_xtensor2", &example3_xtensor2);
    m.def("example3_xtensor2_colmajor", &example3_xtensor2_colmajor);
    m.def("example3_xfixed3", &example3_xfixed3);
    m.def("example3_xfixed2", &example3_xfixed2);
    m.def("example3_xfixed2_colmajor", &example3_xfixed2_colmajor);

    m.def("array_addition", &array_addition);
    m.def("array_subtraction", &array_subtraction);
    m.def("array_multiplication", &array_multiplication);
    m.def("array_division", &array_division);

    m.def("readme_example1", &readme_example1);

    m.def("compare_shapes", &compare_shapes);

    // Layout-specific functions
    m.def("row_major_tensor", &row_major_tensor);
    m.def("col_major_array", &col_major_array);

    // Scalar support
    m.def("xscalar", &xscalar);

    m.def("test_rm", &test_rm);
}
