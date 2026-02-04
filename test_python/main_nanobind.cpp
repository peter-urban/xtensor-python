/***************************************************************************
* Copyright (c) Wolf Vollprecht, Johan Mabille and Sylvain Corlay          *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include <numeric>
#include <complex>
#include <cmath>

#include "xtensor/core/xmath.hpp"
#include "xtensor/containers/xarray.hpp"
#include "xtensor/containers/xtensor.hpp"
#include "xtensor/containers/xfixed.hpp"
#include "xtensor/containers/xadapt.hpp"
#include "xtensor/views/xstrided_view.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/complex.h>

#include "xtensor-python/nanobind/pytensor.hpp"
#include "xtensor-python/nanobind/pynative_casters.hpp"
#include "xtensor-python/nanobind/pyvectorize.hpp"

namespace nb = nanobind;
using complex_t = std::complex<double>;

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

double readme_example2(double i, double j)
{
    return std::sin(i) - std::cos(j);
}

// Complex overload functions
auto complex_overload(const xt::xarray<std::complex<double>>& a)
{
    return a;
}

auto no_complex_overload(const xt::xarray<double>& a)
{
    return a;
}

auto complex_overload_reg(const std::complex<double>& a)
{
    return a;
}

auto no_complex_overload_reg(const double& a)
{
    return a;
}

// Vectorize Examples
int add(int i, int j)
{
    return i + j;
}

// Type string helpers for int_overload
template <class T> std::string typestring() { return "Unknown"; }
template <> std::string typestring<uint8_t>() { return "uint8"; }
template <> std::string typestring<int8_t>() { return "int8"; }
template <> std::string typestring<uint16_t>() { return "uint16"; }
template <> std::string typestring<int16_t>() { return "int16"; }
template <> std::string typestring<uint32_t>() { return "uint32"; }
template <> std::string typestring<int32_t>() { return "int32"; }
template <> std::string typestring<uint64_t>() { return "uint64"; }
template <> std::string typestring<int64_t>() { return "int64"; }

template <class T>
inline std::string int_overload(xt::xarray<T>& m)
{
    return typestring<T>();
}

// Class C with properties for testing reference semantics
class C
{
public:
    using array_type = xt::xarray<double, xt::layout_type::row_major>;
    C() : m_array{0, 0, 0, 0} {}
    array_type& array() { return m_array; }
private:
    array_type m_array;
};

// Test native casters struct
struct test_native_casters
{
    using array_type = xt::xarray<double>;
    array_type a = xt::ones<double>({50, 50});

    const auto& get_array()
    {
        return a;
    }

    auto get_strided_view()
    {
        return xt::strided_view(a, {xt::range(0, 1), xt::range(0, 3, 2)});
    }

    auto get_array_adapter()
    {
        using shape_type = std::vector<size_t>;
        shape_type shape = {2, 2};
        shape_type stride = {3, 2};
        return xt::adapt(a.data(), 4, xt::no_ownership(), shape, stride);
    }

    auto get_tensor_adapter()
    {
        using shape_type = std::array<size_t, 2>;
        shape_type shape = {2, 2};
        shape_type stride = {3, 2};
        return xt::adapt(a.data(), 4, xt::no_ownership(), shape, stride);
    }

    auto get_owning_array_adapter()
    {
        size_t size = 100;
        int* data = new int[size];
        std::fill(data, data + size, 1);

        using shape_type = std::vector<size_t>;
        shape_type shape = {size};
        return xt::adapt(std::move(data), size, xt::acquire_ownership(), shape);
    }
};

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
    m.def("readme_example2", xt::nanobind::pyvectorize(readme_example2));

    m.def("compare_shapes", &compare_shapes);

    // Layout-specific functions
    m.def("row_major_tensor", &row_major_tensor);
    m.def("col_major_array", &col_major_array);

    // Scalar support
    m.def("xscalar", &xscalar);

    // Vectorize examples
    m.def("vectorize_example1", xt::nanobind::pyvectorize(add));
    m.def("rect_to_polar", xt::nanobind::pyvectorize([](complex_t x) { return std::abs(x); }));

    // Complex overload functions
    m.def("complex_overload", no_complex_overload);
    m.def("complex_overload", complex_overload);
    m.def("complex_overload_reg", no_complex_overload_reg);
    m.def("complex_overload_reg", complex_overload_reg);

    // Int overload functions
    m.def("int_overload", int_overload<uint8_t>);
    m.def("int_overload", int_overload<int8_t>);
    m.def("int_overload", int_overload<uint16_t>);
    m.def("int_overload", int_overload<int16_t>);
    m.def("int_overload", int_overload<uint32_t>);
    m.def("int_overload", int_overload<int32_t>);
    m.def("int_overload", int_overload<uint64_t>);
    m.def("int_overload", int_overload<int64_t>);

    // Simple array/tensor functions for bad argument tests
    m.def("simple_array", [](xt::xarray<int>) { return 1; });
    m.def("simple_tensor", [](pytensor<int, 1>) { return 2; });

    // Different shape overloads
    m.def("diff_shape_overload", [](pytensor<int, 1> a) { return 1; });
    m.def("diff_shape_overload", [](pytensor<int, 2> a) { return 2; });

    // Class C with properties
    nb::class_<C>(m, "C")
        .def(nb::init<>())
        .def_prop_ro(
            "copy",
            [](C& self) { return self.array(); },
            nb::rv_policy::copy
        )
        .def_prop_ro(
            "ref",
            [](C& self) -> C::array_type& { return self.array(); },
            nb::rv_policy::reference_internal
        );

    // Test native casters class
    nb::class_<test_native_casters>(m, "test_native_casters")
        .def(nb::init<>())
        .def("get_array", &test_native_casters::get_array, nb::rv_policy::reference_internal)
        .def("get_strided_view", &test_native_casters::get_strided_view, nb::keep_alive<0, 1>())
        .def("get_array_adapter", &test_native_casters::get_array_adapter, nb::keep_alive<0, 1>())
        .def("get_tensor_adapter", &test_native_casters::get_tensor_adapter, nb::keep_alive<0, 1>())
        .def("get_owning_array_adapter", &test_native_casters::get_owning_array_adapter)
        .def("view_keep_alive_member_function", [](test_native_casters& self, xt::xarray<double>& a) {
                return xt::reshape_view(a, {a.size(), });
            }, nb::keep_alive<0, 2>());

    m.def("test_rm", &test_rm);
}
