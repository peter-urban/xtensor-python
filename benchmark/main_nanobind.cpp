/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include <complex>
#include <cmath>
#include <numeric>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/complex.h>

#include "xtensor/containers/xtensor.hpp"
#include "xtensor/containers/xarray.hpp"
#include "xtensor/containers/xadapt.hpp"
#include "xtensor/views/xstrided_view.hpp"
#include "xtensor/views/xview.hpp"
#include "xtensor/core/xmath.hpp"
#include "xtensor/generators/xbuilder.hpp"

#include "xtensor-python/nanobind/pytensor.hpp"
#include "xtensor-python/nanobind/pynative_casters.hpp"
#include "xtensor-python/nanobind/pyvectorize.hpp"

namespace nb = nanobind;
using complex_t = std::complex<double>;
using xt::nanobind::pytensor;

NB_MODULE(benchmark_xtensor_nanobind, m)
{
    m.doc() = "Benchmark module for xtensor nanobind bindings";

    // ========================================================================
    // Basic sum operations (comparison with pybind11)
    // ========================================================================

    // Sum using pytensor (zero-copy, contiguous data)
    m.def("sum_tensor", [](pytensor<double, 1> const& x) {
        return xt::sum(x)();
    });

    // Sum using native ndarray (nanobind's built-in array type)
    m.def("nanobind_sum_array", [](nb::ndarray<double, nb::ndim<1>> const& x) {
        // Use xt::adapt to wrap nanobind array and use xt::sum for SIMD
        auto adapted = xt::adapt(x.data(), x.size(), xt::no_ownership(), std::vector<size_t>{x.size()});
        return xt::sum(adapted)();
    });

    // ========================================================================
    // Auto conversion benchmarks (xtensor by value - creates copy)
    // ========================================================================

    // Takes xtensor by const reference - automatic conversion from numpy
    // This measures the copy overhead when converting numpy -> xtensor
    m.def("auto_convert_xtensor_input", [](const xt::xtensor<double, 1>& x) {
        return xt::sum(x)();
    });

    // Takes xarray by const reference - automatic conversion from numpy
    m.def("auto_convert_xarray_input", [](const xt::xarray<double>& x) {
        return xt::sum(x)();
    });

    // ========================================================================
    // Return value benchmarks (xtensor return - creates copy back to numpy)
    // ========================================================================

    // Returns xtensor - automatic conversion to numpy
    m.def("return_xtensor", [](size_t size) {
        xt::xtensor<double, 1> result = xt::arange<double>(static_cast<double>(size));
        return result;
    });

    // Returns xarray - automatic conversion to numpy
    m.def("return_xarray", [](size_t size) {
        xt::xarray<double> result = xt::arange<double>(static_cast<double>(size));
        return result;
    });

    // Returns pytensor - zero-copy return
    m.def("return_pytensor", [](size_t size) {
        pytensor<double, 1> result = xt::arange<double>(static_cast<double>(size));
        return result;
    });

    // ========================================================================
    // Inplace operation benchmarks (by reference)
    // ========================================================================

    // Inplace multiply by 2 using pytensor reference
    m.def("inplace_multiply_pytensor", [](pytensor<double, 1>& x) {
        x *= 2.0;
    });

    // Inplace add using pytensor reference
    m.def("inplace_add_pytensor", [](pytensor<double, 1>& x, double value) {
        x += value;
    });

    // Inplace operation on 2D pytensor
    m.def("inplace_multiply_pytensor_2d", [](pytensor<double, 2>& x) {
        x *= 2.0;
    });

    // ========================================================================
    // View operation benchmarks
    // ========================================================================

    // Sum on a strided view (non-contiguous memory access)
    m.def("sum_strided_view", [](pytensor<double, 1>& x) {
        auto view = xt::strided_view(x, {xt::range(0, xt::placeholders::_, 2)});
        return xt::sum(view)();
    });

    // Math operations on views
    m.def("math_on_view_pytensor", [](pytensor<double, 1>& x) {
        auto view = xt::strided_view(x, {xt::range(0, xt::placeholders::_, 2)});
        return xt::sum(xt::sin(view) + xt::cos(view))();
    });

    // Math operations on xarray via auto-conversion (includes copy overhead)
    m.def("math_on_xarray", [](const xt::xarray<double>& x) {
        return xt::sum(xt::sin(x) + xt::cos(x))();
    });

    // Math operations on xtensor via auto-conversion (includes copy overhead)
    m.def("math_on_xtensor", [](const xt::xtensor<double, 1>& x) {
        return xt::sum(xt::sin(x) + xt::cos(x))();
    });

    // Math operations on pytensor (no copy)
    m.def("math_on_pytensor", [](pytensor<double, 1> const& x) {
        return xt::sum(xt::sin(x) + xt::cos(x))();
    });

    // ========================================================================
    // Vectorize benchmarks
    // ========================================================================

    m.def("rect_to_polar", xt::nanobind::pyvectorize([](complex_t x) { return std::abs(x); }));

    // ========================================================================
    // Combined input/output benchmark (full round-trip)
    // ========================================================================

    // xtensor input, xtensor output - measures full copy overhead
    m.def("roundtrip_xtensor", [](const xt::xtensor<double, 1>& x) {
        xt::xtensor<double, 1> result = x * 2.0 + 1.0;
        return result;
    });

    // pytensor input, pytensor output - zero-copy operations
    m.def("roundtrip_pytensor", [](pytensor<double, 1> const& x) {
        pytensor<double, 1> result = x * 2.0 + 1.0;
        return result;
    });

    // ========================================================================
    // Large array benchmarks (to highlight memory overhead)
    // ========================================================================

    // Process large array with xtensor (copy)
    m.def("process_large_xtensor", [](const xt::xtensor<double, 2>& x) {
        return xt::sum(x)();
    });

    // Process large array with pytensor (no copy)
    m.def("process_large_pytensor", [](pytensor<double, 2> const& x) {
        return xt::sum(x)();
    });

    // ========================================================================
    // Type Conversion Benchmarks (automatic dtype conversion)
    // ========================================================================

    // int32 → double conversion
    m.def("type_convert_int32_to_double", [](pytensor<int32_t, 1> const& x) {
        xt::xtensor<double, 1> result = xt::cast<double>(x);
        return xt::sum(result)();
    });

    // float32 → double conversion
    m.def("type_convert_float32_to_double", [](pytensor<float, 1> const& x) {
        xt::xtensor<double, 1> result = xt::cast<double>(x);
        return xt::sum(result)();
    });

    // Accept double, work with it directly (no conversion)
    m.def("type_no_convert_double", [](pytensor<double, 1> const& x) {
        return xt::sum(x)();
    });

    // ========================================================================
    // Broadcasting Operations Benchmarks
    // ========================================================================

    // Broadcast scalar to array
    m.def("broadcast_scalar_add", [](pytensor<double, 1> const& x, double scalar) {
        pytensor<double, 1> result = x + scalar;
        return result;
    });

    // Broadcast 1D to 2D (row broadcast)
    m.def("broadcast_1d_to_2d", [](pytensor<double, 2> const& x, pytensor<double, 1> const& row) {
        pytensor<double, 2> result = x + row;
        return result;
    });

    // Broadcast with reduction
    m.def("broadcast_and_reduce", [](pytensor<double, 2> const& x, pytensor<double, 1> const& row) {
        auto broadcasted = x * row;
        return xt::sum(broadcasted)();
    });

    // ========================================================================
    // Slicing Operations Benchmarks
    // ========================================================================

    // Basic slice operation (contiguous)
    m.def("slice_contiguous", [](pytensor<double, 1> const& x, size_t start, size_t end) {
        auto view = xt::view(x, xt::range(start, end));
        return xt::sum(view)();
    });

    // Strided slice operation (every nth element)
    m.def("slice_strided", [](pytensor<double, 1> const& x, size_t step) {
        auto view = xt::view(x, xt::range(size_t(0), xt::placeholders::_, step));
        return xt::sum(view)();
    });

    // 2D slice operation (sub-matrix)
    m.def("slice_2d_submatrix", [](pytensor<double, 2> const& x, 
                                   size_t row_start, size_t row_end,
                                   size_t col_start, size_t col_end) {
        auto view = xt::view(x, xt::range(row_start, row_end), xt::range(col_start, col_end));
        return xt::sum(view)();
    });

    // Slice and modify (inplace on view)
    m.def("slice_and_modify", [](pytensor<double, 1>& x, size_t start, size_t end, double value) {
        auto view = xt::view(x, xt::range(start, end));
        view += value;
    });
}
