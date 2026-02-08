/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#define FORCE_IMPORT_ARRAY
#include "xtensor/containers/xtensor.hpp"
#include "xtensor/containers/xarray.hpp"
#include "xtensor/containers/xadapt.hpp"
#include "xtensor/views/xstrided_view.hpp"
#include "xtensor/views/xview.hpp"
#include "xtensor/core/xmath.hpp"
#include "xtensor/generators/xbuilder.hpp"
#include "xtensor-python/pyarray.hpp"
#include "xtensor-python/pytensor.hpp"
#include "xtensor-python/pyvectorize.hpp"

#include <functional>

using complex_t = std::complex<double>;

namespace py = pybind11;

PYBIND11_MODULE(benchmark_xtensor_python, m)
{
    xt::import_numpy();

    m.doc() = "Benchmark module for xtensor python bindings";

    // ========================================================================
    // Basic sum operations (original benchmarks)
    // ========================================================================

    m.def("sum_array", [](xt::pyarray<double> const& x) {
        return xt::sum(x)();
    }, py::arg("x").noconvert());

    m.def("sum_tensor", [](xt::pytensor<double, 1> const& x) {
        return xt::sum(x)();
    }, py::arg("x").noconvert());

    m.def("pybind_sum_array", [](py::array_t<double> const& x) {
        // Use xt::adapt to wrap numpy array and use xt::sum for SIMD
        auto adapted = xt::adapt(x.data(0), x.size(), xt::no_ownership(), std::vector<size_t>{static_cast<size_t>(x.size())});
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
        xt::pytensor<double, 1> result = xt::arange<double>(static_cast<double>(size));
        return result;
    });

    // Returns pyarray - zero-copy return
    m.def("return_pyarray", [](size_t size) {
        xt::pyarray<double> result = xt::arange<double>(static_cast<double>(size));
        return result;
    });

    // ========================================================================
    // Inplace operation benchmarks (by reference)
    // ========================================================================

    // Inplace multiply by 2 using pytensor reference
    m.def("inplace_multiply_pytensor", [](xt::pytensor<double, 1>& x) {
        x *= 2.0;
    }, py::arg("x").noconvert());

    // Inplace add using pytensor reference
    m.def("inplace_add_pytensor", [](xt::pytensor<double, 1>& x, double value) {
        x += value;
    }, py::arg("x").noconvert(), py::arg("value"));

    // Inplace multiply using pyarray reference
    m.def("inplace_multiply_pyarray", [](xt::pyarray<double>& x) {
        x *= 2.0;
    }, py::arg("x").noconvert());

    // Inplace operation on 2D pytensor
    m.def("inplace_multiply_pytensor_2d", [](xt::pytensor<double, 2>& x) {
        x *= 2.0;
    }, py::arg("x").noconvert());

    // ========================================================================
    // View operation benchmarks
    // ========================================================================

    // Sum on a strided view (non-contiguous memory access)
    m.def("sum_strided_view", [](xt::pytensor<double, 1>& x) {
        auto view = xt::strided_view(x, {xt::range(0, xt::placeholders::_, 2)});
        return xt::sum(view)();
    }, py::arg("x").noconvert());

    // Math operations on views
    m.def("math_on_view_pytensor", [](xt::pytensor<double, 1>& x) {
        auto view = xt::strided_view(x, {xt::range(0, xt::placeholders::_, 2)});
        return xt::sum(xt::sin(view) + xt::cos(view))();
    }, py::arg("x").noconvert());

    // Math operations on xarray via auto-conversion (includes copy overhead)
    m.def("math_on_xarray", [](const xt::xarray<double>& x) {
        return xt::sum(xt::sin(x) + xt::cos(x))();
    });

    // Math operations on xtensor via auto-conversion (includes copy overhead)
    m.def("math_on_xtensor", [](const xt::xtensor<double, 1>& x) {
        return xt::sum(xt::sin(x) + xt::cos(x))();
    });

    // Math operations on pytensor (no copy)
    m.def("math_on_pytensor", [](xt::pytensor<double, 1> const& x) {
        return xt::sum(xt::sin(x) + xt::cos(x))();
    }, py::arg("x").noconvert());

    // Math operations on pyarray (no copy)
    m.def("math_on_pyarray", [](xt::pyarray<double> const& x) {
        return xt::sum(xt::sin(x) + xt::cos(x))();
    }, py::arg("x").noconvert());

    // ========================================================================
    // Vectorize benchmarks
    // ========================================================================

    m.def("rect_to_polar", xt::pyvectorize([](complex_t x) { return std::abs(x); }));

    m.def("pybind_rect_to_polar", [](py::array a) {
        if (py::isinstance<py::array_t<complex_t>>(a))
            return py::vectorize([](complex_t x) { return std::abs(x); })(a);
        else
            throw py::type_error("rect_to_polar unhandled type");
    });

    // ========================================================================
    // Combined input/output benchmark (full round-trip)
    // ========================================================================

    // xtensor input, xtensor output - measures full copy overhead
    m.def("roundtrip_xtensor", [](const xt::xtensor<double, 1>& x) {
        xt::xtensor<double, 1> result = x * 2.0 + 1.0;
        return result;
    });

    // pytensor input, pytensor output - zero-copy operations
    m.def("roundtrip_pytensor", [](xt::pytensor<double, 1> const& x) {
        xt::pytensor<double, 1> result = x * 2.0 + 1.0;
        return result;
    }, py::arg("x").noconvert());

    // pyarray input, pyarray output - zero-copy operations
    m.def("roundtrip_pyarray", [](xt::pyarray<double> const& x) {
        xt::pyarray<double> result = x * 2.0 + 1.0;
        return result;
    }, py::arg("x").noconvert());

    // ========================================================================
    // Large array benchmarks (to highlight memory overhead)
    // ========================================================================

    // Process large array with xtensor (copy)
    m.def("process_large_xtensor", [](const xt::xtensor<double, 2>& x) {
        return xt::sum(x)();
    });

    // Process large array with pytensor (no copy)
    m.def("process_large_pytensor", [](xt::pytensor<double, 2> const& x) {
        return xt::sum(x)();
    }, py::arg("x").noconvert());

    // Process large array with pyarray (no copy)
    m.def("process_large_pyarray", [](xt::pyarray<double> const& x) {
        return xt::sum(x)();
    }, py::arg("x").noconvert());

    // ========================================================================
    // Type Conversion Benchmarks (automatic dtype conversion)
    // ========================================================================

    // int32 → double conversion
    m.def("type_convert_int32_to_double", [](xt::pytensor<int32_t, 1> const& x) {
        xt::xtensor<double, 1> result = xt::cast<double>(x);
        return xt::sum(result)();
    }, py::arg("x").noconvert());

    // float32 → double conversion
    m.def("type_convert_float32_to_double", [](xt::pytensor<float, 1> const& x) {
        xt::xtensor<double, 1> result = xt::cast<double>(x);
        return xt::sum(result)();
    }, py::arg("x").noconvert());

    // Accept double, work with it directly (no conversion)
    m.def("type_no_convert_double", [](xt::pytensor<double, 1> const& x) {
        return xt::sum(x)();
    }, py::arg("x").noconvert());

    // ========================================================================
    // Broadcasting Operations Benchmarks
    // ========================================================================

    // Broadcast scalar to array
    m.def("broadcast_scalar_add", [](xt::pytensor<double, 1> const& x, double scalar) {
        xt::pytensor<double, 1> result = x + scalar;
        return result;
    }, py::arg("x").noconvert(), py::arg("scalar"));

    // Broadcast 1D to 2D (row broadcast)
    m.def("broadcast_1d_to_2d", [](xt::pytensor<double, 2> const& x, xt::pytensor<double, 1> const& row) {
        xt::pytensor<double, 2> result = x + row;
        return result;
    }, py::arg("x").noconvert(), py::arg("row").noconvert());

    // Broadcast with reduction
    m.def("broadcast_and_reduce", [](xt::pytensor<double, 2> const& x, xt::pytensor<double, 1> const& row) {
        auto broadcasted = x * row;
        return xt::sum(broadcasted)();
    }, py::arg("x").noconvert(), py::arg("row").noconvert());

    // ========================================================================
    // Slicing Operations Benchmarks
    // ========================================================================

    // Basic slice operation (contiguous)
    m.def("slice_contiguous", std::function<double(xt::pytensor<double, 1> const&, size_t, size_t)>(
        [](xt::pytensor<double, 1> const& x, size_t start, size_t end) {
            auto view = xt::view(x, xt::range(start, end));
            return xt::sum(view)();
        }), py::arg("x").noconvert(), py::arg("start"), py::arg("end"));

    // Strided slice operation (every nth element)
    m.def("slice_strided", std::function<double(xt::pytensor<double, 1> const&, size_t)>(
        [](xt::pytensor<double, 1> const& x, size_t step) {
            auto view = xt::view(x, xt::range(size_t(0), xt::placeholders::_, step));
            return xt::sum(view)();
        }), py::arg("x").noconvert(), py::arg("step"));

    // 2D slice operation (sub-matrix)
    m.def("slice_2d_submatrix", std::function<double(xt::pytensor<double, 2> const&, size_t, size_t, size_t, size_t)>(
        [](xt::pytensor<double, 2> const& x, 
           size_t row_start, size_t row_end,
           size_t col_start, size_t col_end) {
            auto view = xt::view(x, xt::range(row_start, row_end), xt::range(col_start, col_end));
            return xt::sum(view)();
        }), py::arg("x").noconvert(), py::arg("row_start"), py::arg("row_end"), py::arg("col_start"), py::arg("col_end"));

    // Slice and modify (inplace on view)
    m.def("slice_and_modify", std::function<void(xt::pytensor<double, 1>&, size_t, size_t, double)>(
        [](xt::pytensor<double, 1>& x, size_t start, size_t end, double value) {
            auto view = xt::view(x, xt::range(start, end));
            view += value;
        }), py::arg("x").noconvert(), py::arg("start"), py::arg("end"), py::arg("value"));

    // ========================================================================
    // Reference Test (verify pytensor is not copied)
    // ========================================================================

    // Modify array in-place - Python can verify the modification
    m.def("reference_test_modify", [](xt::pytensor<double, 2>& x) {
        // Set specific pattern: x[i,j] = i * 1000 + j
        for (size_t i = 0; i < x.shape(0); ++i) {
            for (size_t j = 0; j < x.shape(1); ++j) {
                x(i, j) = static_cast<double>(i * 1000 + j);
            }
        }
        // No return - Python verifies the array was modified in-place
    }, py::arg("x").noconvert());

    // ========================================================================
    // Pure C++ Benchmarks (pytensor vs native xtensor, no Python overhead)
    // ========================================================================

    // Benchmark: create native xtensor, do math, return sum
    m.def("cpp_native_xtensor_math", [](size_t size) {
        xt::xtensor<double, 1> x = xt::ones<double>({size});
        auto result = xt::sin(x) + xt::cos(x);
        return xt::sum(result)();
    });

    // Benchmark: use pytensor (already allocated), do math, return sum
    m.def("cpp_pytensor_math", [](xt::pytensor<double, 1>& x) {
        auto result = xt::sin(x) + xt::cos(x);
        return xt::sum(result)();
    }, py::arg("x").noconvert());

    // Benchmark: create native xtensor 2D, do reduction
    m.def("cpp_native_xtensor_2d_sum", [](size_t rows, size_t cols) {
        xt::xtensor<double, 2> x = xt::ones<double>({rows, cols});
        return xt::sum(x)();
    });

    // Benchmark: pytensor 2D sum (already allocated)
    m.def("cpp_pytensor_2d_sum", [](xt::pytensor<double, 2> const& x) {
        return xt::sum(x)();
    }, py::arg("x").noconvert());

    // Benchmark: native xtensor strided view sum
    m.def("cpp_native_strided_view_sum", [](size_t size) {
        xt::xtensor<double, 1> x = xt::ones<double>({size});
        auto view = xt::strided_view(x, {xt::range(0, xt::placeholders::_, 2)});
        return xt::sum(view)();
    });

    // Benchmark: pytensor strided view sum
    m.def("cpp_pytensor_strided_view_sum", [](xt::pytensor<double, 1> const& x) {
        auto view = xt::strided_view(x, {xt::range(0, xt::placeholders::_, 2)});
        return xt::sum(view)();
    }, py::arg("x").noconvert());

    // Benchmark: native xtensor broadcast + reduce
    m.def("cpp_native_broadcast_reduce", [](size_t rows, size_t cols) {
        xt::xtensor<double, 2> x = xt::ones<double>({rows, cols});
        xt::xtensor<double, 1> row = xt::arange<double>(static_cast<double>(cols));
        auto broadcasted = x * row;
        return xt::sum(broadcasted)();
    });

    // Benchmark: pytensor broadcast + reduce
    m.def("cpp_pytensor_broadcast_reduce", [](xt::pytensor<double, 2> const& x, xt::pytensor<double, 1> const& row) {
        auto broadcasted = x * row;
        return xt::sum(broadcasted)();
    }, py::arg("x").noconvert(), py::arg("row").noconvert());

    // ========================================================================
    // No-convert benchmarks for fair NumPy comparison
    // These reject arrays that would require dtype conversion
    // ========================================================================

    m.def("noconvert_sum_tensor", [](xt::pytensor<double, 1> const& x) {
        return xt::sum(x)();
    }, py::arg("x").noconvert());

    m.def("noconvert_sum_tensor_2d", [](xt::pytensor<double, 2> const& x) {
        return xt::sum(x)();
    }, py::arg("x").noconvert());

    m.def("noconvert_inplace_multiply", [](xt::pytensor<double, 1>& x) {
        x *= 2.0;
    }, py::arg("x").noconvert());

    m.def("noconvert_math", [](xt::pytensor<double, 1> const& x) {
        return xt::sum(xt::sin(x) + xt::cos(x))();
    }, py::arg("x").noconvert());

    // ========================================================================
    // Diagnostic benchmarks to isolate type caster overhead
    // These help identify WHERE the xarray slowdown occurs
    // ========================================================================

    // Just retrieve the array and return size (test array import overhead)
    m.def("diag_ndarray_only", [](py::array_t<double, py::array::c_style | py::array::forcecast> const& arr) {
        return arr.size();
    });

    // Allocate xtensor from_shape only (no data copy)
    m.def("diag_xtensor_alloc_only", [](py::array_t<double, py::array::c_style | py::array::forcecast> const& arr) {
        auto buf = arr.request();
        xt::xtensor<double, 1> result = xt::xtensor<double, 1>::from_shape({static_cast<size_t>(buf.shape[0])});
        return result.size();
    });

    // Allocate xarray from_shape only (no data copy)
    m.def("diag_xarray_alloc_only", [](py::array_t<double, py::array::c_style | py::array::forcecast> const& arr) {
        auto buf = arr.request();
        std::vector<std::size_t> shape(buf.ndim);
        for (size_t i = 0; i < buf.ndim; ++i) {
            shape[i] = static_cast<size_t>(buf.shape[i]);
        }
        xt::xarray<double> result = xt::xarray<double>::from_shape(shape);
        return result.size();
    });

    // Full xtensor conversion (alloc + copy) - manual implementation
    m.def("diag_xtensor_full", [](py::array_t<double, py::array::c_style | py::array::forcecast> const& arr) {
        auto buf = arr.request();
        xt::xtensor<double, 1> result = xt::xtensor<double, 1>::from_shape({static_cast<size_t>(buf.shape[0])});
        std::copy(static_cast<double*>(buf.ptr), static_cast<double*>(buf.ptr) + buf.size, result.data());
        return xt::sum(result)();
    });

    // Full xarray conversion (alloc + copy) - manual implementation
    m.def("diag_xarray_full", [](py::array_t<double, py::array::c_style | py::array::forcecast> const& arr) {
        auto buf = arr.request();
        std::vector<std::size_t> shape(buf.ndim);
        for (size_t i = 0; i < buf.ndim; ++i) {
            shape[i] = static_cast<size_t>(buf.shape[i]);
        }
        xt::xarray<double> result = xt::xarray<double>::from_shape(shape);
        std::copy(static_cast<double*>(buf.ptr), static_cast<double*>(buf.ptr) + buf.size, result.data());
        return xt::sum(result)();
    });

    // Test if the slowdown is in Type::from_shape by using pre-sized containers
    m.def("diag_xtensor_presized", [](py::array_t<double, py::array::c_style | py::array::forcecast> const& arr) {
        auto buf = arr.request();
        xt::xtensor<double, 1> result = xt::xtensor<double, 1>::from_shape({static_cast<size_t>(buf.shape[0])});
        std::copy(static_cast<double*>(buf.ptr), static_cast<double*>(buf.ptr) + buf.size, result.data());
        return xt::sum(result)();
    });

    m.def("diag_xarray_presized", [](py::array_t<double, py::array::c_style | py::array::forcecast> const& arr) {
        auto buf = arr.request();
        std::vector<std::size_t> shape(buf.ndim);
        for (size_t i = 0; i < buf.ndim; ++i) {
            shape[i] = static_cast<size_t>(buf.shape[i]);
        }
        xt::xarray<double> result(shape);
        std::copy(static_cast<double*>(buf.ptr), static_cast<double*>(buf.ptr) + buf.size, result.data());
        return xt::sum(result)();
    });

    // Test memcpy vs std::copy
    m.def("diag_xtensor_memcpy", [](py::array_t<double, py::array::c_style | py::array::forcecast> const& arr) {
        auto buf = arr.request();
        xt::xtensor<double, 1> result = xt::xtensor<double, 1>::from_shape({static_cast<size_t>(buf.shape[0])});
        std::memcpy(result.data(), buf.ptr, buf.size * sizeof(double));
        return xt::sum(result)();
    });

    m.def("diag_xarray_memcpy", [](py::array_t<double, py::array::c_style | py::array::forcecast> const& arr) {
        auto buf = arr.request();
        std::vector<std::size_t> shape(buf.ndim);
        for (size_t i = 0; i < buf.ndim; ++i) {
            shape[i] = static_cast<size_t>(buf.shape[i]);
        }
        xt::xarray<double> result = xt::xarray<double>::from_shape(shape);
        std::memcpy(result.data(), buf.ptr, buf.size * sizeof(double));
        return xt::sum(result)();
    });
}
