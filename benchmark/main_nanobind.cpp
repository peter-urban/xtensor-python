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
#include <memory>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/complex.h>

namespace nb = nanobind;
using complex_t = std::complex<double>;

NB_MODULE(benchmark_xtensor_nanobind, m)
{
    m.doc() = "Benchmark module for nanobind bindings (baseline comparison)";

    m.def("nanobind_sum_array", [](nb::ndarray<double, nb::ndim<1>, nb::c_contig> const& x) {
        double sum = 0;
        size_t size = x.size();
        const double* data = x.data();
        for (size_t i = 0; i < size; ++i)
            sum += data[i];
        return sum;
    });

    m.def("nanobind_rect_to_polar", [](nb::ndarray<nb::numpy, complex_t, nb::ndim<1>> const& a) {
        size_t n = a.shape(0);
        auto result = std::make_unique<double[]>(n);
        const complex_t* data = a.data();
        for (size_t i = 0; i < n; ++i) {
            result[i] = std::abs(data[i]);
        }
        double* raw = result.release();
        nb::capsule deleter(raw, [](void* p) noexcept { delete[] static_cast<double*>(p); });
        return nb::ndarray<nb::numpy, double, nb::ndim<1>>(raw, {n}, deleter);
    });
}
