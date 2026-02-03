/***************************************************************************
 * Copyright (c) Wolf Vollprecht, Johan Mabille and Sylvain Corlay          *
 * Copyright (c) QuantStack                                                 *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

// Backwards compatibility header - includes pybind11 version by default
// and brings pyarray into xt namespace
#ifndef XTENSOR_PYTHON_PYARRAY_HPP
#define XTENSOR_PYTHON_PYARRAY_HPP

#include "xtensor-python/pybind11/pyarray.hpp"

namespace xt
{
    using pybind11::pyarray;
    using pybind11::import_numpy;
}

#endif
