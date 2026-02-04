/***************************************************************************
 * Copyright (c) 2025 Peter Urban, Ghent University, Urcoustics             *
 *                                                                          *
 * This code was derived with assistance from ChatGPT-5 Codex, utilizing    *
 * xtensor-python's pyarray.hpp, pycontainer.hpp, and related components.   *
 *                                                                          *
 * Original xtensor-python copyright:                                       *
 * Copyright (c) 2016-2024 Wolf Vollprecht, Johan Mabille,                  *
 *                         Sylvain Corlay, and QuantStack                   *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * This file adapts xtensor-python's pyarray implementation for use with    *
 * nanobind instead of pybind11. The original pyarray.hpp is licensed       *
 * under the BSD 3-Clause License (see below).                              *
 *                                                                          *
 * Note: This code requires a C++17 compiler.                               *
 *                                                                          *
 * Usage: Include this header in your C++ project to utilize the pyarray    *
 *        class for seamless interoperability between C++ and Python using  *
 *        nanobind. The header provides:                                    *
 *        - A pyarray class compatible with xt::xarray for numpy array      *
 *          access as references (for speed and in-place operations).       *
 *          (with namespace this is xt::nanobind::pyarray)                  *
 *        - Type casters for xt::xarray (which always copy/move memory)     *
 *          and pyarray expressions.                                        *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/
/*                                                                          *
 * BSD 3-Clause License (for xtensor-python derived portions)              *
 *                                                                          *
 * Redistribution and use in source and binary forms, with or without       *
 * modification, are permitted provided that the following conditions       *
 * are met:                                                                 *
 *                                                                          *
 * 1. Redistributions of source code must retain the above copyright        *
 *    notice, this list of conditions and the following disclaimer.         *
 *                                                                          *
 * 2. Redistributions in binary form must reproduce the above copyright     *
 *    notice, this list of conditions and the following disclaimer in the   *
 *    documentation and/or other materials provided with the distribution.  *
 *                                                                          *
 * 3. Neither the name of the copyright holder nor the names of its         *
 *    contributors may be used to endorse or promote products derived       *
 *    from this software without specific prior written permission.         *
 *                                                                          *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS      *
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT        *
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR    *
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT     *
 * HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,   *
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED *
 * TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR   *
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF   *
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING     *
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS       *
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.             *
 ****************************************************************************/

#ifndef XTENSOR_PYTHON_NANOBIND_PYARRAY_HPP
#define XTENSOR_PYTHON_NANOBIND_PYARRAY_HPP

#include <algorithm>
#include <cstddef>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "xtensor/containers/xarray.hpp"
#include "xtensor/containers/xbuffer_adaptor.hpp"
#include "xtensor/core/xiterator.hpp"
#include "xtensor/core/xsemantic.hpp"
#include "xtensor/core/xfunction.hpp"
#include "xtensor/utils/xutils.hpp"
#include "xtensor/views/xview.hpp"
#include "xtensor/views/xbroadcast.hpp"
#include "xtensor/views/xindex_view.hpp"
#include "xtensor/reducers/xreducer.hpp"

#include "pycontainer.hpp"
#include "pynative_casters.hpp"
#include "../xtensor_python_config.hpp"

// Forward declarations
namespace xt
{
    namespace nanobind
    {
        template <class T, layout_type L = layout_type::dynamic>
        class pyarray;
    }
}

// xcontainer_inner_types must be defined before pyarray class
namespace xt
{
    template <class T, layout_type L>
    struct xiterable_inner_types<nanobind::pyarray<T, L>>
        : xcontainer_iterable_types<nanobind::pyarray<T, L>>
    {
    };

    template <class T, layout_type L>
    struct xcontainer_inner_types<nanobind::pyarray<T, L>>
    {
        using storage_type = xbuffer_adaptor<T*>;
        using reference = typename storage_type::reference;
        using const_reference = typename storage_type::const_reference;
        using size_type = typename storage_type::size_type;
        using shape_type = std::vector<std::ptrdiff_t>;
        using strides_type = shape_type;
        using backstrides_type = shape_type;
        using inner_shape_type = shape_type;
        using inner_strides_type = strides_type;
        using inner_backstrides_type = backstrides_type;
        using temporary_type = nanobind::pyarray<T, L>;
        static constexpr layout_type layout = L;
    };
}

namespace xt
{
    namespace nanobind
    {
        // Expression tag for nanobind pyarray
        struct pyarray_expression_tag : xt::xtensor_expression_tag
        {
        };

        namespace detail
        {
            // Helper to detect if a type is iterable (has begin/end)
            template <class T, class = void>
            struct is_iterable : std::false_type {};
            
            template <class T>
            struct is_iterable<T, std::void_t<
                decltype(std::begin(std::declval<T&>())),
                decltype(std::end(std::declval<T&>()))
            >> : std::true_type {};
            
            template <class T>
            inline constexpr bool is_iterable_v = is_iterable<T>::value;

            // Helper trait to detect if a type is a valid shape type:
            // - Must be iterable (have begin/end)
            // - Must NOT be the pyarray itself
            // - Must NOT be an xexpression
            template <class S, class PyT>
            struct is_pyarray_shape_type : std::integral_constant<bool,
                is_iterable_v<std::decay_t<S>> &&
                !std::is_same_v<std::decay_t<S>, PyT> &&
                !std::is_base_of_v<xexpression<std::decay_t<S>>, std::decay_t<S>>>
            {
            };

            template <class S, class PyT>
            inline constexpr bool is_pyarray_shape_type_v = is_pyarray_shape_type<S, PyT>::value;

            // ndarray type helper for pyarray (dynamic dimensions)
            // Default (including dynamic) uses row_major (c_contig) since that's NumPy's default
            template <class Scalar, layout_type Layout>
            struct pyarray_ndarray_type_helper
            {
                using type = ::nanobind::ndarray<Scalar, ::nanobind::numpy, ::nanobind::c_contig>;
            };

            template <class Scalar>
            struct pyarray_ndarray_type_helper<Scalar, layout_type::row_major>
            {
                using type = ::nanobind::ndarray<Scalar, ::nanobind::numpy, ::nanobind::c_contig>;
            };

            template <class Scalar>
            struct pyarray_ndarray_type_helper<Scalar, layout_type::column_major>
            {
                using type = ::nanobind::ndarray<Scalar, ::nanobind::numpy, ::nanobind::f_contig>;
            };

            // Stride computation utilities for dynamic-size arrays
            template <class Shape, class Strides>
            inline void compute_dynamic_strides_row_major(const Shape& shape, Strides& strides)
            {
                using size_type = typename Shape::value_type;
                const std::size_t rank = shape.size();
                strides.resize(rank);
                
                if (rank == 0)
                {
                    return;
                }

                strides.back() = size_type(1);
                for (std::ptrdiff_t axis = static_cast<std::ptrdiff_t>(rank) - 2; axis >= 0; --axis)
                {
                    auto next_axis = static_cast<std::size_t>(axis + 1);
                    strides[static_cast<std::size_t>(axis)] = strides[next_axis] * std::max(shape[next_axis], size_type(1));
                }
            }

            template <class Shape, class Strides>
            inline void compute_dynamic_strides_column_major(const Shape& shape, Strides& strides)
            {
                using size_type = typename Shape::value_type;
                const std::size_t rank = shape.size();
                strides.resize(rank);
                
                if (rank == 0)
                {
                    return;
                }

                strides.front() = size_type(1);
                for (std::size_t axis = 1; axis < rank; ++axis)
                {
                    auto previous_axis = axis - 1;
                    strides[axis] = strides[previous_axis] * std::max(shape[previous_axis], size_type(1));
                }
            }

        } // namespace detail

        /**
         * @class pyarray
         * @brief Multidimensional container providing the xtensor container semantics wrapping a numpy array.
         *
         * pyarray is similar to the xarray container in that it has a dynamic dimensionality.
         * Reshapes of a pyarray container are reflected in the underlying numpy array.
         *
         * @tparam T The type of the element stored in the pyarray.
         * @tparam L The layout type (row_major, column_major, or dynamic).
         * @sa pytensor
         */
        template <class T, layout_type L>
        class pyarray : public pycontainer<pyarray<T, L>>,
                        public xcontainer_semantic<pyarray<T, L>>
        {
        public:

            using self_type = pyarray<T, L>;
            using semantic_base = xcontainer_semantic<self_type>;
            using base_type = pycontainer<self_type>;
            using storage_type = typename base_type::storage_type;
            using value_type = typename base_type::value_type;
            using reference = typename base_type::reference;
            using const_reference = typename base_type::const_reference;
            using pointer = typename base_type::pointer;
            using const_pointer = typename base_type::const_pointer;
            using size_type = typename base_type::size_type;
            using difference_type = typename base_type::difference_type;
            using shape_type = typename base_type::shape_type;
            using strides_type = typename base_type::strides_type;
            using backstrides_type = typename base_type::backstrides_type;
            using inner_shape_type = typename base_type::inner_shape_type;
            using inner_strides_type = typename base_type::inner_strides_type;
            using inner_backstrides_type = typename base_type::inner_backstrides_type;

            using expression_tag = pyarray_expression_tag;
            constexpr static std::size_t rank = SIZE_MAX;  // Dynamic rank
            constexpr static layout_type static_layout = L;

            // nanobind-specific types
            using scalar_type = std::remove_const_t<T>;
            using ndarray_scalar_type = std::conditional_t<std::is_const_v<T>, const scalar_type, scalar_type>;
            using ndarray_type = typename detail::pyarray_ndarray_type_helper<ndarray_scalar_type, L>::type;

            // Constructors
            pyarray();
            pyarray(const value_type& t);
            pyarray(nested_initializer_list_t<T, 1> t);
            pyarray(nested_initializer_list_t<T, 2> t);
            pyarray(nested_initializer_list_t<T, 3> t);
            pyarray(nested_initializer_list_t<T, 4> t);
            pyarray(nested_initializer_list_t<T, 5> t);

            template <class S, std::enable_if_t<detail::is_pyarray_shape_type_v<S, self_type>, int> = 0>
            explicit pyarray(S&& shape, layout_type l = L);
            template <class S, std::enable_if_t<detail::is_pyarray_shape_type_v<S, self_type>, int> = 0>
            explicit pyarray(S&& shape, const_reference value, layout_type l = L);
            explicit pyarray(const shape_type& shape, const strides_type& strides, const_reference value);
            explicit pyarray(const shape_type& shape, const strides_type& strides);

            // nanobind-specific constructors
            explicit pyarray(ndarray_type array);

            // xarray container conversion (adopts ownership of data)
            template <class EC, layout_type ArrayLayout, class SC, class Tag,
                      std::enable_if_t<!std::is_const_v<T>, int> = 0>
            pyarray(xt::xarray_container<EC, ArrayLayout, SC, Tag> arr);

            template <class S = shape_type>
            static pyarray from_shape(S&& shape);

            // Copy/move
            pyarray(const self_type& rhs);
            self_type& operator=(const self_type& rhs);

            pyarray(self_type&&) = default;
            self_type& operator=(self_type&& e) = default;

            // Expression conversion
            template <class E>
            pyarray(const xexpression<E>& e);

            template <class E>
            self_type& operator=(const xexpression<E>& e);

            // xarray container assignment (adopts ownership of data)
            template <class EC, layout_type ArrayLayout, class SC, class Tag,
                      std::enable_if_t<!std::is_const_v<T>, int> = 0>
            self_type& operator=(xt::xarray_container<EC, ArrayLayout, SC, Tag> arr);

            using base_type::begin;
            using base_type::end;

            // Static validation methods (pybind11 interface compatibility)
            static self_type ensure(::nanobind::handle h);
            static bool check_(::nanobind::handle h);

            // Use semantic_base operators to avoid ambiguity
            using semantic_base::operator+=;
            using semantic_base::operator-=;
            using semantic_base::operator*=;
            using semantic_base::operator/=;
            using semantic_base::operator|=;
            using semantic_base::operator&=;
            using semantic_base::operator^=;

            // nanobind-specific methods
            bool is_valid() const noexcept;
            ndarray_type& ndarray() noexcept;
            const ndarray_type& ndarray() const noexcept;

            void reset_from_ndarray(ndarray_type array);

        private:

            inner_shape_type m_shape;
            inner_strides_type m_strides;
            inner_backstrides_type m_backstrides;
            storage_type m_storage;
            ndarray_type m_array;

            void init_array(const shape_type& shape, const strides_type& strides);
            void init_from_ndarray();

            // Helper to adopt an xarray_container's storage
            template <class XArray>
            void adopt_xarray_container(std::unique_ptr<XArray> owned_array);

            inner_shape_type& shape_impl() noexcept;
            const inner_shape_type& shape_impl() const noexcept;
            inner_strides_type& strides_impl() noexcept;
            const inner_strides_type& strides_impl() const noexcept;
            inner_backstrides_type& backstrides_impl() noexcept;
            const inner_backstrides_type& backstrides_impl() const noexcept;

            storage_type& storage_impl() noexcept;
            const storage_type& storage_impl() const noexcept;

            layout_type default_dynamic_layout() const;

            friend class xcontainer<pyarray<T, L>>;
            friend class pycontainer<pyarray<T, L>>;
        };

        /**************************
         * pyarray implementation *
         **************************/

        /**
         * @name Constructors
         */
        //@{
        /**
         * Allocates an uninitialized pyarray with a single element (scalar).
         */
        template <class T, layout_type L>
        inline pyarray<T, L>::pyarray()
            : base_type()
        {
            // Create a 0-dimensional array with 1 element (scalar)
            shape_type shape;  // empty shape = 0-dimensional
            strides_type strides;  // empty strides
            init_array(shape, strides);
            // Default initialize the scalar value
            if (m_storage.size() > 0)
            {
                m_storage[0] = value_type();
            }
        }

        /**
         * Allocates a pyarray with a single scalar value.
         */
        template <class T, layout_type L>
        inline pyarray<T, L>::pyarray(const value_type& t)
            : base_type()
        {
            base_type::resize(xt::shape<shape_type>(t), default_dynamic_layout());
            nested_copy(m_storage.begin(), t);
        }

        /**
         * Allocates a pyarray with nested initializer lists.
         */
        template <class T, layout_type L>
        inline pyarray<T, L>::pyarray(nested_initializer_list_t<T, 1> t)
            : base_type()
        {
            base_type::resize(xt::shape<shape_type>(t), default_dynamic_layout());
            L == layout_type::row_major ? nested_copy(m_storage.begin(), t) : nested_copy(this->template begin<layout_type::row_major>(), t);
        }

        template <class T, layout_type L>
        inline pyarray<T, L>::pyarray(nested_initializer_list_t<T, 2> t)
            : base_type()
        {
            base_type::resize(xt::shape<shape_type>(t), default_dynamic_layout());
            L == layout_type::row_major ? nested_copy(m_storage.begin(), t) : nested_copy(this->template begin<layout_type::row_major>(), t);
        }

        template <class T, layout_type L>
        inline pyarray<T, L>::pyarray(nested_initializer_list_t<T, 3> t)
            : base_type()
        {
            base_type::resize(xt::shape<shape_type>(t), default_dynamic_layout());
            L == layout_type::row_major ? nested_copy(m_storage.begin(), t) : nested_copy(this->template begin<layout_type::row_major>(), t);
        }

        template <class T, layout_type L>
        inline pyarray<T, L>::pyarray(nested_initializer_list_t<T, 4> t)
            : base_type()
        {
            base_type::resize(xt::shape<shape_type>(t), default_dynamic_layout());
            L == layout_type::row_major ? nested_copy(m_storage.begin(), t) : nested_copy(this->template begin<layout_type::row_major>(), t);
        }

        template <class T, layout_type L>
        inline pyarray<T, L>::pyarray(nested_initializer_list_t<T, 5> t)
            : base_type()
        {
            base_type::resize(xt::shape<shape_type>(t), default_dynamic_layout());
            L == layout_type::row_major ? nested_copy(m_storage.begin(), t) : nested_copy(this->template begin<layout_type::row_major>(), t);
        }

        /**
         * Allocates an uninitialized pyarray with the specified shape and layout.
         * @param shape the shape of the pyarray
         * @param l the layout_type of the pyarray
         */
        template <class T, layout_type L>
        template <class S, std::enable_if_t<detail::is_pyarray_shape_type_v<S, pyarray<T, L>>, int>>
        inline pyarray<T, L>::pyarray(S&& shape, layout_type l)
            : base_type()
        {
            using std::begin;
            using std::end;
            shape_type converted_shape;
            converted_shape.reserve(std::distance(begin(shape), end(shape)));
            for (auto s : shape)
            {
                converted_shape.push_back(static_cast<typename shape_type::value_type>(s));
            }
            strides_type strides;
            if (l == layout_type::column_major)
            {
                detail::compute_dynamic_strides_column_major(converted_shape, strides);
            }
            else
            {
                detail::compute_dynamic_strides_row_major(converted_shape, strides);
            }
            init_array(converted_shape, strides);
        }

        /**
         * Allocates a pyarray with the specified shape and layout. Elements
         * are initialized to the specified value.
         * @param shape the shape of the pyarray
         * @param value the value of the elements
         * @param l the layout_type of the pyarray
         */
        template <class T, layout_type L>
        template <class S, std::enable_if_t<detail::is_pyarray_shape_type_v<S, pyarray<T, L>>, int>>
        inline pyarray<T, L>::pyarray(S&& shape,
                                      const_reference value,
                                      layout_type l)
            : base_type()
        {
            using std::begin;
            using std::end;
            shape_type converted_shape;
            converted_shape.reserve(std::distance(begin(shape), end(shape)));
            for (auto s : shape)
            {
                converted_shape.push_back(static_cast<typename shape_type::value_type>(s));
            }
            strides_type strides;
            if (l == layout_type::column_major)
            {
                detail::compute_dynamic_strides_column_major(converted_shape, strides);
            }
            else
            {
                detail::compute_dynamic_strides_row_major(converted_shape, strides);
            }
            init_array(converted_shape, strides);
            std::fill(m_storage.begin(), m_storage.end(), value);
        }

        /**
         * Allocates an uninitialized pyarray with the specified shape and strides.
         * Elements are initialized to the specified value.
         * @param shape the shape of the pyarray
         * @param strides the strides of the pyarray
         * @param value the value of the elements
         */
        template <class T, layout_type L>
        inline pyarray<T, L>::pyarray(const shape_type& shape,
                                      const strides_type& strides,
                                      const_reference value)
            : base_type()
        {
            init_array(shape, strides);
            std::fill(m_storage.begin(), m_storage.end(), value);
        }

        /**
         * Allocates an uninitialized pyarray with the specified shape and strides.
         * @param shape the shape of the pyarray
         * @param strides the strides of the pyarray
         */
        template <class T, layout_type L>
        inline pyarray<T, L>::pyarray(const shape_type& shape,
                                      const strides_type& strides)
            : base_type()
        {
            init_array(shape, strides);
        }

        /**
         * Constructs a pyarray from a nanobind ndarray.
         * @param array the nanobind ndarray to wrap
         */
        template <class T, layout_type L>
        inline pyarray<T, L>::pyarray(ndarray_type array)
            : base_type()
        {
            m_array = std::move(array);
            init_from_ndarray();
        }

        /**
         * Constructs a pyarray by adopting an xarray_container's storage.
         * The xarray's data is moved into this pyarray and managed via a capsule.
         * @param arr the xarray_container to adopt
         */
        template <class T, layout_type L>
        template <class EC, layout_type ArrayLayout, class SC, class Tag, std::enable_if_t<!std::is_const_v<T>, int>>
        inline pyarray<T, L>::pyarray(xt::xarray_container<EC, ArrayLayout, SC, Tag> arr)
            : base_type()
        {
            adopt_xarray_container(
                std::make_unique<xt::xarray_container<EC, ArrayLayout, SC, Tag>>(std::move(arr)));
        }

        /**
         * Allocates and returns a pyarray with the specified shape.
         * @param shape the shape of the pyarray
         */
        template <class T, layout_type L>
        template <class S>
        inline pyarray<T, L> pyarray<T, L>::from_shape(S&& shape)
        {
            auto shp = xtl::forward_sequence<shape_type, S>(shape);
            return self_type(shp);
        }
        //@}

        /**
         * @name Copy semantic
         */
        //@{
        /**
         * The copy constructor.
         */
        template <class T, layout_type L>
        inline pyarray<T, L>::pyarray(const self_type& rhs)
            : base_type(), semantic_base(rhs)
        {
            init_array(rhs.shape(), rhs.strides());
            std::copy(rhs.storage().cbegin(), rhs.storage().cend(), this->storage().begin());
        }

        /**
         * The assignment operator.
         */
        template <class T, layout_type L>
        inline auto pyarray<T, L>::operator=(const self_type& rhs) -> self_type&
        {
            self_type tmp(rhs);
            *this = std::move(tmp);
            return *this;
        }
        //@}

        /**
         * @name Extended copy semantic
         */
        //@{
        /**
         * The extended copy constructor.
         */
        template <class T, layout_type L>
        template <class E>
        inline pyarray<T, L>::pyarray(const xexpression<E>& e)
            : base_type()
        {
            shape_type shape = xtl::forward_sequence<shape_type, decltype(e.derived_cast().shape())>(e.derived_cast().shape());
            strides_type strides;
            detail::compute_dynamic_strides_row_major(shape, strides);
            init_array(shape, strides);
            semantic_base::assign(e);
        }

        /**
         * The extended assignment operator.
         */
        template <class T, layout_type L>
        template <class E>
        inline auto pyarray<T, L>::operator=(const xexpression<E>& e) -> self_type&
        {
            return semantic_base::operator=(e);
        }

        /**
         * Assigns an xarray_container by adopting its storage.
         * The xarray's data is moved into this pyarray and managed via a capsule.
         * @param arr the xarray_container to adopt
         * @return reference to this pyarray
         */
        template <class T, layout_type L>
        template <class EC, layout_type ArrayLayout, class SC, class Tag, std::enable_if_t<!std::is_const_v<T>, int>>
        inline auto pyarray<T, L>::operator=(xt::xarray_container<EC, ArrayLayout, SC, Tag> arr) -> self_type&
        {
            adopt_xarray_container(
                std::make_unique<xt::xarray_container<EC, ArrayLayout, SC, Tag>>(std::move(arr)));
            return *this;
        }
        //@}

        /**
         * Attempts to create a pyarray from a Python handle.
         * If the conversion fails, returns an invalid pyarray (is_valid() == false).
         * @param h the Python handle to convert
         * @return a pyarray wrapping the handle, or an invalid pyarray on failure
         */
        template <class T, layout_type L>
        inline auto pyarray<T, L>::ensure(::nanobind::handle h) -> self_type
        {
            if (!h.is_valid())
            {
                return self_type();
            }

            try
            {
                // Try to cast using nanobind's ndarray caster
                ::nanobind::detail::make_caster<ndarray_type> caster;
                if (caster.from_python(h, 0, nullptr))
                {
                    self_type result;
                    result.reset_from_ndarray(std::move(caster.value));
                    return result;
                }
            }
            catch (...)
            {
                // Fall through to return invalid pyarray
            }

            return self_type();
        }

        /**
         * Checks if a Python handle can be converted to this pyarray type.
         * @param h the Python handle to check
         * @return true if the handle can be converted, false otherwise
         */
        template <class T, layout_type L>
        inline bool pyarray<T, L>::check_(::nanobind::handle h)
        {
            if (!h.is_valid())
            {
                return false;
            }

            try
            {
                ::nanobind::detail::make_caster<ndarray_type> caster;
                return caster.from_python(h, 0, nullptr);
            }
            catch (...)
            {
                return false;
            }
        }

        template <class T, layout_type L>
        inline bool pyarray<T, L>::is_valid() const noexcept
        {
            return m_array.is_valid();
        }

        template <class T, layout_type L>
        inline auto pyarray<T, L>::ndarray() noexcept -> ndarray_type&
        {
            return m_array;
        }

        template <class T, layout_type L>
        inline auto pyarray<T, L>::ndarray() const noexcept -> const ndarray_type&
        {
            return m_array;
        }

        template <class T, layout_type L>
        inline void pyarray<T, L>::reset_from_ndarray(ndarray_type array)
        {
            m_array = std::move(array);
            init_from_ndarray();
        }

        template <class T, layout_type L>
        inline void pyarray<T, L>::init_array(const shape_type& shape, const strides_type& strides)
        {
            const std::size_t rank = shape.size();
            
            // Compute total size
            size_type total_size = 1;
            for (std::size_t i = 0; i < rank; ++i)
            {
                total_size *= static_cast<size_type>(shape[i]);
            }

            // Allocate memory and create capsule owner
            scalar_type* raw_ptr = nullptr;
            ::nanobind::object owner;
            if (total_size > 0)
            {
                raw_ptr = new scalar_type[total_size];
                owner = ::nanobind::capsule(
                    raw_ptr,
                    [](void* p) noexcept { delete[] static_cast<scalar_type*>(p); });
            }

            // Create shape array for nanobind
            std::vector<size_t> nb_shape(rank);
            for (std::size_t i = 0; i < rank; ++i)
            {
                nb_shape[i] = static_cast<size_t>(shape[i]);
            }

            // Create strides array for nanobind (in elements, not bytes)
            std::vector<int64_t> nb_strides(rank);
            for (std::size_t i = 0; i < rank; ++i)
            {
                nb_strides[i] = static_cast<int64_t>(strides[i]);
            }

            // Determine order
            char order = 'C';
            if constexpr (L == layout_type::column_major)
            {
                order = 'F';
            }
            else if constexpr (L == layout_type::dynamic)
            {
                // Check actual layout
                if (xt::nanobind::detail::is_column_major(strides, shape))
                {
                    order = 'F';
                }
            }

            // Create nanobind ndarray
            m_array = ndarray_type(
                static_cast<pointer>(raw_ptr),
                rank,
                rank > 0 ? nb_shape.data() : nullptr,
                owner.ptr(),
                rank > 0 ? nb_strides.data() : nullptr,
                ::nanobind::dtype<ndarray_scalar_type>(),
                ::nanobind::device::cpu::value,
                0,
                order);

            // Initialize member variables
            m_shape = shape;
            m_strides = strides;
            m_backstrides.resize(rank);
            adapt_strides(m_shape, m_strides, m_backstrides);
            m_storage = storage_type(raw_ptr, total_size);
        }

        template <class T, layout_type L>
        inline void pyarray<T, L>::init_from_ndarray()
        {
            if (!m_array.is_valid())
            {
                m_shape.clear();
                m_strides.clear();
                m_backstrides.clear();
                m_storage = storage_type(nullptr, 0);
                return;
            }

            const std::size_t rank = m_array.ndim();

            // Copy shape
            m_shape.resize(rank);
            for (std::size_t i = 0; i < rank; ++i)
            {
                m_shape[i] = static_cast<typename shape_type::value_type>(m_array.shape(i));
            }

            // Copy strides (nanobind strides are in elements)
            m_strides.resize(rank);
            const int64_t* stride_ptr = m_array.stride_ptr();
            if (stride_ptr != nullptr)
            {
                for (std::size_t i = 0; i < rank; ++i)
                {
                    m_strides[i] = static_cast<typename strides_type::value_type>(stride_ptr[i]);
                }
            }
            else
            {
                // Compute default strides
                if constexpr (L == layout_type::column_major)
                {
                    detail::compute_dynamic_strides_column_major(m_shape, m_strides);
                }
                else
                {
                    detail::compute_dynamic_strides_row_major(m_shape, m_strides);
                }
            }

            // Validate layout if not dynamic
            if constexpr (L != layout_type::dynamic)
            {
                bool layout_ok = false;
                if constexpr (L == layout_type::row_major)
                {
                    layout_ok = xt::nanobind::detail::is_row_major(m_strides, m_shape);
                }
                else
                {
                    layout_ok = xt::nanobind::detail::is_column_major(m_strides, m_shape);
                }
                if (!layout_ok)
                {
                    throw std::runtime_error("NumPy: passing container with bad strides for layout (is it a view?).");
                }
            }

            // Compute backstrides
            m_backstrides.resize(rank);
            adapt_strides(m_shape, m_strides, m_backstrides);

            // Compute total size
            size_type total_size = 1;
            for (std::size_t i = 0; i < rank; ++i)
            {
                total_size *= static_cast<size_type>(m_shape[i]);
            }
            m_storage = storage_type(m_array.data(), total_size);
        }

        template <class T, layout_type L>
        template <class XArray>
        inline void pyarray<T, L>::adopt_xarray_container(std::unique_ptr<XArray> owned_array)
        {
            const std::size_t rank = owned_array->dimension();
            
            // Get shape and strides
            shape_type shape(rank);
            strides_type strides(rank);
            for (std::size_t i = 0; i < rank; ++i)
            {
                shape[i] = static_cast<typename shape_type::value_type>(owned_array->shape()[i]);
                strides[i] = static_cast<typename strides_type::value_type>(owned_array->strides()[i]);
            }

            // Compute total size
            size_type total_size = 1;
            for (std::size_t i = 0; i < rank; ++i)
            {
                total_size *= static_cast<size_type>(shape[i]);
            }

            // Get data pointer
            scalar_type* data_ptr = owned_array->data();

            // Create shape/strides arrays for nanobind
            std::vector<size_t> nb_shape(rank);
            std::vector<int64_t> nb_strides(rank);
            for (std::size_t i = 0; i < rank; ++i)
            {
                nb_shape[i] = static_cast<size_t>(shape[i]);
                nb_strides[i] = static_cast<int64_t>(strides[i]);
            }

            // Determine order
            char order = 'C';
            if (xt::nanobind::detail::is_column_major(strides, shape))
            {
                order = 'F';
            }

            // Create capsule that owns the xarray
            XArray* raw_ptr = owned_array.release();
            ::nanobind::object owner = ::nanobind::capsule(
                raw_ptr,
                [](void* p) noexcept { delete static_cast<XArray*>(p); });

            // Create nanobind ndarray
            m_array = ndarray_type(
                data_ptr,
                rank,
                rank > 0 ? nb_shape.data() : nullptr,
                owner.ptr(),
                rank > 0 ? nb_strides.data() : nullptr,
                ::nanobind::dtype<ndarray_scalar_type>(),
                ::nanobind::device::cpu::value,
                0,
                order);

            // Initialize member variables
            m_shape = std::move(shape);
            m_strides = std::move(strides);
            m_backstrides.resize(rank);
            adapt_strides(m_shape, m_strides, m_backstrides);
            m_storage = storage_type(data_ptr, total_size);
        }

        template <class T, layout_type L>
        inline auto pyarray<T, L>::shape_impl() noexcept -> inner_shape_type&
        {
            return m_shape;
        }

        template <class T, layout_type L>
        inline auto pyarray<T, L>::shape_impl() const noexcept -> const inner_shape_type&
        {
            return m_shape;
        }

        template <class T, layout_type L>
        inline auto pyarray<T, L>::strides_impl() noexcept -> inner_strides_type&
        {
            return m_strides;
        }

        template <class T, layout_type L>
        inline auto pyarray<T, L>::strides_impl() const noexcept -> const inner_strides_type&
        {
            return m_strides;
        }

        template <class T, layout_type L>
        inline auto pyarray<T, L>::backstrides_impl() noexcept -> inner_backstrides_type&
        {
            return m_backstrides;
        }

        template <class T, layout_type L>
        inline auto pyarray<T, L>::backstrides_impl() const noexcept -> const inner_backstrides_type&
        {
            return m_backstrides;
        }

        template <class T, layout_type L>
        inline auto pyarray<T, L>::storage_impl() noexcept -> storage_type&
        {
            return m_storage;
        }

        template <class T, layout_type L>
        inline auto pyarray<T, L>::storage_impl() const noexcept -> const storage_type&
        {
            return m_storage;
        }

        template <class T, layout_type L>
        inline layout_type pyarray<T, L>::default_dynamic_layout() const
        {
            return L == layout_type::dynamic ? layout_type::row_major : L;
        }

    } // namespace nanobind

    // Specialization of expression assigner base for nanobind pyarray
    template <>
    class xexpression_assigner_base<nanobind::pyarray_expression_tag>
        : public xexpression_assigner_base<xtensor_expression_tag>
    {
    };

    namespace detail
    {
        template <class F, class... E>
        struct select_xfunction_expression<nanobind::pyarray_expression_tag, F, E...>
        {
            using type = xfunction<F, E...>;
        };
    }

    template <class From, class T, layout_type L>
    struct has_assign_conversion<From, nanobind::pyarray<T, L>> : std::false_type
    {
    };

    namespace extension
    {
        // Expression base for pyarray expressions
        struct nanobind_pyarray_expression_base
        {
            using expression_tag = nanobind::pyarray_expression_tag;
        };

        template <class F, class... CT>
        struct xfunction_base_impl<nanobind::pyarray_expression_tag, F, CT...>
        {
            using type = nanobind_pyarray_expression_base;
        };

        template <class CT, class... S>
        struct xview_base_impl<nanobind::pyarray_expression_tag, CT, S...>
        {
            using type = nanobind_pyarray_expression_base;
        };

        template <class F, class CT, class X, class O>
        struct xreducer_base_impl<nanobind::pyarray_expression_tag, F, CT, X, O>
        {
            using type = nanobind_pyarray_expression_base;
        };

        template <class CT, class I>
        struct xindex_view_base_impl<nanobind::pyarray_expression_tag, CT, I>
        {
            using type = nanobind_pyarray_expression_base;
        };

        template <class CT, class X>
        struct xbroadcast_base_impl<nanobind::pyarray_expression_tag, CT, X>
        {
            using type = nanobind_pyarray_expression_base;
        };

        template <class CT, class S, layout_type L, class FST>
        struct xstrided_view_base_impl<nanobind::pyarray_expression_tag, CT, S, L, FST>
        {
            using type = nanobind_pyarray_expression_base;
        };
    }

} // namespace xt

// Type casters for nanobind
NAMESPACE_BEGIN(NB_NAMESPACE)
NAMESPACE_BEGIN(detail)

    // Type caster for xt::nanobind::pyarray
    template <typename T, xt::layout_type L>
    struct type_caster<xt::nanobind::pyarray<T, L>>
    {
        using Type = xt::nanobind::pyarray<T, L>;
        using ndarray_type = typename Type::ndarray_type;

        NB_TYPE_CASTER(Type, const_name("numpy.ndarray"))

        bool from_python(handle src, uint8_t flags, cleanup_list* cleanup) noexcept
        {
            try
            {
                make_caster<ndarray_type> caster;
                if (!caster.from_python(src, flags, cleanup))
                {
                    return false;
                }
                value.reset_from_ndarray(std::move(caster.value));
                return true;
            }
            catch (...)
            {
                return false;
            }
        }

        static handle from_cpp(const Type& src, rv_policy policy, cleanup_list* cleanup) noexcept
        {
            return from_cpp_impl(&src, policy, cleanup);
        }

        static handle from_cpp(Type& src, rv_policy policy, cleanup_list* cleanup) noexcept
        {
            return from_cpp_impl(&src, policy, cleanup);
        }

        static handle from_cpp(Type&& src, rv_policy policy, cleanup_list* cleanup) noexcept
        {
            // For rvalue, copy the data
            if (policy == rv_policy::automatic || policy == rv_policy::automatic_reference)
            {
                policy = rv_policy::copy;
            }
            return from_cpp_impl(&src, policy, cleanup);
        }

    private:
        template <typename CType>
        static handle from_cpp_impl(CType* src, rv_policy policy, cleanup_list* cleanup) noexcept
        {
            if (policy == rv_policy::automatic || policy == rv_policy::automatic_reference)
            {
                policy = rv_policy::reference;
            }

            // For pyarray, we can return a reference to the internal ndarray
            if (!src->is_valid())
            {
                return handle();
            }

            return make_caster<ndarray_type>::from_cpp(src->ndarray(), policy, cleanup);
        }
    };

    // Type caster for xexpression<pyarray>
    template <typename T, xt::layout_type L>
    struct type_caster<xt::xexpression<xt::nanobind::pyarray<T, L>>>
        : type_caster<xt::nanobind::pyarray<T, L>>
    {
        using Type = xt::xexpression<xt::nanobind::pyarray<T, L>>;

        operator Type&()
        {
            return this->value;
        }

        operator const Type&()
        {
            return this->value;
        }
    };

NAMESPACE_END(detail)
NAMESPACE_END(NB_NAMESPACE)

#endif
