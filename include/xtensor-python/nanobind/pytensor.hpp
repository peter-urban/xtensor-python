/***************************************************************************
 * Copyright (c) 2025 Peter Urban, Ghent University, Urcoustics             *
 *                                                                          *
 * This code was derived with assistance from ChatGPT-5 Codex, utilizing    *
 * xtensor-python's pytensor.hpp, pycontainer.hpp, and related components.  *
 *                                                                          *
 * Original xtensor-python copyright:                                       *
 * Copyright (c) 2016-2024 Wolf Vollprecht, Johan Mabille,                  *
 *                         Sylvain Corlay, and QuantStack                   *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * This file adapts xtensor-python's pytensor implementation for use with   *
 * nanobind instead of pybind11. The original pytensor.hpp is licensed      *
 * under the BSD 3-Clause License (see below).                              *
 *                                                                          *
 * Note: This code requires a C++17 compiler.                               *
 *                                                                          *
 * Usage: Include this header in your C++ project to utilize the pytensor   *
 *        class for seamless interoperability between C++ and Python using  *
 *        nanobind. The header provides:                                    *
 *        - A pytensor class compatible with xt::xtensor for numpy array    *
 *          access as references (for speed and in-place operations).       *
 *          (with namespace this is xt::nanobind::pytensor)                 *
 *        - Type casters for xt::xtensor (which always copy/move memory)    *
 *          and pytensor expressions.                                       *
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

#ifndef XTENSOR_PYTHON_NANOBIND_PYTENSOR_HPP
#define XTENSOR_PYTHON_NANOBIND_PYTENSOR_HPP

#include <algorithm>
#include <array>
#include <cstddef>
#include <initializer_list>
#include <memory>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "xtensor/containers/xbuffer_adaptor.hpp"
#include "xtensor/containers/xtensor.hpp"
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
        template <class T, std::size_t N, layout_type L = layout_type::dynamic>
        class pytensor;
    }
}

// xcontainer_inner_types must be defined before pytensor class
namespace xt
{
    template <class T, std::size_t N, layout_type L>
    struct xiterable_inner_types<nanobind::pytensor<T, N, L>>
        : xcontainer_iterable_types<nanobind::pytensor<T, N, L>>
    {
    };

    template <class T, std::size_t N, layout_type L>
    struct xcontainer_inner_types<nanobind::pytensor<T, N, L>>
    {
        using storage_type = xbuffer_adaptor<T*>;
        using reference = typename storage_type::reference;
        using const_reference = typename storage_type::const_reference;
        using size_type = typename storage_type::size_type;
        using shape_type = std::array<std::ptrdiff_t, N>;
        using strides_type = shape_type;
        using backstrides_type = shape_type;
        using inner_shape_type = shape_type;
        using inner_strides_type = strides_type;
        using inner_backstrides_type = backstrides_type;
        using temporary_type = nanobind::pytensor<T, N, L>;
        static constexpr layout_type layout = L;
    };
}

namespace xt
{
    namespace nanobind
    {
        // Expression tag for nanobind pytensor
        struct pytensor_expression_tag : xt::xtensor_expression_tag
        {
        };

        // Forward declaration
        template <class T, std::size_t N, layout_type L>
        class pytensor;

        namespace detail
        {
            // Helper trait to detect if a type is NOT a pytensor or xexpression
            template <class S, class PyT>
            struct is_shape_type : std::integral_constant<bool,
                !std::is_same_v<std::decay_t<S>, PyT> &&
                !std::is_base_of_v<xexpression<std::decay_t<S>>, std::decay_t<S>>>
            {
            };

            template <class S, class PyT>
            inline constexpr bool is_shape_type_v = is_shape_type<S, PyT>::value;

            // ndarray type helper for different layouts
            // Default (including dynamic) uses row_major (c_contig) since that's NumPy's default
            template <class Scalar, std::size_t N, layout_type Layout>
            struct ndarray_type_helper
            {
                using type = ::nanobind::ndarray<Scalar, ::nanobind::ndim<N>, ::nanobind::numpy, ::nanobind::c_contig>;
            };

            template <class Scalar, std::size_t N>
            struct ndarray_type_helper<Scalar, N, layout_type::row_major>
            {
                using type = ::nanobind::ndarray<Scalar, ::nanobind::ndim<N>, ::nanobind::numpy, ::nanobind::c_contig>;
            };

            template <class Scalar, std::size_t N>
            struct ndarray_type_helper<Scalar, N, layout_type::column_major>
            {
                using type = ::nanobind::ndarray<Scalar, ::nanobind::ndim<N>, ::nanobind::numpy, ::nanobind::f_contig>;
            };

            // Stride computation utilities
            template <class Shape, class Strides>
            inline void compute_strides_row_major(const Shape& shape, Strides& strides)
            {
                using size_type = typename Shape::value_type;
                constexpr std::size_t rank = std::tuple_size_v<Shape>;
                if constexpr (rank == 0)
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
            inline void compute_strides_column_major(const Shape& shape, Strides& strides)
            {
                using size_type = typename Shape::value_type;
                constexpr std::size_t rank = std::tuple_size_v<Shape>;
                if constexpr (rank == 0)
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
         * @class pytensor
         * @brief Multidimensional container providing the xtensor container semantics wrapping a numpy array.
         *
         * pytensor is similar to the xtensor container in that it has a static dimensionality.
         *
         * Unlike the pyarray container, pytensor cannot be reshaped with a different number of dimensions
         * and reshapes are not reflected on the Python side. However, pytensor has benefits compared to pyarray
         * in terms of performances. pytensor shapes are stack-allocated which makes iteration upon pytensor
         * faster than with pyarray.
         *
         * @tparam T The type of the element stored in the pytensor.
         * @tparam N The number of dimensions.
         * @tparam L The layout type (row_major, column_major, or dynamic).
         * @sa pyarray
         */
        template <class T, std::size_t N, layout_type L>
        class pytensor : public pycontainer<pytensor<T, N, L>>,
                         public xcontainer_semantic<pytensor<T, N, L>>
        {
        public:

            using self_type = pytensor<T, N, L>;
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

            using expression_tag = pytensor_expression_tag;
            constexpr static std::size_t rank = N;
            constexpr static layout_type static_layout = L;

            // nanobind-specific types
            using scalar_type = std::remove_const_t<T>;
            using ndarray_scalar_type = std::conditional_t<std::is_const_v<T>, const scalar_type, scalar_type>;
            using ndarray_type = typename detail::ndarray_type_helper<ndarray_scalar_type, N, L>::type;

            // Constructors
            pytensor();
            pytensor(nested_initializer_list_t<T, N> t);

            template <class S, std::enable_if_t<detail::is_shape_type_v<S, self_type>, int> = 0>
            explicit pytensor(S&& shape, layout_type l = L);
            template <class S, std::enable_if_t<detail::is_shape_type_v<S, self_type>, int> = 0>
            explicit pytensor(S&& shape, const_reference value, layout_type l = L);
            explicit pytensor(const shape_type& shape, const strides_type& strides, const_reference value);
            explicit pytensor(const shape_type& shape, const strides_type& strides);

            // nanobind-specific constructors
            explicit pytensor(ndarray_type array);

            // xtensor container conversion (adopts ownership of data)
            template <class EC, layout_type TensorLayout, class Tag,
                      std::enable_if_t<!std::is_const_v<T>, int> = 0>
            pytensor(xt::xtensor_container<EC, N, TensorLayout, Tag> tensor);

            template <class S = shape_type>
            static pytensor from_shape(S&& shape);

            // Overload for initializer_list<size_t> to avoid narrowing warnings
            static pytensor from_shape(std::initializer_list<std::size_t> shape);

            // Copy/move
            pytensor(const self_type& rhs);
            self_type& operator=(const self_type& rhs);

            pytensor(self_type&&) = default;
            self_type& operator=(self_type&& e) = default;

            // Expression conversion
            template <class E>
            pytensor(const xexpression<E>& e);

            template <class E>
            self_type& operator=(const xexpression<E>& e);

            // xtensor container assignment (adopts ownership of data)
            template <class EC, layout_type TensorLayout, class Tag,
                      std::enable_if_t<!std::is_const_v<T>, int> = 0>
            self_type& operator=(xt::xtensor_container<EC, N, TensorLayout, Tag> tensor);

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

            void init_tensor(const shape_type& shape, const strides_type& strides);
            void init_from_ndarray();

            // Helper to adopt an xtensor_container's storage
            template <class XTensor>
            void adopt_xtensor_container(std::unique_ptr<XTensor> owned_tensor);

            // Zero-copy reshape helper: updates shape/strides metadata in-place
            void reshape_impl(const shape_type& shape, const strides_type& strides);

            inner_shape_type& shape_impl() noexcept;
            const inner_shape_type& shape_impl() const noexcept;
            inner_strides_type& strides_impl() noexcept;
            const inner_strides_type& strides_impl() const noexcept;
            inner_backstrides_type& backstrides_impl() noexcept;
            const inner_backstrides_type& backstrides_impl() const noexcept;

            storage_type& storage_impl() noexcept;
            const storage_type& storage_impl() const noexcept;

            friend class xcontainer<pytensor<T, N, L>>;
            friend class pycontainer<pytensor<T, N, L>>;
        };

        /***************************
         * pytensor implementation *
         ***************************/

        /**
         * @name Constructors
         */
        //@{
        /**
         * Allocates an uninitialized pytensor that holds 0 elements.
         */
        template <class T, std::size_t N, layout_type L>
        inline pytensor<T, N, L>::pytensor()
            : base_type()
        {
            m_shape = xtl::make_sequence<shape_type>(N, size_type(0));
            m_strides = xtl::make_sequence<strides_type>(N, size_type(0));
            m_backstrides = xtl::make_sequence<backstrides_type>(N, size_type(0));
            m_storage = storage_type(nullptr, size_type(0));
        }

        /**
         * Allocates a pytensor with a nested initializer list.
         */
        template <class T, std::size_t N, layout_type L>
        inline pytensor<T, N, L>::pytensor(nested_initializer_list_t<T, N> t)
            : base_type()
        {
            base_type::resize(xt::shape<shape_type>(t), layout_type::row_major);
            nested_copy(m_storage.begin(), t);
        }

        /**
         * Allocates an uninitialized pytensor with the specified shape and layout.
         * @param shape the shape of the pytensor
         * @param l the layout_type of the pytensor
         */
        template <class T, std::size_t N, layout_type L>
        template <class S, std::enable_if_t<detail::is_shape_type_v<S, pytensor<T, N, L>>, int>>
        inline pytensor<T, N, L>::pytensor(S&& shape, layout_type l)
            : base_type()
        {
            shape_type converted_shape;
            auto it = std::begin(shape);
            for (std::size_t i = 0; i < N; ++i, ++it)
            {
                converted_shape[i] = static_cast<typename shape_type::value_type>(*it);
            }
            strides_type strides;
            if (l == layout_type::column_major)
            {
                detail::compute_strides_column_major(converted_shape, strides);
            }
            else
            {
                detail::compute_strides_row_major(converted_shape, strides);
            }
            init_tensor(converted_shape, strides);
        }

        /**
         * Allocates a pytensor with the specified shape and layout. Elements
         * are initialized to the specified value.
         * @param shape the shape of the pytensor
         * @param value the value of the elements
         * @param l the layout_type of the pytensor
         */
        template <class T, std::size_t N, layout_type L>
        template <class S, std::enable_if_t<detail::is_shape_type_v<S, pytensor<T, N, L>>, int>>
        inline pytensor<T, N, L>::pytensor(S&& shape,
                                           const_reference value,
                                           layout_type l)
            : base_type()
        {
            shape_type converted_shape;
            auto it = std::begin(shape);
            for (std::size_t i = 0; i < N; ++i, ++it)
            {
                converted_shape[i] = static_cast<typename shape_type::value_type>(*it);
            }
            strides_type strides;
            if (l == layout_type::column_major)
            {
                detail::compute_strides_column_major(converted_shape, strides);
            }
            else
            {
                detail::compute_strides_row_major(converted_shape, strides);
            }
            init_tensor(converted_shape, strides);
            std::fill(m_storage.begin(), m_storage.end(), value);
        }

        /**
         * Allocates an uninitialized pytensor with the specified shape and strides.
         * Elements are initialized to the specified value.
         * @param shape the shape of the pytensor
         * @param strides the strides of the pytensor
         * @param value the value of the elements
         */
        template <class T, std::size_t N, layout_type L>
        inline pytensor<T, N, L>::pytensor(const shape_type& shape,
                                           const strides_type& strides,
                                           const_reference value)
            : base_type()
        {
            init_tensor(shape, strides);
            std::fill(m_storage.begin(), m_storage.end(), value);
        }

        /**
         * Allocates an uninitialized pytensor with the specified shape and strides.
         * @param shape the shape of the pytensor
         * @param strides the strides of the pytensor
         */
        template <class T, std::size_t N, layout_type L>
        inline pytensor<T, N, L>::pytensor(const shape_type& shape,
                                           const strides_type& strides)
            : base_type()
        {
            init_tensor(shape, strides);
        }

        /**
         * Constructs a pytensor from a nanobind ndarray.
         * @param array the nanobind ndarray to wrap
         */
        template <class T, std::size_t N, layout_type L>
        inline pytensor<T, N, L>::pytensor(ndarray_type array)
            : base_type()
        {
            m_array = std::move(array);
            init_from_ndarray();
        }

        /**
         * Constructs a pytensor by adopting an xtensor_container's storage.
         * The xtensor's data is moved into this pytensor and managed via a capsule.
         * @param tensor the xtensor_container to adopt
         */
        template <class T, std::size_t N, layout_type L>
        template <class EC, layout_type TensorLayout, class Tag, std::enable_if_t<!std::is_const_v<T>, int>>
        inline pytensor<T, N, L>::pytensor(xt::xtensor_container<EC, N, TensorLayout, Tag> tensor)
            : base_type()
        {
            adopt_xtensor_container(
                std::make_unique<xt::xtensor_container<EC, N, TensorLayout, Tag>>(std::move(tensor)));
        }

        /**
         * Allocates and returns a pytensor with the specified shape.
         * @param shape the shape of the pytensor
         */
        template <class T, std::size_t N, layout_type L>
        template <class S>
        inline pytensor<T, N, L> pytensor<T, N, L>::from_shape(S&& shape)
        {
            detail::check_dims<shape_type>::run(shape.size());
            auto shp = xtl::forward_sequence<shape_type, S>(shape);
            return self_type(shp);
        }

        /**
         * Allocates and returns a pytensor with the specified shape.
         * Overload for initializer_list<size_t> to avoid narrowing warnings.
         * @param shape the shape of the pytensor
         */
        template <class T, std::size_t N, layout_type L>
        inline pytensor<T, N, L> pytensor<T, N, L>::from_shape(std::initializer_list<std::size_t> shape)
        {
            detail::check_dims<shape_type>::run(shape.size());
            shape_type shp;
            std::transform(shape.begin(), shape.end(), shp.begin(),
                [](std::size_t v) { return static_cast<std::ptrdiff_t>(v); });
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
        template <class T, std::size_t N, layout_type L>
        inline pytensor<T, N, L>::pytensor(const self_type& rhs)
            : base_type(), semantic_base(rhs)
        {
            init_tensor(rhs.shape(), rhs.strides());
            std::copy(rhs.storage().cbegin(), rhs.storage().cend(), this->storage().begin());
        }

        /**
         * The assignment operator.
         */
        template <class T, std::size_t N, layout_type L>
        inline auto pytensor<T, N, L>::operator=(const self_type& rhs) -> self_type&
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
        template <class T, std::size_t N, layout_type L>
        template <class E>
        inline pytensor<T, N, L>::pytensor(const xexpression<E>& e)
            : base_type()
        {
            shape_type shape = xtl::forward_sequence<shape_type, decltype(e.derived_cast().shape())>(e.derived_cast().shape());
            strides_type strides;
            detail::compute_strides_row_major(shape, strides);
            init_tensor(shape, strides);
            semantic_base::assign(e);
        }

        /**
         * The extended assignment operator.
         */
        template <class T, std::size_t N, layout_type L>
        template <class E>
        inline auto pytensor<T, N, L>::operator=(const xexpression<E>& e) -> self_type&
        {
            return semantic_base::operator=(e);
        }

        /**
         * Assigns an xtensor_container by adopting its storage.
         * The xtensor's data is moved into this pytensor and managed via a capsule.
         * @param tensor the xtensor_container to adopt
         * @return reference to this pytensor
         */
        template <class T, std::size_t N, layout_type L>
        template <class EC, layout_type TensorLayout, class Tag, std::enable_if_t<!std::is_const_v<T>, int>>
        inline auto pytensor<T, N, L>::operator=(xt::xtensor_container<EC, N, TensorLayout, Tag> tensor) -> self_type&
        {
            adopt_xtensor_container(
                std::make_unique<xt::xtensor_container<EC, N, TensorLayout, Tag>>(std::move(tensor)));
            return *this;
        }
        //@}

        /**
         * Attempts to create a pytensor from a Python handle.
         * If the conversion fails, returns an invalid pytensor (is_valid() == false).
         * @param h the Python handle to convert
         * @return a pytensor wrapping the handle, or an invalid pytensor on failure
         */
        template <class T, std::size_t N, layout_type L>
        inline auto pytensor<T, N, L>::ensure(::nanobind::handle h) -> self_type
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
                // Fall through to return invalid pytensor
            }

            return self_type();
        }

        /**
         * Checks if a Python handle can be converted to this pytensor type.
         * @param h the Python handle to check
         * @return true if the handle can be converted, false otherwise
         */
        template <class T, std::size_t N, layout_type L>
        inline bool pytensor<T, N, L>::check_(::nanobind::handle h)
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

        template <class T, std::size_t N, layout_type L>
        inline bool pytensor<T, N, L>::is_valid() const noexcept
        {
            return m_array.is_valid();
        }

        template <class T, std::size_t N, layout_type L>
        inline auto pytensor<T, N, L>::ndarray() noexcept -> ndarray_type&
        {
            return m_array;
        }

        template <class T, std::size_t N, layout_type L>
        inline auto pytensor<T, N, L>::ndarray() const noexcept -> const ndarray_type&
        {
            return m_array;
        }

        template <class T, std::size_t N, layout_type L>
        inline void pytensor<T, N, L>::reset_from_ndarray(ndarray_type array)
        {
            m_array = std::move(array);
            init_from_ndarray();
        }

        template <class T, std::size_t N, layout_type L>
        inline void pytensor<T, N, L>::init_tensor(const shape_type& shape, const strides_type& strides)
        {
            // Compute total size
            size_type total_size = 1;
            for (std::size_t i = 0; i < N; ++i)
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
            std::array<size_t, N> nb_shape{};
            for (std::size_t i = 0; i < N; ++i)
            {
                nb_shape[i] = static_cast<size_t>(shape[i]);
            }

            // Create strides array for nanobind (in elements, not bytes)
            std::array<int64_t, N> nb_strides{};
            for (std::size_t i = 0; i < N; ++i)
            {
                nb_strides[i] = static_cast<int64_t>(strides[i]);
            }

            // Determine order
            char order = 'C';
            if constexpr (L == layout_type::column_major)
            {
                order = 'F';
            }

            // Create nanobind ndarray
            m_array = ndarray_type(
                static_cast<pointer>(raw_ptr),
                N,
                N > 0 ? nb_shape.data() : nullptr,
                owner.ptr(),
                N > 0 ? nb_strides.data() : nullptr,
                ::nanobind::dtype<ndarray_scalar_type>(),
                ::nanobind::device::cpu::value,
                0,
                order);

            // Initialize member variables
            m_shape = shape;
            m_strides = strides;
            adapt_strides(m_shape, m_strides, m_backstrides);
            m_storage = storage_type(raw_ptr, total_size);
        }

        template <class T, std::size_t N, layout_type L>
        inline void pytensor<T, N, L>::init_from_ndarray()
        {
            if (!m_array.is_valid())
            {
                m_shape.fill(0);
                m_strides.fill(0);
                m_backstrides.fill(0);
                m_storage = storage_type(nullptr, 0);
                return;
            }

            if (m_array.ndim() != N)
            {
                throw std::runtime_error("NumPy ndarray has incorrect number of dimensions");
            }

            // Copy shape
            for (std::size_t i = 0; i < N; ++i)
            {
                m_shape[i] = static_cast<typename shape_type::value_type>(m_array.shape(i));
            }

            // Copy strides (nanobind strides are in elements)
            const int64_t* stride_ptr = m_array.stride_ptr();
            if (stride_ptr != nullptr)
            {
                for (std::size_t i = 0; i < N; ++i)
                {
                    m_strides[i] = static_cast<typename strides_type::value_type>(stride_ptr[i]);
                }
            }
            else
            {
                // Compute default strides
                if constexpr (L == layout_type::column_major)
                {
                    detail::compute_strides_column_major(m_shape, m_strides);
                }
                else
                {
                    detail::compute_strides_row_major(m_shape, m_strides);
                }
            }

            // Validate layout if not dynamic
            if constexpr (L != layout_type::dynamic)
            {
                bool layout_ok = false;
                if constexpr (L == layout_type::row_major)
                {
                    layout_ok = detail::is_row_major(m_strides, m_shape);
                }
                else
                {
                    layout_ok = detail::is_column_major(m_strides, m_shape);
                }
                if (!layout_ok)
                {
                    throw std::runtime_error("NumPy: passing container with bad strides for layout (is it a view?).");
                }
            }

            adapt_strides(m_shape, m_strides, m_backstrides);

            // Compute buffer size directly from shape to avoid circular dependency
            // (this->size() may call storage().size() for contiguous layouts, but storage isn't set yet)
            size_type total_size = 1;
            for (std::size_t i = 0; i < N; ++i)
            {
                total_size *= static_cast<size_type>(m_shape[i]);
            }
            m_storage = storage_type(m_array.data(), total_size);
        }

        /**
         * Zero-copy reshape implementation: updates shape/strides metadata in-place.
         * This does not copy data; it only reinterprets the existing buffer with new metadata.
         * 
         * Note: The caller (pycontainer::reshape) is responsible for verifying that the
         * new shape has the same total number of elements as the current shape.
         * 
         * @param shape the new shape
         * @param strides the new strides
         */
        template <class T, std::size_t N, layout_type L>
        inline void pytensor<T, N, L>::reshape_impl(const shape_type& shape, const strides_type& strides)
        {
            // Update shape and strides metadata
            m_shape = shape;
            m_strides = strides;
            // adapt_strides updates m_backstrides based on the new shape and strides
            // This is necessary for correct reverse iteration behavior in xtensor
            adapt_strides(m_shape, m_strides, m_backstrides);
            
            // The storage size doesn't change since total elements are the same
            // (verified by the caller in pycontainer::reshape).
            // m_storage already points to the correct data with the correct total size
            // We don't need to reallocate or recreate the storage adaptor
            
            // Note: We don't update m_array here because:
            // 1. The pytensor's local metadata (m_shape, m_strides) is what xtensor uses for iteration
            // 2. The underlying data pointer in m_storage is unchanged
            // 3. The m_array is mainly used for ownership and Python interop, not for xtensor ops
        }

        /**
         * Adopts an xtensor_container's storage into this pytensor.
         * The xtensor is moved into a capsule for memory management.
         * @tparam XTensor the xtensor_container type
         * @param owned_tensor unique_ptr to the xtensor to adopt
         */
        template <class T, std::size_t N, layout_type L>
        template <class XTensor>
        inline void pytensor<T, N, L>::adopt_xtensor_container(std::unique_ptr<XTensor> owned_tensor)
        {
            static_assert(!std::is_const_v<T>, "pytensor::adopt_xtensor_container requires mutable tensor");
            static_assert(XTensor::rank == N, "xtensor rank mismatch for pytensor adoption");

            using xtensor_value_type = typename XTensor::value_type;
            static_assert(
                std::is_same_v<std::remove_const_t<xtensor_value_type>, scalar_type>,
                "xtensor value_type mismatch for pytensor adoption");

            auto* raw_tensor = owned_tensor.get();

            // Extract shape from xtensor
            std::array<size_t, N> nb_shape{};
            if constexpr (N > 0)
            {
                const auto& xt_shape = raw_tensor->shape();
                for (std::size_t axis = 0; axis < N; ++axis)
                {
                    nb_shape[axis] = static_cast<size_t>(xt_shape[axis]);
                }
            }

            // Extract strides from xtensor
            std::array<int64_t, N> nb_strides{};
            if constexpr (N > 0)
            {
                const auto& xt_strides = raw_tensor->strides();
                for (std::size_t axis = 0; axis < N; ++axis)
                {
                    nb_strides[axis] = static_cast<int64_t>(xt_strides[axis]);
                }
            }

            // Verify xtensor is contiguous
            const bool xtensor_row_major = detail::is_row_major(nb_strides, nb_shape);
            const bool xtensor_column_major = detail::is_column_major(nb_strides, nb_shape);

            if (!xtensor_row_major && !xtensor_column_major)
            {
                throw std::runtime_error("pytensor requires contiguous xtensor to adopt storage");
            }

            // Verify layout compatibility
            if constexpr (L == layout_type::row_major)
            {
                if (!xtensor_row_major)
                {
                    throw std::runtime_error("Expected row-major xtensor for pytensor row-major layout");
                }
            }
            else if constexpr (L == layout_type::column_major)
            {
                if (!xtensor_column_major)
                {
                    throw std::runtime_error("Expected column-major xtensor for pytensor column-major layout");
                }
            }

            // Determine order for nanobind
            char order = 'C';
            if constexpr (L == layout_type::column_major)
            {
                order = 'F';
            }
            else if constexpr (L == layout_type::dynamic)
            {
                order = xtensor_column_major ? 'F' : 'C';
            }

            // Create capsule to manage xtensor lifetime
            ::nanobind::object owner = ::nanobind::capsule(
                raw_tensor,
                [](void* raw) noexcept { delete static_cast<XTensor*>(raw); });

            owned_tensor.release();

            // Create nanobind ndarray
            m_array = ndarray_type(
                static_cast<pointer>(raw_tensor->data()),
                static_cast<size_t>(N),
                N > 0 ? nb_shape.data() : nullptr,
                owner.ptr(),
                N > 0 ? nb_strides.data() : nullptr,
                ::nanobind::dtype<ndarray_scalar_type>(),
                ::nanobind::device::cpu::value,
                0,
                order);

            // Initialize from the new ndarray
            init_from_ndarray();
        }

        template <class T, std::size_t N, layout_type L>
        inline auto pytensor<T, N, L>::shape_impl() noexcept -> inner_shape_type&
        {
            return m_shape;
        }

        template <class T, std::size_t N, layout_type L>
        inline auto pytensor<T, N, L>::shape_impl() const noexcept -> const inner_shape_type&
        {
            return m_shape;
        }

        template <class T, std::size_t N, layout_type L>
        inline auto pytensor<T, N, L>::strides_impl() noexcept -> inner_strides_type&
        {
            return m_strides;
        }

        template <class T, std::size_t N, layout_type L>
        inline auto pytensor<T, N, L>::strides_impl() const noexcept -> const inner_strides_type&
        {
            return m_strides;
        }

        template <class T, std::size_t N, layout_type L>
        inline auto pytensor<T, N, L>::backstrides_impl() noexcept -> inner_backstrides_type&
        {
            return m_backstrides;
        }

        template <class T, std::size_t N, layout_type L>
        inline auto pytensor<T, N, L>::backstrides_impl() const noexcept -> const inner_backstrides_type&
        {
            return m_backstrides;
        }

        template <class T, std::size_t N, layout_type L>
        inline auto pytensor<T, N, L>::storage_impl() noexcept -> storage_type&
        {
            return m_storage;
        }

        template <class T, std::size_t N, layout_type L>
        inline auto pytensor<T, N, L>::storage_impl() const noexcept -> const storage_type&
        {
            return m_storage;
        }

    } // namespace nanobind

} // namespace xt

// xt:: specializations for pytensor
namespace xt
{
    template <class T>
    struct temporary_type_from_tag<nanobind::pytensor_expression_tag, T>
    {
        using I = std::decay_t<T>;
        using value_type = std::remove_const_t<typename I::value_type>;
        using shape_type = typename I::shape_type;
        static constexpr std::size_t rank = std::tuple_size<shape_type>::value;
        static constexpr layout_type base_layout = layout_remove_any(I::static_layout);
        static constexpr layout_type tensor_layout =
            base_layout == layout_type::dynamic ? layout_type::row_major : base_layout;
        using type = xt::xtensor<value_type, rank, tensor_layout>;
    };

    namespace extension
    {
        struct nanobind_expression_base
        {
            using expression_tag = nanobind::pytensor_expression_tag;
        };

        template <class F, class... CT>
        struct xfunction_base_impl<nanobind::pytensor_expression_tag, F, CT...>
        {
            using type = nanobind_expression_base;
        };

        template <class CT, class... S>
        struct xview_base_impl<nanobind::pytensor_expression_tag, CT, S...>
        {
            using type = nanobind_expression_base;
        };

        template <class F, class CT, class X, class O>
        struct xreducer_base_impl<nanobind::pytensor_expression_tag, F, CT, X, O>
        {
            using type = nanobind_expression_base;
        };

        template <class CT, class I>
        struct xindex_view_base_impl<nanobind::pytensor_expression_tag, CT, I>
        {
            using type = nanobind_expression_base;
        };

        template <class CT, class X>
        struct xbroadcast_base_impl<nanobind::pytensor_expression_tag, CT, X>
        {
            using type = nanobind_expression_base;
        };

        template <class CT, class S, layout_type L, class FST>
        struct xstrided_view_base_impl<nanobind::pytensor_expression_tag, CT, S, L, FST>
        {
            using type = nanobind_expression_base;
        };
    }

    template <>
    class xexpression_assigner_base<nanobind::pytensor_expression_tag>
        : public xexpression_assigner_base<xtensor_expression_tag>
    {
    };

    namespace detail
    {
        template <class T, std::size_t N, layout_type Layout>
        struct is_crtp_base_of_impl<xexpression, nanobind::pytensor<T, N, Layout>> : std::true_type
        {
        };
    }

    namespace detail
    {
        template <class F, class... E>
        struct select_xfunction_expression<nanobind::pytensor_expression_tag, F, E...>
        {
            using type = xfunction<F, E...>;
        };
    }

    template <class From, class T, std::size_t N, layout_type Layout>
    struct has_assign_conversion<From, nanobind::pytensor<T, N, Layout>> : std::false_type
    {
    };
}

// nanobind type casters
NAMESPACE_BEGIN(NB_NAMESPACE)
NAMESPACE_BEGIN(detail)

    template <class T, std::size_t N, xt::layout_type Layout>
    struct type_caster<xt::nanobind::pytensor<T, N, Layout>>
    {
        using tensor_type = xt::nanobind::pytensor<T, N, Layout>;
        using scalar_type = std::remove_const_t<T>;
        // Use the tensor's own ndarray_type which respects the layout
        using ndarray_type = typename tensor_type::ndarray_type;

        NB_TYPE_CASTER(tensor_type, type_caster<ndarray_type>::Name)

        bool from_python(handle src, uint8_t flags, cleanup_list* cleanup) noexcept
        {
            make_caster<ndarray_type> caster;
            flags = flags_for_local_caster<ndarray_type>(flags);
            const bool allow_conversion =
                (flags & static_cast<uint8_t>(::nanobind::detail::cast_flags::convert)) != 0;

            if (caster.from_python(src, flags, cleanup))
            {
                try
                {
                    value.reset_from_ndarray(std::move(caster.value));
                }
                catch (...)
                {
                    return false;
                }
                return true;
            }

            if (!allow_conversion)
            {
                return false;
            }

            tensor_type sequence_tensor;
            if (!try_convert_sequence(src.ptr(), sequence_tensor))
            {
                return false;
            }

            value = std::move(sequence_tensor);
            return true;
        }

        static handle from_cpp(const tensor_type& tensor, rv_policy policy, cleanup_list* cleanup) noexcept
        {
            const ndarray_type& array = tensor.ndarray();
            return make_caster<ndarray_type>::from_cpp(&array, policy, cleanup);
        }

    private:
        static bool try_convert_sequence(PyObject* obj, tensor_type& result)
        {
            if constexpr (N == 0)
            {
                try
                {
                    result = tensor_type{};
                    result() = ::nanobind::cast<scalar_type>(::nanobind::handle(obj));
                    return true;
                }
                catch (...)
                {
                    PyErr_Clear();
                    return false;
                }
            }
            else
            {
                std::array<size_t, N> shape{};
                std::vector<scalar_type> values;
                values.reserve(8);

                if (!flatten_sequence(obj, 0, shape, values))
                {
                    return false;
                }

                size_t expected_size = std::accumulate(shape.begin(), shape.end(), size_t{1}, std::multiplies<size_t>{});
                if (expected_size != values.size())
                {
                    if (!(expected_size == 0 && values.empty()))
                    {
                        return false;
                    }
                }

                typename tensor_type::shape_type xt_shape{};
                for (size_t axis = 0; axis < N; ++axis)
                {
                    xt_shape[axis] = static_cast<typename tensor_type::size_type>(shape[axis]);
                }

                tensor_type tmp = tensor_type::from_shape(xt_shape);
                std::copy(values.begin(), values.end(), tmp.begin());
                result = std::move(tmp);
                return true;
            }
        }

        static bool flatten_sequence(PyObject* obj,
                                     size_t axis,
                                     std::array<size_t, N>& shape,
                                     std::vector<scalar_type>& values)
        {
            if (axis == N)
            {
                try
                {
                    values.push_back(::nanobind::cast<scalar_type>(::nanobind::handle(obj)));
                    return true;
                }
                catch (...)
                {
                    PyErr_Clear();
                    return false;
                }
            }

            if (!PySequence_Check(obj) || PyUnicode_Check(obj) || PyBytes_Check(obj))
            {
                return false;
            }

            PyObject* seq = PySequence_Fast(obj, "pytensor expects a sequence");
            if (seq == nullptr)
            {
                PyErr_Clear();
                return false;
            }

            Py_ssize_t length = PySequence_Fast_GET_SIZE(seq);
            size_t extent = static_cast<size_t>(length);

            if (shape[axis] == 0)
            {
                shape[axis] = extent;
            }
            else if (shape[axis] != extent)
            {
                Py_DECREF(seq);
                return false;
            }

            PyObject** items = PySequence_Fast_ITEMS(seq);
            for (Py_ssize_t i = 0; i < length; ++i)
            {
                if (!flatten_sequence(items[i], axis + 1, shape, values))
                {
                    Py_DECREF(seq);
                    return false;
                }
            }

            Py_DECREF(seq);
            return true;
        }
    };

    // Type caster for xexpression<pytensor>
    template <class T, std::size_t N, xt::layout_type Layout>
    struct type_caster<xt::xexpression<xt::nanobind::pytensor<T, N, Layout>>>
        : type_caster<xt::nanobind::pytensor<T, N, Layout>>
    {
        using expression_type = xt::xexpression<xt::nanobind::pytensor<T, N, Layout>>;
        using base_caster = type_caster<xt::nanobind::pytensor<T, N, Layout>>;

        operator expression_type&()
        {
            return this->value;
        }

        operator const expression_type&()
        {
            return this->value;
        }
    };

NAMESPACE_END(detail)
NAMESPACE_END(NB_NAMESPACE)

#endif
