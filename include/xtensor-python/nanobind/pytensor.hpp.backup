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
 * Note: This code requires a C++20 compiler (tested with GCC 14, clang 16, *
 *       clang 17,clang-cl 19).                                             *
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

#pragma once

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
#include <nanobind/nb_python.h>
#include <nanobind/ndarray.h>

#include <xtl/xsequence.hpp>

#include <xtensor/containers/xarray.hpp>
#include <xtensor/containers/xbuffer_adaptor.hpp>
#include <xtensor/containers/xtensor.hpp>
#include <xtensor/containers/xscalar.hpp>
#include <xtensor/core/xassign.hpp>
#include <xtensor/core/xfunction.hpp>
#include <xtensor/core/xeval.hpp>
#include <xtensor/reducers/xreducer.hpp>
#include <xtensor/views/xindex_view.hpp>
#include <xtensor/views/xview.hpp>

namespace xt
{
    namespace nanobind
    {
        struct pytensor_expression_tag : xt::xtensor_expression_tag
        {
        };

        namespace detail
        {
            template <class TensorRef, class IndicesContainer, std::size_t Dim>
            class pytensor_index_view
            {
            public:
                using tensor_type = std::remove_reference_t<TensorRef>;
                using value_type = typename tensor_type::value_type;
                using size_type = typename tensor_type::size_type;

                pytensor_index_view(TensorRef tensor, IndicesContainer& indices)
                    : m_tensor(tensor)
                    , m_indices(indices)
                {
                }

                template <class Expression>
                pytensor_index_view& operator=(const Expression& expression)
                {
                    auto evaluated = xt::eval(expression);
                    auto expected_size = m_indices.size();
                    if (evaluated.size() != expected_size)
                    {
                        throw std::runtime_error("pytensor index_view assignment size mismatch");
                    }

                    for (std::size_t idx = 0; idx < expected_size; ++idx)
                    {
                        assign_element(m_indices[idx], static_cast<value_type>(evaluated(idx)));
                    }
                    return *this;
                }

            private:
                template <class Index>
                void assign_element(const Index& index, const value_type& value)
                {
                    assign_element_impl(index, value, std::make_index_sequence<Dim>{});
                }

                template <class Index, std::size_t... Axis>
                void assign_element_impl(const Index& index, const value_type& value, std::index_sequence<Axis...>)
                {
                    m_tensor(static_cast<size_type>(index[Axis])...) = value;
                }

                TensorRef m_tensor;
                IndicesContainer& m_indices;
            };
        } // namespace detail

        namespace detail
        {
            template <class Strides, class Shape>
            inline bool is_row_major(const Strides& strides, const Shape& shape)
            {
                using size_type = typename Shape::value_type;
                constexpr std::size_t rank = std::tuple_size_v<Shape>;
                if constexpr (rank <= 1)
                {
                    return true;
                }

                auto expected = static_cast<std::make_signed_t<size_type>>(1);
                for (std::ptrdiff_t axis = static_cast<std::ptrdiff_t>(rank) - 1; axis >= 0; --axis)
                {
                    if (static_cast<std::make_signed_t<size_type>>(strides[static_cast<size_type>(axis)]) != expected)
                    {
                        return false;
                    }
                    expected *= static_cast<std::make_signed_t<size_type>>(shape[static_cast<size_type>(axis)]);
                }
                return true;
            }

            template <class Strides, class Shape>
            inline bool is_column_major(const Strides& strides, const Shape& shape)
            {
                using size_type = typename Shape::value_type;
                constexpr std::size_t rank = std::tuple_size_v<Shape>;
                if constexpr (rank <= 1)
                {
                    return true;
                }

                auto expected = static_cast<std::make_signed_t<size_type>>(1);
                for (size_type axis = 0; axis < rank; ++axis)
                {
                    if (static_cast<std::make_signed_t<size_type>>(strides[axis]) != expected)
                    {
                        return false;
                    }
                    expected *= static_cast<std::make_signed_t<size_type>>(shape[axis]);
                }
                return true;
            }

            template <class Shape, class Strides>
            inline void compute_contiguous_strides_row_major(Shape& shape, Strides& strides)
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
                    auto next_axis = static_cast<size_type>(axis + 1);
                    strides[static_cast<size_type>(axis)] = strides[next_axis] * std::max(shape[next_axis], size_type(1));
                }
            }

            template <class Shape, class Strides>
            inline void compute_contiguous_strides_column_major(Shape& shape, Strides& strides)
            {
                using size_type = typename Shape::value_type;
                constexpr std::size_t rank = std::tuple_size_v<Shape>;
                if constexpr (rank == 0)
                {
                    return;
                }

                strides.front() = size_type(1);
                for (size_type axis = 1; axis < rank; ++axis)
                {
                    auto previous_axis = static_cast<size_type>(axis - 1);
                    strides[axis] = strides[previous_axis] * std::max(shape[previous_axis], size_type(1));
                }
            }
        } // namespace detail

        namespace detail
        {
            template <class Scalar, std::size_t N, layout_type Layout>
            struct ndarray_type_helper
            {
                using type = ::nanobind::ndarray<
                    Scalar,
                    ::nanobind::ndim<N>,
                    ::nanobind::numpy,
                    ::nanobind::any_contig>;
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
        }

        template <class T, std::size_t N, layout_type Layout = layout_type::dynamic>
        class pytensor
            : public xtensor_adaptor<
                  xt::xbuffer_adaptor<
                      std::conditional_t<std::is_const_v<T>, const std::remove_const_t<T>*, std::remove_const_t<T>*>,
                      xt::no_ownership,
                      std::allocator<std::remove_const_t<T>>>,
                  N,
                  layout_type::dynamic>
        {
        private:
            using base_type = xtensor_adaptor<
                xt::xbuffer_adaptor<
                    std::conditional_t<std::is_const_v<T>, const std::remove_const_t<T>*, std::remove_const_t<T>*>,
                    xt::no_ownership,
                    std::allocator<std::remove_const_t<T>>>,
                N,
                layout_type::dynamic>;

        public:
            using self_type = pytensor<T, N, Layout>;
            using scalar_type = std::remove_const_t<T>;
            using ndarray_scalar_type = std::conditional_t<std::is_const_v<T>, const scalar_type, scalar_type>;
            using ndarray_type = typename detail::ndarray_type_helper<ndarray_scalar_type, N, Layout>::type;
            static_assert(
                Layout != layout_type::dynamic
                    || std::is_same_v<
                        ndarray_type,
                        ::nanobind::ndarray<
                            ndarray_scalar_type,
                            ::nanobind::ndim<N>,
                            ::nanobind::numpy,
                            ::nanobind::any_contig>>,
                "pytensor ndarray_type helper mismatch for dynamic layout");
            using buffer_type = typename base_type::storage_type;
            using size_type = typename base_type::size_type;
            using shape_type = typename base_type::shape_type;
            using strides_type = typename base_type::strides_type;
            using value_type = scalar_type;
            using reference = std::conditional_t<std::is_const_v<T>, const scalar_type&, scalar_type&>;
            using const_reference = const scalar_type&;
            using pointer = std::conditional_t<std::is_const_v<T>, const scalar_type*, scalar_type*>;
            using const_pointer = const scalar_type*;
            using expression_tag = pytensor_expression_tag;
            static constexpr layout_type static_layout = Layout;

            pytensor()
                : base_type(buffer_type{})
            {
                shape_type shape{};
                strides_type strides{};
                if constexpr (std::tuple_size_v<shape_type> > 0)
                {
                    shape.fill(size_type(0));
                    strides.fill(size_type(0));
                }
                this->resize(shape, strides);
            }

            explicit pytensor(ndarray_type array)
                : base_type(buffer_type{})
            {
                reset_from_ndarray(std::move(array));
            }

            template <class ShapeLike,
                      std::enable_if_t<!std::is_same_v<std::decay_t<ShapeLike>, self_type>
                                            && !std::is_same_v<std::decay_t<ShapeLike>, ndarray_type>
                                            && !std::is_base_of_v<xt::xexpression<std::decay_t<ShapeLike>>, std::decay_t<ShapeLike>>,
                                        int> = 0>
            explicit pytensor(ShapeLike&& shape_like)
                : base_type(buffer_type{})
            {
                auto temporary = from_shape(std::forward<ShapeLike>(shape_like));
                *this = std::move(temporary);
            }

            template <class ShapeLike,
                      class Value,
                      std::enable_if_t<!std::is_same_v<std::decay_t<ShapeLike>, self_type>
                                            && !std::is_same_v<std::decay_t<ShapeLike>, ndarray_type>
                                            && !std::is_base_of_v<xt::xexpression<std::decay_t<ShapeLike>>, std::decay_t<ShapeLike>>
                                            && std::is_convertible_v<Value, scalar_type>
                                            && !std::is_const_v<T>,
                                        int> = 0>
            pytensor(ShapeLike&& shape_like, Value&& value)
                : base_type(buffer_type{})
            {
                auto temporary = from_shape(std::forward<ShapeLike>(shape_like));
                temporary.fill(static_cast<scalar_type>(std::forward<Value>(value)));
                *this = std::move(temporary);
            }

            template <std::enable_if_t<!std::is_const_v<T>, int> = 0>
            pytensor(const shape_type& shape, const value_type& value)
                : base_type(buffer_type{})
            {
                auto temporary = from_shape(shape);
                temporary.fill(value);
                *this = std::move(temporary);
            }

            template <class Integral,
                      class Value,
                      std::enable_if_t<std::is_integral_v<std::decay_t<Integral>>
                                           && std::is_convertible_v<Value, scalar_type>
                                           && !std::is_const_v<T>,
                                       int> = 0>
            pytensor(std::initializer_list<Integral> shape_list, Value&& value)
                : base_type(buffer_type{})
            {
                auto temporary = from_shape(shape_list);
                temporary.fill(static_cast<scalar_type>(std::forward<Value>(value)));
                *this = std::move(temporary);
            }

            template <class EC,
                      xt::layout_type TensorLayout,
                      class Tag,
                      std::enable_if_t<!std::is_const_v<T>, int> = 0>
            pytensor(xt::xtensor_container<EC, N, TensorLayout, Tag> tensor)
                : base_type(buffer_type{})
            {
                adopt_xtensor_container(
                    std::make_unique<xt::xtensor_container<EC, N, TensorLayout, Tag>>(std::move(tensor)));
            }

            pytensor(const pytensor&) = default;
            pytensor(pytensor&&) noexcept = default;
            pytensor& operator=(const pytensor&) = default;
            pytensor& operator=(pytensor&&) noexcept = default;

            template <class E, std::enable_if_t<!std::is_const_v<T>, int> = 0>
            pytensor(const xexpression<E>& expression)
                : base_type(buffer_type{})
            {
                assign_expression(expression.derived_cast());
            }

            template <class E, std::enable_if_t<!std::is_const_v<T>, int> = 0>
            pytensor& operator=(const xexpression<E>& expression)
            {
                assign_expression(expression.derived_cast());
                return *this;
            }

            template <class EC,
                      xt::layout_type TensorLayout,
                      class Tag,
                      std::enable_if_t<!std::is_const_v<T>, int> = 0>
            pytensor& operator=(xt::xtensor_container<EC, N, TensorLayout, Tag> tensor)
            {
                adopt_xtensor_container(
                    std::make_unique<xt::xtensor_container<EC, N, TensorLayout, Tag>>(std::move(tensor)));
                return *this;
            }

            template <std::enable_if_t<!std::is_const_v<T>, int> = 0>
            pytensor& operator=(typename base_type::temporary_type&& rhs)
            {
                base_type::operator=(std::move(rhs));
                return *this;
            }

            using base_type::operator+=;
            using base_type::operator-=;
            using base_type::operator*=;
            using base_type::operator/=;

            ~pytensor() = default;

            template <class ShapeLike>
            static self_type from_shape(ShapeLike&& shape_like)
            {
                static_assert(!std::is_const_v<T>, "pytensor::from_shape requires a mutable tensor type");

                shape_type shape = xtl::forward_sequence<shape_type, ShapeLike>(std::forward<ShapeLike>(shape_like));

                strides_type strides{};
                if constexpr (std::tuple_size_v<strides_type> > 0)
                {
                    if constexpr (Layout == layout_type::column_major)
                    {
                        detail::compute_contiguous_strides_column_major(shape, strides);
                    }
                    else
                    {
                        detail::compute_contiguous_strides_row_major(shape, strides);
                    }
                }

                constexpr std::size_t rank = std::tuple_size_v<shape_type>;
                std::array<size_t, rank> shape_buffer{};
                size_t total_size = 1;
                if constexpr (rank > 0)
                {
                    for (std::size_t axis = 0; axis < rank; ++axis)
                    {
                        size_t extent = static_cast<size_t>(shape[axis]);
                        shape_buffer[axis] = extent;
                        total_size *= extent;
                    }
                }

                constexpr std::size_t stride_rank = std::tuple_size_v<strides_type>;
                std::array<int64_t, stride_rank> stride_buffer{};
                if constexpr (stride_rank > 0)
                {
                    for (std::size_t axis = 0; axis < stride_rank; ++axis)
                    {
                        stride_buffer[axis] = static_cast<int64_t>(strides[axis]);
                    }
                }

                scalar_type* raw_ptr = nullptr;
                ::nanobind::object owner;
                if (total_size > 0)
                {
                    raw_ptr = new scalar_type[total_size];
                    owner = ::nanobind::capsule(
                        raw_ptr,
                        [](void* p) noexcept { delete[] static_cast<scalar_type*>(p); });
                }

                char order = 'C';
                if constexpr (Layout == layout_type::column_major)
                {
                    order = 'F';
                }

                ndarray_type array(
                    static_cast<pointer>(raw_ptr),
                    rank,
                    rank > 0 ? shape_buffer.data() : nullptr,
                    owner.ptr(),
                    stride_rank > 0 ? stride_buffer.data() : nullptr,
                    ::nanobind::dtype<ndarray_scalar_type>(),
                    (int) ::nanobind::device::cpu::value,
                    0,
                    order);

                return self_type(std::move(array));
            }

            template <class Integral,
                      std::enable_if_t<std::is_integral_v<std::decay_t<Integral>>, int> = 0>
            static self_type from_shape(std::initializer_list<Integral> shape_list)
            {
                shape_type shape{};
                std::size_t axis = 0;

                for (auto extent : shape_list)
                {
                    if (axis >= shape.size())
                    {
                        throw std::runtime_error("pytensor::from_shape received too many dimensions");
                    }
                    shape[axis++] = static_cast<size_type>(extent);
                }

                for (; axis < shape.size(); ++axis)
                {
                    shape[axis] = size_type(0);
                }

                return from_shape(std::move(shape));
            }

            template <class Array>
            void reset_from_ndarray(Array&& array [[maybe_unused]])
            {
                ndarray_type nb_array(std::forward<Array>(array));

                auto shape = compute_shape(nb_array);
                auto strides = compute_strides(nb_array, shape);

                if constexpr (Layout == layout_type::row_major)
                {
                    if (!detail::is_row_major(strides, shape))
                    {
                        throw std::runtime_error("Expected row-major contiguous array for pytensor");
                    }
                }
                else if constexpr (Layout == layout_type::column_major)
                {
                    if (!detail::is_column_major(strides, shape))
                    {
                        throw std::runtime_error("Expected column-major contiguous array for pytensor");
                    }
                }

                this->reset_buffer(nb_array.data(), static_cast<size_type>(nb_array.size()));
                this->resize(shape, strides);
                m_array = std::move(nb_array);

                if constexpr (Layout == layout_type::dynamic)
                {
                    this->mutable_layout() = deduce_layout(m_array, shape, strides);
                }
                else
                {
                    this->mutable_layout() = Layout;
                }
            }

            void reset_from_ndarray(ndarray_type array)
            {
                reset_from_ndarray<ndarray_type>(std::move(array));
            }

            template <class Array>
            void bind(Array&& array)
            {
                reset_from_ndarray(std::forward<Array>(array));
            }

            void bind(ndarray_type array)
            {
                reset_from_ndarray(std::move(array));
            }

            bool is_valid() const noexcept
            {
                return m_array.is_valid();
            }

            ndarray_type& ndarray() noexcept
            {
                return m_array;
            }

            const ndarray_type& ndarray() const noexcept
            {
                return m_array;
            }

        private:
            template <class Expression>
            void assign_expression(const Expression& expression)
            {
                using expression_type = std::decay_t<Expression>;
                static_assert(!std::is_same_v<expression_type, self_type>, "pytensor expression assignment should use copy/move operators");

                if constexpr (Layout == layout_type::column_major)
                {
                    auto owned_tensor = std::make_unique<xt::xtensor<scalar_type, N, layout_type::column_major>>(expression);
                    adopt_xtensor_container(std::move(owned_tensor));
                }
                else if constexpr (Layout == layout_type::row_major)
                {
                    auto owned_tensor = std::make_unique<xt::xtensor<scalar_type, N, layout_type::row_major>>(expression);
                    adopt_xtensor_container(std::move(owned_tensor));
                }
                else
                {
                    layout_type selected_layout = layout_type::row_major;
                    if constexpr (requires { expression.layout(); })
                    {
                        auto expression_layout = expression.layout();
                        if (expression_layout == layout_type::column_major)
                        {
                            selected_layout = layout_type::column_major;
                        }
                    }

                    if (selected_layout == layout_type::column_major)
                    {
                        auto owned_tensor = std::make_unique<xt::xtensor<scalar_type, N, layout_type::column_major>>(expression);
                        adopt_xtensor_container(std::move(owned_tensor));
                    }
                    else
                    {
                        auto owned_tensor = std::make_unique<xt::xtensor<scalar_type, N, layout_type::row_major>>(expression);
                        adopt_xtensor_container(std::move(owned_tensor));
                    }
                }
            }

            template <class XTensor>
            void adopt_xtensor_container(std::unique_ptr<XTensor> owned_tensor)
            {
                static_assert(!std::is_const_v<T>, "pytensor::adopt_xtensor_container requires mutable tensor");
                static_assert(XTensor::rank == N, "xtensor rank mismatch for pytensor adoption");

                using xtensor_value_type = typename XTensor::value_type;
                static_assert(
                    std::is_same_v<std::remove_const_t<xtensor_value_type>, scalar_type>,
                    "xtensor value_type mismatch for pytensor adoption");

                auto* raw_tensor = owned_tensor.get();

                std::array<size_t, N> shape{};
                if constexpr (N > 0)
                {
                    const auto& xt_shape = raw_tensor->shape();
                    for (std::size_t axis = 0; axis < N; ++axis)
                    {
                        shape[axis] = static_cast<size_t>(xt_shape[axis]);
                    }
                }

                std::array<int64_t, N> stride_buffer{};
                if constexpr (N > 0)
                {
                    const auto& xt_strides = raw_tensor->strides();
                    for (std::size_t axis = 0; axis < N; ++axis)
                    {
                        stride_buffer[axis] = static_cast<int64_t>(xt_strides[axis]);
                    }
                }

                const bool xtensor_row_major = detail::is_row_major(stride_buffer, shape);
                const bool xtensor_column_major = detail::is_column_major(stride_buffer, shape);

                if (!xtensor_row_major && !xtensor_column_major)
                {
                    throw std::runtime_error("pytensor requires contiguous xtensor to adopt storage");
                }

                if constexpr (Layout == layout_type::row_major)
                {
                    if (!xtensor_row_major)
                    {
                        throw std::runtime_error("Expected row-major xtensor for pytensor row-major layout");
                    }
                }
                else if constexpr (Layout == layout_type::column_major)
                {
                    if (!xtensor_column_major)
                    {
                        throw std::runtime_error("Expected column-major xtensor for pytensor column-major layout");
                    }
                }

                char order = 'C';
                if constexpr (Layout == layout_type::column_major)
                {
                    order = 'F';
                }
                else if constexpr (Layout == layout_type::dynamic)
                {
                    order = xtensor_column_major ? 'F' : 'C';
                }

                ::nanobind::object owner = ::nanobind::capsule(
                    raw_tensor,
                    [](void* raw) noexcept { delete static_cast<XTensor*>(raw); });

                owned_tensor.release();

                ndarray_type array(
                    static_cast<pointer>(raw_tensor->data()),
                    static_cast<size_t>(N),
                    N > 0 ? shape.data() : nullptr,
                    owner.ptr(),
                    N > 0 ? stride_buffer.data() : nullptr,
                    ::nanobind::dtype<ndarray_scalar_type>(),
                    ::nanobind::device::cpu::value,
                    0,
                    order);

                reset_from_ndarray(std::move(array));
            }

            static shape_type compute_shape(const ndarray_type& array)
            {
                shape_type shape{};
                if constexpr (std::tuple_size_v<shape_type> > 0)
                {
                    for (std::size_t axis = 0; axis < shape.size(); ++axis)
                    {
                        shape[axis] = static_cast<size_type>(array.shape(axis));
                    }
                }
                return shape;
            }

            static strides_type compute_strides(const ndarray_type& array, const shape_type& shape)
            {
                strides_type strides{};
                if constexpr (std::tuple_size_v<strides_type> == 0)
                {
                    return strides;
                }

                const int64_t* strides_ptr = array.stride_ptr();
                if (strides_ptr == nullptr)
                {
                    if constexpr (Layout == layout_type::column_major)
                    {
                        detail::compute_contiguous_strides_column_major(shape, strides);
                    }
                    else
                    {
                        detail::compute_contiguous_strides_row_major(shape, strides);
                    }
                }
                else
                {
                    for (std::size_t axis = 0; axis < strides.size(); ++axis)
                    {
                        const auto raw_stride = strides_ptr[axis];

                        using stride_value_type = typename strides_type::value_type;
                        if constexpr (std::is_signed_v<stride_value_type>)
                        {
                            strides[axis] = static_cast<stride_value_type>(raw_stride);
                        }
                        else
                        {
                            if (raw_stride < 0)
                            {
                                throw std::runtime_error("pytensor: negative strides are not supported");
                            }
                            strides[axis] = static_cast<stride_value_type>(raw_stride);
                        }
                    }
                }
                return strides;
            }

            static layout_type deduce_layout(const ndarray_type& array,
                                             const shape_type& shape,
                                             const strides_type& strides)
            {
                if constexpr (Layout != layout_type::dynamic)
                {
                    return Layout;
                }

                if constexpr (std::tuple_size_v<shape_type> <= 1)
                {
                    return layout_type::row_major;
                }

                if (array.stride_ptr() == nullptr)
                {
                    return layout_type::row_major;
                }

                if (detail::is_row_major(strides, shape))
                {
                    return layout_type::row_major;
                }

                if (detail::is_column_major(strides, shape))
                {
                    return layout_type::column_major;
                }

                return layout_type::dynamic;
            }

            ndarray_type m_array;
        };
    } // namespace nanobind
} // namespace xt

namespace xt
{
    template <class T, std::size_t N, layout_type Layout>
    struct xcontainer_inner_types<nanobind::pytensor<T, N, Layout>>
    {
        using tensor_type = nanobind::pytensor<T, N, Layout>;
        using storage_type = typename tensor_type::buffer_type;
        using reference = typename tensor_type::reference;
        using const_reference = typename tensor_type::const_reference;
        using size_type = typename tensor_type::size_type;
        using shape_type = typename tensor_type::shape_type;
        using strides_type = typename tensor_type::strides_type;
        using backstrides_type = typename tensor_type::strides_type;
        using inner_shape_type = shape_type;
        using inner_strides_type = strides_type;
        using inner_backstrides_type = backstrides_type;
        using temporary_type = tensor_type;
        static constexpr layout_type layout = Layout;
    };

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
    }

    template <>
    class xexpression_assigner_base<nanobind::pytensor_expression_tag>
        : public xexpression_assigner_base<xtensor_expression_tag>
    {
    };

    template <class T, std::size_t N, layout_type Layout>
    struct xiterable_inner_types<nanobind::pytensor<T, N, Layout>>
        : xcontainer_iterable_types<nanobind::pytensor<T, N, Layout>>
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

    template <class T, std::size_t N, layout_type Layout, class Indices>
    inline auto index_view(nanobind::pytensor<T, N, Layout>& tensor, Indices& indices)
    {
        return nanobind::detail::pytensor_index_view<nanobind::pytensor<T, N, Layout>&, Indices, N>(tensor, indices);
    }
}

NAMESPACE_BEGIN(NB_NAMESPACE)
NAMESPACE_BEGIN(detail)

    template <class T, std::size_t N, xt::layout_type Layout>
    struct type_caster<xt::nanobind::pytensor<T, N, Layout>>
    {
        using tensor_type = xt::nanobind::pytensor<T, N, Layout>;
        using scalar_type = std::remove_const_t<T>;
        using ndarray_scalar_type = std::conditional_t<std::is_const_v<T>, const scalar_type, scalar_type>;
        using ndarray_type = ::nanobind::ndarray<
            ndarray_scalar_type,
            ::nanobind::ndim<N>,
            ::nanobind::numpy,
            ::nanobind::any_contig>;

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
            if constexpr (tensor_type::rank == 0)
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
                std::array<size_t, tensor_type::rank> shape{};
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
                for (size_t axis = 0; axis < tensor_type::rank; ++axis)
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
                                     std::array<size_t, tensor_type::rank>& shape,
                                     std::vector<scalar_type>& values)
        {
            if (axis == tensor_type::rank)
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

    template <class Tensor>
    class pytensor_xtensor_caster_impl
    {
    public:
        using tensor_type = Tensor;
        using value_type = typename tensor_type::value_type;
        using scalar_type = std::remove_const_t<value_type>;
        static constexpr std::size_t rank = tensor_type::rank;
        static constexpr xt::layout_type layout = tensor_type::static_layout;

        using pytensor_bridge = xt::nanobind::pytensor<value_type, rank, layout>;
        using ndarray_type = typename pytensor_bridge::ndarray_type;
        using ndarray_scalar_type = typename pytensor_bridge::ndarray_scalar_type;

        NB_TYPE_CASTER(tensor_type, type_caster<ndarray_type>::Name)

        bool from_python(handle src, uint8_t flags, cleanup_list* cleanup) noexcept
        {
            ::nanobind::detail::make_caster<pytensor_bridge> caster;
            flags = ::nanobind::detail::flags_for_local_caster<pytensor_bridge>(flags);
            if (!caster.from_python(src, flags, cleanup))
            {
                return false;
            }

            try
            {
                value = tensor_type(caster.value);
            }
            catch (...)
            {
                return false;
            }
            return true;
        }

        static handle from_cpp(const tensor_type& tensor, rv_policy policy, cleanup_list* cleanup) noexcept
        {
            rv_policy effective_policy = policy;
            if (effective_policy == rv_policy::automatic)
            {
                effective_policy = rv_policy::move;
            }
            else if (effective_policy == rv_policy::automatic_reference)
            {
                effective_policy = rv_policy::reference;
            }

            std::array<size_t, rank> shape{};
            std::array<int64_t, rank> stride_buffer{};

            if constexpr (rank > 0)
            {
                const auto& tensor_shape = tensor.shape();
                const auto& tensor_strides = tensor.strides();
                for (std::size_t axis = 0; axis < rank; ++axis)
                {
                    shape[axis] = static_cast<size_t>(tensor_shape[axis]);
                    stride_buffer[axis] = static_cast<int64_t>(tensor_strides[axis]);
                }
            }

            auto deduce_order = [&]() -> char {
                if constexpr (layout == xt::layout_type::row_major)
                {
                    return 'C';
                }
                else if constexpr (layout == xt::layout_type::column_major)
                {
                    return 'F';
                }
                else
                {
                    if constexpr (rank <= 1)
                    {
                        return 'C';
                    }

                    auto compute_expected = [](const std::array<size_t, rank>& extents,
                                               bool reverse) {
                        std::array<int64_t, rank> expected{};
                        if constexpr (rank == 0)
                        {
                            return expected;
                        }

                        int64_t accumulator = 1;
                        if (reverse)
                        {
                            for (std::ptrdiff_t axis = static_cast<std::ptrdiff_t>(rank) - 1; axis >= 0; --axis)
                            {
                                expected[static_cast<std::size_t>(axis)] = accumulator;
                                accumulator *= static_cast<int64_t>(std::max<size_t>(extents[static_cast<std::size_t>(axis)], 1));
                            }
                        }
                        else
                        {
                            for (std::size_t axis = 0; axis < rank; ++axis)
                            {
                                expected[axis] = accumulator;
                                accumulator *= static_cast<int64_t>(std::max<size_t>(extents[axis], 1));
                            }
                        }
                        return expected;
                    };

                    const auto expected_row = compute_expected(shape, true);
                    const auto expected_col = compute_expected(shape, false);

                    bool row_major = true;
                    bool column_major = true;

                    for (std::size_t axis = 0; axis < rank; ++axis)
                    {
                        if (stride_buffer[axis] != expected_row[axis])
                        {
                            row_major = false;
                        }
                        if (stride_buffer[axis] != expected_col[axis])
                        {
                            column_major = false;
                        }
                    }

                    if (row_major)
                    {
                        return 'C';
                    }
                    if (column_major)
                    {
                        return 'F';
                    }
                    return 'A';
                }
            };

            auto create_array = [&](auto* data_ptr,
                                    ::nanobind::handle owner_handle,
                                    rv_policy array_policy) -> handle {
                const size_t* shape_ptr = nullptr;
                const int64_t* stride_ptr = nullptr;
                if constexpr (rank > 0)
                {
                    shape_ptr = shape.data();
                    stride_ptr = stride_buffer.data();
                }

                ndarray_type array(
                    data_ptr,
                    static_cast<size_t>(rank),
                    shape_ptr,
                    owner_handle,
                    stride_ptr,
                    ::nanobind::dtype<ndarray_scalar_type>(),
                    ::nanobind::device::cpu::value,
                    0,
                    deduce_order());

                return ::nanobind::detail::make_caster<ndarray_type>::from_cpp(array, array_policy, cleanup);
            };

            using non_const_tensor = std::remove_const_t<tensor_type>;

            if (effective_policy == rv_policy::move)
            {
                if constexpr (!std::is_const_v<typename tensor_type::value_type>)
                {
                    auto* moved_tensor = new non_const_tensor(std::move(const_cast<non_const_tensor&>(tensor)));
                    ::nanobind::object owner = ::nanobind::capsule(
                        moved_tensor,
                        [](void* raw) noexcept { delete static_cast<non_const_tensor*>(raw); });
                    return create_array(moved_tensor->data(), owner, rv_policy::reference);
                }
                else
                {
                    effective_policy = rv_policy::reference;
                }
            }

            if (effective_policy == rv_policy::copy)
            {
                auto* copied_tensor = new non_const_tensor(tensor);
                ::nanobind::object owner = ::nanobind::capsule(
                    copied_tensor,
                    [](void* raw) noexcept { delete static_cast<non_const_tensor*>(raw); });
                return create_array(copied_tensor->data(), owner, rv_policy::reference);
            }

            if (effective_policy == rv_policy::reference_internal && cleanup != nullptr && cleanup->self() != nullptr)
            {
                return create_array(const_cast<scalar_type*>(tensor.data()), ::nanobind::borrow(cleanup->self()), rv_policy::reference);
            }

            return create_array(const_cast<scalar_type*>(tensor.data()), ::nanobind::handle(), effective_policy);
        }
    };

    template <class T, std::size_t N, xt::layout_type Layout>
    struct type_caster<xt::xtensor<T, N, Layout>>
        : pytensor_xtensor_caster_impl<xt::xtensor<T, N, Layout>>
    {
    };

NAMESPACE_END(detail)
NAMESPACE_END(NB_NAMESPACE)
