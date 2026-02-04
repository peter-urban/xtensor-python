/***************************************************************************
* Copyright (c) Wolf Vollprecht, Johan Mabille and Sylvain Corlay          *
* Copyright (c) QuantStack                                                 *
* Copyright (c) Peter Urban, Ghent University                              *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XTENSOR_PYTHON_NANOBIND_XTENSOR_TYPE_CASTER_HPP
#define XTENSOR_PYTHON_NANOBIND_XTENSOR_TYPE_CASTER_HPP

#include <cstddef>
#include <algorithm>
#include <array>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "xtensor/containers/xarray.hpp"
#include "xtensor/containers/xtensor.hpp"
#include "xtensor/containers/xfixed.hpp"

#include "pycontainer.hpp"

NAMESPACE_BEGIN(NB_NAMESPACE)
NAMESPACE_BEGIN(detail)

    // Helper to get ndarray type based on layout
    template <typename T, xt::layout_type L>
    struct nanobind_array_getter_impl
    {
        using type = ::nanobind::ndarray<T, ::nanobind::numpy, ::nanobind::any_contig>;
    };

    template <typename T>
    struct nanobind_array_getter_impl<T, xt::layout_type::row_major>
    {
        using type = ::nanobind::ndarray<T, ::nanobind::numpy, ::nanobind::c_contig>;
    };

    template <typename T>
    struct nanobind_array_getter_impl<T, xt::layout_type::column_major>
    {
        using type = ::nanobind::ndarray<T, ::nanobind::numpy, ::nanobind::f_contig>;
    };

    // Get ndarray type for a given xtensor type
    template <class T>
    struct nanobind_array_getter
    {
    };

    template <class T, xt::layout_type L>
    struct nanobind_array_getter<xt::xarray<T, L>>
    {
        using type = typename nanobind_array_getter_impl<T, L>::type;
    };

    template <class T, std::size_t N, xt::layout_type L>
    struct nanobind_array_getter<xt::xtensor<T, N, L>>
    {
        using type = typename nanobind_array_getter_impl<T, L>::type;
    };

    template <class T, class FSH, xt::layout_type L>
    struct nanobind_array_getter<xt::xtensor_fixed<T, FSH, L>>
    {
        using type = typename nanobind_array_getter_impl<T, L>::type;
    };

    // Dimension checker for xtensor types
    template <class T>
    struct nanobind_array_dim_checker
    {
        template <class Array>
        static bool run(const Array& /*arr*/)
        {
            return true;
        }
    };

    template <class T, std::size_t N, xt::layout_type L>
    struct nanobind_array_dim_checker<xt::xtensor<T, N, L>>
    {
        template <class Array>
        static bool run(const Array& arr)
        {
            return arr.ndim() == N;
        }
    };

    template <class T, class FSH, xt::layout_type L>
    struct nanobind_array_dim_checker<xt::xtensor_fixed<T, FSH, L>>
    {
        template <class Array>
        static bool run(const Array& arr)
        {
            return arr.ndim() == FSH::size();
        }
    };

    // Shape checker for xtensor_fixed
    template <class T>
    struct nanobind_array_shape_checker
    {
        template <class Array>
        static bool run(const Array& /*arr*/)
        {
            return true;
        }
    };

    template <class T, class FSH, xt::layout_type L>
    struct nanobind_array_shape_checker<xt::xtensor_fixed<T, FSH, L>>
    {
        template <class Array>
        static bool run(const Array& arr)
        {
            auto shape = FSH();
            for (std::size_t i = 0; i < FSH::size(); ++i)
            {
                if (arr.shape(i) != shape[i])
                {
                    return false;
                }
            }
            return true;
        }
    };

    // Base class of type_caster for strided expressions
    template <class Type>
    class xtensor_type_caster_base
    {
    public:
        using value_type = typename Type::value_type;
        using scalar_type = std::remove_const_t<value_type>;
        using ndarray_type = typename nanobind_array_getter<Type>::type;

        NB_TYPE_CASTER(Type, type_caster<ndarray_type>::Name)

        bool from_python(handle src, uint8_t flags, cleanup_list* cleanup) noexcept
        {
            make_caster<ndarray_type> caster;
            flags = flags_for_local_caster<ndarray_type>(flags);

            if (!caster.from_python(src, flags, cleanup))
            {
                return false;
            }

            auto& arr = caster.value;

            if (!nanobind_array_dim_checker<Type>::run(arr))
            {
                return false;
            }

            if (!nanobind_array_shape_checker<Type>::run(arr))
            {
                return false;
            }

            try
            {
                std::vector<std::size_t> shape(arr.ndim());
                for (std::size_t i = 0; i < arr.ndim(); ++i)
                {
                    shape[i] = arr.shape(i);
                }
                value = Type::from_shape(shape);
                std::copy(arr.data(), arr.data() + arr.size(), value.data());
            }
            catch (...)
            {
                return false;
            }

            return true;
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
            // For rvalue, use move policy
            if (policy == rv_policy::automatic || policy == rv_policy::automatic_reference)
            {
                policy = rv_policy::move;
            }

            return from_cpp_impl(&src, policy, cleanup);
        }

    private:
        template <typename CType>
        static handle from_cpp_impl(CType* src, rv_policy policy, cleanup_list* cleanup) noexcept
        {
            if (policy == rv_policy::automatic || policy == rv_policy::automatic_reference)
            {
                policy = rv_policy::copy;
            }

            // Determine layout order
            char order = 'C';
            if constexpr (Type::static_layout == xt::layout_type::column_major)
            {
                order = 'F';
            }
            else if constexpr (Type::static_layout == xt::layout_type::dynamic)
            {
                // Try to detect from strides
                if (src->strides().size() > 1)
                {
                    if (xt::nanobind::detail::is_column_major(src->strides(), src->shape()))
                    {
                        order = 'F';
                    }
                }
            }

            const std::size_t rank = src->dimension();
            std::vector<size_t> shape(rank);
            std::vector<int64_t> strides(rank);

            for (std::size_t i = 0; i < rank; ++i)
            {
                shape[i] = src->shape(i);
                strides[i] = static_cast<int64_t>(src->strides()[i]);
            }

            auto create_array = [&](auto* data_ptr, ::nanobind::handle owner_handle) -> handle {
                ndarray_type array(
                    data_ptr,
                    rank,
                    rank > 0 ? shape.data() : nullptr,
                    owner_handle,
                    rank > 0 ? strides.data() : nullptr,
                    ::nanobind::dtype<scalar_type>(),
                    ::nanobind::device::cpu::value,
                    0,
                    order);

                return make_caster<ndarray_type>::from_cpp(array, rv_policy::reference, cleanup);
            };

            using non_const_type = std::remove_const_t<std::remove_pointer_t<CType>>;

            if (policy == rv_policy::move)
            {
                auto* moved = new non_const_type(std::move(*const_cast<non_const_type*>(src)));
                ::nanobind::object owner = ::nanobind::capsule(
                    moved,
                    [](void* raw) noexcept { delete static_cast<non_const_type*>(raw); });
                return create_array(moved->data(), owner);
            }

            if (policy == rv_policy::copy)
            {
                auto* copied = new non_const_type(*src);
                ::nanobind::object owner = ::nanobind::capsule(
                    copied,
                    [](void* raw) noexcept { delete static_cast<non_const_type*>(raw); });
                return create_array(copied->data(), owner);
            }

            if (policy == rv_policy::take_ownership)
            {
                ::nanobind::object owner = ::nanobind::capsule(
                    const_cast<non_const_type*>(src),
                    [](void* raw) noexcept { delete static_cast<non_const_type*>(raw); });
                return create_array(const_cast<scalar_type*>(src->data()), owner);
            }

            if (policy == rv_policy::reference_internal && cleanup != nullptr && cleanup->self() != nullptr)
            {
                return create_array(const_cast<scalar_type*>(src->data()), ::nanobind::borrow(cleanup->self()));
            }

            // reference policy
            return create_array(const_cast<scalar_type*>(src->data()), ::nanobind::handle());
        }
    };

NAMESPACE_END(detail)
NAMESPACE_END(NB_NAMESPACE)

#endif
