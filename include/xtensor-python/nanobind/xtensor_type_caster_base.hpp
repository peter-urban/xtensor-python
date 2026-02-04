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
#include "xtensor/containers/xadapt.hpp"
#include "xtensor/views/xstrided_view.hpp"

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

    // For strided views - use any_contig since views can have arbitrary strides
    template <class CT, class S, xt::layout_type L, class FST>
    struct nanobind_array_getter<xt::xstrided_view<CT, S, L, FST>>
    {
        using value_t = typename xt::xstrided_view<CT, S, L, FST>::value_type;
        using type = ::nanobind::ndarray<value_t, ::nanobind::numpy>;
    };

    // For array adapters - use layout-aware getter
    template <class EC, xt::layout_type L, class SC, class Tag>
    struct nanobind_array_getter<xt::xarray_adaptor<EC, L, SC, Tag>>
    {
        using value_t = typename EC::value_type;
        using type = typename nanobind_array_getter_impl<value_t, L>::type;
    };

    // For tensor adapters - use layout-aware getter
    template <class EC, std::size_t N, xt::layout_type L, class Tag>
    struct nanobind_array_getter<xt::xtensor_adaptor<EC, N, L, Tag>>
    {
        using value_t = typename EC::value_type;
        using type = typename nanobind_array_getter_impl<value_t, L>::type;
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

    // Specialized type caster for strided views (output-only, cannot load from Python)
    template <class CT, class S, xt::layout_type L, class FST>
    class xstrided_view_type_caster
    {
    public:
        using Type = xt::xstrided_view<CT, S, L, FST>;
        using value_type = typename Type::value_type;
        using scalar_type = std::remove_const_t<value_type>;
        using ndarray_type = ::nanobind::ndarray<scalar_type, ::nanobind::numpy>;

        NB_TYPE_CASTER(Type, const_name("numpy.ndarray"))

        // Strided views cannot be loaded from Python
        bool from_python(handle /*src*/, uint8_t /*flags*/, cleanup_list* /*cleanup*/) noexcept
        {
            return false;
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

            const std::size_t rank = src->dimension();
            std::vector<size_t> shape(rank);
            std::vector<int64_t> strides(rank);

            for (std::size_t i = 0; i < rank; ++i)
            {
                shape[i] = src->shape(i);
                // Strides are already in elements (DLPack convention)
                strides[i] = static_cast<int64_t>(src->strides()[i]);
            }

            // Detect layout order
            char order = 'C';
            if (rank > 1)
            {
                if (xt::nanobind::detail::is_column_major(src->strides(), src->shape()))
                {
                    order = 'F';
                }
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

            // Get raw data pointer
            // For strided views that have data interface, use data() + data_offset()
            // This is more reliable than &*begin() for views with complex iterators
            auto get_data_ptr = [&]() -> scalar_type* {
                // Check if the underlying expression type has a data interface
                // and the storage type is not a flat_expression_adaptor
                using xexpr_type = typename Type::xexpression_type;
                using storage_type = typename Type::storage_type;
                constexpr bool has_data = xt::detail::provides_data_interface<xexpr_type, storage_type>::value;
                
                if constexpr (has_data)
                {
                    // Use data() + data_offset() for types that support it
                    return const_cast<scalar_type*>(src->data() + src->data_offset());
                }
                else
                {
                    // Fallback to &*begin() for types without data interface
                    return const_cast<scalar_type*>(&(*src->begin()));
                }
            };

            if (policy == rv_policy::copy)
            {
                // Make a copy as an xarray
                using result_type = xt::xarray<scalar_type>;
                auto* copied = new result_type(*src);
                
                // Compute shape and strides from the COPIED array (not src)
                const std::size_t copy_rank = copied->dimension();
                std::vector<size_t> copy_shape(copy_rank);
                std::vector<int64_t> copy_strides(copy_rank);
                for (std::size_t i = 0; i < copy_rank; ++i)
                {
                    copy_shape[i] = copied->shape(i);
                    copy_strides[i] = static_cast<int64_t>(copied->strides()[i]);
                }
                
                // Detect layout order for the copy
                char copy_order = 'C';
                if (copy_rank > 1)
                {
                    if (xt::nanobind::detail::is_column_major(copied->strides(), copied->shape()))
                    {
                        copy_order = 'F';
                    }
                }
                
                ::nanobind::object owner = ::nanobind::capsule(
                    copied,
                    [](void* raw) noexcept { delete static_cast<result_type*>(raw); });
                    
                ndarray_type array(
                    copied->data(),
                    copy_rank,
                    copy_rank > 0 ? copy_shape.data() : nullptr,
                    owner,
                    copy_rank > 0 ? copy_strides.data() : nullptr,
                    ::nanobind::dtype<scalar_type>(),
                    ::nanobind::device::cpu::value,
                    0,
                    copy_order);

                return make_caster<ndarray_type>::from_cpp(array, rv_policy::reference, cleanup);
            }

            if (policy == rv_policy::reference_internal && cleanup != nullptr && cleanup->self() != nullptr)
            {
                return create_array(get_data_ptr(), ::nanobind::borrow(cleanup->self()));
            }

            // reference policy - just return a reference to the underlying data
            return create_array(get_data_ptr(), ::nanobind::handle());
        }
    };

    // Specialized type caster for xarray_adaptor (output-only, cannot load from Python)
    template <class EC, xt::layout_type L, class SC, class Tag>
    class xarray_adaptor_type_caster
    {
    public:
        using Type = xt::xarray_adaptor<EC, L, SC, Tag>;
        using value_type = typename Type::value_type;
        using scalar_type = std::remove_const_t<value_type>;
        using ndarray_type = typename nanobind_array_getter<Type>::type;

        NB_TYPE_CASTER(Type, const_name("numpy.ndarray"))

        // Adapters cannot be loaded from Python
        bool from_python(handle /*src*/, uint8_t /*flags*/, cleanup_list* /*cleanup*/) noexcept
        {
            return false;
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

            // Determine layout order
            char order = 'C';
            if constexpr (L == xt::layout_type::column_major)
            {
                order = 'F';
            }
            else if constexpr (L == xt::layout_type::dynamic)
            {
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
                // Strides are already in elements (DLPack convention)
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

            if (policy == rv_policy::copy)
            {
                using result_type = xt::xarray<scalar_type, L>;
                auto* copied = new result_type(*src);
                
                // Compute shape and strides from the COPIED array
                const std::size_t copy_rank = copied->dimension();
                std::vector<size_t> copy_shape(copy_rank);
                std::vector<int64_t> copy_strides(copy_rank);
                for (std::size_t i = 0; i < copy_rank; ++i)
                {
                    copy_shape[i] = copied->shape(i);
                    copy_strides[i] = static_cast<int64_t>(copied->strides()[i]);
                }
                
                char copy_order = 'C';
                if (copy_rank > 1 && xt::nanobind::detail::is_column_major(copied->strides(), copied->shape()))
                {
                    copy_order = 'F';
                }
                
                ::nanobind::object owner = ::nanobind::capsule(
                    copied,
                    [](void* raw) noexcept { delete static_cast<result_type*>(raw); });
                    
                ndarray_type array(
                    copied->data(),
                    copy_rank,
                    copy_rank > 0 ? copy_shape.data() : nullptr,
                    owner,
                    copy_rank > 0 ? copy_strides.data() : nullptr,
                    ::nanobind::dtype<scalar_type>(),
                    ::nanobind::device::cpu::value,
                    0,
                    copy_order);

                return make_caster<ndarray_type>::from_cpp(array, rv_policy::reference, cleanup);
            }

            if (policy == rv_policy::reference_internal && cleanup != nullptr && cleanup->self() != nullptr)
            {
                return create_array(const_cast<scalar_type*>(src->data()), ::nanobind::borrow(cleanup->self()));
            }

            // reference policy
            return create_array(const_cast<scalar_type*>(src->data()), ::nanobind::handle());
        }
    };

    // Specialized type caster for xtensor_adaptor (output-only, cannot load from Python)
    template <class EC, std::size_t N, xt::layout_type L, class Tag>
    class xtensor_adaptor_type_caster
    {
    public:
        using Type = xt::xtensor_adaptor<EC, N, L, Tag>;
        using value_type = typename Type::value_type;
        using scalar_type = std::remove_const_t<value_type>;
        using ndarray_type = typename nanobind_array_getter<Type>::type;

        NB_TYPE_CASTER(Type, const_name("numpy.ndarray"))

        // Adapters cannot be loaded from Python
        bool from_python(handle /*src*/, uint8_t /*flags*/, cleanup_list* /*cleanup*/) noexcept
        {
            return false;
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

            // Determine layout order
            char order = 'C';
            if constexpr (L == xt::layout_type::column_major)
            {
                order = 'F';
            }

            const std::size_t rank = N;
            std::vector<size_t> shape(rank);
            std::vector<int64_t> strides(rank);

            for (std::size_t i = 0; i < rank; ++i)
            {
                shape[i] = src->shape(i);
                // Strides are already in elements (DLPack convention)
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

            if (policy == rv_policy::copy)
            {
                using result_type = xt::xtensor<scalar_type, N, L>;
                auto* copied = new result_type(*src);
                
                // Compute shape and strides from the COPIED array
                const std::size_t copy_rank = copied->dimension();
                std::vector<size_t> copy_shape(copy_rank);
                std::vector<int64_t> copy_strides(copy_rank);
                for (std::size_t i = 0; i < copy_rank; ++i)
                {
                    copy_shape[i] = copied->shape(i);
                    copy_strides[i] = static_cast<int64_t>(copied->strides()[i]);
                }
                
                char copy_order = 'C';
                if (copy_rank > 1 && xt::nanobind::detail::is_column_major(copied->strides(), copied->shape()))
                {
                    copy_order = 'F';
                }
                
                ::nanobind::object owner = ::nanobind::capsule(
                    copied,
                    [](void* raw) noexcept { delete static_cast<result_type*>(raw); });
                    
                ndarray_type array(
                    copied->data(),
                    copy_rank,
                    copy_rank > 0 ? copy_shape.data() : nullptr,
                    owner,
                    copy_rank > 0 ? copy_strides.data() : nullptr,
                    ::nanobind::dtype<scalar_type>(),
                    ::nanobind::device::cpu::value,
                    0,
                    copy_order);

                return make_caster<ndarray_type>::from_cpp(array, rv_policy::reference, cleanup);
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
