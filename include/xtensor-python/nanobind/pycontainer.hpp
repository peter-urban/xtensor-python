/***************************************************************************
* Copyright (c) Wolf Vollprecht, Johan Mabille and Sylvain Corlay          *
* Copyright (c) QuantStack                                                 *
* Copyright (c) Peter Urban, Ghent University                              *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XTENSOR_PYTHON_NANOBIND_PYCONTAINER_HPP
#define XTENSOR_PYTHON_NANOBIND_PYCONTAINER_HPP

#include <algorithm>
#include <cstddef>
#include <functional>
#include <numeric>
#include <sstream>
#include <stdexcept>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "xtensor/containers/xcontainer.hpp"
#include "xtl/xsequence.hpp"

namespace xt
{
    namespace nanobind
    {
        // Forward declaration
        template <class D>
        class pycontainer;

        namespace detail
        {
            // Layout detection utilities (can be moved to shared header later)
            // These handle unit dimensions (size=1) which may have stride=0 from compute_strides
            template <class Strides, class Shape>
            inline bool is_row_major(const Strides& strides, const Shape& shape)
            {
                using size_type = typename Shape::value_type;
                const std::size_t rank = shape.size();
                if (rank <= 1)
                {
                    return true;
                }

                auto expected = static_cast<std::make_signed_t<size_type>>(1);
                for (std::ptrdiff_t axis = static_cast<std::ptrdiff_t>(rank) - 1; axis >= 0; --axis)
                {
                    auto axis_idx = static_cast<size_type>(axis);
                    auto stride_val = static_cast<std::make_signed_t<size_type>>(strides[axis_idx]);
                    auto shape_val = static_cast<std::make_signed_t<size_type>>(shape[axis_idx]);
                    
                    // For unit dimensions, accept either expected stride or 0
                    if (shape_val == 1)
                    {
                        if (stride_val != expected && stride_val != 0)
                        {
                            return false;
                        }
                    }
                    else if (stride_val != expected)
                    {
                        return false;
                    }
                    expected *= shape_val;
                }
                return true;
            }

            template <class Strides, class Shape>
            inline bool is_column_major(const Strides& strides, const Shape& shape)
            {
                using size_type = typename Shape::value_type;
                const std::size_t rank = shape.size();
                if (rank <= 1)
                {
                    return true;
                }

                auto expected = static_cast<std::make_signed_t<size_type>>(1);
                for (size_type axis = 0; axis < rank; ++axis)
                {
                    auto stride_val = static_cast<std::make_signed_t<size_type>>(strides[axis]);
                    auto shape_val = static_cast<std::make_signed_t<size_type>>(shape[axis]);
                    
                    // For unit dimensions, accept either expected stride or 0
                    if (shape_val == 1)
                    {
                        if (stride_val != expected && stride_val != 0)
                        {
                            return false;
                        }
                    }
                    else if (stride_val != expected)
                    {
                        return false;
                    }
                    expected *= shape_val;
                }
                return true;
            }

            template <class S>
            struct check_dims
            {
                static bool run(std::size_t)
                {
                    return true;
                }
            };

            template <class T, std::size_t N>
            struct check_dims<std::array<T, N>>
            {
                static bool run(std::size_t new_dim)
                {
                    if (new_dim != N)
                    {
                        std::ostringstream err_msg;
                        err_msg << "Invalid conversion to pycontainer, expecting a container of dimension "
                                << N << ", got a container of dimension " << new_dim << ".";
                        throw std::runtime_error(err_msg.str());
                    }
                    return new_dim == N;
                }
            };

        } // namespace detail

        /**
         * @class pycontainer
         * @brief Base class for xtensor containers wrapping numpy arrays via nanobind.
         *
         * The pycontainer class should not be instantiated directly. Instead, users should
         * use pytensor and pyarray instances.
         *
         * @tparam D The derived type, i.e. the inheriting class for which pycontainer
         *           provides the interface.
         */
        template <class D>
        class pycontainer : public xcontainer<D>
        {
        public:

            using derived_type = D;

            using base_type = xcontainer<D>;
            using inner_types = xcontainer_inner_types<D>;
            using storage_type = typename inner_types::storage_type;
            using value_type = typename storage_type::value_type;
            using reference = typename storage_type::reference;
            using const_reference = typename storage_type::const_reference;
            using pointer = typename storage_type::pointer;
            using const_pointer = typename storage_type::const_pointer;
            using size_type = typename storage_type::size_type;
            using difference_type = typename storage_type::difference_type;

            using shape_type = typename inner_types::shape_type;
            using strides_type = typename inner_types::strides_type;
            using backstrides_type = typename inner_types::backstrides_type;
            using inner_shape_type = typename inner_types::inner_shape_type;
            using inner_strides_type = typename inner_types::inner_strides_type;

            using iterable_base = xcontainer<D>;

            using iterator = typename iterable_base::iterator;
            using const_iterator = typename iterable_base::const_iterator;

            using stepper = typename iterable_base::stepper;
            using const_stepper = typename iterable_base::const_stepper;

            template <class S = shape_type>
            void resize(const S& shape);
            template <class S = shape_type>
            void resize(const S& shape, layout_type l);
            template <class S = shape_type>
            void resize(const S& shape, const strides_type& strides);

            template <class S = shape_type>
            auto& reshape(S&& shape, layout_type layout = base_type::static_layout) &;

            layout_type layout() const;
            bool is_contiguous() const noexcept;

            using base_type::operator();
            using base_type::operator[];
            using base_type::begin;
            using base_type::end;

        protected:

            pycontainer();
            ~pycontainer() = default;

            pycontainer(const pycontainer&) = default;
            pycontainer& operator=(const pycontainer&) = default;

            pycontainer(pycontainer&&) = default;
            pycontainer& operator=(pycontainer&&) = default;

            derived_type& derived_cast();
            const derived_type& derived_cast() const;

            size_type get_buffer_size() const;
        };

        /******************************
         * pycontainer implementation *
         ******************************/

        template <class D>
        inline pycontainer<D>::pycontainer()
        {
        }

        template <class D>
        inline auto pycontainer<D>::derived_cast() -> derived_type&
        {
            return *static_cast<derived_type*>(this);
        }

        template <class D>
        inline auto pycontainer<D>::derived_cast() const -> const derived_type&
        {
            return *static_cast<const derived_type*>(this);
        }

        template <class D>
        inline auto pycontainer<D>::get_buffer_size() const -> size_type
        {
            const size_type& (*min)(const size_type&, const size_type&) = std::min<size_type>;
            size_type min_stride = this->strides().empty() ? size_type(1) :
                std::max(size_type(1), std::accumulate(this->strides().cbegin(),
                                                       this->strides().cend(),
                                                       std::numeric_limits<size_type>::max(),
                                                       min));
            return min_stride * this->size();
        }

        /**
         * Resizes the container.
         * @param shape the new shape
         */
        template <class D>
        template <class S>
        inline void pycontainer<D>::resize(const S& shape)
        {
            if (shape.size() != this->dimension() || !std::equal(std::begin(shape), std::end(shape), std::begin(this->shape())))
            {
                resize(shape, layout_type::row_major);
            }
        }

        /**
         * Resizes the container.
         * @param shape the new shape
         * @param l the new layout
         */
        template <class D>
        template <class S>
        inline void pycontainer<D>::resize(const S& shape, layout_type l)
        {
            strides_type strides = xtl::make_sequence<strides_type>(shape.size(), size_type(1));
            compute_strides(shape, l, strides);
            resize(shape, strides);
        }

        /**
         * Resizes the container.
         * @param shape the new shape
         * @param strides the new strides
         */
        template <class D>
        template <class S>
        inline void pycontainer<D>::resize(const S& shape, const strides_type& strides)
        {
            detail::check_dims<shape_type>::run(shape.size());
            derived_type tmp(xtl::forward_sequence<shape_type, decltype(shape)>(shape), strides);
            *static_cast<derived_type*>(this) = std::move(tmp);
        }

        template <class D>
        template <class S>
        inline auto& pycontainer<D>::reshape(S&& shape, layout_type layout) &
        {
            if (compute_size(shape) != this->size())
            {
                throw std::runtime_error("Cannot reshape with incorrect number of elements (" 
                    + std::to_string(this->size()) + " vs " + std::to_string(compute_size(shape)) + ")");
            }
            detail::check_dims<shape_type>::run(shape.size());
            
            // For nanobind, we need to create a new tensor with the reshaped view
            // This is different from pybind11 which uses PyArray_Newshape
            layout = default_assignable_layout(layout);
            
            strides_type new_strides = xtl::make_sequence<strides_type>(shape.size(), size_type(1));
            compute_strides(shape, layout, new_strides);
            
            // Create new tensor and copy data
            derived_type tmp(xtl::forward_sequence<shape_type, S>(std::forward<S>(shape)), new_strides);
            std::copy(this->storage().cbegin(), this->storage().cend(), tmp.storage().begin());
            *static_cast<derived_type*>(this) = std::move(tmp);
            
            return *this;
        }

        /**
         * Return the layout_type of the container
         * @return layout_type of the container
         */
        template <class D>
        inline layout_type pycontainer<D>::layout() const
        {
            if (detail::is_row_major(this->strides(), this->shape()))
            {
                return layout_type::row_major;
            }
            else if (detail::is_column_major(this->strides(), this->shape()))
            {
                return layout_type::column_major;
            }
            else
            {
                return layout_type::dynamic;
            }
        }

        /**
         * Return whether or not the container uses contiguous buffer
         * @return Boolean for contiguous buffer
         */
        template <class D>
        inline bool pycontainer<D>::is_contiguous() const noexcept
        {
            if (this->strides().size() == 0)
            {
                return true;
            }
            
            auto l = layout();
            if (l == layout_type::row_major)
            {
                return 1 == this->strides().back();
            }
            else if (l == layout_type::column_major)
            {
                return 1 == this->strides().front();
            }
            else
            {
                return false;
            }
        }

    } // namespace nanobind

} // namespace xt

#endif
