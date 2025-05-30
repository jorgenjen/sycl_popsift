#pragma once

// #include "sycl/queue.hpp"

#include <sycl/device.hpp>
#include <sycl/kernel.hpp>
#include <sycl/kernel_bundle.hpp>
#include <sycl/queue.hpp>

namespace popsift {

/* This computation is needed very frequently when a dim3 grid block is
 * initialized. It ensure that the tail is not forgotten.
 */
inline int grid_divide_cuda(int size, int divider) { return size / divider + (size % divider != 0 ? 1 : 0); }
// same as above just for sycl. Global and local. Local must perfectly divide global and hence we need to ensure
// that tail is included if it exists

// TODO: Change this to return size_t as it is mainly used in range (global)
inline int grid_divide(int size, int divider)
{
    return size % divider != 0 ? size + (divider - (size % divider)) : size;
}

inline size_t getPreferredAlignment(sycl::queue& q)
{
    // Query the device's preferred alignment (in bits, convert to bytes)
    auto align_bits = q.get_device().get_info<sycl::info::device::mem_base_addr_align>();
    return align_bits / 8;
}

// USAGE INFO
// need t oassign a struct or class that you declare before the kernel launch
// and use for the class and use in the kernel lauch as the template argument
// kernel_lauch is tied to a launch of a kernel not a kernel name (sub_groups)
// could vary from launch of same kernel (especially when templated)

// Fetches the actual sub_group size that will be used for the ori_par kernel
template<class kernel_launch>
inline auto get_kernel_subgroup_size(sycl::queue& Q) // not sure if we want to inline
{
    auto kernel_id = sycl::get_kernel_id<kernel_launch>();
    auto kernel_bundle = sycl::get_kernel_bundle<sycl::bundle_state::executable>(Q.get_context());
    auto kernel = kernel_bundle.get_kernel(kernel_id);
    return kernel.template get_info<sycl::info::kernel_device_specific::max_sub_group_size>(Q.get_device());
}
}
