#pragma once

// #include "sycl/queue.hpp"

#include "assist.h" // For POP_FATAL

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

// Queue selection
sycl::queue initQueue();
// Matrix supported verification

inline bool supportsJointMatrixMatch(sycl::queue& Q)
{
#if !USE_JOINT_MATRIX
    return false; // As joint matrix is in this case not compiled
#endif
    sycl::device dev = Q.get_device();
    auto combinations = dev.get_info<sycl::ext::oneapi::experimental::info::device::matrix_combinations>();

    for(const auto& combo : combinations)
    {
        if(combo.atype == sycl::ext::oneapi::experimental::matrix::matrix_type::fp16 &&
           combo.btype == sycl::ext::oneapi::experimental::matrix::matrix_type::fp16 &&
           combo.ctype == sycl::ext::oneapi::experimental::matrix::matrix_type::fp32 &&
           combo.dtype == sycl::ext::oneapi::experimental::matrix::matrix_type::fp32 && combo.msize == 16 &&
           combo.nsize == 16 && combo.ksize == 16)
        {
            return true; // Found it! And we support current JointMatrix matching
        }
    }
    return false; // Not supoprting the used matrix type in matrix matching code
}

// struct sg_region_blocks
// {
//     // Full block dimensions
//     int width;
//     int height;
//
//     // Information about remainder
//     int bottom_row_height; // Uses width
//
//     // Not sure if it would be faster to compute this based on with and height in the kernel
//     int x_remainder;
//     int y_remainder;
//
//     // Uses remainder to determine which pixel each work item is responsible for
//     int col_pixel_length;     // How many pixels per SG (multiple of sg_widht)
//     int corner_pixel_length;  // starts at end of col_pixel for final SG
//     int second_corner_length; // Does full rows but not same as rest to not overload corner
// };
//
// struct persistent_pyramid_octave_config
// {
//     bool use_persistent_block;
//     sg_region_blocks sg_block;
//     // int num_corner_blocks; // From corner walking up column (per column in case of two)
//
//     // Might add this but I don't think it's needed
//     // int num_col_for_col;   // How many columns are used for dealing with remainder_column
//     sycl::range<2> global;
//     sycl::range<2> local;
//
//     // To make it self contained
//     void reconfigure(int width, int height);
//     persistent_pyramid_octave_config() = delete;
//
// #if !USE_ROOT_GROUP
//     bool* wg_sync_flags; // work group status flags for synchronization
//     int persistent_sync_size;
//     sycl::queue _device_queue; // Queue for managing the device malloced bool array
//
//     persistent_pyramid_octave_config(int width, int height, sycl::queue Q);
// #else
//     persistent_pyramid_octave_config(int width, int height);
// #endif
// };
//
// // This should be part of experimental as it is relying on root group
// persistent_pyramid_octave_config compute_persistent_sg_region_block(int width, int height);

}
