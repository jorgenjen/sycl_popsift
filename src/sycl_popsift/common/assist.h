#pragma once

#include "sycl/queue.hpp"
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

}
