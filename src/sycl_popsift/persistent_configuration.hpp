#pragma once

#include <sycl/queue.hpp>

namespace popsift {

struct sg_region_blocks
{
    // Full block dimensions
    int width;
    int height;

    // Information about remainder
    int bottom_row_height; // Uses width

    // Not sure if it would be faster to compute this based on with and height in the kernel
    int x_remainder;
    int y_remainder;

    // Uses remainder to determine which pixel each work item is responsible for
    int col_pixel_length;     // How many pixels per SG (multiple of sg_widht)
    int corner_pixel_length;  // starts at end of col_pixel for final SG
    int second_corner_length; // Does full rows but not same as rest to not overload corner
};

struct persistent_pyramid_octave_config
{
    bool use_persistent_block;
    sg_region_blocks sg_block;
    // int num_corner_blocks; // From corner walking up column (per column in case of two)

    // Might add this but I don't think it's needed
    // int num_col_for_col;   // How many columns are used for dealing with remainder_column
    sycl::range<2> global;
    sycl::range<2> local;

    // To make it self contained
    void reconfigure(int width, int height);
    persistent_pyramid_octave_config() = delete;

#if !USE_ROOT_GROUP
    bool* wg_sync_flags; // work group status flags for synchronization
    int persistent_sync_size;
    sycl::queue _device_queue; // Queue for managing the device malloced bool array

    persistent_pyramid_octave_config(int width, int height, sycl::queue Q);
#else
    persistent_pyramid_octave_config(int width, int height);
#endif
};

} // namespace popsift
