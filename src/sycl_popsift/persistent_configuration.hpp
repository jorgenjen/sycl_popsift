#pragma once

#include "sycl_popsift/use_root_group_macro.h"

#include <sycl/queue.hpp>

namespace popsift {

// Need to cleaan up these structs as alot of asttirbutes are not in use

struct sg_region_blocks
{
    // Full block dimensions
    int width;
    int height;

    // Information about remainder
    int bottom_row_height; // Uses width

#if !USE_ROOT_GROUP
    unsigned char* wg_sync_state; // work group status flags for synchronization
#endif

    // Uses remainder to determine which pixel each work item is responsible for
    // int col_pixel_length;     // How many pixels per SG (multiple of sg_widht)
    // int corner_pixel_length;  // starts at end of col_pixel for final SG
    // int second_corner_length; // Does full rows but not same as rest to not overload corner
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

    int x_remainder;
    int y_remainder;

    // To make it self contained
    void reconfigure(int width, int height);

#if USE_ROOT_GROUP
    persistent_pyramid_octave_config();
    persistent_pyramid_octave_config(int width, int height);
#else
    // unsigned char* wg_sync_state; // work group status flags for synchronization
    int persistent_sync_size;
    sycl::queue _device_queue; // Queue for managing the device malloced bool array

    persistent_pyramid_octave_config(sycl::queue Q);
    persistent_pyramid_octave_config(int width, int height, sycl::queue Q);

    ~persistent_pyramid_octave_config();
#endif

  private:
    inline void compute_size(int width, int height); // To not duplicate code
};

} // namespace popsift
