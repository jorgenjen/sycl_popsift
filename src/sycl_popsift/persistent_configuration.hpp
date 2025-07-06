#pragma once

#include "sycl_popsift/persistent_config_macros.h"

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
    // unsigned char* wg_sync_state; // work group status flags for synchronization

    // Need to be int for use of atomic_ref
    int* wg_sync_state; // work group status flags for synchronization
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

    // If we can use local mem for horiz and vert
    // Depends on how much shared memory that is available
    bool local_mem_horiz = false;       // Uses row prefetch should work on almost all GPU's
    bool local_mem_vert = false;        // Can store full window span * 2 or span*2 + 2 (based on if we do span or not)
    bool local_mem_buffer_vert = false; // When we can't store full vert window and hide latency by row prefetch
    int local_mem_size;                 // The version that had the largest memory need

    int x_remainder;
    int y_remainder;

    sycl::queue _device_queue; // for array when we have that and device info (mem size)
    // To make it self contained
    void reconfigure(int width, int height, int largest_span);

    persistent_pyramid_octave_config(sycl::queue Q);
    persistent_pyramid_octave_config(int width, int height, int largest_span, sycl::queue Q);

#if !USE_ROOT_GROUP
    // unsigned char* wg_sync_state; // work group status flags for synchronization
    int persistent_sync_size;
    sycl::event _zeroed_event; // To ensure the memset is done before we start incrementing

    ~persistent_pyramid_octave_config();
#endif

  private:
    inline void compute_size(int width, int height, int largest_span); // To not duplicate code
};

} // namespace popsift
