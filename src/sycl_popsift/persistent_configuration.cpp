
#include "sycl_popsift/persistent_configuration.hpp"

#include "sycl_popsift/common/debug_macros.hpp"
#include "sycl_popsift/popsift.hpp"
#include "sycl_popsift/use_root_group_macro.h"

#include <cstdio>

namespace popsift {

#define DEBUGG_LOG 0

#if USE_PERSISTENT
// inline void persistent_pyramid_octave_config::compute_size(int height, int width)
inline void persistent_pyramid_octave_config::compute_size(int width, int height)
{
    // TODO: IT should be sub_group width for kernel x 13

    // IMPORTANT:
    // TODO: Figure out how many registers the kernel uses per thread and how many registers each thread can use to then
    // figure out the how many sub_groups that can reside on a compute_unit based on register usage Currently it assumes
    // that registers is not a bottleneck which might not always be the case
    // int sg_width = 32; // set to 32 for now

    int max_total_sg = PopSift::sg_per_cu * PopSift::num_cu;
    int max_sg_per_cu = PopSift::sg_per_cu; // Generic would like to figoure out for kernel specificaly if possible

    constexpr int start_height = 13; // This could start out as smaller but not sure how far down it is worth it to use
                                     // it (Could test and graph that and include in results)
    sg_block.width = 32;             // Replace 32 with sg widht of device
    sg_block.height = start_height;

    int x_blocks = width / sg_block.width;
    int y_blocks;

    x_remainder = width % sg_block.width;

    int total_x = x_remainder == 0 ? x_blocks : x_blocks + 1;
    int total_y;

    int total_blocks;
    int num_col_sg;
    int right_col_pixels;

    // We can cover a whole wave for this octave
    // Find max block size that covers a wave
    while(true) // Terminated in if
    {
        y_blocks = height / sg_block.height;

        y_remainder = height % sg_block.height;

        total_y = y_remainder == 0 ? y_blocks : y_blocks + 1;

        total_blocks = total_x * total_y;

        if(total_blocks <= max_total_sg)
        {
            if(sg_block.height == start_height)
            {
                // Did not cover a full wave with initial block size hence not usnig
                use_persistent_block = false;
                // break;
                return;
            }
            use_persistent_block = true;
            // We have reached a block size that is large enough to cover no more than one wave

            break;
        }
        sg_block.height++;
    }

    // if(use_persistent_block) // Not needed as we return when that is not the case
    // {
// for more complex and probably less efficient scheduling requiering more code in kernel
#if false
    // Use simple wraping for remainder column poor coaleced reads but each work-item is used at all times besides
    // for corner with this division
    // Look at bottom for file for initial outline of a more coaleced way of spliting the column work

    // Could use second column to do remainder column if we have enough free sub_groups

    // int total_col_pixels = x_remainder * height;

    sg_block.bottom_row_height = height % sg_block.height;

    if(x_remainder != 0)
    {
        int total_col_pixels = x_remainder * height;

        int total_full_width = total_col_pixels / sg_block.width;
        int corner_pixels = total_col_pixels % sg_block.width;

        int col_sg_full_width = total_full_width / y_blocks;
        int corner_full_width = total_full_width % y_blocks;

        sg_block.col_pixel_length = col_sg_full_width * sg_block.width;
        sg_block.corner_pixel_length = corner_full_width * sg_block.width + corner_pixels;

        if(sg_block.corner_pixel_length > sg_block.col_pixel_length)
        {
            // Try again but this time using two sub_groups for corner
            col_sg_full_width = total_full_width / (y_blocks - 1); // One less static_cast<size_t>(given to corn)er
            corner_full_width = total_full_width % (y_blocks - 1);

            sg_block.col_pixel_length = col_sg_full_width * sg_block.width;
            sg_block.second_corner_length = ((corner_full_width + 1) / 2) * sg_block.width; // Posetive integer ceil;
            sg_block.corner_pixel_length = ((corner_full_width / 2) * sg_block.width) + corner_pixels; // Floor division
        }
        else
        {
            // Only need one corner
            sg_block.second_corner_length = 0; // Meaning it's not in use
        }
    }
    else
    {
        sg_block.col_pixel_length = 0;
        sg_block.corner_pixel_length = 0;
        sg_block.second_corner_length = 0;
    }
#endif

    // Figure out global and local
    // Want to use work_groups to ensure SG located in neigbourhood are on same CU allowing for better L1
    // utilization

    // Find biggest functioning work_group that allows for full occupancy in our configuration

    int free_sg_per_cu = (max_total_sg - (x_blocks + 1) * (y_blocks + 1)) / PopSift::num_cu;

    int lead_sg_count = 0;
    int lead_x = 0;
    int lead_y = 0;
    for(int x = 1; x < 8; x++)
    {
        if(total_x % x != 0)
            continue;

        for(int y = 1; y < 8; y++)
        {
            int sg_count = x * y;
            if(total_y % y != 0)
                continue;

            if(max_sg_per_cu % sg_count <= free_sg_per_cu)
            {
                // Accepted
                if(lead_sg_count < sg_count)
                {
                    lead_sg_count = sg_count;
                    lead_x = x;
                    lead_y = y;
                }
            }
        }
    }

    local = {static_cast<size_t>(lead_y), static_cast<size_t>(lead_x * sg_block.width)};
    global = {static_cast<size_t>(total_y), static_cast<size_t>(total_x * sg_block.width)};
    // }
}

// Basic contruructors
#if USE_ROOT_GROUP
persistent_pyramid_octave_config::persistent_pyramid_octave_config()
  : use_persistent_block(false)
{}
#else
persistent_pyramid_octave_config::persistent_pyramid_octave_config(sycl::queue Q)
  : _device_queue(Q)
  , use_persistent_block(false)
  , persistent_sync_size(0)
{}
#endif

// Constructors hat compute (not in use currently)
// Computes regions that allows the compute to be done in one wave. Computes the smallest regions per sub-group that
// results in one wave. computes a 2d block that is used for both horiz and vert
#if USE_ROOT_GROUP
persistent_pyramid_octave_config::persistent_pyramid_octave_config(int width, int height)
#else
persistent_pyramid_octave_config::persistent_pyramid_octave_config(int width, int height, sycl::queue Q)
  : _device_queue(Q)
  , persistent_sync_size(0)
#endif
{
    reconfigure(width, height);
}

void persistent_pyramid_octave_config::reconfigure(int width, int height)
{
    // printf("REconfigurint this persistent config!! \n\n");
    compute_size(width, height);

    // fprintf(stderr,
    //         "w=%d h=%d -->Local(%zu, %zu) -- global (%zu, %zu) -- sg_region --  width = %d - height = %d -- "
    //         "x_remainder = %d -- "
    //         "y_remainder = %d\n\n",
    //         width,
    //         height,
    //         local[0],
    //         local[1],
    //         global[0],
    //         global[1],
    //         sg_block.width,
    //         sg_block.height,
    //         x_remainder,
    // y_remainder);

#if !USE_ROOT_GROUP
    if(use_persistent_block)
    {
        sycl::range<2> work_group_grid = global / local;
        size_t wg_grid_size = work_group_grid.size();
        if(wg_grid_size > persistent_sync_size)
        {
            sycl::free(sg_block.wg_sync_state, _device_queue);
            sg_block.wg_sync_state =
              sycl_common::malloc_devT<unsigned char>(work_group_grid.size(),
                                                      __FILE__,
                                                      __LINE__,
                                                      "Failed to allocate persistent blocks synchronization array",
                                                      _device_queue);

            persistent_sync_size = wg_grid_size; // Update value
        }
    }
#endif
}

#if !USE_ROOT_GROUP
persistent_pyramid_octave_config::~persistent_pyramid_octave_config()
{
    if(sg_block.wg_sync_state)
    {
        sycl::free(sg_block.wg_sync_state, _device_queue);
    }
}
#endif

#endif // if USE_PERSISTENT

} // namespace popsift
