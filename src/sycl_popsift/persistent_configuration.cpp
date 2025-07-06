
#include "sycl_popsift/persistent_configuration.hpp"

#include "sycl_popsift/common/debug_macros.hpp"
#include "sycl_popsift/persistent_config_macros.h"
#include "sycl_popsift/popsift.hpp"

#include <cstdio>
#include <iostream>

namespace popsift {

#define DEBUGG_LOG 0

#if USE_PERSISTENT
// inline void persistent_pyramid_octave_config::compute_size(int height, int width)

inline void persistent_pyramid_octave_config::compute_size(int width, int height, int largest_span)
{
    // TODO: IT should be sub_group width for kernel x 13

    // IMPORTANT:
    // TODO: Figure out how many registers the kernel uses per thread and how many registers each thread can use to then
    // figure out the how many sub_groups that can reside on a compute_unit based on register usage Currently it assumes
    // that registers is not a bottleneck which might not always be the case
    // int sg_width = 32; // set to 32 for now

    int max_total_sg = PopSift::sg_per_cu * PopSift::num_cu;
    int max_sg_per_cu = PopSift::sg_per_cu; // Generic would like to figoure out for kernel specificaly if possible

    // This could start out as smaller but not sure how far down it is worth it to use
    // it (Could test and graph that and include in results)
    int start_height =
      largest_span + 1; // makes first vert iteration of vert safe to load in above without checking bounds

    sg_block.width = 32; // Replace 32 with sg widht of device
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

#if MULTI_ROW_WG
    int lead_sg_count = 0;
    int lead_x = 0;
    int lead_y = 0;
    for(int x = 1; x <= 8; x++)
    {
        if(total_x % x != 0)
            continue;

        for(int y = 1; y <= 8; y++)
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

    int wg_per_cu = max_sg_per_cu / lead_sg_count; // Number of work_groups per compute unit
#else

    // Async_wrok_group copy only works when load is identical for all work_items in work_group hence multiple rows does
    // not work(or becomes difficult to program (not too bad just set stride and they could load in one after the other
    // for rows results in less intuitive memory layout)

    // Easier to limit our selves to one row per work-group will result in more work_group synchronization which is not
    // great and less chance for cache hits (most likely)

    // This configuration results in larger vert memory usage per compute unit (should still work most cases)
    int lead_x = 0;
    for(int x = 1; x <= 32; x++)
    {
        if(total_x % x != 0)
            continue;
        int sg_count = x;

        if(max_sg_per_cu % x <= free_sg_per_cu)
        {
            // accepted
            lead_x = x;
        }
    }

    local = {1, static_cast<size_t>(lead_x * sg_block.width)};
    global = {static_cast<size_t>(total_y), static_cast<size_t>(total_x * sg_block.width)};

    int wg_per_cu = max_sg_per_cu / lead_x; // Number of work_groups per compute unit
#endif

    auto device = _device_queue.get_device();
    int device_local_mem_size = static_cast<float>(device.get_info<sycl::info::device::local_mem_size>());

    //
    // each sub_group uses a sliding window of height (span * 2 +1) and widht of sg_widht. one row is added for async
    // loading of next. loca([0])*local[1]) final part is for async loading prev level for doing Difference of Gaussian
    // (DoG) on the fly to reuse data better

#define SKIP_SPAN 1 // Skipping doing iteration where offset == span as filter[span] == 0 hence does not change result
#if MINIMAL_WINDOW
    // Remove part of window that is equal to sspan as filter[span] seems to always be zero
    // largest_span << 1 is for whole thing as dist around self is span - 1 and self is one and buffer row is 1
    // So same as ((largest_span - 1) << 1) + 1 + 1;

    int vert_local_size = ((largest_span << 1) * local[1]) * local[0] + local[0] * local[1];
#else
// +1 is for self; second + 1 is for free buffer row; largest_span * 2 is for span range around self
#if SKIP_SPAN
    int vert_local_size = (((largest_span << 1) * local[1]) * local[0] + local[0] * local[1]) * sizeof(float);
#else
    int vert_local_size = ((((largest_span << 1) + 1 + 1) * local[1]) * local[0] + local[0] * local[1]) * sizeof(float);
#endif
#endif

    // Used when we don't have enough memory for a full window and must use prefetched rows instead
    // -> Two rows for current work and four rows to prefetch next two (so we have two current async rows loading in)
    // -> final part of sum is for DoG so we can load in prev level intermediate and do DoG on the fly and write that
    // back in addition to intermediate
    // --> For final level we don't even need to write back intermediate and can only write back DoG as once we have DoG
    // we never use the Data Array again as we only use DoG aray
    int vert_buffer_local_size = ((local[1] * 6) * local[0] + local[0] * local[1]) * sizeof(float);

    // The size of this one is quite constant with respect to image sizes so should work on most GPU's
    // Also quite constant with respect to number of Compute Units on the GPU as it's a sliding window
    // Takes around 150-180k bytes which is too much as max tends to be around 100k bytes

    // Should add minimal here aswell when I support that in the horiz part as I believe that the same is true there
    int horiz_local_size =
      (((largest_span << 1) + local[1]) * (local[0] * 2)) * sizeof(float); // Buffering of horiz rows

    // int horiz_local_isze = ((largest_span * 2) + local[1]) * 2) *local[]

    std::printf("w=%d h=%d largest_pan = %d wg_per_cu = %d --> Vert_local_size = %d ---- Horiz_local_size = %d "
                "local(%zu, %zu) global(%zu, %zu) -- vert_buffer_local_size = %d -- Local_mem_size = %d\n",
                width,
                height,
                largest_span,
                wg_per_cu,
                vert_local_size,
                horiz_local_size,
                local[0],
                local[1],
                global[0],
                global[1],
                vert_buffer_local_size,
                device_local_mem_size);
    int max_mem_per_wg = device_local_mem_size / wg_per_cu; // Assumes local mem is per CU which it is for GPU's

    // local_mem_vert = (max_mem_per_wg >= vert_local_size) &&  ((largest_span << 1) + 1)   // If doing more complex
    // vert start for better cache we need more things to be true

    local_mem_vert = max_mem_per_wg >= vert_local_size;
    local_mem_buffer_vert =
      local_mem_vert ? false : max_mem_per_wg >= vert_buffer_local_size; // Set to false when window works

    local_mem_horiz = max_mem_per_wg >= horiz_local_size;

    if(local_mem_horiz && local_mem_vert)
    {
        local_mem_size = sycl::max(vert_local_size, horiz_local_size);
    }
    else if(!local_mem_vert && local_mem_buffer_vert)
    {
        local_mem_size = local_mem_horiz ? sycl::max(horiz_local_size, vert_buffer_local_size) : vert_buffer_local_size;
    }
    else if(local_mem_horiz)
    {
        local_mem_size = horiz_local_size;
    }
    else if(local_mem_vert)
    {
        // Unlikely to run
        local_mem_size = vert_local_size;
    }
    else
    {
        local_mem_size = 0; // No shared memory used in this case
    }

    std::printf("Final local mem size = %d -- local(%zu, %zu)\n", local_mem_size, local[0], local[1]);

    // }
}

// Basic contruructors
persistent_pyramid_octave_config::persistent_pyramid_octave_config(sycl::queue Q)
  : _device_queue(Q)
  , use_persistent_block(false)
  , persistent_sync_size(0)
{}

// Constructors hat compute (not in use currently)
// Computes regions that allows the compute to be done in one wave. Computes the smallest regions per sub-group that
// results in one wave. computes a 2d block that is used for both horiz and vert
persistent_pyramid_octave_config::persistent_pyramid_octave_config(int width,
                                                                   int height,
                                                                   int largest_span,
                                                                   sycl::queue Q)
  : _device_queue(Q)
  , use_persistent_block(false)
  , persistent_sync_size(0)
{
    reconfigure(width, height, largest_span);
}

void persistent_pyramid_octave_config::reconfigure(int width, int height, int largest_span)
{
    // printf("REconfigurint this persistent config!! \n\n");
    compute_size(width, height, largest_span);

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

    // Figure out the maximum shared memory size per work_group

#if !USE_ROOT_GROUP
    if(use_persistent_block)
    {
        sycl::range<2> work_group_grid = global / local;
        size_t wg_grid_size = work_group_grid.size();
        if(wg_grid_size > persistent_sync_size)
        {
            sycl::free(sg_block.wg_sync_state, _device_queue);
            sg_block.wg_sync_state =
              sycl_common::malloc_devT<int>(work_group_grid.size(),
                                            __FILE__,
                                            __LINE__,
                                            "Failed to allocate persistent blocks synchronization array",
                                            _device_queue);
            _zeroed_event = _device_queue.memset(sg_block.wg_sync_state, 0, sizeof(int) * wg_grid_size);

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
