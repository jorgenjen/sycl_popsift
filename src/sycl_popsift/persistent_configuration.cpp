
#include "sycl_popsift/persistent_configuration.hpp"

#include "sycl_popsift/popsift.hpp"
#include "sycl_popsift/use_root_group_macro.h"

namespace popsift {

#define DEBUGG_LOG 0
// Computes regions that allows the compute to be done in one wave. Computes the smallest regions per sub-group that
// results in one wave. computes a 2d block that is used for both horiz and vert
#if !USE_ROOT_GROUP
persistent_pyramid_octave_config::persistent_pyramid_octave_config(int width, int height, sycl::queue Q)
#else
persistent_pyramid_octave_config::persistent_pyramid_octave_config(int width, int height)
#endif
{
    // Compute based on num_cu and sg_per_cu

    // Minimum block is 32x13 (32 along x for contigous reads of one sg)
    // TODO: IT should be sub_group width for kernel x 13

    // IMPORTANT:
    // TODO: Figure out how many registers the kernel uses per thread and how many registers each thread can use to then
    // figure out the how many sub_groups that can reside on a compute_unit based on register usage Currently it assumes
    // that registers is not a bottleneck which might not always be the case
    // int sg_width = 32; // set to 32 for now
    int max_total_sg = PopSift::sg_per_cu * PopSift::num_cu;
    int max_sg_per_cu = PopSift::sg_per_cu; // Generic would like to figoure out for kernel specificaly if possible

    persistent_pyramid_octave_config sg_region;

    constexpr int start_height = 13; // This could start out as smaller but not sure how far down it is worth it to use
                                     // it (Could test and graph that and include in results)
    sg_region.sg_block.width = 32;   // Replace 32 with sg widht of device
    sg_region.sg_block.height = start_height;

    int x_blocks = width / sg_region.sg_block.width;
    int y_blocks;

    sg_region.sg_block.x_remainder = width % sg_region.sg_block.width;
    // int y_remainder;

    int total_x = sg_region.sg_block.x_remainder == 0 ? x_blocks : x_blocks + 1;
    int total_y;

    int total_blocks;
    int num_col_sg;
    int right_col_pixels;

    // We can cover a whole wave for this octave
    // Find max block size that covers a wave
    while(true) // Terminated in if
    {
        y_blocks = height / sg_region.sg_block.height;

        sg_region.sg_block.y_remainder = height % sg_region.sg_block.height;

        total_y = sg_region.sg_block.y_remainder == 0 ? y_blocks : y_blocks + 1;

        // if(sg_region.sg_block.x_remainder == 0 && sg_region.sg_block.y_remainder == 0)
        //     total_blocks = x_blocks * y_blocks;
        // else if(sg_region.sg_block.x_remainder == 0)
        //     total_blocks = x_blocks * (y_blocks + 1);
        // else if(sg_region.sg_block.x_remainder == 0)
        //     total_blocks = (x_blocks + 1) * y_blocks;
        // else
        //     total_blocks = (x_blocks + 1) * (y_blocks + 1);

        total_blocks = total_x * total_y;

        if(total_blocks <= max_total_sg)
        {
            if(sg_region.sg_block.height == start_height)
            {
                // Did not cover a full wave with initial block size hence not usnig
                sg_region.use_persistent_block = false;
                break;
            }
            sg_region.use_persistent_block = true;
// We have reached a block size that is large enough to cover no more than one wave
#if DEBUGG_LOG
            printf("WE DONE --> total_blocks = %d - Max_total_sg = %d -- num_col_sg = %d -- x_blocks = %d -- "
                   "main_region = %d -- "
                   "x_remainder = %d -- y_remainder = %d\n",
                   total_blocks,
                   max_total_sg,
                   num_col_sg,
                   x_blocks,
                   x_blocks * y_blocks,
                   sg_region.sg_block.x_remainder,
                   sg_region.sg_block.y_remainder);
#endif

            break;
        }
        sg_region.sg_block.height++;
    }

    if(sg_region.use_persistent_block)
    {
        // Use simple wraping for remainder column poor coaleced reads but each work-item is used at all times besides
        // for corner with this division
        // Look at bottom for file for initial outline of a more coaleced way of spliting the column work

        // Could use second column to do remainder column if we have enough free sub_groups

        // int total_col_pixels = x_remainder * height;

        sg_region.sg_block.bottom_row_height = height % sg_region.sg_block.height;

        if(sg_region.sg_block.x_remainder != 0)
        {
            int total_col_pixels = sg_region.sg_block.x_remainder * height;

            int total_full_width = total_col_pixels / sg_region.sg_block.width;
            int corner_pixels = total_col_pixels % sg_region.sg_block.width;

            int col_sg_full_width = total_full_width / y_blocks;
            int corner_full_width = total_full_width % y_blocks;

            sg_region.sg_block.col_pixel_length = col_sg_full_width * sg_region.sg_block.width;
            sg_region.sg_block.corner_pixel_length = corner_full_width * sg_region.sg_block.width + corner_pixels;

#if DEBUGG_LOG
            printf(
              "\n col_pixels = %d -- col_pixel_length = %d -- corner_pixel = %d -- total_col_pixels = %d -- Normal "
              "block pixel count = %d\n\t Corner_pixels = %d -- corner_full_widht = %d -- col_sg_full_width = %d\n",
              total_col_pixels,
              sg_region.sg_block.col_pixel_length,
              sg_region.sg_block.corner_pixel_length,
              total_col_pixels,
              sg_region.sg_block.width * sg_region.sg_block.height,
              corner_pixels,
              corner_full_width,
              col_sg_full_width);
#endif

            if(sg_region.sg_block.corner_pixel_length > sg_region.sg_block.col_pixel_length)
            {
                // Try again but this time using two sub_groups for corner
                col_sg_full_width = total_full_width / (y_blocks - 1); // One less static_cast<size_t>(given to corn)er
                corner_full_width = total_full_width % (y_blocks - 1);

                sg_region.sg_block.col_pixel_length = col_sg_full_width * sg_region.sg_block.width;
                sg_region.sg_block.second_corner_length =
                  ((corner_full_width + 1) / 2) * sg_region.sg_block.width; // Posetive integer ceil;
                sg_region.sg_block.corner_pixel_length =
                  ((corner_full_width / 2) * sg_region.sg_block.width) + corner_pixels; // Floor division
            }
            else
            {
                // Only need one corner
                sg_region.sg_block.second_corner_length = 0; // Meaning it's not in use
            }

#if DEBUGG_LOG
            printf(
              "\n col_pixels = %d -- col_pixel_length = %d -- corner_pixel = %d -- total_col_pixels = %d -- Normal "
              "block pixel count = %d\n\t Corner_pixels = %d -- corner_full_widht = %d -- col_sg_full_width = %d -- "
              "second_corner_length = %d \n",
              total_col_pixels,
              sg_region.sg_block.col_pixel_length,
              sg_region.sg_block.corner_pixel_length,
              total_col_pixels,
              sg_region.sg_block.width * sg_region.sg_block.height,
              corner_pixels,
              corner_full_width,
              col_sg_full_width,
              sg_region.sg_block.second_corner_length);
#endif
        }
        else
        {
            sg_region.sg_block.col_pixel_length = 0;
            sg_region.sg_block.corner_pixel_length = 0;
            sg_region.sg_block.second_corner_length = 0;
        }

#if DEBUGG_LOG
        printf("x_blocks = %d -- y_blocks = %d\n", x_blocks, y_blocks);
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

#if DEBUGG_LOG
        printf("lead_sg_count = %d -- lead_x = %d -- lead_y = %d -- free_sg_per_cu = %d -- max_total_sg = %d\n",
               lead_sg_count,
               lead_x,
               lead_y,
               free_sg_per_cu,
               max_total_sg);
#endif

        sg_region.local = {static_cast<size_t>(lead_y), static_cast<size_t>(lead_x * sg_region.sg_block.width)};
        sg_region.global = {static_cast<size_t>(total_y), static_cast<size_t>(total_x * sg_region.sg_block.width)};

#if DEBUGG_LOG
        printf("\n\nWidth = %d height= %d --> normal_block (%d, %d), final_row_height = %d -- col_pixel_length = %d -- "
               "corner_pixel = %d -- corner_2_pixel = %d\n",
               width,
               height,
               sg_region.sg_block.width,
               sg_region.sg_block.height,
               sg_region.sg_block.bottom_row_height,
               sg_region.sg_block.col_pixel_length,
               sg_region.sg_block.corner_pixel_length,
               sg_region.sg_block.second_corner_length);

        printf("\tLocal (%zu, %zu) -- Global (%zu, %zu)\n",
               sg_region.local[0],
               sg_region.local[1],
               sg_region.global[0],
               sg_region.global[1]);
        printf("\ty_remainder = %d == %d\n\n\n", sg_region.sg_block.y_remainder, sg_region.sg_block.bottom_row_height);
#endif
    }

    return sg_region;
}
}
