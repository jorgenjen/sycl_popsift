#include "sycl_popsift/common/assist.h"

#include "sycl_popsift/common/debug_macros.hpp" // For POP_FATAL
#include "sycl_popsift/popsift.hpp"

namespace popsift {

sycl::queue initQueue()
{
#if QUEUE_PROFILING
    sycl::property_list queue_proplist = sycl::property_list{sycl::property::queue::enable_profiling{}};
#else
    sycl::property_list queue_proplist = {};
#endif

#ifndef CPU_ONLY
    // should probably also have a compile time flag --experimental to enable this feature
    if constexpr(USE_BINDLESS_INPUT && USE_BINDLESS_ARRAY &&
                 sycl::any_device_has<sycl::aspect::ext_oneapi_bindless_images>() &&
                 sycl::any_device_has<sycl::aspect::ext_oneapi_bindless_sampled_image_fetch_2d>() &&
                 sycl::any_device_has<sycl::aspect::ext_oneapi_image_array>())
    {
        // Running with bindless image -- need to find gpu with that aspect (needed incase of multi gpu system)

        for(sycl::device dev : sycl::device::get_devices(sycl::info::device_type::gpu))
        {
            // Find GPU with the aspect (incase of multigpu system)
            if(dev.has(sycl::aspect::ext_oneapi_bindless_images) && dev.has(sycl::aspect::ext_oneapi_image_array) &&
               dev.has(sycl::aspect::ext_oneapi_bindless_sampled_image_fetch_2d))
            {
                std::cout << "Running on: " << dev.get_info<sycl::info::device::name>() << std::endl
                          << "\t--> supports ext_oneapi_bindless_images: YES" << std::endl
                          << "\t--> supports ext_oneapi_image_array: YES" << std::endl;

                // We always select first gpu that had the aspect (might be a way to select the best one)
                // but most systems will be single gpu anyways
                return sycl::queue(sycl::context{dev}, dev, queue_proplist);
            }
        }
        // Did not return hence we did not find a matching device to that was on the compiled system
        POP_FATAL("Could not find device with support for  ext_oneapi_bindless_images and ext_oneapi_image_array "
                  "Such a device was available at compile time... Please re-compile")
    }
    else if constexpr(USE_BINDLESS_INPUT && sycl::any_device_has<sycl::aspect::ext_oneapi_bindless_images>() &&
                      sycl::any_device_has<sycl::aspect::ext_oneapi_bindless_sampled_image_fetch_2d>())
    {
        // In case it only supports bindless we can use it for upscaling still
        for(sycl::device dev : sycl::device::get_devices(sycl::info::device_type::gpu))
        {
            // Find GPU with the aspect (incase of multigpu system)
            if(dev.has(sycl::aspect::ext_oneapi_bindless_images) &&
               dev.has(sycl::aspect::ext_oneapi_bindless_sampled_image_fetch_2d))
            {
                std::cout << "Running on: " << dev.get_info<sycl::info::device::name>() << std::endl
                          << "\t--> supports ext_oneapi_bindless_images: YES" << std::endl
                          << "\t--> supports ext_oneapi_image_array: "
                          << (dev.has(sycl::aspect::ext_oneapi_image_array)
                                ? "YES... But not in use due to being NO at compile time..."
                                : "NO")
                          << std::endl
                          << std::endl;

                return sycl::queue(sycl::context{dev}, dev, queue_proplist);
                // We always select first gpu that had the aspect (might be a way to select the best one)
                // but most systems will be single gpu anyways
            }
        }

        // Did not return hence we did not find a matching device to that was on the compiled system
        POP_FATAL("Could not find device with ext_oneapi_bindless_images support... Such a device was  available "
                  "at compile time... Please re-compile")
    }
    else
    {
        try
        {
            // Did not have bindless aspect during compile time so we just try to select any GPU
            // If there is no GPU it will throw exception and use CPU in catch

            sycl::device dev = sycl::device{sycl::gpu_selector_v};
            return sycl::queue(sycl::context{dev}, dev, queue_proplist);
        }
        catch(sycl::exception const& ex)
        {
            std::cout << "No GPU found falling back to CPU... Exception thrown: " << ex.what() << std::endl;

            // Could use defualt selector but not sure how would handle fpga... hence cpu selector
            sycl::device dev = sycl::device{sycl::cpu_selector_v};
            return sycl::queue(sycl::context{dev}, dev);
        }
    }

    // Could go back to using runtime selection of bindless or not, but using compiletime for now. Makes it less
    // portable but don't think it's that portable between systems anyways... (Without compiling on the system
    // ofcourse)

#else
    fprintf(stderr, "Running in CPU_ONLY mode\n");
    try
    {
        // For in order queue use this (usefull for debugging)
        // sycl::device cpu_dev = sycl::device{sycl::cpu_selector_v};
        // _device_queue = sycl::queue(
        //   cpu_dev, sycl::property_list{sycl::property::queue::in_order{},
        //   sycl::property::queue::enable_profiling{}});

        sycl::device dev = sycl::device{sycl::cpu_selector_v};
        _device_queue = sycl::queue(sycl::context{dev}, dev, queue_proplist);
    }
    catch(const sycl::exception& e)
    {
        std::cerr << "Failed to create CPU queue: " << e.what() << std::endl;
        throw;
    }

    // _device_queue = std::make_shared<sycl::queue>(sycl::context{dev}, dev);
#endif
}

#define DEBUGG_LOG 0
// Computes regions that allows the compute to be done in one wave. Computes the smallest regions per sub-group that
// results in one wave. computes a 2d block that is used for both horiz and vert
// persistent_pyramid_octave_config compute_persistent_sg_region_block(int width, int height)
// {
//     // Compute based on num_cu and sg_per_cu
//
//     // Minimum block is 32x13 (32 along x for contigous reads of one sg)
//     // TODO: IT should be sub_group width for kernel x 13
//
//     // IMPORTANT:
//     // TODO: Figure out how many registers the kernel uses per thread and how many registers each thread can use to
//     then
//     // figure out the how many sub_groups that can reside on a compute_unit based on register usage Currently it
//     assumes
//     // that registers is not a bottleneck which might not always be the case
//     // int sg_width = 32; // set to 32 for now
//     int max_total_sg = PopSift::sg_per_cu * PopSift::num_cu;
//     int max_sg_per_cu = PopSift::sg_per_cu; // Generic would like to figoure out for kernel specificaly if possible
//
//     persistent_pyramid_octave_config sg_region;
//
//     constexpr int start_height = 13; // This could start out as smaller but not sure how far down it is worth it to
//     use
//                                      // it (Could test and graph that and include in results)
//     sg_region.sg_block.width = 32;   // Replace 32 with sg widht of device
//     sg_region.sg_block.height = start_height;
//
//     int x_blocks = width / sg_region.sg_block.width;
//     int y_blocks;
//
//     sg_region.sg_block.x_remainder = width % sg_region.sg_block.width;
//     // int y_remainder;
//
//     int total_x = sg_region.sg_block.x_remainder == 0 ? x_blocks : x_blocks + 1;
//     int total_y;
//
//     int total_blocks;
//     int num_col_sg;
//     int right_col_pixels;
//
//     // We can cover a whole wave for this octave
//     // Find max block size that covers a wave
//     while(true) // Terminated in if
//     {
//         y_blocks = height / sg_region.sg_block.height;
//
//         sg_region.sg_block.y_remainder = height % sg_region.sg_block.height;
//
//         total_y = sg_region.sg_block.y_remainder == 0 ? y_blocks : y_blocks + 1;
//
//         // if(sg_region.sg_block.x_remainder == 0 && sg_region.sg_block.y_remainder == 0)
//         //     total_blocks = x_blocks * y_blocks;
//         // else if(sg_region.sg_block.x_remainder == 0)
//         //     total_blocks = x_blocks * (y_blocks + 1);
//         // else if(sg_region.sg_block.x_remainder == 0)
//         //     total_blocks = (x_blocks + 1) * y_blocks;
//         // else
//         //     total_blocks = (x_blocks + 1) * (y_blocks + 1);
//
//         total_blocks = total_x * total_y;
//
//         if(total_blocks <= max_total_sg)
//         {
//             if(sg_region.sg_block.height == start_height)
//             {
//                 // Did not cover a full wave with initial block size hence not usnig
//                 sg_region.use_persistent_block = false;
//                 break;
//             }
//             sg_region.use_persistent_block = true;
// // We have reached a block size that is large enough to cover no more than one wave
// #if DEBUGG_LOG
//             printf("WE DONE --> total_blocks = %d - Max_total_sg = %d -- num_col_sg = %d -- x_blocks = %d -- "
//                    "main_region = %d -- "
//                    "x_remainder = %d -- y_remainder = %d\n",
//                    total_blocks,
//                    max_total_sg,
//                    num_col_sg,
//                    x_blocks,
//                    x_blocks * y_blocks,
//                    sg_region.sg_block.x_remainder,
//                    sg_region.sg_block.y_remainder);
// #endif
//
//             break;
//         }
//         sg_region.sg_block.height++;
//     }
//
//     if(sg_region.use_persistent_block)
//     {
//         // Use simple wraping for remainder column poor coaleced reads but each work-item is used at all times
//         besides
//         // for corner with this division
//         // Look at bottom for file for initial outline of a more coaleced way of spliting the column work
//
//         // Could use second column to do remainder column if we have enough free sub_groups
//
//         // int total_col_pixels = x_remainder * height;
//
//         sg_region.sg_block.bottom_row_height = height % sg_region.sg_block.height;
//
//         if(sg_region.sg_block.x_remainder != 0)
//         {
//             int total_col_pixels = sg_region.sg_block.x_remainder * height;
//
//             int total_full_width = total_col_pixels / sg_region.sg_block.width;
//             int corner_pixels = total_col_pixels % sg_region.sg_block.width;
//
//             int col_sg_full_width = total_full_width / y_blocks;
//             int corner_full_width = total_full_width % y_blocks;
//
//             sg_region.sg_block.col_pixel_length = col_sg_full_width * sg_region.sg_block.width;
//             sg_region.sg_block.corner_pixel_length = corner_full_width * sg_region.sg_block.width + corner_pixels;
//
// #if DEBUGG_LOG
//             printf(
//               "\n col_pixels = %d -- col_pixel_length = %d -- corner_pixel = %d -- total_col_pixels = %d -- Normal "
//               "block pixel count = %d\n\t Corner_pixels = %d -- corner_full_widht = %d -- col_sg_full_width = %d\n",
//               total_col_pixels,
//               sg_region.sg_block.col_pixel_length,
//               sg_region.sg_block.corner_pixel_length,
//               total_col_pixels,
//               sg_region.sg_block.width * sg_region.sg_block.height,
//               corner_pixels,
//               corner_full_width,
//               col_sg_full_width);
// #endif
//
//             if(sg_region.sg_block.corner_pixel_length > sg_region.sg_block.col_pixel_length)
//             {
//                 // Try again but this time using two sub_groups for corner
//                 col_sg_full_width = total_full_width / (y_blocks - 1); // One less static_cast<size_t>(given to
//                 corn)er corner_full_width = total_full_width % (y_blocks - 1);
//
//                 sg_region.sg_block.col_pixel_length = col_sg_full_width * sg_region.sg_block.width;
//                 sg_region.sg_block.second_corner_length =
//                   ((corner_full_width + 1) / 2) * sg_region.sg_block.width; // Posetive integer ceil;
//                 sg_region.sg_block.corner_pixel_length =
//                   ((corner_full_width / 2) * sg_region.sg_block.width) + corner_pixels; // Floor division
//             }
//             else
//             {
//                 // Only need one corner
//                 sg_region.sg_block.second_corner_length = 0; // Meaning it's not in use
//             }
//
// #if DEBUGG_LOG
//             printf(
//               "\n col_pixels = %d -- col_pixel_length = %d -- corner_pixel = %d -- total_col_pixels = %d -- Normal "
//               "block pixel count = %d\n\t Corner_pixels = %d -- corner_full_widht = %d -- col_sg_full_width = %d -- "
//               "second_corner_length = %d \n",
//               total_col_pixels,
//               sg_region.sg_block.col_pixel_length,
//               sg_region.sg_block.corner_pixel_length,
//               total_col_pixels,
//               sg_region.sg_block.width * sg_region.sg_block.height,
//               corner_pixels,
//               corner_full_width,
//               col_sg_full_width,
//               sg_region.sg_block.second_corner_length);
// #endif
//         }
//         else
//         {
//             sg_region.sg_block.col_pixel_length = 0;
//             sg_region.sg_block.corner_pixel_length = 0;
//             sg_region.sg_block.second_corner_length = 0;
//         }
//
// #if DEBUGG_LOG
//         printf("x_blocks = %d -- y_blocks = %d\n", x_blocks, y_blocks);
// #endif
//
//         // Figure out global and local
//         // Want to use work_groups to ensure SG located in neigbourhood are on same CU allowing for better L1
//         // utilization
//
//         // Find biggest functioning work_group that allows for full occupancy in our configuration
//
//         int free_sg_per_cu = (max_total_sg - (x_blocks + 1) * (y_blocks + 1)) / PopSift::num_cu;
//
//         int lead_sg_count = 0;
//         int lead_x = 0;
//         int lead_y = 0;
//         for(int x = 1; x < 8; x++)
//         {
//             if(total_x % x != 0)
//                 continue;
//
//             for(int y = 1; y < 8; y++)
//             {
//                 int sg_count = x * y;
//                 if(total_y % y != 0)
//                     continue;
//
//                 if(max_sg_per_cu % sg_count <= free_sg_per_cu)
//                 {
//                     // Accepted
//                     if(lead_sg_count < sg_count)
//                     {
//                         lead_sg_count = sg_count;
//                         lead_x = x;
//                         lead_y = y;
//                     }
//                 }
//             }
//         }
//
// #if DEBUGG_LOG
//         printf("lead_sg_count = %d -- lead_x = %d -- lead_y = %d -- free_sg_per_cu = %d -- max_total_sg = %d\n",
//                lead_sg_count,
//                lead_x,
//                lead_y,
//                free_sg_per_cu,
//                max_total_sg);
// #endif
//
//         sg_region.local = {static_cast<size_t>(lead_y), static_cast<size_t>(lead_x * sg_region.sg_block.width)};
//         sg_region.global = {static_cast<size_t>(total_y), static_cast<size_t>(total_x * sg_region.sg_block.width)};
//
// #if DEBUGG_LOG
//         printf("\n\nWidth = %d height= %d --> normal_block (%d, %d), final_row_height = %d -- col_pixel_length = %d
//         -- "
//                "corner_pixel = %d -- corner_2_pixel = %d\n",
//                width,
//                height,
//                sg_region.sg_block.width,
//                sg_region.sg_block.height,
//                sg_region.sg_block.bottom_row_height,
//                sg_region.sg_block.col_pixel_length,
//                sg_region.sg_block.corner_pixel_length,
//                sg_region.sg_block.second_corner_length);
//
//         printf("\tLocal (%zu, %zu) -- Global (%zu, %zu)\n",
//                sg_region.local[0],
//                sg_region.local[1],
//                sg_region.global[0],
//                sg_region.global[1]);
//         printf("\ty_remainder = %d == %d\n\n\n", sg_region.sg_block.y_remainder,
//         sg_region.sg_block.bottom_row_height);
// #endif
//     }
//
//     return sg_region;
// }
}
