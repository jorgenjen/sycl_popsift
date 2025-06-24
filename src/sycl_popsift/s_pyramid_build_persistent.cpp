#include "sycl_popsift/common/assist.h"
#include "sycl_popsift/persistent_configuration.hpp"
#include "sycl_popsift/popsift.hpp"
#include "sycl_popsift/sift_pyramid.hpp"
#include "sycl_popsift/use_root_group_macro.h" // If we are using root group or handcrafted wg syncrinozation

namespace syclexp = sycl::ext::oneapi::experimental;

namespace popsift {

// Region that a sub-group takes responsibility for
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
// };

// This should be part of experimental as it is relying on root group

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

template<bool REMAINDER_COL>
static inline void horiz_bindless_input(float* intermediate,
                                        syclexp::sampled_image_handle src,
                                        const float* filter,
                                        const int span,
                                        const int dst_w,
                                        const int write_x,
                                        int write_y,
                                        float read_x,
                                        float read_y,
                                        int base_pos)
{
    if constexpr(REMAINDER_COL)
    {
        if(write_x >= dst_w)
            return;
    }

    float out = 0.0f;

    // #pragma unroll
    //     for(int offset = span; offset > 0; offset--)
    //     {
    //         const float g = filter[offset];
    //         const float offrel = float(offset) / dst_w; // relative offset
    //         const float v1 = syclexp::sample_image<float>(src, sycl::float2{read_x - offrel, read_y});
    //         const float v2 = syclexp::sample_image<float>(src, sycl::float2{read_x + offrel, read_y});
    //         out += ((v1 + v2) * g);
    //     }
    //
    //     const float& g = filter[0];
    //     const float v3 = syclexp::sample_image<float>(src, sycl::float2{read_x, read_y});
    //     out += (v3 * g);
    //
    //     // if(write_x < 120 && write_x > 110 && write_y < 120 && write_y > 110)
    //     if(write_x < 900 && write_x > 890 && write_y < 500 && write_y > 490)
    //     {
    //         syclexp::printf("write(%d, %d) --> out = %f -- read_x = %f - read_y = %f -- wave bindless\n",
    //                         write_x,
    //                         write_y,
    //                         out,
    //                         read_x,
    //                         read_y);
    //     }
    //
    //     intermediate[write_x + write_y * dst_w] = out * 255.0f;

#if true

#pragma unroll
    for(int offset = span; offset > 0; offset--)
    {
        const float g = filter[offset];
        const float offrel = float(offset) / dst_w; // relative offset
        const float v1 = syclexp::sample_image<float>(src, sycl::float2{read_x - offrel, read_y});
        const float v2 = syclexp::sample_image<float>(src, sycl::float2{read_x + offrel, read_y});
        out += ((v1 + v2) * g);

        // const float v1 = buffer[base_pos - offset];
        // const float v2 = buffer[base_pos + offset];
        // out += ((v1 + v2) * g);
    }

    const float& g = filter[0];
    const float v3 = syclexp::sample_image<float>(src, sycl::float2{read_x, read_y});
    out += (v3 * g);
    // out += (buffer[base_pos] * filter[0]);

    // if(write_x < 120 && write_x > 110 && write_y < 120 && write_y > 110)
    // if(write_x < 900 && write_x > 890 && write_y < 500 && write_y > 490)
    // {
    //     syclexp::printf("write(%d, %d) --> out = %f -- read_x = %f - read_y = %f wave bindless\n",
    //                     write_x,
    //                     write_y,
    //                     out,
    //                     read_x,
    //                     read_y);
    // }

    intermediate[write_x + write_y * dst_w] = out * 255.0f;
#endif

    // #pragma unroll
    //     for(int offset = span; offset > 1; offset--)
    //     {
    //         const float g = filter[offset];
    //         const float offrel = float(offset) / dst_w; // relative offset
    //         const float v1 = syclexp::sample_image<float>(src, sycl::float2{read_x - offrel, read_y});
    //         const float v2 = syclexp::sample_image<float>(src, sycl::float2{read_x + offrel, read_y});
    //         out += ((v1 + v2) * g);
    //
    //         // const float v1 = buffer[base_pos - offset];
    //         // const float v2 = buffer[base_pos + offset];
    //         // out += ((v1 + v2) * g);
    //     }
    //
    //     const float& g = filter[0];
    //     const float v3 = syclexp::sample_image<float>(src, sycl::float2{read_x, read_y});
    //     out += (v3 * g);
    //     // out += (buffer[base_pos] * filter[0]);
    //
    //     // if(write_x < 120 && write_x > 110 && write_y < 120 && write_y > 110)
    //     if(write_x < 900 && write_x > 890 && write_y < 500 && write_y > 490)
    //     {
    //         syclexp::printf("write(%d, %d) --> out = %f -- read_x = %f - read_y = %f wave bindless\n",
    //                         write_x,
    //                         write_y,
    //                         out,
    //                         read_x,
    //                         read_y);
    //     }
    //
    //     intermediate[write_x + write_y * dst_w] = out * 255.0f;
}

template<bool REMAINDER_COL>
static inline void horiz_local_mem(float* intermediate,
                                   sycl::local_accessor<float, 1> buffer,
                                   const float* filter,
                                   const int span,
                                   const int dst_w,
                                   const int write_x,
                                   int write_y,
                                   int base_pos)
{
    if constexpr(REMAINDER_COL)
    {
        if(write_x >= dst_w)
            return;
    }

    float out = 0.0f;

#pragma unroll
    for(int offset = span; offset > 0; offset--)
    {
        const float g = filter[offset];
        // const float offrel = float(offset) / dst_w; // relative offset
        // const float v1 = syclexp::sample_image<float>(src, sycl::float2{read_x - offrel, read_y});
        // const float v2 = syclexp::sample_image<float>(src, sycl::float2{read_x + offrel, read_y});
        // out += ((v1 + v2) * g);

        const float v1 = buffer[base_pos - offset];
        const float v2 = buffer[base_pos + offset];
        out += ((v1 + v2) * g);
    }

    // const float& g = filter[0];
    // const float v3 = syclexp::sample_image<float>(src, sycl::float2{read_x, read_y});
    // out += (v3 * g);
    out += (buffer[base_pos] * filter[0]);

    // if(write_x < 120 && write_x > 110 && write_y < 120 && write_y > 110)
    // if(write_x < 900 && write_x > 890 && write_y < 500 && write_y > 490)
    // {
    //     syclexp::printf("write(%d, %d) --> out = %f -- wave local mem\n", write_x, write_y, out);
    // }

    intermediate[write_x + write_y * dst_w] = out * 255.0f;
}

static inline void vert_persistent(float* intermediate,
                                   sycl::local_accessor<float, 1> buffer,
                                   const float* filter,
                                   const int span,
                                   const int dst_w,
                                   const int write_x,
                                   int write_y,
                                   int base_pos)
{
    // Vert kernel that resues registers
}

#define USE_ATOMIC_SYNC 1 // For using atomic ref on the work-group state used for synchronizatio

// synchronizes vert execution so that ll data needed to do horiz is available and correct
static inline void vert_sync_for_horiz(int* wg_sync_state, sycl::nd_item<2>& it, int wait_on_state)
{
#if USE_ROOT_GROUP
    sycl::group_barrier(root);
#else
    // Use local hand crafted sychronization
    sycl::group group = it.get_group();
    sycl::group_barrier(group); // Ensure all have done horiz

    if(it.get_local_linear_id() == 0) // only one per work_group
    {
        // Convert to int to use less registers (might help)
        int group_pos = it.get_group(1);
        int group_final_index = it.get_group_range(1) - 1;
        int group_linear_pos = it.get_group_linear_id();
#if USE_ATOMIC_SYNC
        // Need to be a 4 byte wide data type like int or unsigned int for this to work...
        sycl::atomic_ref<int,
                         sycl::memory_order_relaxed,
                         sycl::memory_scope_device,
                         sycl::access::address_space::global_space>(wg_sync_state[group_linear_pos])++;

        // Active wait-- spin lock
        if(group_pos == 0)
        {
            // left border -- only depends on right

            sycl::atomic_ref<int,
                             sycl::memory_order_relaxed,
                             sycl::memory_scope_device,
                             sycl::access::address_space::global_space>
              right(wg_sync_state[group_linear_pos + 1]);

            while(right < wait_on_state) {}
        }
        else if(group_pos == group_final_index)
        {
            // right most border -- only depends on left

            sycl::atomic_ref<int,
                             sycl::memory_order_relaxed,
                             sycl::memory_scope_device,
                             sycl::access::address_space::global_space>
              left(wg_sync_state[group_linear_pos - 1]);
            while(left < wait_on_state) {}
        }
        else
        {
            // Normal in the middle  -- depends on left and right
            sycl::atomic_ref<int,
                             sycl::memory_order_relaxed,
                             sycl::memory_scope_device,
                             sycl::access::address_space::global_space>
              left(wg_sync_state[group_linear_pos - 1]);

            sycl::atomic_ref<int,
                             sycl::memory_order_relaxed,
                             sycl::memory_scope_device,
                             sycl::access::address_space::global_space>
              right(wg_sync_state[group_linear_pos + 1]);

            while(left < wait_on_state && right < wait_on_state) {}
        }

#else
        // BUG: This deadlocks probably does not get the update and the value is stale
        // --> Seems like we need atomic might work with atomic for only read or write not sure seems risky
        // --> Using atomic for both with the one above instead of this one

        // Use normal memory think that should be fine for global aswell...
        // Could cause deadlock if it never let's other's work or if it does not properly update the value on
        // each iteration and just continues to read the initial stale value -- if so atomics would be required

        // int group_pos = it.get_group(1);
        // int group_final_index = it.get_group_range(1) - 1;
        // int group_linear_pos = it.get_group_linear_id();
        // work group leader
        wg_sync_state[group_linear_pos]++; // Signal horiz done

        // Active wait -- spin lock
        if(group_pos == 0)
        {
            // left border -- only depends on right
            while(wg_sync_state[group_linear_pos + 1] < 1) {}
        }
        else if(group_pos == group_final_index)
        {
            // right most border -- only depends on left
            while(wg_sync_state[group_linear_pos - 1] < 1) {}
        }
        else
        {
            // Normal in the middle  -- depends on left and right
            while(wg_sync_state[group_linear_pos - 1] < 1 && wg_sync_state[group_linear_pos + 1] < 1) {}
        }
#endif
    }
    sycl::group_barrier(group); // Wait for wg leader to finish spin lock ensuring dependencies are done
#endif
}

// synchronizes horiz execution so that all data needed to do vert is available and correct
static inline void horiz_sync_for_vert(int* wg_sync_state, sycl::nd_item<2>& it, int wait_on_state)
{
#if USE_ROOT_GROUP
    sycl::group_barrier(root);
#else
    // Use local hand crafted sychronization -- Volatile requires it all to be scheduled in one wave
    sycl::group group = it.get_group();
    sycl::group_barrier(group); // Ensure all have done horiz

    if(it.get_local_linear_id() == 0) // only one per work_group
    {
        // Convert to int to use less registers (might help)
        int group_pos_0 = it.get_group(0);
        int group_pos_1 = it.get_group(1);
        int group_range_0 = it.get_group_range(0);
        int group_range_1 = it.get_group_range(1);
        // int group_linear_pos = it.get_group_linear_id();
        // Need to be a 4 byte wide data type like int or unsigned int for this to work...
        sycl::atomic_ref<int,
                         sycl::memory_order_relaxed,
                         sycl::memory_scope_device,
                         sycl::access::address_space::global_space>(wg_sync_state[it.get_group_linear_id()])++;

        // Active wait-- spin lock
        if(group_pos_0 == 0)
        {
            // top border -- only depends on wg below

            sycl::atomic_ref<int,
                             sycl::memory_order_relaxed,
                             sycl::memory_scope_device,
                             sycl::access::address_space::global_space>
              below(wg_sync_state[(group_pos_0 + 1) * group_range_1 + group_pos_1]);

            while(below < wait_on_state) {}
        }
        else if(group_pos_0 == group_range_0 - 1)
        {
            // right most border -- only depends on left

            sycl::atomic_ref<int,
                             sycl::memory_order_relaxed,
                             sycl::memory_scope_device,
                             sycl::access::address_space::global_space>
              above(wg_sync_state[(group_pos_0 - 1) * group_range_1 + group_pos_1]);
            while(above < wait_on_state) {}
        }
        else
        {
            // Normal in the middle  -- depends on left and right
            sycl::atomic_ref<int,
                             sycl::memory_order_relaxed,
                             sycl::memory_scope_device,
                             sycl::access::address_space::global_space>
              above(wg_sync_state[(group_pos_0 - 1) * group_range_1 + group_pos_1]);

            sycl::atomic_ref<int,
                             sycl::memory_order_relaxed,
                             sycl::memory_scope_device,
                             sycl::access::address_space::global_space>
              below(wg_sync_state[(group_pos_0 + 1) * group_range_1 + group_pos_1]);

            while(above < wait_on_state && below < wait_on_state) {}
        }
    }
    sycl::group_barrier(group); // Wait for wg leader to finish spin lock ensuring dependencies are done
#endif
}

namespace normalizedSource {

// Used for ImageBindless
// Only used on input image (initial)
// And only works for it due to  filter and span selection

// aspect::ext_oneapi_bindless_sampled_image_fetch_2d
// This aspect is required to use sampled image need to add a check for that earlier in selection
// template<bool if_required>

#define USE_SHARED_MEM_FOR_INPUT 1
template<bool REMAINDER_COL, bool REMAINDER_ROW>
class BuildOctave
{
  private:
    syclexp::sampled_image_handle src;
    float** data_array; // Need to be array of all dst data
    float** dog_array;
    float* intermediate;
    popsift::GaussInfo* d_gauss;
    sycl::local_accessor<float, 1> buffer;
    const sg_region_blocks sg_region;
    const int dst_w;
    const int dst_h;
    const float shift;
    const int levels;

  public:
    BuildOctave(syclexp::sampled_image_handle src,
                float** data_array,
                float** dog_array,
                float* intermediate,
                popsift::GaussInfo* d_gauss,
                sycl::local_accessor<float, 1> buffer,
                const sg_region_blocks sg_region,
                const int dst_w,
                const int dst_h,
                const float shift,
                const int levels)

      : src(src)
      , data_array(data_array)
      , dog_array(dog_array)
      , intermediate(intermediate)
      , d_gauss(d_gauss)
      , buffer(buffer)
      , sg_region(sg_region)
      , dst_w(dst_w)
      , dst_h(dst_h)
      , shift(shift)
      , levels(levels) {};

    inline void operator()(sycl::nd_item<2> it) const
    {
        const auto sg_width = it.get_sub_group().get_max_local_range()[0]; // 32 in cuda

        // Used for input only
        const float* filter = &d_gauss->dd.filter[0];
        const int span = d_gauss->dd.span[0];

        const int write_x = it.get_global_id(1); // Constant in normal block
        // int write_y = it.get_group(0) * sg_region.height; // Changes in normal block aswell
        int write_y = it.get_global_id(0) * sg_region.height; // Changes in normal block aswell

        const float read_x = (write_x + shift) / dst_w;
        float read_y = (write_y + shift) / dst_h;

        // Not sure if there is a point of using this for input level -- As we can't async load
        const int base_pos =
          (it.get_local_range(1) + (span << 1)) * (it.get_local_id(0) << 1) + it.get_local_id(1) + span;

        // Second buffer row (there are two per row in the work-group)
        const int base_pos_2 =
          (it.get_local_range(1) + (span << 1)) * ((it.get_local_id(0) << 1) + 1) + it.get_local_id(1) + span;

        // const int rel_span = ((1 / dst_w) * span); // Relative span value used for offset
        const float rel_span = float(span) / dst_w; // Relative span value used for offset

        // for(int i = 0; i < sg_region.height; i++)

        // const float read_y_increment = 1.0f / dst_h; // Does not result in the same as recompute due to
        // accumulation of floating point error

        int loop_end = write_y + sg_region.height;

        if constexpr(REMAINDER_ROW)
        {
            if(loop_end >= dst_h)
                loop_end = dst_h; // Limit to last pixel
        }

#if USE_ROOT_GROUP
        auto root = it.ext_oneapi_get_root_group(); // Root group all work_items running kernel
#endif
        for(; write_y < loop_end; ++write_y) // Modifies write_y want that later
        {
            // read_y += read_y_increment; // Floating point error accumulation hence not using
            read_y = (write_y + shift) / dst_h;

#if USE_SHARED_MEM_FOR_INPUT
            buffer[base_pos] = syclexp::sample_image<float>(src, sycl::float2{read_x, read_y}); // every one does this

            if(it.get_local_id(1) < span)
            {
                // load left side (lenght of span)
                buffer[base_pos - span] = syclexp::sample_image<float>(src, sycl::float2{read_x - rel_span, read_y});
            }
            else if(it.get_local_id(1) >= (it.get_local_range(1) - span))
            {
                buffer[base_pos + span] = syclexp::sample_image<float>(src, sycl::float2{read_x + rel_span, read_y});
            }

            // Here would be good to do async load of next row but does not seem to be possible to do with bindless
            // images But for remaining parts it will be not sure if we should use local mem for this part however
            sycl::group_barrier(it.get_group()); // Ensure all is loaded before we do horiz

            horiz_local_mem<REMAINDER_COL>(intermediate, buffer, filter, span, dst_w, write_x, write_y, base_pos);

#else
            horiz_bindless_input<REMAINDER_COL>(
              intermediate, src, filter, span, dst_w, write_x, write_y, read_x, read_y, base_pos);

#endif

            // Second row buffer in use: Same as above otherwise

            write_y++;
            if(write_y >= loop_end)
                break;

            // read_y += read_y_increment; // Floating point error accumulation hence not using
            read_y = (write_y + shift) / dst_h;

#if USE_SHARED_MEM_FOR_INPUT
            buffer[base_pos_2] = syclexp::sample_image<float>(src, sycl::float2{read_x, read_y});

            if(it.get_local_id(1) < span)
            {
                buffer[base_pos_2 - span] = syclexp::sample_image<float>(src, sycl::float2{read_x - rel_span, read_y});
            }
            else if(it.get_local_id(1) >= (it.get_local_range(1) - span))
            {
                buffer[base_pos_2 + span] = syclexp::sample_image<float>(src, sycl::float2{read_x + rel_span, read_y});
            }

            sycl::group_barrier(it.get_group());

            horiz_local_mem<REMAINDER_COL>(intermediate, buffer, filter, span, dst_w, write_x, write_y, base_pos_2);
#else
            horiz_bindless_input<REMAINDER_COL>(
              intermediate, src, filter, span, dst_w, write_x, write_y, read_x, read_y, base_pos);
#endif
        }
        // Synchronize and then do horiz
        horiz_sync_for_vert(sg_region.wg_sync_state, it, 1);

        // Start doing Vert then later we do horiz on data_array so not using sampled image then we can use async
        // load of next row Do vert for this one then make loop over the levels for the rest with horiz from prev
        // and vert from intermediate

        // Vert

        // float int6 = intermediate[write_y * dst_w + write_x];
        // float int11 = intermediate[(write_y - 1) * dst_w + write_x];
        // float int10 = intermediate[(write_y - 2) * dst_w + write_x];
        // float int9 = intermediate[(write_y - 3) * dst_w + write_x];
        // float int8 = intermediate[(write_y - 4) * dst_w + write_x];
        // float int7 = intermediate[(write_y - 5) * dst_w + write_x];
        // float int6 = intermediate[(write_y - 6) * dst_w + write_x];
        // float int5 = intermediate[(write_y - 7) * dst_w + write_x];
        // float int4 = intermediate[(write_y - 8) * dst_w + write_x];
        // float int3 = intermediate[(write_y - 9) * dst_w + write_x];
        // float int2 = intermediate[(write_y - 10) * dst_w + write_x];
        // float int1 = intermediate[(write_y - 11) * dst_w + write_x];
        // float int0 = intermediate[(write_y - 12) * dst_w + write_x];

        for(; write_y >= (it.get_global_id(0) * sg_region.height); write_y--)
        {
            // Move the other way for potential better cache
        }

        vert_sync_for_horiz(sg_region.wg_sync_state, it, 2);

        // Then we do DoG -- Or do DoG as we are storing the next level as we have the value in register and only
        // need to load one value (prev_level) and then we can store DoG as we go giving another justification for
        // doing it the persistent way and perhaps faster :D
    }
};

} // namespace normalizedSource

#if USE_PERSISTENT
// bool Pyramid::build_octave_one_wave_input(const Config& conf,
sycl::event Pyramid::build_octave_one_wave_input(const Config& conf,
                                                 ImageBase* base,
                                                 sycl::event d_gauss_write,
                                                 sycl::event img_write)
{
    Octave& oct_obj = _octaves[0];
    persistent_pyramid_octave_config& sg_region = oct_obj._sg_region;

    const int width = oct_obj.getWidth();
    const int height = oct_obj.getHeight();
    // persistent_pyramid_octave_config sg_region = compute_persistent_sg_region_block(width, height);

    // fprintf(stderr,
    //         "Local(%zu, %zu) -- global (%zu, %zu) -- sg_region --  width = %d - height = %d -- x_remainder = %d
    //         -- " "y_remainder = %d\n\n", sg_region.local[0], sg_region.local[1], sg_region.global[0],
    //         sg_region.global[1],
    //         sg_region.sg_block.width,
    //         sg_region.sg_block.height,
    //         sg_region.sg_block.x_remainder,
    //         sg_region.sg_block.y_remainder);

    // if(!sg_region.use_persistent_block)
    //     return false; // Could not use persistent block

#if true
    for(auto& plat : sycl::platform::get_platforms())
    {
        std::cout << "CUDA‐SYCL platform name: " << plat.get_info<sycl::info::platform::name>() << "\n"
                  << "Reported version:     " << plat.get_info<sycl::info::platform::version>() << "\n";
    }

#if SYCL_EXT_ONEAPI_ROOT_GROUP == 1
    printf("ROOT GPOUP SUPPPOERTED\n");
#else
    printf("ROOT GROUP NOT SUPPORTED\n");
#endif

#endif

    // Just for test
    if(!sg_region.use_persistent_block)
        return sycl::event(); // Could not use persistent block

    // Use persistent block

    // Lauch kernel that starts from input image -- quite likely to be compatible with persistent_block

    // sycl::event dependency = prev_oct_obj.getLevelCompleteEvent(_levels - PREV_LEVEL);

    // This ^ is the event we need to write to ensure interoperability between block and non block versions running.
    // So for from prev version this would also need to be our dependency (not sure if that is good enough not sure
    // this would work in case of other kernels being in flight...)

#if USE_ROOT_GROUP
    auto props = syclexp::properties{syclexp::use_root_sync}; // Does not seem to be supported by cuda backend for dpc++
#endif
    if constexpr(USE_BINDLESS_INPUT && sycl::any_device_has<sycl::aspect::ext_oneapi_bindless_images>() &&
                 sycl::any_device_has<sycl::aspect::ext_oneapi_bindless_sampled_image_fetch_2d>())
    {
        // auto kernel_id = sycl::get_kernel_id<horiz_if>();
        // auto kernel_bundle =
        // sycl::get_kernel_bundle<sycl::bundle_state::executable>(_device_queue.get_context()); auto kernel =
        // kernel_bundle.get_kernel(kernel_id);
        //
        // auto compiled_num_sg = kernel.template
        // get_info<sycl::info::kernel_device_specific::compile_num_sub_groups>(
        //   _device_queue.get_device());
        // auto max_num_sg =
        //   kernel.template
        //   get_info<sycl::info::kernel_device_specific::max_num_sub_groups>(_device_queue.get_device());
        // auto prefered_wg_multiple =
        //   kernel.template get_info<sycl::info::kernel_device_specific::preferred_work_group_size_multiple>(
        //     _device_queue.get_device());

        // Bindless version
        const float shift = 0.5f * sycl::pow(2.0f, conf.getUpscaleFactor());

        const bool col = sg_region.x_remainder != 0;
        const bool row = sg_region.y_remainder != 0;

        auto device = _device_queue.get_device();

        // Get local memory size in bytes
        size_t local_mem_size = device.get_info<sycl::info::device::local_mem_size>();

        const int buffer_size = (sg_region.local[1] + (Pyramid::largest_span << 2)) * (sg_region.local[0] << 2);
        const int vert_buffer_size =
          ((sg_region.local[1] * 13) * sg_region.local[0]); // might be better to store the 13 values in registers
                                                            // might be problematic if register pressure get's too high
        printf("Local(%zu, %zu) -- global(%zu, %zu) --- Local mem size = %zu -- largest span = %d\n",
               sg_region.local[0],
               sg_region.local[1],
               sg_region.global[0],
               sg_region.global[1],
               local_mem_size,
               largest_span);
        printf("THIS IS SHIFT = %f -- widht=%d -- height=%d  -- col = %d -- row = %d -- Buffer_size = %d -- "
               "vert_buffer_size = %d",
               shift,
               width,
               height,
               col,
               row,
               buffer_size,
               vert_buffer_size);

        if(col && row)
        {
            // printf("We doing col and row whop whop\n");
            // sycl::event e = _device_queue.submit([&](sycl::handler& cgh) {
            return _device_queue.submit([&](sycl::handler& cgh) { // for TEST
                cgh.depends_on({d_gauss_write, img_write, sg_region._zeroed_event});

                // TODO: Need to figure out what type of buffer I need for vert and assign the largest one to use

                auto buffer = sycl::local_accessor<float, 1>(buffer_size, cgh);

                cgh.parallel_for(sycl::nd_range{sg_region.global, sg_region.local},
#if USE_ROOT_GROUP
                                 props,
#endif
                                 normalizedSource::BuildOctave<true, true>(base->getInputImage(),
                                                                           oct_obj.getDataArray(),
                                                                           oct_obj.getDogArray(),
                                                                           oct_obj.getIntermediate(),
                                                                           _d_gauss,
                                                                           buffer,
                                                                           sg_region.sg_block,
                                                                           width,
                                                                           height,
                                                                           shift,
                                                                           _levels));
            });
        }
        else if(col)
        {
            // sycl::event e = _device_queue.submit([&](sycl::handler& cgh) {
            return _device_queue.submit([&](sycl::handler& cgh) { // for TEST
                cgh.depends_on({d_gauss_write, img_write, sg_region._zeroed_event});

                // TODO: Need to figure out what type of buffer I need for vert and assign the largest one to use

                auto buffer = sycl::local_accessor<float, 1>(buffer_size, cgh);

                cgh.parallel_for(sycl::nd_range{sg_region.global, sg_region.local},
#if USE_ROOT_GROUP
                                 props,
#endif
                                 normalizedSource::BuildOctave<true, false>(base->getInputImage(),
                                                                            oct_obj.getDataArray(),
                                                                            oct_obj.getDogArray(),
                                                                            oct_obj.getIntermediate(),
                                                                            _d_gauss,
                                                                            buffer,
                                                                            sg_region.sg_block,
                                                                            width,
                                                                            height,
                                                                            shift,
                                                                            _levels));
            });
        }
        else if(row)
        {
            // sycl::event e = _device_queue.submit([&](sycl::handler& cgh) {
            return _device_queue.submit([&](sycl::handler& cgh) { // for TEST
                cgh.depends_on({d_gauss_write, img_write, sg_region._zeroed_event});

                // TODO: Need to figure out what type of buffer I need for vert and assign the largest one to use

                auto buffer = sycl::local_accessor<float, 1>(buffer_size, cgh);

                cgh.parallel_for(sycl::nd_range{sg_region.global, sg_region.local},
#if USE_ROOT_GROUP
                                 props,
#endif
                                 normalizedSource::BuildOctave<false, true>(base->getInputImage(),
                                                                            oct_obj.getDataArray(),
                                                                            oct_obj.getDogArray(),
                                                                            oct_obj.getIntermediate(),
                                                                            _d_gauss,
                                                                            buffer,
                                                                            sg_region.sg_block,
                                                                            width,
                                                                            height,
                                                                            shift,
                                                                            _levels));
            });
        }
        else
        {
            // sycl::event e = _device_queue.submit([&](sycl::handler& cgh) {
            return _device_queue.submit([&](sycl::handler& cgh) { // for TEST
                cgh.depends_on({d_gauss_write, img_write, sg_region._zeroed_event});

                // TODO: Need to figure out what type of buffer I need for vert and assign the largest one to use

                auto buffer = sycl::local_accessor<float, 1>(buffer_size, cgh);

                cgh.parallel_for(sycl::nd_range{sg_region.global, sg_region.local},
#if USE_ROOT_GROUP
                                 props,
#endif
                                 normalizedSource::BuildOctave<false, false>(base->getInputImage(),
                                                                             oct_obj.getDataArray(),
                                                                             oct_obj.getDogArray(),
                                                                             oct_obj.getIntermediate(),
                                                                             _d_gauss,
                                                                             buffer,
                                                                             sg_region.sg_block,
                                                                             width,
                                                                             height,
                                                                             shift,
                                                                             _levels));
            });
        }

        // _device_queue.wait(); // For testing
    }
    else
    {
        // Normal inpute image not bindless

        // sycl::event e = _device_queue.parallel_for(
        //   sycl::nd_range{global, local},
        //   {d_gauss_write, img_write},
        //   absoluteSource::Horiz<0, true>(base->getInputFloat(), oct_obj.getIntermediate(), _d_gauss, width,
        //   height, 0));
    }

    return sycl::event();
}
#endif // if USE_PERSISTENT

} // namespace popsift

#if false
        // Position to write to (image that has the size of scale up)
        const int write_x = it.get_global_id(1);
        // const int write_y = it.get_global_id(0) * dst_w;
        const int write_y = it.get_group(0);
        // Cant use it.get_global_range(1) inplace of dst_w due to if if_required width !=
        it.get_global_range(1) and
          // hence positions would be off could be used in else case but not sure if it matters much (probs not)

          if constexpr(if_required)
        {
            // Destination width was not perfectly divisible with it.get_local_range(1)
            if(write_x >= dst_w)
                return;
        }

        const float* filter = &d_gauss->dd.filter[0];
        const int span = d_gauss->dd.span[0];
        const float read_x = (write_x + shift) / dst_w;
        const float read_y = (write_y + shift) / dst_h;

        // Could pass dimensions as a int2 and do vector wise
        // const sycl::float2 read_pos = sycl::float2{(write_x + shift) / dst_w, (write_y + shift) /
        dst_h
    };

    float out = 0.0f;

    // Look into sycl mad or fma (multiply-and-add instruction done in one clock cycle)
    // is probably done by the compiler anyways though

#pragma unroll
    for(int offset = span; offset > 1; offset--)
    {
        const float g = filter[offset];
        const float offrel = float(offset) / dst_w; // relative offset
        const float v1 = syclexp::sample_image<float>(src, sycl::float2{read_x - offrel, read_y});
        const float v2 = syclexp::sample_image<float>(src, sycl::float2{read_x + offrel, read_y});
        out += ((v1 + v2) * g);
    }

    const float& g = filter[0];
    const float v3 = syclexp::sample_image<float>(src, sycl::float2{read_x, read_y});
    out += (v3 * g);

    dst_data[write_x + write_y * dst_w] = out * 255.0f;
#endif
