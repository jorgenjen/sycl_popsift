#include "sycl_popsift/common/assist.h"
#include "sycl_popsift/common/debug_macros.hpp"
#include "sycl_popsift/non_sycl/sift_conf.hpp"
#include "sycl_popsift/popsift.hpp"
#include "sycl_popsift/s_image.hpp"
#include "sycl_popsift/sift_octave.hpp"
#include "sycl_popsift/sift_pyramid.hpp"

#include <cstdio>
#include <ios>
#include <vector>

/* It makes no sense whatsoever to change this value */
#define PREV_LEVEL 3

using std::cerr;
using std::cout;
using std::endl;

namespace popsift {

// namespace gauss { // is only for one function get_by_2_pick_every_second
//                   // only used once so I just use a lambda instead of a functor in namespace
//
// }

struct Downscale; // For Profiing tools

// not sure if we want the se to be inline they were in CUDA popsift
inline sycl::event Pyramid::downscale_from_prev_octave(int octave)
{
    Octave& oct_obj = _octaves[octave];
    Octave& prev_oct_obj = _octaves[octave - 1];

    // downscaled with and height (current for this octave)
    const int dst_width = oct_obj.getWidth();
    const int dst_height = oct_obj.getHeight();

    const int src_width = prev_oct_obj.getWidth();
    const int src_height = prev_oct_obj.getHeight();

    float* src_data = prev_oct_obj.getDataArrayHost()[_levels - PREV_LEVEL];
    float* dst_data = oct_obj.getDataArrayHost()[0];

    sycl::range local{2, 64};
    sycl::range global{(size_t)grid_divide(dst_height, local[0]), (size_t)grid_divide(dst_width, local[1])};

    sycl::event dependency = prev_oct_obj.getLevelCompleteEvent(_levels - PREV_LEVEL);

    // This kernel is almost 3 times slower than the texture kernel of cuda version...
    // And they are almost identical so it does not make sense. Don't get how a texture could make that much of a
    // differences so need to compare the ptx
    return _device_queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(dependency);
        cgh.parallel_for<Downscale>(sycl::nd_range(global, local), [=](sycl::nd_item<2> it) {
            int x = it.get_global_id(1);
            int y = it.get_global_id(0);

            // better to have in one or two? -- Probs don't matter
            if(x >= dst_width)
                return;
            if(y >= dst_height)
                return;

            // Don't need clamp just an upper check not sure if it matters
            const int read_x = sycl::clamp(x << 1, 0, src_width);
            const int read_y = sycl::clamp(y << 1, 0, src_height);

            // calamp ensures src access is always safe
            dst_data[x + y * dst_width] = src_data[read_x + read_y * src_width];
        });
    });
}

// Seems like a bit of an odd way to make the kernel?
class make_dog; // Dont necessarily need to name it could also just stay anonymous

// Not sure if thes shoould be inline or not...
// inline void Pyramid::dogs_from_blurred(int octave, int max_level, sycl::event octave_complete)
sycl::event Pyramid::dogs_from_blurred(int octave, int max_level, sycl::event octave_complete)
{
    Octave& oct_obj = _octaves[octave];

    const int width = oct_obj.getWidth();
    const int height = oct_obj.getHeight();

    sycl::range local{1, 1024};
    sycl::range global{(size_t)grid_divide(height, local[0]), (size_t)grid_divide(width, local[1])};

    float** data_array = oct_obj.getDataArray();
    float** dog_array = oct_obj.getDogArray();

    return _device_queue.parallel_for<make_dog>(
      sycl::nd_range{global, local},
      {octave_complete, oct_obj.getDataArrayWriteEvent(), oct_obj.getDogArrayWriteEvent()},
      [=](sycl::nd_item<2> it) {
          int x = it.get_global_id(1);
          int y = it.get_global_id(0);
          if(x >= width)
              return;

          // Reverse for potentially more cache hits for first access
          float upper = data_array[max_level - 1][x + y * width];
          for(int level = max_level - 2; level >= 0; --level)
          {
              const float lower = data_array[level][x + y * width];

              dog_array[level][x + y * width] = upper - lower;
              upper = lower;
          }
      });
}

inline sycl::event Pyramid::horiz_from_prev_level(int octave, int level, GaussTableChoice useInterpolatedGauss)
{
    switch(useInterpolatedGauss)
    {
        // case Interpolated_FromPrevious: horiz_from_prev_level_pairs(octave, level, stream); break;
        case Interpolated_FromPrevious: cout << "horiz_from_prev_level_pairs not implemented yet"; break;
        case NotInterpolated_FromPrevious: return horiz_from_prev_level_basic(octave, level); break;
        default: POP_FATAL("Missing case in horizontal Gauss filter from previous level"); break;
    }
    return sycl::event(); // just to return for now to avoid warning for compiler
}

sycl::event Pyramid::vert_from_interm(int octave,
                                      int level,
                                      GaussTableChoice useInterpolatedGauss,
                                      sycl::event intm_write)
{
    Octave& oct_obj = _octaves[octave];

    const int width = oct_obj.getWidth();
    const int height = oct_obj.getHeight();

    switch(useInterpolatedGauss)
    {
        // case Interpolated_FromPrevious: vert_from_interm_pairs(octave, level, stream); break;
        case Interpolated_FromPrevious: cout << "vert_from_interm_paris not implemented yet"; break;
        case NotInterpolated_FromPrevious: return vert_from_interm_basic(octave, level, intm_write); break;
        default: {
            POP_FATAL("Missing case in vertical Gauss filter from intermediate buffer");
        }
        break;
    }
    return sycl::event(); // just to return for now to avoid warning for compiler
}

// Region that a sub-group takes responsibility for
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
};

// This should be part of experimental as it is relying on root group

// Computes regions that allows the compute to be done in one wave. Computes the largest regions per sub-group that
// results in one wave. computes a 2d block that is used for both horiz and vert
persistent_pyramid_octave_config compute_persistent_sg_region_block(int width, int height)
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

    int x_remainder = width % sg_region.sg_block.width;
    int y_remainder;

    int total_blocks;
    int num_col_sg;
    int right_col_pixels;

    // We can cover a whole wave for this octave
    // Find max block size that covers a wave
    while(true) // Terminated in if
    {
        y_blocks = height / sg_region.sg_block.height;

        total_blocks = x_blocks * y_blocks + x_blocks + y_blocks + 1; // Minimum needed

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
            printf("WE DONE --> total_blocks = %d - Max_total_sg = %d -- num_col_sg = %d -- x_blocks = %d -- "
                   "main_region = %d -- "
                   "x_remainder = %d -- y_remainder = %d\n",
                   total_blocks,
                   max_total_sg,
                   num_col_sg,
                   x_blocks,
                   x_blocks * y_blocks,
                   x_remainder,
                   y_remainder);

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

        int total_col_pixels = x_remainder * height;

        int total_full_width = total_col_pixels / sg_region.sg_block.width;
        int corner_pixels = total_col_pixels % sg_region.sg_block.width;

        int col_sg_full_width = total_full_width / y_blocks;
        int corner_full_width = total_full_width % y_blocks;

        sg_region.sg_block.col_pixel_length = col_sg_full_width * sg_region.sg_block.width;
        sg_region.sg_block.corner_pixel_length = corner_full_width * sg_region.sg_block.width + corner_pixels;

        // sg_region.sg_block.bottom_row_height = y_remainder;
        sg_region.sg_block.bottom_row_height = height % sg_region.sg_block.height;

        printf("\n col_pixels = %d -- col_pixel_length = %d -- corner_pixel = %d -- total_col_pixels = %d -- Normal "
               "block pixel count = %d\n\t Corner_pixels = %d -- corner_full_widht = %d -- col_sg_full_width = %d\n",
               total_col_pixels,
               sg_region.sg_block.col_pixel_length,
               sg_region.sg_block.corner_pixel_length,
               total_col_pixels,
               sg_region.sg_block.width * sg_region.sg_block.height,
               corner_pixels,
               corner_full_width,
               col_sg_full_width);

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

        printf("\n col_pixels = %d -- col_pixel_length = %d -- corner_pixel = %d -- total_col_pixels = %d -- Normal "
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

        printf("x_blocks = %d -- y_blocks = %d\n", x_blocks, y_blocks);

        // Figure out global and local
        // Want to use work_groups to ensure SG located in neigbourhood are on same CU allowing for better L1
        // utilization

        // Find biggest functioning work_group that allows for full occupancy in our configuration

        int free_sg_per_cu = (max_total_sg - (x_blocks + 1) * (y_blocks + 1)) / PopSift::num_cu;

        int total_x = x_blocks + 1;
        int total_y = x_blocks + 1;

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
        printf("lead_sg_count = %d -- lead_x = %d -- lead_y = %d -- free_sg_per_cu = %d -- max_total_sg = %d\n",
               lead_sg_count,
               lead_x,
               lead_y,
               free_sg_per_cu,
               max_total_sg);

        sg_region.local = {static_cast<size_t>(lead_y), static_cast<size_t>(lead_x * sg_region.sg_block.width)};
        sg_region.global = {static_cast<size_t>(total_y), static_cast<size_t>(total_x * sg_region.sg_block.width)};

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
        printf("\ty_remainder = %d == %d\n\n\n", y_remainder, sg_region.sg_block.bottom_row_height);
    }

    return sg_region;
}
// }

// Kernel that does persistent way:

void Pyramid::build_pyramid(const Config& conf,
                            ImageBase* base_img,
                            sycl::event d_gauss_write,
                            sycl::event img_transfer)
{
    GaussTableChoice gaussTableChoice;

#if USE_PERSISTENT
    persistent_pyramid_octave_config sg_region;
#endif

    if(conf.getGaussMode() == Config::VLFeat_Relative)
        gaussTableChoice = Interpolated_FromPrevious;
    else
        gaussTableChoice = NotInterpolated_FromPrevious;

    for(int octave = 0; octave < _num_octaves; octave++)
    {
        Octave& oct_obj = _octaves[octave];
#if USE_PERSISTENT
        sg_region = compute_persistent_sg_region_block(oct_obj.getWidth(), oct_obj.getHeight());

        if(sg_region.use_persistent_block == true)
        {
            // Launch using persistent block
        }
        else
#endif
        {
            for(int level = 0; level < _levels; level++)
            {
                if(level == 0)
                {
                    if(octave == 0)
                    {
                        sycl::event horiz = horiz_from_input_image(conf, base_img, d_gauss_write, img_transfer);
                        // Storing event to class only for profiling not needed for normal use
#if QUEUE_PROFILING
                        _input_horiz_event = horiz; // copy it for use later
#endif
                        oct_obj._level_complete_events[0] = vert_from_interm(octave, 0, gaussTableChoice, horiz);
                    }
                    else
                    {
                        oct_obj._level_complete_events[0] = downscale_from_prev_octave(octave);
                    }
                }
                else
                {
                    // Depends on set level event from prev level
                    sycl::event horiz = horiz_from_prev_level(octave, level, gaussTableChoice);

                    oct_obj._level_complete_events[level] = vert_from_interm_basic(octave, level, horiz);
                }
            }
        }
    }

#if USE_PERSISTENT
    if(!sg_region.use_persistent_block) // Don't need DoG kernel if using persistent block as it's embedded
#endif
    {
        for(int octave = 0; octave < _num_octaves; octave++)
        {
            Octave& oct_obj = _octaves[octave];

            oct_obj._dog_done_event = dogs_from_blurred(octave, _levels, oct_obj._level_complete_events[_levels - 1]);
        }
    }
}

} // namespace popsift

#if false

// Too complex would require multiple different versions  with template so takes too much time to develop hence using simple 
// chain that would result in less ideal memory coaleced reads which was the goal and plan with this segmentation that is not complete 

    if(sg_region.use_persistent_block)
    {
        // TODO: Figure out col block size

        // Now we know the number of blocks we have to divide to column into should strive to make it a multiple of 32
        // but that will not be possible so should make all a multiple of 32/sg_widht besides the last one in the corner
        // that is not -- It will not be a row multiple but it will be wrap around just to have full usage of the
        // threads

        // Four types of shecduling. Main region, Bottom row, Rightmost col and corner.
        // Row blocks are all sg_widht wide and each row block has a pixel multiple of sg_widht for full usage. Corner
        // takes the remaining pixels of Need to pass x_remainder for col start postion and wraping to be computed what
        // is passed as col block is just the lenght and each work_item need to compute the positions that they are
        // responsible for

        y_remainder = height % sg_region.sg_block.height;

        if(x_remainder != 0)
        {
            // Need to deal with column and possibly corner

            int num_rows_per_sg = sg_region.sg_block.width / x_remainder; // 1
            int sg_remainder = sg_region.sg_block.width % x_remainder;    // 5

            if(sg_remainder > x_remainder)
            {
                // Need to think differently
            }
            else
            {
                // num_rows_per_sg is always one here

                // Need a set of main rows and a set of rows that will fill full rows for remainder

                // Need full remainder rows
                int full_rows_sg_remainder = 1;

                while((full_rows_sg_remainder * x_remainder) % sg_remainder != 0)
                {
                    full_rows_sg_remainder++;
                }

                int num_coalesced_rows = (full_rows_sg_remainder * x_remainder) / sg_remainder;

                int total_rows = num_coalesced_rows + full_rows_sg_remainder;

                if(total_rows < sg_region.sg_block.height)
                {
                    // It's small enough now we need to figure out if it's usable or if it leaves to large a remainder
                    int max_rows =
                      ((sg_region.sg_block.width * sg_region.sg_block.height) * remainder_percentage) / x_remainder;

                    int inner_block_count = total_rows
                }

                if(total_blocks + y_blocks + 1 <= max_total_sg)
                {
                    // Can use two columns for remainder col;
                    // Not sure if we want to do that
                }

                if(y_remainder %) {}
                else
                {
                    // Do encoding that is chaing of pixels less coaleced memory reads so less optimal
                }
            }
            else
            {
                // No need for corner and column
            }

            // if(right_col_pixels != 0)
            // {
            //     int sg_for_col = max_total_sg - total_blocks - x_blocks; // Free sg that can use for col
            //     // Might not be need for corner
            //     if(right_col_pixels % sg_region.sg_block.width == 0 &&
            //        (right_col_pixels / sg_region.sg_block.width) % sg_for_col == 0)
            //     {
            //         // No need for corner
            //     }
            //     else
            //     {
            //         // Save one for corner
            //         int rows_per_col_sg = (right_col_pixels / sg_region.sg_block.width) / (sg_for_col - 1);
            //
            //         int rows_for_corner = (right_col_pixels / sg_region.sg_block.width) % (sg_for_col - 1);
            //         int reminder_col_pixels = right_col_pixels % sg_region.sg_block.width;
            //     }
            // }
        }
        // else we don't need col so should launch a different version throught templte and local and global does not
        // need it
        // printf("Free blocks for col =  %d", sg_for_col);
    }
#endif
