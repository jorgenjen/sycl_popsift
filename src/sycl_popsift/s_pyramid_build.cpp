#include "sycl_popsift/common/assist.h"
#include "sycl_popsift/common/debug_macros.hpp"
#include "sycl_popsift/non_sycl/sift_conf.hpp"
#include "sycl_popsift/popsift.hpp"
#include "sycl_popsift/s_image.hpp"
#include "sycl_popsift/sift_octave.hpp"
#include "sycl_popsift/sift_pyramid.hpp"

#include <cstdio>
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
struct sg_region_block
{
    int width;
    int height;
};

struct persistent_pyramid_config
{
    bool use_persistent_block;
    sg_region_block sg_block;
    sycl::range<2> global;
    sycl::range<2> local;
};

// This should be part of experimental as it is relying on root group

// Computes regions that allows the compute to be done in one wave. Computes the largest regions per sub-group that
// results in one wave. computes a 2d block that is used for both horiz and vert
persistent_pyramid_config compute_persistent_sg_region_block(int width, int height, float remainder_percentage)
{
    // Compute based on num_cu and sg_per_cu

    // Minimum block is 32x13 (32 along x for contigous reads of one sg)
    // TODO: IT should be sub_group width for kernel x 13

    // IMPORTANT:
    // TODO: Figure out how many registers the kernel uses per thread and how many registers each thread can use to then
    // figure out the how many sub_groups that can reside on a compute_unit based on register usage Currently it assumes
    // that registers is not a bottleneck which might not always be the case
    int sg_width = 32; // set to 32 for now
    int max_total_sg = PopSift::sg_per_cu * PopSift::num_cu;

    int x_blocks = width / sg_width;
    int y_blocks = height / 13;
    int block_pixels = sg_width * 13; // how many pixels each sg is responsible for in non edge case
    int total_blocks = x_blocks * y_blocks;

    int x_remainder = width % sg_width;
    int y_remainder = height % 13;
    int remainder_pixels =
      x_remainder * (x_blocks * sg_width) + y_remainder * (y_blocks * 13) + x_remainder * y_remainder;
    persistent_pyramid_config sg_region;
    // if((x_blocks * y_blocks) < max_total_sg * 0.95)
    if(total_blocks < max_total_sg * 0.95)
    {
        // Can't be a proper full wave hence we don't use block for persistent threads pyramid building
        sg_region.use_persistent_block = false;
        return sg_region;
    }
    sg_region.sg_block.width = sg_width;
    sg_region.sg_block.height = 13;

    // We can cover a whole wave for this octave
    // Find max block size that covers a wave

    int remaining_blocks = 0;
    int remainder_blocks = 0;
    // Grow along y first as reuse is more valuable along that direction
    // while(total_blocks > max_total_sg)
    while(true) // Terminated in if
    {
        // int new_height = sg_region.sg_block.height + 1; // Test size increase
        sg_region.sg_block.height++;
        y_blocks = height / sg_region.sg_block.height;

        block_pixels = sg_region.sg_block.width * sg_region.sg_block.height;
        y_remainder = height % sg_region.sg_block.height;

        // This compute is wrong somehow... FIX: This problem line giving wrong remainder pixels
        remainder_pixels = x_remainder * (y_blocks * sg_region.sg_block.height) + y_remainder * width; // y takes corner

        int divisor_remainder = block_pixels * remainder_percentage;
        remainder_blocks = (remainder_pixels + divisor_remainder - 1) / divisor_remainder; // Positive integer ceil
        // Should probably do this ^ more exact and consider the division of work that is natural and that I will use

        total_blocks = x_blocks * y_blocks + remainder_blocks;

        // if(total_blocks > max_total_sg)
        if(total_blocks <= max_total_sg) //
        {
            // We are finaly using less than or equal to the number of sub_groups available
            printf("EXIT -- main_blocks(%d, %d) -- block(%d, %d) -- total_blocks=%d -- Remainder blocks = %d -- "
                   "max_sg=%d -- remainder_pixels = %d block_pixls = %d w=%d h=%d\n ",
                   x_blocks,
                   y_blocks,
                   sg_region.sg_block.width,
                   sg_region.sg_block.height,
                   total_blocks,
                   remainder_blocks,
                   max_total_sg,
                   remainder_pixels,
                   block_pixels,
                   width,
                   height);
            printf("DEBUGG: remainder(%d, %d) -- blocks(%d, %d) -- sg_block(%d, %d) --> Remainder_pixels = %d\n",
                   x_remainder,
                   y_remainder,
                   x_blocks,
                   y_blocks,
                   sg_region.sg_block.width,
                   sg_region.sg_block.height,
                   remainder_pixels);
            break;
        }

        //
    }

    int free_blocks = max_total_sg - total_blocks;
    int bottom_row_pixels = y_remainder * width; // Whole widht remainder row
    int right_col_pixels =
      x_remainder * (y_blocks * sg_region.sg_block.height); // non-remainder y-region of remainder col to avoid overlap

    float col_percent = right_col_pixels == 0 ? 0.0 : (float)right_col_pixels / remainder_pixels;
    float row_percent = bottom_row_pixels == 0 ? 0.0 : (float)bottom_row_pixels / remainder_pixels;
    // Find work division of the remainder region

    int total_remainder = free_blocks + remainder_blocks;

    int corner_pixels = x_remainder * y_remainder;

    int corner_sg = corner_pixels != 0 ? 1 : 0; // If we need a sub_group for corner or not
    // Need to template based on this I think for the different modes
    // int num_row_sg = sycl::floor(total_remainder * row_percent);
    int num_row_sg = x_blocks; //

    int num_col_sg = total_remainder - num_row_sg - corner_sg;

    printf("\nWI - total = %d -- col_percent = %f -- row_percent = %f ---- num_col = %d -- num_row = %d ----- "
           "row_pixels=%d col_pixels=%d -- Remainder_pixels = %d\n",
           total_remainder,
           col_percent,
           row_percent,
           num_col_sg,
           num_row_sg,
           bottom_row_pixels,
           right_col_pixels,
           remainder_pixels);
    // Split remainder along based on the number assigned to col and row

    return sg_region;
}

// Kernel that does persistent way:

void Pyramid::build_pyramid(const Config& conf,
                            ImageBase* base_img,
                            sycl::event d_gauss_write,
                            sycl::event img_transfer)
{
    GaussTableChoice gaussTableChoice;

    if(conf.getGaussMode() == Config::VLFeat_Relative)
        gaussTableChoice = Interpolated_FromPrevious;
    else
        gaussTableChoice = NotInterpolated_FromPrevious;

    for(int octave = 0; octave < _num_octaves; octave++)
    {
        Octave& oct_obj = _octaves[octave];
        compute_persistent_sg_region_block(oct_obj.getWidth(), oct_obj.getHeight(), 0.95);

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

    for(int octave = 0; octave < _num_octaves; octave++) //
    {
        Octave& oct_obj = _octaves[octave];

        oct_obj._dog_done_event = dogs_from_blurred(octave, _levels, oct_obj._level_complete_events[_levels - 1]);
    }
}

} // namespace popsift
