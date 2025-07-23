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

// }

// Kernel that does persistent way:

void Pyramid::build_pyramid(const Config& conf,
                            ImageBase* base_img,
                            sycl::event d_gauss_write,
                            sycl::event img_transfer)
{
    GaussTableChoice gaussTableChoice;

#ifdef USE_PERSISTENT
    bool use_persistent_block;
#endif

    if(conf.getGaussMode() == Config::VLFeat_Relative)
        gaussTableChoice = Interpolated_FromPrevious;
    else
        gaussTableChoice = NotInterpolated_FromPrevious;

    for(int octave = 0; octave < _num_octaves; octave++)
    {
        Octave& oct_obj = _octaves[octave];
#ifdef USE_PERSISTENT
        // sg_region = compute_persistent_sg_region_block(oct_obj.getWidth(), oct_obj.getHeight());

        // if(octave == 0)
        // {
        //     use_persistent_block = build_octave_one_wave_input(conf, base_img, d_gauss_write, img_transfer);
        // }
        // else
        // {
        //     // TODO: Add version for from prev octave
        //     // use_persistent_block = build_octave_one_wave_prev_octave(octave);
        //     use_persistent_block = false;
        // }

        // // if(sg_region.use_persistent_block == true)
        // if(use_persistent_block)
        // {
        //     // Launch using persistent block
        // }
        // else
#endif
        {
            for(int level = 0; level < _levels; level++)
            {
                if(level == 0)
                {
                    if(octave == 0)
                    {
#define ONLY_HORIZ false
                        // fprintf(stderr, "PRE ONE WAVE \n");
                        // build_octave_one_wave_input(conf, base_img, d_gauss_write, img_transfer);
#if ONLY_HORIZ
                        sycl::event horiz = build_octave_one_wave_input(conf, base_img, d_gauss_write, img_transfer);

#else

                        // WORKS AS BOTH HORIZ AND VERT FOR THIS LEVEL
                        // oct_obj._level_complete_events[0] =
                        //   build_octave_one_wave_input(conf, base_img, d_gauss_write, img_transfer);
#endif

                        // fprintf(stderr, "AFTER ONE WAVE \n");

                        // sycl::event horiz = horiz_from_input_image(conf, base_img, d_gauss_write, img_transfer);

                        // Storing event to class only for profiling not needed for normal use

                        sycl::event horiz = horiz_from_input_image(conf, base_img, d_gauss_write, img_transfer);
#if QUEUE_PROFILING
                        _input_horiz_event = horiz;
// #if ONLY_HORIZ
//                         // _input_horiz_event = horiz; // copy it for use later
// #else
//                         // _input_horiz_event = oct_obj._level_complete_events[0]; // When working as both horiz and
//                         // vert
// #endif
#endif
#if ONLY_HORIZ
                        oct_obj._level_complete_events[0] = vert_from_interm(octave, 0, gaussTableChoice, horiz);
#endif

                        // My own test case We do horiz normaly and then Vert with wave code

                        // sycl::event horiz = horiz_from_input_image(conf, base_img, d_gauss_write, img_transfer);
                        // sycl::event horiz = build_octave_one_wave_input(conf, base_img, d_gauss_write, img_transfer);

                        // _input_horiz_event = horiz; // copy it for use later

                        // horiz.wait();
                        // double frame_start =
                        //   horiz.template get_profiling_info<sycl::info::event_profiling::command_start>();
                        //
                        // double frame_end =
                        //   horiz.template get_profiling_info<sycl::info::event_profiling::command_end>();
                        //
                        // double frame_time = frame_end - frame_start;
                        //
                        // std::printf(
                        //   "Time to compute first horiz = %lf ns == %lf ms\n\n", frame_time, frame_time / 1000000);

                        // horiz.wait();

                        // popsift::sycl_common::print_region(oct_obj.getIntermediate(),
                        //                                    "AFTER HORIZ o=0 l=0",
                        //                                    0,
                        //                                    10,
                        //                                    0,
                        //                                    120,
                        //                                    oct_obj.getWidth(),
                        //                                    _device_queue);

                        // Test if vert part works alone

                        oct_obj._level_complete_events[0] =
                          build_octave_one_wave_input(conf, base_img, d_gauss_write, img_transfer);

                        // _input_horiz_event = oct_obj._level_complete_events[0]; // For one wave (both horiz and vert)

                        // oct_obj._level_complete_events[0].wait(); // Just for testing
                        // oct_obj._level_complete_events[0] = vert_from_interm(octave, 0, gaussTableChoice, horiz);

                        // oct_obj._level_complete_events[0].wait();

                        // popsift::sycl_common::print_region(oct_obj.getDataArrayHost()[0],
                        //                                    "AFTER HORIZ o=0 l=0",
                        //                                    0,
                        //                                    10,
                        //                                    0,
                        //                                    120,
                        //                                    oct_obj.getWidth(),
                        //                                    _device_queue);
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

#ifdef USE_PERSISTENT
    // // if(!sg_region.use_persistent_block)
    // if(!use_persistent_block) // Don't need DoG kernel if using persistent block as it's embedded
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
