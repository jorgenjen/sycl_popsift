#include "sycl_popsift/common/assist.h"
#include "sycl_popsift/common/debug_macros.hpp"
#include "sycl_popsift/non_sycl/sift_conf.hpp"
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

    // sycl::range local{64, 2};
    // sycl::range global{(size_t)grid_divide(dst_width, local.get(0)), (size_t)grid_divide(dst_height, local.get(1))};
    sycl::range local{2, 64};
    sycl::range global{(size_t)grid_divide(dst_height, local[0]), (size_t)grid_divide(dst_width, local[1])};

    // printf("\n\n\tIN downscale_from_prev_octave GLOBAL(%zu, %zu), OCTAVE=%d\n", global[0], global[1], octave);
    // sycl::event dependency = prev_oct_obj.getLevelEvent(_levels - PREV_LEVEL);
    sycl::event dependency = prev_oct_obj._level_complete_events[_levels - PREV_LEVEL];

    return _device_queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(dependency);
        cgh.parallel_for(sycl::nd_range(global, local), [=](sycl::nd_item<2> it) {
            // int x = it.get_global_id(0);
            // int y = it.get_global_id(1);
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
    // _device_queue.wait(); // just for now

    // POP_SYNC_CHK;
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

    // sycl::range local{1024, 1};
    // sycl::range global{(size_t)grid_divide(width, local.get(0)), (size_t)grid_divide(height, local.get(1))};
    sycl::range local{1, 1024};
    sycl::range global{(size_t)grid_divide(height, local[0]), (size_t)grid_divide(width, local[1])};

    // _device_queue.wait();

    float** data_array = oct_obj.getDataArray();
    float** dog_array = oct_obj.getDogArray();

#define reverse 1
#if reverse

    // Having this go from highest to lowest level would probably be better for cache hits
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
#else

    return _device_queue.parallel_for<make_dog>(
      sycl::nd_range{global, local},
      {octave_complete, oct_obj.getDataArrayWriteEvent(), oct_obj.getDogArrayWriteEvent()},
      [=](sycl::nd_item<2> it) {
          int x = it.get_global_id(1);
          int y = it.get_global_id(0);
          if(x >= width)
              return;

          float upper = data_array[max_level - 1][pos];
          for(int level = max_level - 2; level >= 0; --level)
          {
              const float lower = data_array[level][pos];
              dog_array[level][pos] = upper - lower;
              upper = lower;
          }
      });
#endif
}

inline sycl::event Pyramid::horiz_from_prev_level(int octave, int level, GaussTableChoice useInterpolatedGauss)
// sycl::event prev_level_write)
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

template<typename DerivedImage>
void Pyramid::build_pyramid(const Config& conf,
                            DerivedImage* base_img,
                            sycl::event d_gauss_write,
                            sycl::event img_transfer)
{
    // #if (PYRAMID_PRINT_DEBUG==1)
    //     cerr << "Entering " << __FUNCTION__ << " with base image "  << endl
    //          << "    type size         : " << base->type_size << endl
    //          << "    aligned byte size : " << base->a_width << "x" <<
    //          base->a_height << endl
    //          << "    pitch size        : " << base->pitch << "x" <<
    //          base->a_height << endl
    //          << "    original byte size: " << base->u_width << "x" <<
    //          base->u_height << endl
    //          << "    aligned pix size  : " << base->a_width/base->type_size
    //          << "x" << base->a_height << endl
    //          << "    original pix size : " << base->u_width/base->type_size
    //          << "x" << base->u_height << endl;
    // #endif // (PYRAMID_PRINT_DEBUG==1)

    GaussTableChoice gaussTableChoice;

    if(conf.getGaussMode() == Config::VLFeat_Relative)
        gaussTableChoice = Interpolated_FromPrevious;
    else
        gaussTableChoice = NotInterpolated_FromPrevious;

    std::cout << "Octaves: " << _num_octaves << " Levels: " << _levels << " GaussTableChoice: " << gaussTableChoice
              << std::endl;
    for(int octave = 0; octave < _num_octaves; octave++)
    {
        Octave& oct_obj = _octaves[octave];
        // std::vector<sycl::event>& lvl_events = oct_obj._level_complete_events;

        // Just for print outs
        int w = _octaves[octave].getWidth();
        int h = _octaves[octave].getHeight();

        for(int level = 0; level < _levels; level++)
        {
            if(level == 0)
            {
                if(octave == 0)
                {
                    sycl::event horiz =
                      horiz_from_input_image<DerivedImage>(conf, base_img, d_gauss_write, img_transfer);
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
    fprintf(stderr, "\n\tFinal octave is done so pyramid is done!!\n\n");

    for(int octave = 0; octave < _num_octaves; octave++) //
    {
        Octave& oct_obj = _octaves[octave];

        oct_obj._dog_done_event = dogs_from_blurred(octave, _levels, oct_obj._level_complete_events[_levels - 1]);
    }

    fprintf(stderr, "\n\tDone with DoG's\n\n");
}

} // namespace popsift
