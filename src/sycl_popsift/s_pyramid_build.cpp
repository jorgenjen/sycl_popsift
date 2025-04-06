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
inline sycl::event Pyramid::downscale_from_prev_octave(int octave, sycl::event prev_octave_done)
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

    return _device_queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(prev_octave_done);
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
class make_dog;

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

    return _device_queue.parallel_for<make_dog>(
      sycl::nd_range{global, local},
      {octave_complete, oct_obj.getDataArrayWriteEvent(), oct_obj.getDogArrayWriteEvent()},
      [=](sycl::nd_item<2> it) {
          int x = it.get_global_id(1);
          int y = it.get_global_id(0);
          if(x > width)
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
          if(x > width)
              return;

          float a = data_array[0][x + y * width];
          for(int level = 0; level < max_level - 1; level++)
          {
              const float b = data_array[level + 1][x + y * width];

              dog_array[level][x + y * width] = b - a;
              a = b;
          }
      });
#endif
}

inline sycl::event Pyramid::horiz_from_prev_level(int octave,
                                                  int level,
                                                  GaussTableChoice useInterpolatedGauss,
                                                  sycl::event prev_level_write)
{
    switch(useInterpolatedGauss)
    {
        // case Interpolated_FromPrevious: horiz_from_prev_level_pairs(octave, level, stream); break;
        case Interpolated_FromPrevious: cout << "horiz_from_prev_level_pairs not implemented yet"; break;
        case NotInterpolated_FromPrevious: return horiz_from_prev_level_basic(octave, level, prev_level_write); break;
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

std::vector<sycl::event> Pyramid::build_pyramid(const Config& conf,
                                                Image* base_img,
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

    // cudaDeviceSynchronize();

    GaussTableChoice gaussTableChoice;

    if(conf.getGaussMode() == Config::VLFeat_Relative)
        gaussTableChoice = Interpolated_FromPrevious;
    else
        gaussTableChoice = NotInterpolated_FromPrevious;

    std::cout << "Octaves: " << _num_octaves << " Levels: " << _levels << " GaussTableChoice: " << gaussTableChoice
              << std::endl;
    for(int octave = 0; octave < _num_octaves; octave++)
    {
        // fprintf(stderr, "BEFORE ACCESS OF octave %d", octave);
        Octave& oct_obj = _octaves[octave];

        // Just for print outs
        int w = _octaves[octave].getWidth();
        int h = _octaves[octave].getHeight();

        for(int level = 0; level < _levels; level++)
        {
            if(level == 0)
            {
                if(octave == 0)
                {
                    sycl::event horiz = horiz_from_input_image(conf, base_img, {d_gauss_write, img_transfer});
                    horiz.wait();
                    fprintf(stderr, "past horiz so problems occur in vert??\n\n");
                    oct_obj._level_complete_events[0] = vert_from_interm(octave, 0, gaussTableChoice, horiz);
                    oct_obj._level_complete_events[0].wait();
                    fprintf(stderr, "past horiz so problems occur in vert??\n\n");

                    // horiz.wait();
                    // popsift::sycl_common::print_region(oct_obj.getIntermediate(),
                    //                                    "INTERMEDIATE after horiz first octave -- ",
                    //                                    w - 8,
                    //                                    w,
                    //                                    h - 8,
                    //                                    h,
                    //                                    w,
                    //                                    _device_queue);
                    //
                    // _device_queue.wait();
                    // popsift::sycl_common::print_region(oct_obj.getDataArray()[level],
                    //                                    "First octave after vert -- ",
                    //                                    w - 8,
                    //                                    w,
                    //                                    h - 8,
                    //                                    h,
                    //                                    w,
                    //                                    _device_queue);
                }
                else
                {
                    fprintf(stderr, "BEFORE ACCESS OF PREV OCTAVE!!!!! in octave %d", octave);
                    Octave& prev_oct_obj = _octaves[octave - 1];

                    // fprintf(stderr, "Before downscale to Octave %d", octave);
                    oct_obj._level_complete_events[0] =
                      downscale_from_prev_octave(octave, prev_oct_obj._level_complete_events[_levels - PREV_LEVEL]);

                    oct_obj._level_complete_events[0].wait();
                    fprintf(stderr, "AFTER downscale to Octave %d", octave);
                }
            }
            else
            {
                fprintf(stderr, "Before horiz on octave %d at level %d\n", octave, level);
                sycl::event horiz =
                  horiz_from_prev_level(octave, level, gaussTableChoice, oct_obj._level_complete_events[level - 1]);

                horiz.wait();

                fprintf(stderr, "AFTER WAIT ON horiz level %d at octave %d\n", level, octave);
                // Hope horiz is fine to use even though it goes out of scope after the line but should have been
                // copied by then I think eventough vert_from_interm takes it as reference...
                fprintf(stderr, "RIGHT BEFORE FAILURE level=%d -- octave=%d\n ", level, octave);
                fprintf(stderr,
                        "Event created: %p Status: %d\n",
                        &horiz,
                        horiz.get_info<sycl::info::event::command_execution_status>());

                // oct_obj._level_complete_events[level] = vert_from_interm(octave, level, gaussTableChoice, horiz);
                // oct_obj._level_complete_events[level] = vert_from_interm_basic(octave, level, horiz); // Use directly
                sycl::event tmp_event = vert_from_interm_basic(octave, level, horiz); // Use directly
                tmp_event.wait();
                fprintf(stderr, "AFTER VERT FROM interm %d\n", level);
                oct_obj._level_complete_events[level] = sycl::event();
                fprintf(stderr, "after vector assignment\n");

                // oct_obj._level_complete_events[level].wait();
            }
        }
    }
    // _octaves[_num_octaves - 1]._level_complete_events[_levels - 1].wait();
    _device_queue.wait();
    fprintf(stderr, "\n\tFinal octave is done so pyramid is done!!\n\n");

    std::vector<sycl::event> make_dog_events;
    make_dog_events.reserve(_num_octaves);
    for(int octave = 0; octave < _num_octaves; octave++) //
    {
        Octave& oct_obj = _octaves[octave];

        // Final level must be complete before we can do dogs (aka all levels as final depends on all before it)
        make_dog_events.push_back(dogs_from_blurred(octave, _levels, oct_obj._level_complete_events[_levels - 1]));
        // make_dog_events[octave].wait();
        // fprintf(stderr, "Done octave %d\n", octave);
    }
    _device_queue.wait();

#define me_oct 4
#define me_lvl 5

    Octave& oct = _octaves[me_oct];
    int w = oct.getWidth();
    int h = oct.getHeight();

    popsift::sycl_common::print_region(
      oct.getDataArrayHost()[me_lvl], "Me doggy dog dog  new -> ", w - 8, w, h - 8, h, w, _device_queue);

    return make_dog_events;
}

} // namespace popsift
