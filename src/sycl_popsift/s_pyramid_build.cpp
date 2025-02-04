#include "sycl_popsift/common/assist.h"
#include "sycl_popsift/common/debug_macros.hpp"
#include "sycl_popsift/non_sycl/sift_conf.hpp"
#include "sycl_popsift/s_image.hpp"
#include "sycl_popsift/sift_octave.hpp"
#include "sycl_popsift/sift_pyramid.hpp"

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
inline sycl::event Pyramid::downscale_from_prev_octave(int octave, const sycl::event& prev_octave_done)
{
    Octave& oct_obj = _octaves[octave];
    Octave& prev_oct_obj = _octaves[octave - 1];

    // downscaled with and height (current for this octave)
    const int dst_width = oct_obj.getWidth();
    const int dst_height = oct_obj.getHeight();

    const int src_width = prev_oct_obj.getWidth();
    const int src_height = prev_oct_obj.getHeight();

    float* src_data = prev_oct_obj.getDataArray()[_levels - PREV_LEVEL];
    float* dst_data = oct_obj.getDataArray()[0]; // Level 0 is the subsampled result

    sycl::range local{64, 2};
    sycl::range global{(size_t)grid_divide(dst_width, local.get(0)), (size_t)grid_divide(dst_height, local.get(1))};

    printf("\n\n\tIN downscale_from_prev_octave GLOBAL(%zu, %zu), OCTAVE=%d\n", global[0], global[1], octave);

    return _device_queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(prev_octave_done);
        cgh.parallel_for(sycl::nd_range(global, local), [=](sycl::nd_item<2> it) {
            int x = it.get_global_id(0);
            int y = it.get_global_id(1);

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

inline sycl::event Pyramid::horiz_from_prev_level(int octave,
                                                  int level,
                                                  GaussTableChoice useInterpolatedGauss,
                                                  const sycl::event& prev_level_write)
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

inline sycl::event Pyramid::vert_from_interm(int octave,
                                             int level,
                                             GaussTableChoice useInterpolatedGauss,
                                             const sycl::event& intm_write)
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

void Pyramid::build_pyramid(const Config& conf,
                            Image* base_img,
                            const sycl::event& d_gauss_write,
                            const sycl::event& img_transfer)
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
    for(uint32_t octave = 0; octave < _num_octaves; octave++)
    {
        Octave& oct_obj = _octaves[octave];

        for(int level = 0; level < _levels; level++)
        {
            if(level == 0)
            {
                if(octave == 0)
                {
                    sycl::event horiz = horiz_from_input_image(conf, base_img, {d_gauss_write, img_transfer});
                    oct_obj._level_complete_events[0] = vert_from_interm(octave, 0, gaussTableChoice, horiz);
                }
                else
                {
                    Octave& prev_oct_obj = _octaves[octave - 1];

                    fprintf(stderr, "Before downscale to Octave %d", octave);
                    oct_obj._level_complete_events[0] =
                      downscale_from_prev_octave(octave, prev_oct_obj._level_complete_events[_levels - PREV_LEVEL]);

                    oct_obj._level_complete_events[0].wait();
                    fprintf(stderr, "AFTER downscale to Octave %d", octave);
                }
            }
            else
            {
                sycl::event horiz =
                  horiz_from_prev_level(octave, level, gaussTableChoice, oct_obj._level_complete_events[level - 1]);

                // Hope horiz is fine to use even though it goes out of scope after the line but should have been
                // copied by then I think eventough vert_from_interm takes it as reference...
                oct_obj._level_complete_events[level] = vert_from_interm(octave, level, gaussTableChoice, horiz);
            }
        }
    }

    // for (int octave = 0; octave < _num_octaves; octave++) {
    //     Octave &oct_obj = _octaves[octave];
    //     cudaStream_t stream = oct_obj.getStream();
    //     dogs_from_blurred(octave, _levels, stream);
    // }

    // for (int octave = 0; octave < _num_octaves; octave++) {
    //     Octave &oct_obj = _octaves[octave];
    //     cudaStream_t stream = oct_obj.getStream();
    //     cudaStreamSynchronize(stream);
    // }
}

} // namespace popsift
