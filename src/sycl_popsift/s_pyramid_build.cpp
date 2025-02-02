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
// }

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
}

inline sycl::event Pyramid::vert_from_interm(int octave,
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
    // might have to add event return here for compilation before it's done
}

void Pyramid::build_pyramid(const Config& conf, Image* base_img, sycl::event d_gauss_write)
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
        // cudaStream_t stream = oct_obj.getStream();
        //
        sycl::event prev_level_write[_levels - 1]; // clange extension this is?
        for(int level = 0; level < _levels; level++)
        {
            if(level == 0)
            {
                if(octave == 0)
                {
                    cout << "first ocatve first level" << endl;
                    sycl::event horiz = horiz_from_input_image(conf, base_img, d_gauss_write);
                    _device_queue.wait();
                    prev_level_write[level] = vert_from_interm(octave, 0, gaussTableChoice, horiz);
                    _device_queue.wait();
                }
                else
                {
                    // Octave& prev_oct_obj = _octaves[octave - 1];
                    // cuda::event_wait(prev_oct_obj.getEventScaleDone(), stream, __FILE__, __LINE__);
                    // prev_oct_obj.getEventScaleDone().wait();
                    // cout << "Sclae done ready for next dude to start yay" << endl;
                    // downscale_from_prev_octave(octave, stream);
                }
            }
            else
            {
                if(octave == 0) // TMP: for now only allow octave 0 as I develop the rest
                {
                    // might be problem with event copy but should be fine as it is all copies
                    // and sycl should dela with that fine
                    sycl::event horiz =
                      horiz_from_prev_level(octave, level, gaussTableChoice, prev_level_write[level - 1]);
                    _device_queue.wait();

                    prev_level_write[level] = vert_from_interm(octave, level, gaussTableChoice, horiz);
                    _device_queue.wait();

                    if(level == _levels - PREV_LEVEL)
                    {
                        oct_obj.setEventScaleDone(prev_level_write[level]);
                    }
                }
                // if(level == _levels - PREV_LEVEL)
                // {
                //     cuda::event_record(oct_obj.getEventScaleDone(), stream, __FILE__, __LINE__);
                // }
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
