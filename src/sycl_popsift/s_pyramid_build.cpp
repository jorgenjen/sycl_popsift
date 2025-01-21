#include "sycl_popsift/non_sycl/sift_conf.hpp"
#include "sycl_popsift/s_image.hpp"
#include "sycl_popsift/sift_octave.hpp"
#include "sycl_popsift/sift_pyramid.hpp"

using std::cerr;
using std::cout;
using std::endl;

namespace popsift {

// namespace gauss { // is only for one function get_by_2_pick_every_second
// }

void Pyramid::build_pyramid(const Config& conf, Image* base_img)
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
        for(int level = 0; level < _levels; level++)
        {
            if(level == 0)
            {
                if(octave == 0)
                {
                    cout << "first ocatve first level" << endl;
                    // horiz_from_input_image(conf, base, stream);
                    // vert_from_interm(octave, 0, stream, gaussTableChoice);
                }
                // else
                // {
                //     Octave& prev_oct_obj = _octaves[octave - 1];
                //     cuda::event_wait(prev_oct_obj.getEventScaleDone(), stream, __FILE__, __LINE__);
                //     downscale_from_prev_octave(octave, stream);
                // }
            }
            // else
            // {
            //     horiz_from_prev_level(octave, level, stream, gaussTableChoice);
            //     vert_from_interm(octave, level, stream, gaussTableChoice);
            //     if(level == _levels - PREV_LEVEL)
            //     {
            //         cuda::event_record(oct_obj.getEventScaleDone(), stream, __FILE__, __LINE__);
            //     }
            // }
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
