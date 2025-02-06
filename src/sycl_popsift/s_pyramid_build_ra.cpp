#include "sycl/ext/oneapi/experimental/builtins.hpp"
#include "sycl/ext/oneapi/experimental/graph.hpp"
#include "sycl/usm.hpp"
#include "sycl_popsift/common/assist.h"
#include "sycl_popsift/gauss_filter.hpp"
#include "sycl_popsift/popsift.hpp"
#include "sycl_popsift/sift_pyramid.hpp"

#include <cmath>
#include <cstddef>

namespace popsift {

namespace normalizedSource {

// NOTE: Should probably be writen as a functor instead of a lambda inside of a function
// swap out the texture memory with normal memory or work-group memory (scratch pad memory/shared memory)
// and then the rest of the kernel should be the same just use nd_range and it should be the same as this
// quite straight forward (I think and HOPE!)

// Functor for reusability
class Horiz
{
  private:
    float* input;
    float* intermediate;
    const float* filter;
    const int span;
    const int width;

  public:
    Horiz(float* input, float* intermediate, const float* filter, const int span, const int width)
      : input(input)
      , intermediate(intermediate)
      , filter(filter)
      , span(span)
      , width(width)
    {}

    // Not sure if inlining makes this worse or better...
    // might remove function calls but not sure exactly
    inline void operator()(sycl::nd_item<2> it) const
    {
        // kernel code
        int x = it.get_global_id(0);
        int y = it.get_global_id(1);

        // could have two different kernels one with this and one without
        // depending on if it is perfectly divisible by 128 but might not be worth it... Test
        if(x >= width)
            return;

        float out = 0.0f;

#pragma unroll
        for(int offset = span; offset > 0; offset--)
        {
            const float& g = filter[offset];
            const auto v1_pos = x - offset;
            const auto v2_pos = x + offset;

            // clamp to left for - and clamp to right for + // does the smae as cudaAddressModeClamp for
            // textures in cuda used in popsift
            const float v1 = v1_pos < 0 ? input[y * width] : input[v1_pos + y * width];
            const float v2 = v2_pos >= width ? input[width - 1 + y * width] : input[v2_pos + y * width];
            out += ((v1 + v2) * g);
        }
        const float& g = filter[0];
        const float v3 = input[x + y * width];
        out += (v3 * g);

        intermediate[x + y * width] = out * 255.0f;

        // JUst for verification -- remove!
        // if(x == 0 && y == 0)
        // {
        //     sycl::ext::oneapi::experimental::printf("\n\nSCALED UP AS GIVEN TO KERNEL FUNCTOR width = %d:\n", width);
        //     for(int y = 852 - 13; y < 852; ++y)
        //     {
        //         for(int x = width - 13; x < width; ++x)
        //         {
        //             // printf("\t\tValue at %d %d: %f\n", x, y, tex2D<float>(src_linear_tex, x, y));
        //             sycl::ext::oneapi::experimental::printf("%06.2f ", input[x + y * (width)] * 255.0f);
        //         }
        //         sycl::ext::oneapi::experimental::printf("\n");
        //     }
        //     sycl::ext::oneapi::experimental::printf("\n\n");
        // }
    }
};

} // namespace normalizedSource

sycl::event Pyramid::horiz_from_input_image(const Config& conf, Image* base, std::vector<sycl::event> dependencies)
{
    Octave& oct_obj = _octaves[0];

    const int width = oct_obj.getWidth();
    const int height = oct_obj.getHeight();

    float shift = 0.5f * powf(2.0f, conf.getUpscaleFactor());

    std::size_t grid_x = grid_divide(width, 128); // different from CUDA popsift

    sycl::range global{grid_x, (size_t)height};

    const float* filter = &_d_gauss->dd.filter[0];
    const int span = _d_gauss->dd.span[0];

    std::cout << "INPUT IMAGE -- LEVEL 0" << std::endl;

    return _device_queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(dependencies);
        sycl::range local{128, 1};
        cgh.parallel_for(sycl::nd_range{global, local},
                         normalizedSource::Horiz(base->getInput(), oct_obj.getIntermediate(), filter, span, width));
    });

    // _device_queue.wait(); // temporary waiting here remove in future

    // Just for verification -- Remove!
    // printf("Print intermediate \n");
    // _device_queue.submit([&](sycl::handler& cgh) {
    //     float* intermediate = oct_obj.getIntermediateArray()[0];
    //     cgh.single_task([=]() {
    //         sycl::ext::oneapi::experimental::printf("\n\n");
    //         for(int y = height - 8; y < height; ++y)
    //         {
    //             for(int x = width - 8; x < width; ++x)
    //             {
    //                 // for(int y = 0; y < 13; ++y)
    //                 // {
    //                 //     for(int x = 0; x < 13; ++x)
    //                 //     {
    //                 // printf("\t\tValue at %d %d: %f\n", x, y, tex2D<float>(src_linear_tex, x, y));
    //                 sycl::ext::oneapi::experimental::printf("%10.6f ", intermediate[x + y * (width)]);
    //             }
    //             sycl::ext::oneapi::experimental::printf("\n");
    //         }
    //         sycl::ext::oneapi::experimental::printf("\n\n");
    //     });
    // });

    // _device_queue.wait();
    // print out intermediate here to see that it works like it should !

    // NOTE: Is an error check after kernel that is conditionally set bu an ifdef
    // consider implementing something similar
    // POP_SYNC_CHK;
}

} // namespace popsift
