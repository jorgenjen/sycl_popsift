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

template<bool if_required, bool initial>
class Horiz_old
{
  private:
    float* src;
    float* dst_data;
    popsift::GaussInfo* d_gauss;
    const int width;
    const int height; // not sure if height was needed here (verify)
    const int level;

  public:
    Horiz_old(float* src, float* dst_data, popsift::GaussInfo* d_gauss, const int width, const int height, int level)
      : src(src)
      , dst_data(dst_data)
      , d_gauss(d_gauss)
      , width(width)
      , height(height)
      , level(level)

    {}

    // Not sure if inlining makes this worse or better...
    // might remove function calls but not sure exactly
    inline void operator()(sycl::nd_item<2> it) const
    {
        int x = it.get_global_id(1);
        int y = it.get_global_id(0);

        const float* filter;
        int span;
        if constexpr(initial)
        {
            // is always from source image and level 0 // called once

            // Look into packing the struct differntly to avoid splitting but this might be the best way(idk)
            filter = &d_gauss->dd.filter[0];
            span = d_gauss->dd.span[0];
        }
        else
        {
            filter = &d_gauss->inc.filter[level * GAUSS_ALIGN];
            span = d_gauss->inc.span[level];
        }

        // could have two different kernels one with this and one without
        // depending on if it is perfectly divisible by 128 but might not be worth it... Test

        // Using template so that we can call kernel without if if it's perfectly divisible by 128
        // and hence would not be needed // hopefully it works like this look into
        // NOTE: Look into if template makes this multiple kernels or not if not we might benefit from spliting them and
        // having them as different kernels mby different namespace to separete them
        if constexpr(if_required)
        {
            // Using contexpr so it is evaluated at compile time (should force it to make multiple kernels I think)
            if(x >= width || y >= height)
            {
                return;
            }
        }

        int idx;
        float g;
        float val;
        float out = 0.0f;

        // Look into sycl mad or fma (multiply-and-add instruction done in one clock cycle)
        // is probably done by the compiler anyways though
        for(int offset = span; offset > 0; offset--)
        {
            g = filter[offset];

            idx = x - offset;
            val = idx < 0 ? src[y * width] : src[idx + y * width];

            out += (val * g);

            idx = x + offset;
            val = idx >= width ? src[width - 1 + y * width] : src[idx + y * width];
            out += (val * g);
        }

        g = filter[0];
        val = src[x + y * width];
        out += (val * g);

        dst_data[x + y * width] = out;
    };
};
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
        int x = it.get_global_id(1);
        int y = it.get_global_id(0);

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
    }
};

} // namespace normalizedSource

// Currently not using normalized source so moving  to s_pyramid_bulid_aa.cpp where
// absolut source resides
// sycl::event Pyramid::horiz_from_input_image(const Config& conf, Image* base, std::vector<sycl::event> dependencies)
// {
//     Octave& oct_obj = _octaves[0];
//
//     const int width = oct_obj.getWidth();
//     const int height = oct_obj.getHeight();
//
//     float shift = 0.5f * powf(2.0f, conf.getUpscaleFactor());
//
//     std::size_t grid_x = grid_divide(width, 128); // different from CUDA popsift
//
//     sycl::range global{grid_x, (size_t)height};
//
//     const float* filter = &_d_gauss->dd.filter[0];
//     const int span = _d_gauss->dd.span[0];
//
//     std::cout << "INPUT IMAGE -- LEVEL 0" << std::endl;
//
//     return _device_queue.submit([&](sycl::handler& cgh) {
//         cgh.depends_on(dependencies);
//         sycl::range local{128, 1};
//         cgh.parallel_for(sycl::nd_range{global, local},
//                          normalizedSource::Horiz(base->getInput(), oct_obj.getIntermediate(), filter, span, width));
//     });
//
//     // _device_queue.wait(); // temporary waiting here remove in future
//
//     // Just for verification -- Remove!
//     // printf("Print intermediate \n");
//     // _device_queue.submit([&](sycl::handler& cgh) {
//     //     float* intermediate = oct_obj.getIntermediateArray()[0];
//     //     cgh.single_task([=]() {
//     //         sycl::ext::oneapi::experimental::printf("\n\n");
//     //         for(int y = height - 8; y < height; ++y)
//     //         {
//     //             for(int x = width - 8; x < width; ++x)
//     //             {
//     //                 // for(int y = 0; y < 13; ++y)
//     //                 // {
//     //                 //     for(int x = 0; x < 13; ++x)
//     //                 //     {
//     //                 // printf("\t\tValue at %d %d: %f\n", x, y, tex2D<float>(src_linear_tex, x, y));
//     //                 sycl::ext::oneapi::experimental::printf("%10.6f ", intermediate[x + y * (width)]);
//     //             }
//     //             sycl::ext::oneapi::experimental::printf("\n");
//     //         }
//     //         sycl::ext::oneapi::experimental::printf("\n\n");
//     //     });
//     // });
//
//     // _device_queue.wait();
//     // print out intermediate here to see that it works like it should !
//
//     // NOTE: Is an error check after kernel that is conditionally set bu an ifdef
//     // consider implementing something similar
//     // POP_SYNC_CHK;
// }

} // namespace popsift
