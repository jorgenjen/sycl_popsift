/*
 * Copyright 2016-2017, Simula Research Laboratory
 *           2018-2024, University of Oslo
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "common/assist.h"
#include "gauss_filter.hpp"
#include "sift_constants.hpp"
#include "sift_pyramid.hpp"
#include "sycl/nd_range.hpp"

#include <cmath>

namespace popsift {
namespace absoluteSource {

// SHould use this one instead of Horiz in absolute source as this one makes sense to use in both
// situations and we dont need to divide the image initially, wasting performance.
template<int if_required>
class Horiz
{
  private:
    float* src;
    float* dst_data;
    const float* filter;
    const int span;
    const int width;
    const int height; // not sure if height was needed here (verify)

  public:
    Horiz(float* src, float* dst_data, const float* filter, const int span, const int width, const int height)
      : src(src)
      , dst_data(dst_data)
      , filter(filter)
      , span(span)
      , width(width)
      , height(height)
    {}

    // Not sure if inlining makes this worse or better...
    // might remove function calls but not sure exactly
    inline void operator()(sycl::nd_item<2> it) const
    {
        int x = it.get_global_id(0);
        int y = it.get_global_id(1);

        // could have two different kernels one with this and one without
        // depending on if it is perfectly divisible by 128 but might not be worth it... Test

        // Using template so that we can call kernel without if if it's perfectly divisible by 128
        // and hence would not be needed
        switch(if_required)
        {
            case 1:
                if(x >= width || y >= height)
                {
                    return;
                }
                break;
            default: break; // do nothing
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

template<int if_required>
class Horiz_new
{
  private:
    float* src;
    float* dst_data;
    const float* filter;
    const int span;
    const int width;
    const int height; // not sure if height was needed here (verify)

  public:
    Horiz_new(float* src, float* dst_data, const float* filter, const int span, const int width, const int height)
      : src(src)
      , dst_data(dst_data)
      , filter(filter)
      , span(span)
      , width(width)
      , height(height)
    {}

    // Not sure if inlining makes this worse or better...
    // might remove function calls but not sure exactly
    inline void operator()(sycl::nd_item<2> it) const
    {
        // int x = it.get_global_id(0);
        // int y = it.get_global_id(1);
        int x = it.get_global_id(1);
        int y = it.get_global_id(0);

        // could have two different kernels one with this and one without
        // depending on if it is perfectly divisible by 128 but might not be worth it... Test

        // Using template so that we can call kernel without if if it's perfectly divisible by 128
        // and hence would not be needed
        switch(if_required)
        {
            case 1:
                if(x >= width || y >= height)
                {
                    return;
                }
                break;
            default: break; // do nothing
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

class Vert
{
  private:
    float* intermediate; // or is it intermediate :D IDK
    float* dst_data;
    const float* filter;
    const int span;
    const int width;
    const int height;
    const int level;

  public:
    Vert(float* intermediate,
         float* dst_data,
         const float* filter,
         const int span,
         const int width,
         const int height,
         const int level)
      : intermediate(intermediate)
      , dst_data(dst_data)
      , filter(filter)
      , span(span)
      , width(width)
      , height(height)
      , level(level)
    {}

    // seems to be slightly different from cuda PopSIFT but could be due to the way GPU vs cpu handles
    // floats as the difference is noticable for the intermediate futher out in the decimal values and I gues
    // it compunds making a bigger difference causing differences of up to almost 5? but that does seem like it is
    // way too much.... so Need to investigate what is going on Intermediate after horiz is different some places on
    // the 3 decimal so could be from the way the texture engine is getting the data and reading the uchars as
    // floats and that conversion is not as accurate as the software way I've done here in sycl.. Will assume it is
    // correct for now and move on
    inline void operator()(sycl::nd_item<2> it) const
    {
        int x = it.get_global_id(0);
        int y = it.get_global_id(1);

        // This need to be here  I think. Was not in cuda PopSift probs due to textures making it a non issue
        if(x >= width || y >= height)
            return;

        int idy;
        float g;
        float val;
        float out = 0.0f;

        for(int offset = span; offset > 0; offset--)
        {
            g = filter[offset];

            idy = y - offset;
            val = idy < 0 ? intermediate[x] : intermediate[x + idy * width]; // clamp edge
            out += (val * g);

            idy = y + offset;
            val = idy >= height ? intermediate[x + (height - 1) * width] : intermediate[x + idy * width]; // clamp edge
            out += (val * g);
        }

        g = filter[0];
        val = intermediate[x + y * width];
        out += (val * g);

        dst_data[x + y * width] = out;
    }
};

} // namespace absoluteSource

// Moved from s_pyramid_build_ra.cpp as  I don't use normalized source when using USM
sycl::event Pyramid::horiz_from_input_image(const Config& conf, Image* base, std::vector<sycl::event> dependencies)
{
    Octave& oct_obj = _octaves[0];

    const int width = oct_obj.getWidth();
    const int height = oct_obj.getHeight();

    float shift = 0.5f * powf(2.0f, conf.getUpscaleFactor());

    // Grid divide is different from cuda due to the way nd_range kerneels
    // work in sycl but serves the same purpose
    // sycl::range local{128, 1};
    // sycl::range global{(size_t)grid_divide(width, local[0]), (size_t)height};

    sycl::range local{1, 128};
    sycl::range total{(size_t)height, (size_t)grid_divide(width, local[1])};

    const float* filter = &_d_gauss->dd.filter[0];
    const int span = _d_gauss->dd.span[0];

    // if(global[0] == width)
    if(total[1] == width)
    {
        fprintf(stderr, "Running no if\n");
        // width % 128 = 0 and hence we don't need if check in kernel
        return _device_queue.parallel_for(
          sycl::nd_range{total, local},
          dependencies,
          absoluteSource::Horiz_new<0>(base->getInput(), oct_obj.getIntermediate(), filter, span, width, height));
    }
    else
    {
        return _device_queue.parallel_for(
          sycl::nd_range{total, local},
          dependencies,
          absoluteSource::Horiz<1>(base->getInput(), oct_obj.getIntermediate(), filter, span, width, height));
    }
}

// Should only be called wiht a level > 0
sycl::event Pyramid::horiz_from_prev_level_basic(int octave, int level, sycl::event prev_level_write)
{
    Octave& oct_obj = _octaves[octave];

    const int width = oct_obj.getWidth();
    const int height = oct_obj.getHeight();

    // similar speed: dim3 block( 32,  4 ); dim3 block( 32,  3 ); dim3 block( 32,  2 );
    // (32, 8) most stable good perf on GTX 980 TI -- need to test different for me sycl implementation

    sycl::range local{32, 8}; // coult move inside of submit but probs done by compiler
                              // and replaced .get(0) with the values inline
    sycl::range global{(size_t)grid_divide(width, local.get(0)), (size_t)grid_divide(height, local.get(1))};

    // _device_queue.wait();
    // printf("\nGlobal in horiz from prev level wop wop(%zu, %zu), level=%d:\n", global[0], global[1], level);

    // Not sure if it is better to have these varaibles inside of the submit or not
    float* prev_level = oct_obj.getDataArray()[level - 1]; // src
    // // fprintf(stderr, "\nThis is fine!!!\n");
    // float* cur_intm = oct_obj.getIntermediateArray()[level]; // dst_data
    float* cur_intm = oct_obj.getIntermediate(); // dst_data
    // fprintf(stderr, "\nThis is fine!!!\n");
    const float* filter = &_d_gauss->inc.filter[level * GAUSS_ALIGN];
    // fprintf(stderr, "\nThis is fine!!!\n");
    const int span = _d_gauss->inc.span[level];
    // fprintf(stderr, "\nThis is fine!!!\n");

    sycl::event e = _device_queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(prev_level_write);
        cgh.parallel_for(sycl::nd_range{global, local},
                         absoluteSource::Horiz<1>(prev_level, cur_intm, filter, span, width, height));
    });

    // fprintf(stderr, "\nAFTER BEFORE WAIT!!!\n");
    // e.wait();
    // fprintf(stderr, "\nAFTER WAIT!!! before return LEVEL = %d \n", level);
    // printf("AFTER HORIZ IN 'From prev' -- LEVEL = %d", level);
    return e;

    // _device_queue.wait();
    // absoluteSource::horiz<<<grid, block, 0, stream>>>(
    //   oct_obj.getDataTexPoint(), oct_obj.getIntermediateSurface(), level);
    // POP_SYNC_CHK;
}

sycl::event Pyramid::vert_from_interm_basic(int octave, int level, sycl::event intm_write)
{
    Octave& oct_obj = _octaves[octave];

    const int width = oct_obj.getWidth();
    const int height = oct_obj.getHeight();

    sycl::range local{64, 2};
    sycl::range global{(size_t)grid_divide(width, local.get(0)), (size_t)grid_divide(height, local.get(1))};

    // printf("\n\n\tvert_from_interm_basic GLOBAL(%zu, %zu), LEVEL=%d\n", global[0], global[1], level);
    // printf("\n\tSpan=%d \n", _d_gauss->inc.span[level]);

    // float* intermediate = oct_obj.getIntermediateArray()[level];
    float* intermediate = oct_obj.getIntermediate();
    float* dst_data = oct_obj.getDataArray()[level];
    const int span = _d_gauss->inc.span[level];
    const float* filter = &_d_gauss->inc.filter[level * GAUSS_ALIGN];

    sycl::event e = _device_queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(intm_write); // Set horiz write to intermediate as dependency --
        // Sycl not in order queue by default hence needed
        // std::cout << "Past intm dependency I think" << std::endl;
        cgh.parallel_for(sycl::nd_range(global, local),
                         absoluteSource::Vert(intermediate, dst_data, filter, span, width, height, level));
    });

    // fprintf(stderr, "After Vert_aa was submited\n");
    // e.wait();
    // printf("\n\t AFTER VERT SUBMIT CALL BEFORE RETURN level = %d\n", level);
    return e;

    // _device_queue.wait();

    // _device_queue.submit([&](sycl::handler& cgh) {
    //     cgh.single_task([=]() {
    //         sycl::ext::oneapi::experimental::printf(
    //           "\n\nAfter Vert: y(%d, %d) x(%d, %d)\n", height - 13, height, width - 13, width);
    //         for(int y = height - 13; y < height; ++y)
    //         {
    //             for(int x = width - 13; x < width; ++x)
    //             {
    //                 sycl::ext::oneapi::experimental::printf("%06.2f ", dst_data[x + y * (width)]);
    //             }
    //             sycl::ext::oneapi::experimental::printf("\n");
    //         }
    //         sycl::ext::oneapi::experimental::printf("\n\n");
    //     });
    // });
    // absoluteSource::vert<<<grid, block, 0, stream>>>(oct_obj.getIntermDataTexPoint(), oct_obj.getDataSurface(),
    // level); POP_SYNC_CHK;
}

} // namespace popsift
