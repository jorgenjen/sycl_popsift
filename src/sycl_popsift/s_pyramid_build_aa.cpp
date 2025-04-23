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
#include "sycl/kernel_bundle_enums.hpp"
#include "sycl/nd_range.hpp"
#include "sycl/usm.hpp"
#include "sycl_popsift/s_image.hpp"

#include <cmath>
#include <cstdio>

namespace popsift {

// Alias used for bindlessimages
namespace syclexp = sycl::ext::oneapi::experimental;

namespace normalizedSource {

// Used for ImageBindless
// Only used on input image (initial)
// And only works for it due to  filter and span selection

// aspect::ext_oneapi_bindless_sampled_image_fetch_2d
// This aspect is required to use sampled image need to add a check for that earlier in selection
template<bool if_required>
class Horiz
{
  private:
    syclexp::sampled_image_handle src;
    float* dst_data;
    popsift::GaussInfo* d_gauss;
    const int dst_w;
    const int dst_h;
    const float shift;

  public:
    Horiz(syclexp::sampled_image_handle src,
          float* dst_data,
          popsift::GaussInfo* d_gauss,
          const int dst_w,
          const int dst_h,
          const float shift)
      : src(src)
      , dst_data(dst_data)
      , d_gauss(d_gauss)
      , dst_w(dst_w)
      , dst_h(dst_h)
      , shift(shift)
    {}

    inline void operator()(sycl::nd_item<2> it) const
    {
        // Position to write to (image that has the size of scale up)
        const int write_x = it.get_global_id(1);
        // const int write_y = it.get_global_id(0) * dst_w;
        const int write_y = it.get_group(0);
        // Cant use it.get_global_range(1) inplace of dst_w due to if if_required width != it.get_global_range(1) and
        // hence positions would be off could be used in else case but not sure if it matters much (probs not)

        if constexpr(if_required)
        {
            // Destination width was not perfectly divisible with it.get_local_range(1)
            if(write_x >= dst_w)
                return;
        }

        const float* filter = &d_gauss->dd.filter[0];
        const int span = d_gauss->dd.span[0];
        const float read_x = (write_x + shift) / dst_w;
        const float read_y = (write_y + shift) / dst_h;

        // Could pass dimensions as a int2 and do vector wise
        // const sycl::float2 read_pos = sycl::float2{(write_x + shift) / dst_w, (write_y + shift) / dst_h};

        float out = 0.0f;

        // Look into sycl mad or fma (multiply-and-add instruction done in one clock cycle)
        // is probably done by the compiler anyways though

#pragma unroll
        for(int offset = span; offset > 0; offset--)
        {
            const float g = filter[offset];
            const float offrel = float(offset) / dst_w; // relative offset
            const float v1 = syclexp::sample_image<float>(src, sycl::float2{read_x - offrel, read_y});
            const float v2 = syclexp::sample_image<float>(src, sycl::float2{read_x + offrel, read_y});
            out += ((v1 + v2) * g);
        }

        const float& g = filter[0];
        const float v3 = syclexp::sample_image<float>(src, sycl::float2{read_x, read_y});
        out += (v3 * g);

        dst_data[write_x + write_y * dst_w] = out * 255.0f;
    };
};
} // namespace normalizedSource

namespace absoluteSource {

// Could also use sycl::specialization constaants that would make the decision in JIT compilation stage
// But not sure how well that would work for this kernel setup that is depends on octave w and h
// Think it is better with template and constexpr (should make multiple kernels as set up now)
template<bool if_required, bool initial>
class Horiz
{
  private:
    float* src;
    float* dst_data;
    popsift::GaussInfo* d_gauss;
    const int width;
    const int height; // not sure if height was needed here (verify)
    const int level;

  public:
    Horiz(float* src, float* dst_data, popsift::GaussInfo* d_gauss, const int width, const int height, int level)
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

        // Using template so that we can call kernel without if iff it's perfectly divisible by 128
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

class Vert
{
  private:
    float* intermediate; // or is it intermediate :D IDK
    float* dst_data;
    popsift::GaussInfo* d_gauss;
    const int width;
    const int height;
    const int level;

  public:
    Vert(float* intermediate,
         float* dst_data,
         popsift::GaussInfo* d_gauss,
         const int width,
         const int height,
         const int level)
      : intermediate(intermediate)
      , dst_data(dst_data)
      , d_gauss(d_gauss)
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
        // int x = it.get_global_id(0);
        // int y = it.get_global_id(1);
        int x = it.get_global_id(1);
        int y = it.get_global_id(0);

        const int span = d_gauss->inc.span[level];
        const float* filter = &d_gauss->inc.filter[level * GAUSS_ALIGN];

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

sycl::event Pyramid::horiz_from_input_image(const Config& conf,
                                            ImageBase* base,
                                            sycl::event d_gauss_write,
                                            sycl::event img_write)

{
    Octave& oct_obj = _octaves[0];

    const int width = oct_obj.getWidth();
    const int height = oct_obj.getHeight();

    sycl::range local{1, 128};
    sycl::range global{(size_t)height, (size_t)grid_divide(width, local[1])};

    // _device_queue.wait();

    if constexpr(USE_BINDLESS_INPUT && sycl::any_device_has<sycl::aspect::ext_oneapi_bindless_images>() &&
                 sycl::any_device_has<sycl::aspect::ext_oneapi_bindless_sampled_image_fetch_2d>())
    {
        // Bindless version
        fprintf(stderr, "Yay differey verison -- w=%d h=%d\n", width, height);
        float shift = 0.5f * powf(2.0f, conf.getUpscaleFactor());

        if(global[1] == width)
        {
            fprintf(stderr, "Running no if\n");
            // width % 128 = 0 and hence we don't need if check in kernel
            return _device_queue.parallel_for(
              sycl::nd_range{global, local},
              {d_gauss_write, img_write},
              normalizedSource::Horiz<false>(
                base->getInputImage(), oct_obj.getIntermediate(), _d_gauss, width, height, shift));
        }
        else
        {
            return _device_queue.parallel_for(
              sycl::nd_range{global, local},
              {d_gauss_write, img_write},
              normalizedSource::Horiz<true>(
                base->getInputImage(), oct_obj.getIntermediate(), _d_gauss, width, height, shift));
        }
    }
    else
    {
        // Running USM for input image
        if(global[1] == width)
        {
            fprintf(stderr, "Running no if IN USM MODE!!!\n");
            // width % 128 = 0 and hence we don't need if check in kernel
            return _device_queue.parallel_for(
              sycl::nd_range{global, local},
              {d_gauss_write, img_write},
              absoluteSource::Horiz<0, true>(
                base->getInputFloat(), oct_obj.getIntermediate(), _d_gauss, width, height, 0));
        }
        else
        {
            return _device_queue.parallel_for(
              sycl::nd_range{global, local},
              {d_gauss_write, img_write},
              absoluteSource::Horiz<1, true>(
                base->getInputFloat(), oct_obj.getIntermediate(), _d_gauss, width, height, 0));
        }
    }
}

sycl::event Pyramid::horiz_from_prev_level_basic(int octave, int level)
{
    Octave& oct_obj = _octaves[octave];

    const int width = oct_obj.getWidth();
    const int height = oct_obj.getHeight();

    // similar speed: dim3 block( 32,  4 ); dim3 block( 32,  3 ); dim3 block( 32,  2 );
    // (32, 8) most stable good perf on GTX 980 TI -- need to test different for me sycl implementation

    sycl::range local{8, 32};
    sycl::range global{(size_t)grid_divide(height, local[0]), (size_t)grid_divide(width, local[1])};

    // Should be fine to do arithmetic on getDaraArray as it's only first level of a pointer hence no dereferences
    // needed and we can still use device memory and not shared
    float* prev_level = oct_obj.getDataArrayHost()[level - 1]; // src
    float* cur_intm = oct_obj.getIntermediate();               // dst_data

    // sycl::event dependency = oct_obj.getLevelEvent(level-1); // wrong
    sycl::event prev_lvl_event = oct_obj._level_complete_events[level - 1]; // prev level
    return _device_queue.parallel_for(
      sycl::nd_range{global, local},
      prev_lvl_event,
      absoluteSource::Horiz<1, false>(prev_level, cur_intm, _d_gauss, width, height, level));
}

sycl::event Pyramid::vert_from_interm_basic(int octave, int level, sycl::event intm_write)
{
    Octave& oct_obj = _octaves[octave];

    const int width = oct_obj.getWidth();
    const int height = oct_obj.getHeight();

    sycl::range local{2, 64};
    sycl::range global{(size_t)grid_divide(height, local[0]), (size_t)grid_divide(width, local[1])};

    float* intermediate = oct_obj.getIntermediate();
    float* dst_data = oct_obj.getDataArrayHost()[level]; // Uses host array to get device pointer

    // fprintf(stderr,
    //         "INSIDE VERT_FROM_INTERM_BASIC --> Event created: %p Status: %d\n",
    //         &intm_write,
    //         intm_write.get_info<sycl::info::event::command_execution_status>());

    // _device_queue.wait_and_throw();

    return _device_queue.parallel_for(sycl::nd_range{global, local},
                                      intm_write,
                                      absoluteSource::Vert(intermediate, dst_data, _d_gauss, width, height, level));
}

} // namespace popsift
