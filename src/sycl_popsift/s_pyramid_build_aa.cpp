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
        if(initial)
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
        if(if_required)
        {
            // Had switch before but I believe that if should server the same purpose
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

sycl::event Pyramid::horiz_from_input_image(const Config& conf, Image* base, std::vector<sycl::event> dependencies)
{
    Octave& oct_obj = _octaves[0];

    const int width = oct_obj.getWidth();
    const int height = oct_obj.getHeight();

    float shift = 0.5f * powf(2.0f, conf.getUpscaleFactor());

    sycl::range local{1, 128};
    sycl::range global{(size_t)height, (size_t)grid_divide(width, local[1])};

    const float* filter = &_d_gauss->dd.filter[0];
    const int span = _d_gauss->dd.span[0];

    if(global[1] == width)
    {
        fprintf(stderr, "Running no if\n");
        // width % 128 = 0 and hence we don't need if check in kernel
        return _device_queue.parallel_for(
          sycl::nd_range{global, local},
          dependencies,
          absoluteSource::Horiz<0, true>(base->getInput(), oct_obj.getIntermediate(), _d_gauss, width, height, 0));
    }
    else
    {
        return _device_queue.parallel_for(
          sycl::nd_range{global, local},
          dependencies,
          absoluteSource::Horiz<1, true>(base->getInput(), oct_obj.getIntermediate(), _d_gauss, width, height, 0));
    }
}

sycl::event Pyramid::horiz_from_prev_level_basic(int octave, int level, sycl::event prev_level_write)
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
    float* prev_level = oct_obj.getDataArray()[level - 1]; // src
    float* cur_intm = oct_obj.getIntermediate();           // dst_data

    return _device_queue.parallel_for(
      sycl::nd_range{global, local},
      prev_level_write,
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
    float* dst_data = oct_obj.getDataArray()[level]; // should be fine just pointer arithmetic

    // TODO: Consider adding template argument like in Horiz and either have one for all, w, h and nothing but might be
    // excessive need to test (and figure out if the templates works as I hope making separeate kernels or somother
    // smart thing)
    return _device_queue.parallel_for(sycl::nd_range{global, local},
                                      intm_write,
                                      absoluteSource::Vert(intermediate, dst_data, _d_gauss, width, height, level));
}
} // namespace popsift
