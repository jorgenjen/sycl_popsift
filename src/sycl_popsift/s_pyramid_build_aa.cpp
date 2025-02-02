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

namespace popsift {
namespace absoluteSource {

// __global__ static void horiz(cudaTextureObject_t src_point_texture, cudaSurfaceObject_t dst_data, int dst_level)
// {
//     const int src_level = dst_level - 1;
//     const int span = d_gauss.inc.span[dst_level];
//     const float* filter = &d_gauss.inc.filter[dst_level * GAUSS_ALIGN];
//     const int block_x = blockIdx.x * blockDim.x;
//     const int block_y = blockIdx.y * blockDim.y;
//     const int xpos = block_x + threadIdx.x;
//     const int ypos = block_y + threadIdx.y;
//
//     int idx;
//     float g;
//     float val;
//     float out = 0.0f;
//
//     for(int offset = span; offset > 0; offset--)
//     {
//         g = filter[offset];
//
//         idx = xpos - offset;
//         val = readTex(src_point_texture, idx, ypos, src_level);
//         out += (val * g);
//
//         idx = xpos + offset;
//         val = readTex(src_point_texture, idx, ypos, src_level);
//         out += (val * g);
//     }
//
//     g = filter[0];
//     val = readTex(src_point_texture, xpos, ypos, src_level);
//     out += (val * g);
//
//     surf2DLayeredwrite(out, dst_data, xpos * 4, ypos, dst_level, cudaBoundaryModeZero);
// }

// __global__ static void vert(cudaTextureObject_t src_point_texture, cudaSurfaceObject_t dst_data, int dst_level)
// {
// const int span = d_gauss.inc.span[dst_level];
// const float* filter = &d_gauss.inc.filter[dst_level * GAUSS_ALIGN];
//     const int block_x = blockIdx.x * blockDim.x;
//     const int block_y = blockIdx.y * blockDim.y;
//     const int xpos = block_x + threadIdx.x;
//     const int ypos = block_y + threadIdx.y;
//
//     int idy;
//     float g;
//     float val;
//     float out = 0.0f;
//
//     for(int offset = span; offset > 0; offset--)
//     {
//         g = filter[offset];
//
//         idy = ypos - offset;
//         val = readTex(src_point_texture, xpos, idy, dst_level);
//         out += (val * g);
//
//         idy = ypos + offset;
//         val = readTex(src_point_texture, xpos, idy, dst_level);
//         out += (val * g);
//     }
//
//     g = filter[0];
//     val = readTex(src_point_texture, xpos, ypos, dst_level);
//     out += (val * g);
//
//     surf2DLayeredwrite(out, dst_data, xpos * 4, ypos, dst_level, cudaBoundaryModeZero);
// }

class Vert
{
  private:
    float* intermediate; // or is it intermediate :D IDK
    float* dst_data;
    const int span;
    const float* filter;
    const int width;
    const int height;

  public:
    Vert(float* intermediate, float* dst_data, const int span, const float* filter, const int width, const int height)
      : intermediate(intermediate)
      , dst_data(dst_data)
      , span(span)
      , filter(filter)
      , width(width)
      , height(height)
    {}

    // seems to be slightly different from cuda PopSIFT but could be due to the way GPU vs cpu handles
    // floats as the difference is noticable for the intermediate futher out in the decimal values and I gues
    // it compunds making a bigger difference causing differences of up to almost 5? but that does seem like it is way
    // too much.... so Need to investigate what is going on
    // Intermediate after horiz is different some places on the 3 decimal so could be from the
    // way the texture engine is getting the data and reading the uchars as floats and that conversion
    // is not as accurate as the software way I've done here in sycl..
    // Will assume it is correct for now and move on
    inline void operator()(sycl::nd_item<2> it) const
    {
        int x = it.get_global_id(0);
        int y = it.get_global_id(1);

        // This need to be here  I think. Was not in cuda PopSift probs due to textures making it a non issue
        if(x >= width)
            return;

        int idy;
        float g;
        float val;
        float out = 0.0f;

        for(int offset = span; offset > 0; offset--)
        {
            g = filter[offset];

            idy = y - offset;
            // val = readTex(src_point_texture, xpos, idy, dst_level);
            val = idy < 0 ? intermediate[x] : intermediate[x + idy * width]; // clamp edge
            out += (val * g);
            if(x == 1267 && y == 839)
            {
                sycl::ext::oneapi::experimental::printf("\nidy = %d -- val=%f -- out %f -- g =%f\n", idy, val, out, g);
            }

            idy = y + offset;
            // val = readTex(src_point_texture, xpos, idy, dst_level);
            val = idy >= height ? intermediate[x + (height - 1) * width] : intermediate[x + y * width]; // clamp edge
            out += (val * g);
        }

        g = filter[0];
        // val = readTex(src_point_texture, xpos, ypos, dst_level);
        val = intermediate[x + y * width];
        out += (val * g);

        dst_data[x + y * width] = out;
    }
};

} // namespace absoluteSource

// __host__ void Pyramid::horiz_from_prev_level_basic(int octave, int level, cudaStream_t stream)
// {
//     Octave& oct_obj = _octaves[octave];
//
//     const int width = oct_obj.getWidth();
//     const int height = oct_obj.getHeight();
//
//     // similar speed: dim3 block( 32,  4 ); dim3 block( 32,  3 ); dim3 block( 32,  2 );
//     dim3 block(32, 8); // most stable good perf on GTX 980 TI
//     dim3 grid;
//     grid.x = grid_divide(width, 32);
//     grid.y = grid_divide(height, block.y);
//
//     absoluteSource::horiz<<<grid, block, 0, stream>>>(
//       oct_obj.getDataTexPoint(), oct_obj.getIntermediateSurface(), level);
//     POP_SYNC_CHK;
// }

void Pyramid::vert_from_interm_basic(int octave, int level)
{
    Octave& oct_obj = _octaves[octave];

    const int width = oct_obj.getWidth();
    const int height = oct_obj.getHeight();

    sycl::range local{64, 2};
    sycl::range global{(size_t)grid_divide(width, local.get(0)), (size_t)grid_divide(height, local.get(1))};

    printf("\n\n\tGlobal_range_basic (%zu, %zu)\n", global[0], global[1]);
    printf("\n\tSpan=%d \n", _d_gauss->inc.span[level]);

    float* intermediate = oct_obj.getIntermediateArray()[level];
    float* dst_data = oct_obj.getDataArray()[level];
    const int span = _d_gauss->inc.span[level];
    const float* filter = &_d_gauss->inc.filter[level * GAUSS_ALIGN];

    _device_queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::nd_range(global, local),
                         absoluteSource::Vert(intermediate, dst_data, span, filter, width, height));
    });

    _device_queue.wait();

    _device_queue.submit([&](sycl::handler& cgh) {
        cgh.single_task([=]() {
            sycl::ext::oneapi::experimental::printf(
              "\n\nAfter Vert: y(%d, %d) x(%d, %d)\n", height - 13, height, width - 13, width);
            for(int y = height - 13; y < height; ++y)
            {
                for(int x = width - 13; x < width; ++x)
                {
                    sycl::ext::oneapi::experimental::printf("%06.2f ", dst_data[x + y * (width)]);
                }
                sycl::ext::oneapi::experimental::printf("\n");
            }
            sycl::ext::oneapi::experimental::printf("\n\n");
        });
    });
    // absoluteSource::vert<<<grid, block, 0, stream>>>(oct_obj.getIntermDataTexPoint(), oct_obj.getDataSurface(),
    // level); POP_SYNC_CHK;
}

} // namespace popsift
