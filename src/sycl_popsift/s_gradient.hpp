/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once

// #include "sycl_popsift/sift_constants.hpp"

// #include <cinttypes>
#include <sycl/sycl.hpp>
// #include <sycl/ext/intel/math.hpp> // ilncluding this one here instead of #include <sycl/ext/oneapi/common/math.hpp>
// to avoid conflicts #include <sycl/ext/oneapi/common/algorithm.hpp> // For sycl::clamp, sycl::min, sycl::max

#include <cstdio>

// Should probably not have this as it's own file as I will probably not have many versions like the cuda code
namespace popsift {
/*
 * We are wasting time by computing gradiants on demand several
 * times. We could precompute gradiants for all pixels once, as
 * other code does, but the number of features should be too low
 * to make that feasible. So, we take this performance hit.
 * Especially punishing in the descriptor computation.
 *
 * Also, we are always computing from the closest blur level
 * as Lowe expects us to do. Other implementations compute the
 * gradiant always from the original image, which we think is
 * not in the spirit of the hierarchy is blur levels. That
 * assumption would only hold if we could simply downscale to
 * every first level of every octave ... which is not compatible
 * behaviour.
 */

/* get_gradient() works for both point texture and linear interpolation
 * textures. The reason is that readTex must add 0.5 for coordinates in
 * both cases to access the expected pixel.
 */

#define TO_CLAMP true
static inline void get_gradient(float& grad,
                                float& theta,
                                const int x,
                                const int y,
                                const int width,
                                const int height,
                                float** data,
                                const int level,
                                bool do_print)
{
    // float dx = readTex(layer, x + 1.0f, y, level) - readTex(layer, x - 1.0f, y, level);
    // float dy = readTex(layer, x, y + 1.0f, level) - readTex(layer, x, y - 1.0f, level);

    // Not sure if we need clamping? Think the extremas are not along the edges and hence shoudl be fine?

    // TODO: Look into if we need clamping or not (currently using to be safe)
    // SEEMS TO BE FINE WHEN NOT USING CLAMING

#define XPOS 90.565437f
#define YPOS 137.517151f

#if TO_CLAMP
    const int safe_x = sycl::clamp(x, 0, width - 1);
    const int safe_y = sycl::clamp(y, 0, height - 1);

    const int right_x = sycl::min(x + 1, width - 1) + safe_y * width;
    const int left_x = sycl::max(safe_x - 1, 0) + safe_y * width;

    const int upper_y = safe_x + sycl::min(safe_y + 1, height - 1) * width;
    const int lower_y = safe_x + sycl::max(safe_y - 1, 0) * width;

    float dx = data[level][right_x] - data[level][left_x];
    float dy = data[level][upper_y] - data[level][lower_y];

    grad = sycl::hypot(dx, dy);
    theta = sycl::atan2(dy, dx);
#else

    float dx = data[level][x + 1 + y * width] - data[level][x - 1 + y * width];
    float dy = data[level][x + (y + 1) * width] - data[level][x + (y - 1) * width];
    grad = sycl::hypot(dx, dy);  // Hypotenuse -- sqrt(dx^2 + dy^2)
    theta = sycl::atan2(dy, dx); // Inverse tangent of dy/dx
    // theta = atan2f(dy, dx)       // if using non sycl verson as in orientatoin
    // Need to include it in such a cas e cmath
#endif

    if(do_print)
    {
        // sycl::ext::oneapi::experimental::printf(
        //   "\n\tx=%d y=%d || dx=%f - dy=%f || grad=%f - thetat=%f ||\n", x, y, dx, dy, grad, theta);
        sycl::ext::oneapi::experimental::printf(
          "\n\tx=%d y=%d lvl=%d || dx= %f - %f = %f || dy= %f - %f = %f  || grad=%f - thetat=%f ||\n",
          x,
          y,
          level,
          data[level][x + 1 + y * width],
          data[level][x - 1 + y * width],
          dx,
          data[level][x + (y + 1) * width],
          data[level][x + (y - 1) * width],
          dy,
          grad,
          theta);
    }
}

// Could mby compute the gradient on cpu side right after we are done building the pyramid atleast for the
// first octave and mby second as that is where most of the extremas tend to be and they are done first
// Then we do recompute for the rest
static inline void get_gradient(
  float& grad, float& theta, const int x, const int y, const int width, const int height, float** data, const int level)
{
    // TODO: Look into if we need clamping or not (currently using to be safe)
    // SEEMS TO BE FINE WHEN NOT USING CLAMING

    // #define TO_CLAMP false

#if TO_CLAMP
    const int safe_x = sycl::clamp(x, 0, width - 1);
    const int safe_y = sycl::clamp(y, 0, height - 1);

    const int right_x = sycl::min(x + 1, width - 1) + safe_y * width;
    const int left_x = sycl::max(safe_x - 1, 0) + safe_y * width;

    const int upper_y = safe_x + sycl::min(safe_y + 1, height - 1) * width;
    const int lower_y = safe_x + sycl::max(safe_y - 1, 0) * width;

    float dx = data[level][right_x] - data[level][left_x];
    float dy = data[level][upper_y] - data[level][lower_y];

    grad = sycl::hypot(dx, dy);
    theta = sycl::atan2(dy, dx);
#else

    float dx = data[level][x + 1 + y * width] - data[level][x - 1 + y * width];
    float dy = data[level][x + (y + 1) * width] - data[level][x + (y - 1) * width];
    grad = sycl::hypot(dx, dy);  // Hypotenuse -- sqrt(dx^2 + dy^2)
    theta = sycl::atan2(dy, dx); // Inverse tangent of dy/dx
#endif
}

/* A version of get_gradiant that works for a (32,1,1) threadblock
 * and pulls data to shared memory before computing. Data is pulled
 * less frequently, meaning that we do not rely on the L1 cache.
 */
// __device__ static inline void get_gradiant32(
//   float& grad, float& theta, const int x, const int y, cudaTextureObject_t layer, const int level)
// {
//     const int idx = threadIdx.x;
//
//     __shared__ float x_array[34];
//
//     for(int i = idx; i < 34; i += blockDim.x)
//     {
//         x_array[i] = readTex(layer, x + i - 1.0f, y, level);
//     }
//     __syncthreads();
//
//     const float dx = x_array[idx + 2] - x_array[idx];
//
//     const float dy = readTex(layer, x + idx, y + 1.0f, level) - readTex(layer, x + idx, y - 1.0f, level);
//
//     grad = hypotf(dx, dy); // __fsqrt_rz(dx*dx + dy*dy);
//     theta = atan2f(dy, dx);
// }

}; // namespace popsift
