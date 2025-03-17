/*
 * Copyright 2016-2017, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "sycl/accessor.hpp"
#include "sycl/ext/oneapi/experimental/builtins.hpp"
#include "sycl/group_barrier.hpp"
#include "sycl/kernel_bundle_enums.hpp"
#include "sycl/memory_enums.hpp"
#include "sycl/vector.hpp"
#include "sycl_popsift/common/assist.h"
#include "sycl_popsift/common/debug_macros.hpp"
// #include "common/excl_blk_prefix_sum.h"
// #include "common/warp_bitonic_sort.h"
// #include "s_gradiant.h"
#include "sycl_popsift/common/warp_bitonic_sort.hpp"
#include "sycl_popsift/non_sycl/sift_conf.hpp"
#include "sycl_popsift/sift_constants.hpp"
#include "sycl_popsift/sift_pyramid.hpp"

#include <cinttypes>
#include <cmath>
#include <cstdio>

using namespace popsift;
using namespace std;

/* Smoothing like VLFeat is the default mode.
 * If you choose to undefine it, you get the smoothing approach taken by OpenCV
 */
#define WITH_VLFEAT_SMOOTHING

#define HALF_PRECISION 1
namespace popsift {

template<int half>
inline float divide(const float& a, const float& b)
{
    switch(half)
    {
        case 1: return sycl::half_precision::divide(a, b);
        case 0: return sycl::native::divide(a, b);
        default: return a / b;
    }
}

// base e exponential of x
template<int half>
inline float exp(const float& x)
{
    switch(half)
    {
        case 1: return sycl::half_precision::exp(x);
        case 0: return sycl::native::exp(x);
        default: return sycl::exp(x);
    }
}

inline void get_gradient(
  float& grad, float& theta, const int x, const int y, const float* leveled_layer, const int& w, const int& h)
{
    float dx = leveled_layer[x + 1 + y * w] - leveled_layer[x - 1 + y * w];
    float dy = leveled_layer[x + (y + 1) * w] - leveled_layer[x + (y - 1) * w];
    grad = sycl::hypot(dx, dy); // Hypotenuse -- sqrt(dx^2 + dy^2)
    theta = atan2f(dy, dx);     // Inverse tangent of dy/dx
}

/*
 * Histogram smoothing helper
 */
// inline static float smoothe(const float* const src, const int bin)
inline static float smoothe(const sycl::local_accessor<float, 1> src, const int bin)
{ // removed template as it did nothing
    const int prev = (bin == 0) ? ORI_NBINS - 1 : bin - 1;
    const int next = (bin == ORI_NBINS - 1) ? 0 : bin + 1;

    const float f = (src[prev] + src[bin] + src[next]) / 3.0f;

    return f;
}

class ori_par
{
  private:
    const int octave;
    const int ext_prefix_sum;
    float** layer;
    const int w;
    const int h;
    sycl::local_accessor<float, 1> hist;
    sycl::local_accessor<float, 1> sm_hist; // rename work_group_hist?? or something sycl like
    sycl::local_accessor<float, 1> refined_angle;
    sycl::local_accessor<float, 1> yval;
    DevBuffers* dobuf;

  public:
    ori_par(const int octave,
            const int ext_prefix_sum,
            float** layer,
            const int w,
            const int h,
            sycl::local_accessor<float, 1> hist,
            sycl::local_accessor<float, 1> sm_hist,
            sycl::local_accessor<float, 1> refined_angle,
            sycl::local_accessor<float, 1> yval,
            DevBuffers* dobuf)
      : octave(octave)
      , ext_prefix_sum(ext_prefix_sum)
      , layer(layer)
      , w(w)
      , h(h)
      , hist(hist)
      , sm_hist(sm_hist)
      , refined_angle(refined_angle)
      , yval(yval)
      , dobuf(dobuf)
    {}

    inline void operator()(sycl::nd_item<2> it) const
    {
        // Possition in the grid but 0 is always 1 so should be same as it.get_group(1)
        const int extremum_index = it.get_group(1) * it.get_group(0);

        // Implement this I suppose
        // if(popsift::all(extremum_index >= dct.ext_ct[octave]))
        //     return; // a few trailing warps

        const int iext_off = dobuf->i_ext_off[octave][extremum_index];
        const InitialExtremum* iext = &dobuf->i_ext_dat[octave][iext_off];

        // Initialize hist to zero each work-item does 2 in work-group
        hist[it.get_local_id(1) + 0] = 0.0f;
        hist[it.get_local_id(1) + 32] = 0.0f;

        /* keypoint fractional geometry */
        const float x = iext->xpos;
        const float y = iext->ypos;
        const int level = iext->lpos; // old_level;
        const float sig = iext->sigma;

        /* orientation histogram radius */
        const float sigw = ORI_WINFACTOR * sig;
        const int32_t rad = (int)sycl::round((3.0f * sigw)); // do this needs to be int32_t and not just int?

        const float factor = popsift::divide<HALF_PRECISION>(-0.5f, (sigw * sigw));
        const int sq_thres = rad * rad;

        // int xmin = sycl::max(1,     (int)floor(x - rad));
        // int xmax = sycl::min(w - 2, (int)floor(x + rad));
        // int ymin = sycl::max(1,     (int)floor(y - rad));
        // int ymax = sycl::min(h - 2, (int)floor(y + rad));
        int xmin = sycl::max(1, (int)sycl::round(x) - rad);
        int xmax = sycl::min(w - 2, (int)sycl::round(x) + rad);
        int ymin = sycl::max(1, (int)sycl::round(y) - rad);
        int ymax = sycl::min(h - 2, (int)sycl::round(y) + rad);

        int wx = xmax - xmin + 1;
        int hy = ymax - ymin + 1;
        int loops = wx * hy;

        // Consider computing these values in it's own kernel and store the needed ones in an array that would
        // correspond to their index. Making one thread compute one instead of all threads in work-grop doing it and
        // then everyone can read the value from global memory(and it will be cached) could also mby help reduce
        // register pressure if that is a problem

        sycl::group_barrier(it.get_group());

        // TODO: Make sub-group version
        // Doing the loop for the whole work_group instead of sub-group might make anothe version later that uses the
        // sub_group

        // CUDA code is using the fact that a warp is 32 threads but in sycl we need it to support any...
        // could template to have one that wokrs like cuda and one that works for any sub_group size
        for(int i = it.get_local_id(1); sycl::any_of_group(it.get_group(), i < loops); i += it.get_local_range(1))
        // I think we can just use the condition don't see why they all must enter the loop
        // to then have the condition checked in the first if and does nothing more and exits ... so threads will
        // diverge anyways...
        {
            // Why does it need to run if any
            if(i < loops)
            {
                // Current threads x and y position in image at the level
                // Still safe and no need to clamp
                int yy = i / wx + ymin;
                int xx = i % wx + xmin;

                float grad;
                float theta;
                popsift::get_gradient(grad, theta, xx, yy, layer[level], w, h);

                float dx = xx - x;
                float dy = yy - y;

                int sq_dist = dx * dx + dy * dy;
                if(sq_dist <= sq_thres)
                {
                    float weight = grad * popsift::exp<HALF_PRECISION>(sq_dist * factor);

                    // int bidx = (int)rintf( __fdividef( ORI_NBINS * (theta + M_PI), M_PI2 ) );
                    int bidx =
                      (int)sycl::round(popsift::divide<HALF_PRECISION>(float(ORI_NBINS) * (theta + M_PI), M_PI2));

                    if(bidx > ORI_NBINS)
                    {
                        sycl::ext::oneapi::experimental::printf("Crashing: bin %d theta %f :-)\n", bidx, theta);
                    }
                    if(bidx < 0)
                    {
                        sycl::ext::oneapi::experimental::printf("Crashing: bin %d theta %f :-)\n", bidx, theta);
                    }

                    bidx = (bidx == ORI_NBINS) ? 0 : bidx;

                    sycl::atomic_ref<float,
                                     sycl::memory_order_relaxed,
                                     sycl::memory_scope_work_group, // Change to sub_group for sub_group version
                                     sycl::access::address_space::local_space>(hist[bidx]) += weight;
                }
            }
        }

        sycl::group_barrier(it.get_group());

#ifdef WITH_VLFEAT_SMOOTHING
        for(int i = 0; i < 3; i++)
        {
            sm_hist[it.get_local_id(1) + 0] = smoothe(hist, it.get_local_id(1) + 0);
            sm_hist[it.get_local_id(1) + 32] = smoothe(hist, it.get_local_id(1) + 32);
            sycl::group_barrier(it.get_group());
            hist[it.get_local_id(1) + 0] = smoothe(sm_hist, it.get_local_id(1) + 0);
            hist[it.get_local_id(1) + 32] = smoothe(sm_hist, it.get_local_id(1) + 32);
            sycl::group_barrier(it.get_group());
        }

        sm_hist[it.get_local_id(1) + 0] = hist[it.get_local_id(1) + 0];
        sm_hist[it.get_local_id(1) + 32] = hist[it.get_local_id(1) + 32];
        sycl::group_barrier(it.get_group());
#else  // not WITH_VLFEAT_SMOOTHING
       // TODO: Implement this version aswell
        for(int bin = threadIdx.x; bin < ORI_NBINS; bin += blockDim.x)
        {
            int prev2 = bin - 2;
            int prev1 = bin - 1;
            int next1 = bin + 1;
            int next2 = bin + 2;
            if(prev2 < 0)
                prev2 += ORI_NBINS;
            if(prev1 < 0)
                prev1 += ORI_NBINS;
            if(next1 >= ORI_NBINS)
                next1 -= ORI_NBINS;
            if(next2 >= ORI_NBINS)
                next2 -= ORI_NBINS;
            sm_hist[bin] = (hist[prev2] + hist[next2] + (hist[prev1] + hist[next1]) * 4.0f + hist[bin] * 6.0f) / 16.0f;
        }
        __syncthreads();
#endif // not WITH_VLFEAT_SMOOTHING

        // sub-cell refinement of the histogram cell index, yielding the angle
        // not necessary to initialize, every cell is computed

        for(int bin = it.get_local_id(1); sycl::any_of_group(it.get_group(), bin < ORI_NBINS);
            bin += it.get_local_range(1))
        {
            const int prev = bin == 0 ? ORI_NBINS - 1 : bin - 1;
            const int next = bin == ORI_NBINS - 1 ? 0 : bin + 1;

            bool predicate = (bin < ORI_NBINS) && (sm_hist[bin] > max(sm_hist[prev], sm_hist[next]));

            const float num = predicate ? 3.0f * sm_hist[prev] - 4.0f * sm_hist[bin] + 1.0f * sm_hist[next] : 0.0f;
            // const float num  = predicate ?   2.0f * sm_hist[prev]
            //                                - 4.0f * sm_hist[bin]
            //                                + 2.0f * sm_hist[next]
            //                              : 0.0f;
            const float denB = predicate ? 2.0f * (sm_hist[prev] - 2.0f * sm_hist[bin] + sm_hist[next]) : 1.0f;

            const float newbin = popsift::divide<HALF_PRECISION>(num, denB);

            predicate = (predicate && newbin >= 0.0f && newbin <= 2.0f);

            refined_angle[bin] = predicate ? prev + newbin : -1;
            yval[bin] = predicate ? -(num * num) / (4.0f * denB) + sm_hist[prev] : -INFINITY;
        }
        sycl::group_barrier(it.get_group());

        sycl::vec<int, 2> best_index(it.get_local_id(1), it.get_local_id(1) + 32);

        // BitonicSort
        BitonicSort::Warp32<float> sorter(yval, it);
    }
};

}

void Pyramid::orientation(const Config& conf)
{
    // Wait so that the computation is done before the memcpy
    // Look for ways to make this part faster (less waits the better)
    _device_queue.wait();

    readDescCountersFromDevice().wait();

    int ext_total = 0;
    for(int o : _hct.ext_ct)
    {
        if(o > 0)
        {
            ext_total += o;
        }
    }

    // Something is wrong...
    printf("\n\text_total for all octaves = %d", ext_total);

    // Seems to do nothing in my case...

    // Filter functions are only called if necessary. They are very expensive,
    // therefore add 10% slack.
    // if(conf.getFilterMaxExtrema() > 0 && int(conf.getFilterMaxExtrema() * 1.1) < ext_total)
    // {
    //     ext_total = extrema_filter_grid(conf, ext_total);
    // }

    // TODO: ADd spport for this one -- unlikely to run but need it it will only do something if ext_total is larger
    // than the max_extrema per octave 100 000 by default
    // reallocExtrema(ext_total);

    int ext_ct_prefix_sum = 0;
    for(int octave = 0; octave < _num_octaves; octave++)
    {
        _hct.ext_ps[octave] = ext_ct_prefix_sum;
        ext_ct_prefix_sum += _hct.ext_ct[octave];
    }
    _hct.ext_total = ext_ct_prefix_sum;

    // for( int octave=0; octave<_num_octaves; octave++ )
    for(int octave = _num_octaves - 1; octave >= 0; octave--)
    {
        Octave& oct_obj = _octaves[octave];

        size_t num = _hct.ext_ct[octave];

        if(num > 0)
        {
            // "local needs to divide global perfectly
            sycl::range local{1, 32};
            sycl::range global{1, num * 32};

            _device_queue.submit([&](sycl::handler& cgh) {
                // sycl::local_accessor<float, 1> -- is the type
                auto hist = sycl::local_accessor<float, 1>(64, cgh);
                auto sm_hist = sycl::local_accessor<float, 1>(64, cgh);
                auto refined_angle = sycl::local_accessor<float, 1>(64, cgh);
                auto yval = sycl::local_accessor<float, 1>(64, cgh);
                cgh.parallel_for(sycl::nd_range{global, local},
                                 popsift::ori_par(octave,
                                                  _hct.ext_ps[octave],
                                                  oct_obj.getDataArray(),
                                                  oct_obj.getWidth(),
                                                  oct_obj.getHeight(),
                                                  hist,
                                                  sm_hist,
                                                  refined_angle,
                                                  yval,
                                                  _dobuf));
            });
            // TODO: Understand why this is needed and if I need something similar
            // if(octave != 0)
            // {
            //     cuda::event_record(oct_obj.getEventOriDone(), oct_str, __FILE__, __LINE__);
            //     cuda::event_wait(oct_obj.getEventOriDone(), oct_0_str, __FILE__, __LINE__);
            // }
        }

        /* Compute and set the orientation prefixes on the device */
        // dim3 block;
        // dim3 grid;
        // block.x = 32;
        // block.y = 32;
        // grid.x = 1;

        // Runs after all octav
        // sycl::range local{1, 32, 32};
        // sycl::range global{1, 1, 1};
        //
        // ori_prefix_sum<<<grid, block, 0, oct_0_str>>>(ext_ct_prefix_sum, _num_octaves);
        // POP_SYNC_CHK;
        //
        // cudaDeviceSynchronize();
        _device_queue.wait();
    }
}
