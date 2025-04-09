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

#include <sycl/sycl.hpp>
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
    // switch(half)
    // {
    //     case 1: return sycl::half_precision::divide(a, b);
    //     case 0: return sycl::native::divide(a, b);
    //     default: return a / b;
    // }

    // Using constexpr should guarantee that the branch is taken at compile time (but 99% sure that was already the
    // case)
    if constexpr(half == 1)
        return sycl::half_precision::divide(a, b);
    else if constexpr(half == 0)
        return sycl::native::divide(a, b);
    else
        return a / b;
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

template<bool useSubGroup>
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
    ExtremaCounters* dct;

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
            DevBuffers* dobuf,
            ExtremaCounters* dct)
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
      , dct(dct)
    {}

    inline void operator()(sycl::nd_item<2> it) const
    {
        // Should be optimized away during compile time
        auto group = [&]() {
            if constexpr(useSubGroup)
                return it.get_sub_group();
            else
                return it.get_group();
        }();

        // My cpu reports it supports 4 8 16 32 64 sizes subgroups
        // So the largest is 64 but it uses 8 in the kernel hence need re think this

        if(it.get_global_linear_id() == 0)
        {
            if constexpr(useSubGroup)
                sycl::ext::oneapi::experimental::printf("Sub group size %d\n", group.get_local_range()[0]);
            else
                sycl::ext::oneapi::experimental::printf("work group size %d\n",
                                                        group.get_local_range()[0] * group.get_local_range()[1]);
        }

        // Possition in the grid but 0 is always 1 so should be same as it.get_group(1)
        // This is the extrema index as we are getting index in terms of work_groups
        // const int extremum_index = it.get_group(1) * it.get_group(0);
        const int extremum_index = it.get_group(1); // it.get_grpu(0) is zero

        // Only if the whole group (warp) has extremum_index higher than dct->ext_ct[octave]
        // So the number of extremas in the octave but the kernel params are based on that count
        // So not sure how this could happen? TODO: See if we can remove this check
        if(sycl::all_of_group(group, extremum_index >= dct->ext_ct[octave]))
            return; // A few trailing sub groups

        // This does also seem like a strange way of doig it getting index from one
        // and using it to get the data

        // NOTE: The resulting iext_off should always be the same as extremum_index
        // So not sure why it's done like this seems like we can get rid of it
        // and just use extremum_index
        const int iext_off = dobuf->i_ext_off[octave][extremum_index]; // Should get rid of this must be an artifiact
        // from they added the second layer (octave) to structure of i_ext_dat

        if(iext_off != extremum_index)
            sycl::ext::oneapi::experimental::printf("\n\n\t\tWAHHHHATER FUCKER NO WAYYYYY WHYYYYYYY\n\n");

        const InitialExtremum* iext = &dobuf->i_ext_dat[octave][iext_off];

        // if(sycl::floor(iext->xpos) == 812)
        // if(octave == 0 && group.leader())
        // {
        //     // sycl::ext::oneapi::experimental::printf("WORKYYY %f\n\n", iext->xpos);
        //     sycl::ext::oneapi::experimental::printf("WORKYYY extremum_index %d, %f --> group[1] %d -- group[0] %d
        //     \n\n",
        //                                             extremum_index,
        //                                             dobuf->i_ext_dat[octave][iext_off].xpos,
        //                                             it.get_group(1),
        //                                             it.get_group(0));
        // }

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

        sycl::group_barrier(group);

        for(int i = it.get_local_id(1); sycl::any_of_group(group, i < loops); i += it.get_local_range(1))
        // I think we can just use the condition don't see why they all must enter the loop
        // to then have the condition checked in the first if and does nothing more and exits ... so threads will
        // diverge anyways...
        {
            // This is used to only alow the ones that should continue run...
            // Should try to just have the condition check in the loop itself don't see how it's beenficial
            // To continua if any is true but then mask out the remaining by if  (why not just do mask by for loop)
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

                    if constexpr(useSubGroup)
                    {
                        sycl::atomic_ref<float,
                                         sycl::memory_order_relaxed,
                                         sycl::memory_scope_sub_group,
                                         sycl::access::address_space::local_space>(hist[bidx])
                          .fetch_add(weight);
                    }
                    else
                    {
                        // Work grup version
                        sycl::atomic_ref<float,
                                         sycl::memory_order_relaxed,
                                         sycl::memory_scope_work_group, // Change to sub_group for sub_group version
                                         sycl::access::address_space::local_space>(hist[bidx])
                          .fetch_add(weight);
                    }
                }
            }
        }

        sycl::group_barrier(group);

#ifdef WITH_VLFEAT_SMOOTHING
        for(int i = 0; i < 3; i++)
        {
            sm_hist[it.get_local_id(1) + 0] = smoothe(hist, it.get_local_id(1) + 0);
            sm_hist[it.get_local_id(1) + 32] = smoothe(hist, it.get_local_id(1) + 32);
            sycl::group_barrier(group);
            hist[it.get_local_id(1) + 0] = smoothe(sm_hist, it.get_local_id(1) + 0);
            hist[it.get_local_id(1) + 32] = smoothe(sm_hist, it.get_local_id(1) + 32);
            sycl::group_barrier(group);
        }

        sm_hist[it.get_local_id(1) + 0] = hist[it.get_local_id(1) + 0];
        sm_hist[it.get_local_id(1) + 32] = hist[it.get_local_id(1) + 32];
        sycl::group_barrier(group);
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

        for(int bin = it.get_local_id(1); sycl::any_of_group(group, bin < ORI_NBINS); bin += it.get_local_range(1))
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
        sycl::group_barrier(group);

        sycl::vec<int, 2> best_index(it.get_local_id(1), it.get_local_id(1) + 32);

#define XPOS 26.643719f
#define YPOS 185.853836f

        if(iext->xpos == XPOS && iext->ypos == YPOS && it.get_local_id(1) == 0)
        {
            // printf("\nBEFORE: best_index (%d, %d)\n", best_index.x, best_index.y);
            for(int i = 0; i < 64; i += 2)
            {
                sycl::ext::oneapi::experimental::printf("\tidx=%d --> %.6f  -- ", i, yval[i]);
                sycl::ext::oneapi::experimental::printf("idx=%d --> %.6f\n ", i + 1, yval[i + 1]);
            }
            sycl::ext::oneapi::experimental::printf("\nAFTER");
        }

        // BitonicSort
        if constexpr(useSubGroup)
        {
            // BitonicSort::Warp32<float, sycl::sub_group> sorter(yval, it, group);
            if(it.get_global_linear_id() == 0)
                sycl::ext::oneapi::experimental::printf("OCTAVE WE DOING SORTER ON %d\n\n", octave);
            BitonicSort::Warp32<float, sycl::sub_group> sorter(yval, it, group);
            sorter.sort64(best_index);
        }
        else
        {
            // TODO: Make the work-group version work

            // sycl::vec<int, 2> best_index(it.get_local_id(1), it.get_local_id(1) + 32);
            // BitonicSort::Warp32<float, sycl::group<2>> sorter(yval, it, group);
            // sorter.sort64(best_index);
        }

        sycl::group_barrier(group); // me test syncer :D

        if(iext->xpos == XPOS && iext->ypos == YPOS)
        {
            sycl::ext::oneapi::experimental::printf("\n\tthreadIdx %d --> best_index (%d, %d) --> yval(%f, %f)",
                                                    it.get_local_id(1),
                                                    best_index.x(),
                                                    best_index.y(),
                                                    yval[best_index.x()],
                                                    yval[best_index.y()]);
        }

        // Looks correct
        // return;

        const float best_val = yval[best_index.x()];

        // Zero broadcast as it has higest yvalue
        const float yval_treshold = 0.8 * sycl::group_broadcast(group, best_val, 0);

        // if(iext->xpos == XPOS && iext->ypos == YPOS && group.leader())
        if(iext->xpos == XPOS && iext->ypos == YPOS)
        {
            sycl::ext::oneapi::experimental::printf("\n\tyval_treshold = %f", yval_treshold);
        }

        // Think we compute out of loop to avoid too much compute in branching?
        const bool valid = (best_val >= yval_treshold); // Only larger than threshold is accepted
        bool written = false;

        Extremum* ext = &dobuf->extrema[ext_prefix_sum + extremum_index];

        if(it.get_local_id()[1] < ORIENTATION_MAX_COUNT)
        {
            if(iext->xpos == XPOS && iext->ypos == YPOS)
            {
                sycl::ext::oneapi::experimental::printf(
                  "\n\tHELLO IN LE LOOOP -- local_id = %d max_ori = %d\n", it.get_local_id()[1], ORIENTATION_MAX_COUNT);
            }
            if(valid)
            {
                float chosen_bin = refined_angle[best_index.x()];
                if(chosen_bin >= ORI_NBINS)
                    chosen_bin -= ORI_NBINS;

                // Fast version of a * b + c (approximate)
                // float th = sycl::mad(M_PI2 * chosen_bin, 1.0f / ORI_NBINS, -M_PI);

                constexpr float M_PI2_f = M_PI2;
                constexpr float M_PI_f = M_PI;
                // accurate version of a * b + c (not as fast as sycl::mad)
                float th = sycl::fma(M_PI2_f * chosen_bin, (1.0f / ORI_NBINS), -M_PI_f);
                // float th = sycl::fma((M_PI2 * chosen_bin, 1.0f / ORI_NBINS, -M_PI);

                // sycl::ext::oneapi::experimental::printf("Orientation %f\n", th);
                ext->orientation[it.get_local_id()[1]] = th;
                written = true;
            }
        }

        int angles = [&]() {
            if constexpr(useSubGroup)
            {
                // auto mask = sycl::ext::oneapi::group_ballot(group, written);
                // return sycl::popcount(mask.get_mask());

                // Using extension to use do ballot
                return sycl::ext::oneapi::group_ballot(group, written).count();
                // unsigned mask = sycl::ext::oneapi::extract_bits<unsigned>(ballot_result);
                // return sycl::popcount(mask);
                // return sycl::popcount(sycl::ext::oneapi::group_ballot(group, written));
            }
            else
            {
                uint32_t mask = sycl::reduce_over_group(
                  group, written ? (1u << it.get_local_id()[1]) : 0u, sycl::ext::oneapi::bit_or<uint32_t>());
                return sycl::popcount(mask);
            }
        }();
        // int angles = sycl::popcount(sycl::ext::oneapi::group_ballot(group, written));
        if(it.get_local_id()[1] == 0)
        {
            ext->xpos = iext->xpos;
            ext->ypos = iext->ypos;
            ext->lpos = iext->lpos;
            ext->sigma = iext->sigma;
            ext->octave = octave;
            ext->num_ori = angles;
        }
    }
};

struct ori_par_subgroup;

// Computes the actual sub_group size that will be used for the ori_par kernel
auto get_ori_par_subgroup_size(sycl::queue& Q)
{
    auto kernel_id = sycl::get_kernel_id<ori_par_subgroup>();
    auto kernel_bundle = sycl::get_kernel_bundle<sycl::bundle_state::executable>(Q.get_context());
    auto kernel = kernel_bundle.get_kernel(kernel_id);
    return kernel.get_info<sycl::info::kernel_device_specific::max_sub_group_size>(Q.get_device());
}

}

void Pyramid::orientation(const Config& conf)
{
    // auto max_subgroup = _device_queue.get_device().get_info<sycl::info::device::max_sub_group_size>();

    // returns all slupported and not what it will use for the kernel (my cpu gave 4 8 16 32 64) but used
    // 8 so not too valuable
    // auto max_subgroup = _device_queue.get_device().get_info<sycl::info::device::sub_group_sizes>().back();

    // Hopefully evaluated at compile time
    auto max_subgroup = get_ori_par_subgroup_size(_device_queue);
    bool useSubGroup = max_subgroup >= 32;

    fprintf(stderr, "\n\tWAITING IN ORIENTATION FOR EXTREMA FOR ALL OCTAVE TO FINISH\n");
    // Wait so that the computation is done before the memcpy
    // Look for ways to make this part faster (less waits the better)
    _device_queue.wait();

    fprintf(stderr, "Sub group kernel mas sub_group_size %d", max_subgroup);
    // Need to think about if this is really necessary?
    // As now we neet to wait for all octaes to do extrema before we can do orientation
    // Not sure if we actually need to do this...
    readDescCountersFromDevice().wait();

    int ext_total = 0;
    for(int o : _hct.ext_ct)
    {
        if(o > 0)
        {
            ext_total += o;
        }
    }

    // Works as expected
    printf("\n\text_total for all octaves = %d\n", ext_total);

    // TODO: It is set up to do nothing in current configuration but should consider adding support for it
    // Seems to do nothing in my case...

    // Filter functions are only called if necessary. They are very expensive,
    // therefore add 10% slack.
    // if(conf.getFilterMaxExtrema() > 0 && int(conf.getFilterMaxExtrema() * 1.1) < ext_total)
    // {
    //     ext_total = extrema_filter_grid(conf, ext_total);
    // }

    reallocExtrema(ext_total);

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

            // NOTE: Need to make modificatons when sub-group is not 32
            // as is normaly the case on cpu's

            if(useSubGroup)
            {
                fprintf(stderr, "\n\tUSING SUBGOUP for Orientation\n");
                // Uses sub-group(warp) for synchronization and communication
                _device_queue.submit([&](sycl::handler& cgh) {
                    cgh.depends_on({_dobuf_write, oct_obj._extrema_done_event});
                    // sycl::local_accessor<float, 1> -- is the type (using auto as it's so long)
                    auto hist = sycl::local_accessor<float, 1>(64, cgh);
                    auto sm_hist = sycl::local_accessor<float, 1>(64, cgh);
                    auto refined_angle = sycl::local_accessor<float, 1>(64, cgh);
                    auto yval = sycl::local_accessor<float, 1>(64, cgh);

                    cgh.parallel_for<ori_par_subgroup>(sycl::nd_range{global, local},
                                                       popsift::ori_par<true>(octave,
                                                                              _hct.ext_ps[octave],
                                                                              oct_obj.getDataArray(),
                                                                              oct_obj.getWidth(),
                                                                              oct_obj.getHeight(),
                                                                              hist,
                                                                              sm_hist,
                                                                              refined_angle,
                                                                              yval,
                                                                              _dobuf,
                                                                              _dct));
                });
            }
            else
            {
                fprintf(stderr, "\n\tUSING WORKGRPU for orientation\n");
                // Uses work groip for synchronization and communication
                _device_queue.submit([&](sycl::handler& cgh) {
                    cgh.depends_on({_dobuf_write, oct_obj._extrema_done_event});
                    // sycl::local_accessor<float, 1> -- is the type (using auto as it's so long)
                    auto hist = sycl::local_accessor<float, 1>(64, cgh);
                    auto sm_hist = sycl::local_accessor<float, 1>(64, cgh);
                    auto refined_angle = sycl::local_accessor<float, 1>(64, cgh);
                    auto yval = sycl::local_accessor<float, 1>(64, cgh);
                    // ExtremaCounters* dct = _dct;

                    cgh.parallel_for(sycl::nd_range{global, local},
                                     popsift::ori_par<false>(octave,
                                                             _hct.ext_ps[octave],
                                                             oct_obj.getDataArray(),
                                                             oct_obj.getWidth(),
                                                             oct_obj.getHeight(),
                                                             hist,
                                                             sm_hist,
                                                             refined_angle,
                                                             yval,
                                                             _dobuf,
                                                             _dct));
                });
            }
            // TODO: Understand why this is needed and if I need something similar
            // if(octave != 0)
            // {
            //     cuda::event_record(oct_obj.getEventOriDone(), oct_str, __FILE__, __LINE__);
            //     cuda::event_wait(oct_obj.getEventOriDone(), oct_0_str, __FILE__, __LINE__);
            // }
            _device_queue.wait();

            if(octave == 0)
            {
                _device_queue.single_task([=, dobuf = _dobuf, hct = _hct]() {
                    for(int i = 0; i < num; ++i)
                    {
                        Extremum* ext = &dobuf->extrema[hct.ext_ps[octave] + i];
                        sycl::ext::oneapi::experimental::printf(
                          "Extremum: xpos=%.6f, ypos=%.6f, lpos=%d, sigma=%.6f, octave=%d, num_ori=%d, idx_ori=%d, "
                          "orientation=[%.4f, %.4f, %.4f, %.4f]\n",
                          ext->xpos,
                          ext->ypos,
                          ext->lpos,
                          ext->sigma,
                          ext->octave,
                          ext->num_ori,
                          ext->idx_ori,
                          ext->orientation[0],
                          ext->orientation[1],
                          ext->orientation[2],
                          ext->orientation[3]);
                    }
                });
            }
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
