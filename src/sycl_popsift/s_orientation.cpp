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
#include "sycl_popsift/common/excl_blk_prefix_sum.h"

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
// TODO: Make this into a constexpr if statement instead of switch to ensure it's done at compile time
// Probs belong in assist or something like that
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

// TODO: Remove and use the one in s_gradient.hpp
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

// For small image
#define XPOS 26.643719f
#define YPOS 185.853836f
        // For medium image
        // #define XPOS 1096.967896f
        // #define YPOS 819.321289f

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

        // Needed to for sub_grop using minimal as if it's only two yval that's non -inf
        // and the largets has index 3 the same orientation will be selected twize which we don't want
        // Hence need to set this to 2 in that case (popcount)
        //      --> Is the min of popcount and ORIENTATION_MAX_COUNT
        // int max_count = ORIENTATION_MAX_COUNT;
#define minimal_sort 0

#if minimal_sort
        int my_index = -1; // if unmodified it was not assigned value in minimal
#else
        int my_index = 0; // Will always be modified by non-minimal
#endif
        // BitonicSort
        if constexpr(useSubGroup)
        {
            // BitonicSort::Warp32<float, sycl::sub_group> sorter(yval, it, group);
            if(it.get_global_linear_id() == 0)
                sycl::ext::oneapi::experimental::printf("OCTAVE WE DOING SORTER ON %d\n\n", octave);
            BitonicSort::WorkGroup32<float, sycl::sub_group> sorter(yval, it, group);
#if minimal_sort
            sorter.minimal_sort64(my_index);
#else
            sorter.sort64(my_index);
#endif
        }
        else
        {
            BitonicSort::WorkGroup32<float, sycl::group<2>> sorter(yval, it, group);
            sorter.sort64(my_index);
        }

        sycl::group_barrier(group); // me test syncer :D

        if(iext->xpos == XPOS && iext->ypos == YPOS)
        {
            sycl::ext::oneapi::experimental::printf("\n\tthreadIdx %d  -  best_index %d --> yval = %f",
                                                    (int)it.get_local_id(1),
                                                    my_index,
                                                    my_index != -1 ? yval[my_index] : -INFINITY);
        }

#if minimal_sort
        // Needed to avoid some value potentially being duplicate when running minimal
        const float best_val = my_index != -1 ? yval[my_index] : -INFINITY;
#else

        const float best_val = yval[my_index];
#endif

        // Zero broadcast as it has higest yvalue
        const float yval_treshold = 0.8 * sycl::group_broadcast(group, best_val, 0);

        // if(iext->xpos == XPOS && iext->ypos == YPOS && group.leader())
        if(iext->xpos == XPOS && iext->ypos == YPOS)
        {
            sycl::ext::oneapi::experimental::printf("\n\tyval_treshold = %f", yval_treshold);
        }

        // Think we compute out of loop to avoid too much compute in branching?
        const bool valid = (best_val >= yval_treshold); // Only larger than threshold is accepted
        // A -1 my_index could only be valid if all yval is -INFINITY
        // I don't think that can happen     TODO: Verify that it can't happen
        // This is only concern for miniimal

        bool written = false;

        Extremum* ext = &dobuf->extrema[ext_prefix_sum + extremum_index];

        if(it.get_local_id()[1] < ORIENTATION_MAX_COUNT)
        // if(it.get_local_id()[1] < max_count)
        {
            if(iext->xpos == XPOS && iext->ypos == YPOS)
            {
                sycl::ext::oneapi::experimental::printf(
                  "\n\tHELLO IN LE LOOOP -- local_id = %d max_ori = %d\n", it.get_local_id()[1], ORIENTATION_MAX_COUNT);
            }
            if(valid)
            {
                float chosen_bin = refined_angle[my_index];
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

class ExtremaRead
{
    const Extremum* const _oris;

  public:
    inline explicit ExtremaRead(const Extremum* const d_oris)
      : _oris(d_oris)
    {}

    inline int get(int n) const { return _oris[n].num_ori; }
};

// Makes the code more readable and maintainable and esier to change underlying memory without modifying function
// Seems a bit excessive in this case but should not add overhead
class ExtremaWrt
{
    Extremum* _oris;

  public:
    inline explicit ExtremaWrt(Extremum* d_oris)
      : _oris(d_oris)
    {}

    inline void set(int n, int value) { _oris[n].idx_ori = value; }
};

class ExtremaTot
{
    int& _extrema_counter;

  public:
    inline explicit ExtremaTot(int& extrema_counter)
      : _extrema_counter(extrema_counter)
    {}

    inline void set(int value) { _extrema_counter = value; }
};

class ExtremaWrtMap
{
    int* _featvec_to_extrema_mapper;
    int _max_feat;

  public:
    inline ExtremaWrtMap(int* featvec_to_extrema_mapper, int max_feat)
      : _featvec_to_extrema_mapper(featvec_to_extrema_mapper)
      , _max_feat(max_feat)
    {}

    inline void set(int base, int num, int value)
    {
        int* baseptr = &_featvec_to_extrema_mapper[base];
        do
        {
            num--;
            if(base + num < _max_feat)
            {
                baseptr[num] = value;
            }
        } while(num > 0);
    }
};

// Changing these values are not soupported (could add support for different 0 dim)
// But 32x32 is what makes the most sense or sub_group size in both directons
#define PREFIX_0_DIM 32
#define PREFIX_1_DIM 32 // must be 32 --> atleast for gpu targets

template<bool useSubGroup>
class ori_prefix_sum
{
  private:
    const int total_ext_ct;
    const int num_octaves;
    ExtremaBuffers* dbuf;
    DevBuffers* dobuf;
    ConstInfo* d_consts;
    ExtremaCounters* dct;
    sycl::local_accessor<int, 1> sum;
    sycl::local_accessor<int, 1> loop_total;

  public:
    ori_prefix_sum(const int total_ext_ct,
                   const int num_octaves,
                   ExtremaBuffers* dbuf,
                   DevBuffers* dobuf,
                   ConstInfo* d_consts,
                   ExtremaCounters* dct,
                   sycl::local_accessor<int, 1> sum,
                   sycl::local_accessor<int, 1> loop_total)
      : total_ext_ct(total_ext_ct)
      , num_octaves(num_octaves)
      , dbuf(dbuf)
      , dobuf(dobuf)
      , d_consts(d_consts)
      , dct(dct)
      , sum(sum)
      , loop_total(loop_total)
    {}

    inline void operator()(sycl::nd_item<2> it) const
    {
        sycl::group work_group = it.get_group();

        ExtremaWrtMap mapping_writer(dobuf->feat_to_ext_map, max(d_consts->max_orientations, dbuf->ori_allocated));

        if(it.get_local_linear_id() == 0)
        {
            loop_total[0] = 0;
        }

        if constexpr(!useSubGroup)
        {
            // Need to ensure loop_total is initialized before first use for all work-items
            sycl::group_barrier(work_group);
            // Not needed in sub_group version as that has barriers before first use
        }
        // sycl::group_barrier(work_group); // should not be needed

        const int num = total_ext_ct; // NOT USED
        const int start = it.get_local_linear_id();
        // constexpr int wrap = 1024;
        constexpr int wrap = PREFIX_1_DIM * PREFIX_0_DIM;
        // int wrap = it.get_global_range(1) * it.get_global_range(0);
        const int end = (total_ext_ct & (wrap - 1)) ? (total_ext_ct & ~(wrap - 1)) + wrap : total_ext_ct;
        // End is a factor of 1024 (required so every work-item reaches the barriers)

        if(start == 0)
            sycl::ext::oneapi::experimental::printf("\nEND = %d\n", end);

        for(int x = start; x < end; x += wrap)
        {
            // sycl::group_barrier(work_group); // could be needed to ensure we are not diverged when doing scan

            const bool valid = (x < total_ext_ct);

            int self = (valid) ? dobuf->extrema[x].num_ori : 0;

            if constexpr(useSubGroup)
            {
                sycl::sub_group sub_group = it.get_sub_group();
                // This loop is an exclusive prefix sum for one warp
                int ews = sycl::exclusive_scan_over_group(sub_group, self, sycl::plus<>());

                if(it.get_local_id(1) == PREFIX_1_DIM - 1) // only last adds
                {
                    // store inclusive warp prefix sum in shared mem
                    // to be summed up in next phase
                    sum[it.get_local_id(0)] = ews + self; // making it  inclusive
                }
                sycl::group_barrier(work_group); // write to sum

                int ibs; // inclusive block prefix sum
                if(it.get_local_id(0) == 0)
                {
                    int self = sum[it.get_local_id(1)];

                    // ANother exclusive scan
                    int ebs = sycl::exclusive_scan_over_group(sub_group, self, sycl::plus<>());

                    sum[it.get_local_id(1)] = ebs;
                    ibs = ebs + self;
                }

                sycl::group_barrier(work_group); // write to sum

                if(valid)
                {
                    // start_pos_loop + start_pos_sub_group + work_item exclusive sum in it's sub_group
                    const int ebs = loop_total[0] + sum[it.get_local_id(0)] + ews;

                    dobuf->extrema[x].idx_ori = ebs;

                    mapping_writer.set(ebs, self, x);
                }
                sycl::group_barrier(work_group); // ensure everyone read loop_total before it's updated

                if(it.get_local_id(0) == 0 && it.get_local_id(1) == PREFIX_1_DIM - 1)
                {
                    loop_total[0] += ibs;
                    // sycl::ext::oneapi::experimental::printf(
                    //   "lid=(%d,%d), ibs=%d, self=%d \n", (int)it.get_local_id(0), (int)it.get_local_id(1), ibs,
                    //   self);
                }

                // Ensures consistency of loop_total before next iteration or before we done
                sycl::group_barrier(work_group);
            }
            else
            {
                // ################ WORK_GROUP ######################
                // const int ebs = loop_total[0] + sycl::exclusive_scan_over_group(work_group, self, sycl::plus<>());
                const int ebs = sycl::exclusive_scan_over_group(work_group, self, sycl::plus<>());

                if(valid)
                {
                    dobuf->extrema[x].idx_ori = ebs + loop_total[0];

                    mapping_writer.set(ebs + loop_total[0], self, x); // writer needs offset
                }
                // if(it.get_local_id(0) == PREFIX_0_DIM - 1 && it.get_local_id(1) == PREFIX_1_DIM - 1)
                if(it.get_local_linear_id() == wrap - 1)
                {
                    loop_total[0] += ebs + self;
                    // sycl::ext::oneapi::experimental::printf(
                    //   "\n\tebs = %d | self = %d | loop_total = %d", ebs, self, ebs + self);
                    sycl::ext::oneapi::experimental::printf("lid=(%d,%d), ebs=%d, self=%d | ibs = %d\n",
                                                            (int)it.get_local_id(0),
                                                            (int)it.get_local_id(1),
                                                            ebs,
                                                            self,
                                                            ebs + self);
                }

                sycl::group_barrier(work_group); // ensure loop_total is consistent before next loop iteration
            }
        }

        // End of part that was in separate funcion
        if(it.get_global_linear_id() == 0)
        {
            // Look into if this could be done by multiple work items in paralell to speed it up
            // Curently one work-item does the whole thing the rest does nothing
            dct->ext_ps[0] = 0;
            for(int o = 1; o < MAX_OCTAVES; o++)
            {
                dct->ext_ps[o] = dct->ext_ps[o - 1] + dct->ext_ct[o - 1];
            }

            for(int o = 0; o < MAX_OCTAVES; o++)
            {
                if(dct->ext_ct[o] == 0)
                {
                    dct->ori_ct[o] = 0;
                }
                else
                {
                    int fe = dct->ext_ps[o];         /* first extremum for this octave */
                    int le = dct->ext_ps[o + 1] - 1; /* last  extremum for this octave */
                    int lo_ori_index = dobuf->extrema[fe].idx_ori;
                    int num_ori = dobuf->extrema[le].num_ori;

                    int hi_ori_index = dobuf->extrema[le].idx_ori + num_ori;
                    // sycl::ext::oneapi::experimental::printf(
                    //   "\n\t\tIN ORI_PREFIX_SUM -> hi_ori_index = %d - lo_ori_index = %d", hi_ori_index,
                    //   lo_ori_index);
                    dct->ori_ct[o] = hi_ori_index - lo_ori_index;

                    sycl::ext::oneapi::experimental::printf(
                      "Octave %d: fe=%d, le=%d, lo_ori_index=%d, num_ori=%d, hi_ori_index=%d, ori_ct[o]=%d\n",
                      o,                          // octave index
                      fe,                         // first extremum
                      le,                         // last extremum
                      lo_ori_index,               // low orientation index
                      num_ori,                    // number of orientations
                      hi_ori_index,               // high orientation index
                      hi_ori_index - lo_ori_index // ori_ct[o] value
                    );
                }
            }

            dct->ori_ps[0] = 0;
            for(int o = 1; o < MAX_OCTAVES; o++)
            {
                dct->ori_ps[o] = dct->ori_ps[o - 1] + dct->ori_ct[o - 1];
            }

            dct->ori_total = dct->ori_ps[MAX_OCTAVES - 1] + dct->ori_ct[MAX_OCTAVES - 1];
            dct->ext_total = dct->ext_ps[MAX_OCTAVES - 1] + dct->ext_ct[MAX_OCTAVES - 1];

            sycl::ext::oneapi::experimental::printf("\n\t\tIN ORI_PREFIX_SUM -> ori_total = %d", dct->ori_total);
        }
    }
};

// must be set in namespace scope
struct ori_par_subgroup;
struct ori_prefix_sum_subgroup;
void Pyramid::orientation(const Config& conf)
{
    auto max_subgroup_ori_par = get_kernel_subgroup_size<ori_par_subgroup>(_device_queue);
    bool use_subgroup_ori_par = max_subgroup_ori_par >= 32;
    auto max_subgroup_prefix = get_kernel_subgroup_size<ori_prefix_sum_subgroup>(_device_queue);
    bool use_subgroup_prefix = max_subgroup_prefix >= 32;

    fprintf(stderr, "\n\tWAITING IN ORIENTATION FOR EXTREMA FOR ALL OCTAVE TO FINISH\n");
    // Wait so that the computation is done before the memcpy
    // Look for ways to make this part faster (less waits the better)
    _device_queue.wait();

    fprintf(stderr, "Sub group kernel mas sub_group_size %d -- %d", max_subgroup_ori_par, max_subgroup_prefix);
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

            // if(!use_subgroup_ori_par) // to debug local memory on gpu
            if(use_subgroup_ori_par)
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
                                                       ori_par<true>(octave,
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
        }

        // _device_queue.wait(); // to test one by one

        fprintf(stderr, "\n\tDone Ori par for octave %d", octave);
    }

    // Should remove this and addd ependencies if htare are any
    _device_queue.wait();
    // Could just for sor range for this (unless I need work_grop/sub_group)

    fprintf(stderr, "\n\tWE ARE PAST ORIENTATION");
    // NOTE: We need num_orientation hence we need to wait for prev kernel could try to split up into octave but not
    // sure if that would make it faster Could try full size for first octave and half size kernel for the reset as the
    // earlier octaves always have more extremas than the later octaves

    sycl::range local_prefix{PREFIX_0_DIM, PREFIX_1_DIM};
    sycl::range global_prefix{PREFIX_0_DIM, PREFIX_1_DIM};

    // if(!use_subgroup_prefix)    // to debugg work_group on GPU
    if(use_subgroup_prefix) // Normal
    {
        fprintf(stderr, "Running subgroup\n");
        _device_queue.submit([&, dbuf = _dbuf, dobuf = _dobuf, d_consts = _d_consts, dct = _dct](sycl::handler& cgh) {
            // sycl::local_accessor<int, 1> -- is the type
            auto sum = sycl::local_accessor<int, 1>(32, cgh);
            auto loop_total = sycl::local_accessor<int, 1>(1, cgh);

            cgh.parallel_for<ori_prefix_sum_subgroup>(
              sycl::nd_range{global_prefix, local_prefix},
              ori_prefix_sum<true>(ext_ct_prefix_sum, _num_octaves, dbuf, dobuf, d_consts, dct, sum, loop_total));
        });
    }
    else
    {
        fprintf(stderr, "Running work group\n");
        _device_queue.submit([&, dbuf = _dbuf, dobuf = _dobuf, d_consts = _d_consts, dct = _dct](sycl::handler& cgh) {
            // sycl::local_accessor<int, 1> -- is the type
            auto sum = sycl::local_accessor<int, 1>(32, cgh);
            auto loop_total = sycl::local_accessor<int, 1>(1, cgh);

            cgh.parallel_for(
              sycl::nd_range{global_prefix, local_prefix},
              ori_prefix_sum<false>(ext_ct_prefix_sum, _num_octaves, dbuf, dobuf, d_consts, dct, sum, loop_total));
        });
    }
    _device_queue.wait(); // Required for now first should replace with events
    fprintf(stderr, "\n\t\tPAST PREFIX SUM COMPUTE\n");
}

} // namespace popsift
