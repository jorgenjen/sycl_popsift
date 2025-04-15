/*
 * Copyright 2017, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once
#include "common/assist.h"
#include "s_desc_normalize.h"

// using namespace popsift;
// using namespace std;

class NormalizeRootSift
{
  public:
    template<bool wg_reduce>
    static inline void normalize(float* features, bool ignoreme, sycl::nd_item<2> it, popsift::ConstInfo* d_consts);

    // static inline void normalize_restrict(const float* __restrict__ src_desc, float* __restrict__ dest_desc,
    // popsift::ConstInfo* d_consts);

    template<bool wg_reduce>
    static inline void normalize(
      const float* src_desc, float* dest_desc, bool ignoreme, sycl::nd_item<2> it, popsift::ConstInfo* d_consts);
};

template<bool wg_reduce>
inline void NormalizeRootSift::normalize(float* features,
                                         bool ignoreme,
                                         sycl::nd_item<2> it,
                                         popsift::ConstInfo* d_consts)
{
    normalize<wg_reduce>(features, features, ignoreme, it, d_consts);
}

// __device__ inline void NormalizeRootSift::normalize_restrict(const float* __restrict__ src_desc,
//                                                              float* __restrict__ dst_desc)
// {
//     normalize(src_desc, dst_desc, false);
// }

template<bool wg_reduce>
inline void NormalizeRootSift::normalize(
  const float* src_desc, float* dst_desc, bool ignoreme, sycl::nd_item<2> it, popsift::ConstInfo* d_consts)
{
    // sycl::ext::oneapi::experimental::printf("Hello");

    // const float4* ptr4 = (const float4*)src_desc;
    const sycl::vec<float, 4>* ptr4 = reinterpret_cast<const sycl::vec<float, 4>*>(src_desc);

    sycl::vec<float, 4> descr = ptr4[it.get_local_id(1)];

    float sum = descr.x() + descr.z() + descr.y() + descr.w();

    // sum += popsift::shuffle_down(sum, 16);
    // sum += popsift::shuffle_down(sum, 8);
    // sum += popsift::shuffle_down(sum, 4);
    // sum += popsift::shuffle_down(sum, 2);
    // sum += popsift::shuffle_down(sum, 1);
    //
    // sum = popsift::shuffle(sum, 0);

    // Only works when sub_gorup is 32...
    // TODO: Add support for work group to this an many other parts of descriptor code)

    if constexpr(wg_reduce)
    {
        sum = sycl::reduce_over_group(it.get_group(), sum, sycl::plus<float>());
    }
    else
    {
        // Sum of the whole descriptor with sub_group
        sum = sycl::reduce_over_group(it.get_sub_group(), sum, sycl::plus<float>());
    }

    // float val;
    // val = scalbnf(__fsqrt_rn(__fdividef(descr.x, sum)), d_consts.norm_multi);
    // descr.x = val;
    // val = scalbnf(__fsqrt_rn(__fdividef(descr.y, sum)), d_consts.norm_multi);
    // descr.y = val;
    // val = scalbnf(__fsqrt_rn(__fdividef(descr.z, sum)), d_consts.norm_multi);
    // descr.z = val;
    // val = scalbnf(__fsqrt_rn(__fdividef(descr.w, sum)), d_consts.norm_multi);
    // descr.w = val;

    // Not sure if using tmp val makes it faster like they did in cuda
    // Verify if the rounding is important or not

    // Not sure if I should use sycl::native or sycl::half_precision
    // sqrt is in both above and sycl ...
    descr.x() = sycl::ldexp(sycl::sqrt(sycl::native::divide(descr.x(), sum)), d_consts->norm_multi);
    descr.y() = sycl::ldexp(sycl::sqrt(sycl::native::divide(descr.y(), sum)), d_consts->norm_multi);
    descr.z() = sycl::ldexp(sycl::sqrt(sycl::native::divide(descr.z(), sum)), d_consts->norm_multi);
    descr.w() = sycl::ldexp(sycl::sqrt(sycl::native::divide(descr.w(), sum)), d_consts->norm_multi);

    if constexpr(wg_reduce)
    {
        // Work group has jump of one hence non are ignored
        sycl::vec<float, 4>* out4 = reinterpret_cast<sycl::vec<float, 4>*>(dst_desc);
        out4[it.get_local_id(1)] = descr;
    }
    else
    {
        if(!ignoreme)
        {
            sycl::vec<float, 4>* out4 = reinterpret_cast<sycl::vec<float, 4>*>(dst_desc);
            out4[it.get_local_id(1)] = descr;
        }
    }
}
