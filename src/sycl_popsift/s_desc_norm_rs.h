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
#include "sycl_popsift/sift_desc_config.hpp" // For FeatureType

// using namespace popsift;
// using namespace std;

class NormalizeRootSift
{
  public:
    template<bool wg_reduce>
    static inline void normalize(FeatureType* features,
                                 bool ignoreme,
                                 sycl::nd_item<2> it,
                                 popsift::ConstInfo* d_consts);

    // static inline void normalize_restrict(const float* __restrict__ src_desc, float* __restrict__ dest_desc,
    // popsift::ConstInfo* d_consts);

    template<bool wg_reduce>
    static inline void normalize(const FeatureType* src_desc,
                                 FeatureType* dest_desc,
                                 bool ignoreme,
                                 sycl::nd_item<2> it,
                                 popsift::ConstInfo* d_consts);
};

template<bool wg_reduce>
inline void NormalizeRootSift::normalize(FeatureType* features,
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
  const FeatureType* src_desc, FeatureType* dst_desc, bool ignoreme, sycl::nd_item<2> it, popsift::ConstInfo* d_consts)
{
    const sycl::vec<FeatureType, 4>* ptr4 = reinterpret_cast<const sycl::vec<FeatureType, 4>*>(src_desc);

    sycl::vec<FeatureType, 4> descr = ptr4[it.get_local_id(1)];

    FeatureType sum = descr.x() + descr.z() + descr.y() + descr.w();

    if constexpr(wg_reduce)
    {
        sum = sycl::reduce_over_group(it.get_group(), sum, sycl::plus<FeatureType>());
    }
    else
    {
        // Sum of the whole descriptor with sub_group
        sum = sycl::reduce_over_group(it.get_sub_group(), sum, sycl::plus<FeatureType>());
    }

    // Not sure if I should use sycl::native or sycl::half_precision
    descr.x() = sycl::ldexp(sycl::sqrt(sycl::native::divide(descr.x(), sum)), d_consts->norm_multi);
    descr.y() = sycl::ldexp(sycl::sqrt(sycl::native::divide(descr.y(), sum)), d_consts->norm_multi);
    descr.z() = sycl::ldexp(sycl::sqrt(sycl::native::divide(descr.z(), sum)), d_consts->norm_multi);
    descr.w() = sycl::ldexp(sycl::sqrt(sycl::native::divide(descr.w(), sum)), d_consts->norm_multi);

    if constexpr(wg_reduce)
    {
        // Work group has jump of one hence non are ignored
        sycl::vec<FeatureType, 4>* out4 = reinterpret_cast<sycl::vec<FeatureType, 4>*>(dst_desc);
        out4[it.get_local_id(1)] = descr;
    }
    else
    {
        if(!ignoreme)
        {
            sycl::vec<FeatureType, 4>* out4 = reinterpret_cast<sycl::vec<FeatureType, 4>*>(dst_desc);
            out4[it.get_local_id(1)] = descr;
        }
    }
}
