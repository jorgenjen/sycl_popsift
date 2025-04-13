/*
 * Copyright 2017, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once

// #include "s_desc_norm_l2.h" // Not implemented as of now
#include "s_desc_norm_rs.h"
#include "sift_extremum.h"

// template<class T>
// void normalize_histogram()
// {
//     Descriptor* descs = dbuf.desc;
//     const int num_orientations = dct.ori_total;
//
//     int offset = blockIdx.x * 32 + threadIdx.y;
//
//     // all of these threads are useless
//     if(blockIdx.x * 32 >= num_orientations)
//         return;
//
//     offset = (offset < num_orientations) ? offset : num_orientations - 1;
//     Descriptor* desc = &descs[offset];
//
//     bool ignoreme = (offset >= num_orientations);
//
//     T::normalize(desc->features, ignoreme);
// }

template<class T>
class Normalize_histogram
{
  private:
    popsift::Descriptor* descs;
    popsift::ConstInfo* d_consts;
    const int num_orientations;

  public:
    Normalize_histogram(popsift::Descriptor* descs, popsift::ConstInfo* d_consts, const int num_orientations)
      : descs(descs)
      , d_consts(d_consts)
      , num_orientations(num_orientations)
    {}

    inline void operator()(sycl::nd_item<2> it) const
    {
        // Descriptor* descs = dbuf.desc;
        // const int num_orientations = dct.ori_total;

        // int offset = blockIdx.x * 32 + threadIdx.y;

        // So we get one descriptor per sub_group (along dim 1)
        int offset = it.get_group(1) * it.get_local_range(1) + it.get_local_id(0);

        // all of these threads are useless
        // if(blockIdx.x * 32 >= num_orientations)
        if(it.get_group(1) * it.get_local_range(1) >= num_orientations)
            return;

        offset = (offset < num_orientations) ? offset : num_orientations - 1;
        popsift::Descriptor* desc = &descs[offset];

        bool ignoreme = (offset >= num_orientations);

        // Every work-item in same-work grop  pass same pointer
        // We take a new descriptor for each y change
        T::normalize(desc->features, ignoreme, it, d_consts);
    }
};
