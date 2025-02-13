/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "sift_constants.hpp"

#include "common/debug_macros.hpp"
#include "sycl/queue.hpp"
#include "sycl/usm.hpp"

// #include <cuda_runtime.h>

#include <cmath>
#include <iostream>
#include <sstream>

using namespace std;

namespace popsift {

// thread_local ConstInfo h_consts;
ConstInfo h_consts;
// ConstInfo* d_consts = nullptr;
// __device__ __constant__ ConstInfo d_consts;

sycl::event init_constants(float sigma0,
                           int levels,
                           float threshold,
                           float edge_limit,
                           int max_extrema,
                           int normalization_multiplier,
                           sycl::queue& Q,
                           ConstInfo** d_consts)
{
    // cudaError_t err;

    h_consts.sigma0 = sigma0;
    h_consts.sigma_k = powf(2.0f, 1.0f / levels);
    h_consts.edge_limit = edge_limit;
    h_consts.threshold = threshold;
    h_consts.max_extrema = max_extrema;
    h_consts.max_orientations = max_extrema + max_extrema / 4;
    h_consts.norm_multi = normalization_multiplier;

    float dn_step = 1.0f / 8.0f;
    float dn_base = 0.5f * dn_step - 20.0f * dn_step;
    for(int y = 0; y < 40; y++)
    {
        for(int x = 0; x < 40; x++)
        {
            float dnx = dn_base + x * dn_step;
            float dny = dn_base + y * dn_step;
            h_consts.desc_gauss[y][x] = expf(-scalbnf(dnx * dnx + dny * dny, -3));
        }
    }

    for(int i = 0; i < 16; i++)
    {
        const float nx = -1.0f + 1.0f / 16.0f + i * 1.0f / 8.0f;
        h_consts.desc_tile[i] = 1.0f - fabs(nx);
    }

    // TODO: Check that using usm is better than buffer here
    // or look into the  sycl::ext::oneapi::device_global
    // sycl::buffer<ConstInfo> d_consts(h_consts);

    // send constantst to device

    try
    {
        if(*d_consts == nullptr)
            *d_consts = sycl::malloc_device<ConstInfo>(1, Q);
        else
            std::cout << "\n\n\t\td_consts is not nullpointer and hence initialized --> no malloc needed\n\n"
                      << std::endl;
    }
    catch(const sycl::exception& e)
    {
        stringstream ss;
        ss << "Memory allocation failed: " << e.what();

        POP_FATAL(ss.str());

        // Here, d_consts was never assigned, so there's no need to free it
    }
    return Q.memcpy(*d_consts, &h_consts, sizeof(ConstInfo));
    // return sycl::event(); // tmp to avoid compiler warning function is not currently in use

    // TODO: ADd error checking here

    // err = cudaMemcpyToSymbol( d_consts, &h_consts,
    //                           sizeof(ConstInfo), 0,
    //                           cudaMemcpyHostToDevice );
    // POP_CUDA_FATAL_TEST( err, "Failed to upload h_consts to device: " );
}

} // namespace popsift
