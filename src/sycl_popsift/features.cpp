/*
 * Copyright 2016-2017, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "features.hpp"

#include "common/assist.h"
#include "common/debug_macros.hpp"
#include "sift_extremum.h"

// #include <math_constants.h>
#include <sycl/sycl.hpp> // for free and alloc and queue

#include <cerrno>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>

using namespace std;

namespace popsift {

/*************************************************************
 * FeaturesBase
 *************************************************************/

FeaturesBase::FeaturesBase()
  : _num_ext(0)
  , _num_ori(0)
{}

FeaturesBase::~FeaturesBase() = default;

/*************************************************************
 * FeaturesHost
 *************************************************************/

FeaturesHost::FeaturesHost(sycl::queue Q)
  : _device_queue(Q)
  , _ext(nullptr)
  , _ori(nullptr)
{}

FeaturesHost::FeaturesHost(sycl::queue Q, int num_ext, int num_ori)
  : _device_queue(Q)
  , _ext(nullptr)
  , _ori(nullptr)
{
    reset(num_ext, num_ori);
}

FeaturesHost::~FeaturesHost()
{
    // memalign_free(_ext);
    // memalign_free(_ori);

    fprintf(stderr, "FREEING FEATUREHOST\n");
    sycl::free(_ext, _device_queue);
    sycl::free(_ori, _device_queue);
}

void FeaturesHost::reset(int num_ext, int num_ori)
{
    if(_ext != nullptr)
    {
        sycl::free(_ext, _device_queue);
        _ext = nullptr;
    }
    if(_ori != nullptr)
    {
        sycl::free(_ori, _device_queue);
        _ori = nullptr;
    }

    size_t alignment = getPreferredAlignment(_device_queue);
    _ext = sycl::aligned_alloc_host<Feature>(alignment, num_ext, _device_queue);

    _ori = sycl::aligned_alloc_host<Descriptor>(alignment, num_ori, _device_queue);
    // It seems like host allocations use pinned memory as the docs say "The total size of host allocations will be
    // limited by the amount of pinnable-memory on the host on most systems." Hence assuming it's pinned memory
    // There is an extension to use pinned memory but that is for buffers (nothing for USM probs due to it already being
    // implemented for host memory)

    setFeatureCount(num_ext);
    setDescriptorCount(num_ori);
}

void FeaturesHost::print(std::ostream& ostr, bool write_as_uchar) const
{
    for(int i = 0; i < size(); i++)
    {
        _ext[i].print(ostr, write_as_uchar);
    }
}

std::ostream& operator<<(std::ostream& ostr, const FeaturesHost& feature)
{
    feature.print(ostr, false);
    return ostr;
}

/*************************************************************
 * FeaturesDev
 *************************************************************/

FeaturesDev::FeaturesDev(sycl::queue Q)
  : _device_queue(Q)
  , _ext(nullptr)
  , _ori(nullptr)
  , _rev(nullptr)
{}

FeaturesDev::FeaturesDev(sycl::queue Q, int num_ext, int num_ori)
  : _device_queue(Q)
  , _ext(nullptr)
  , _ori(nullptr)
  , _rev(nullptr)
{
    reset(num_ext, num_ori);
}

FeaturesDev::~FeaturesDev()
{
    // cudaFree(_ext);
    // cudaFree(_ori);
    // cudaFree(_rev);

    fprintf(stderr, "DESTRUCTURE OGA FeaturesDev\n");
    sycl::free(_ext, _device_queue);
    sycl::free(_ori, _device_queue);
    sycl::free(_rev, _device_queue);
}

void FeaturesDev::reset(int num_ext, int num_ori)
{
    if(_ext != nullptr)
    {
        sycl::free(_ext, _device_queue);
        _ext = nullptr;
    }
    if(_ori != nullptr)
    {
        sycl::free(_ori, _device_queue);
        _ori = nullptr;
    }
    if(_rev != nullptr)
    {
        sycl::free(_rev, _device_queue);
        _rev = nullptr;
    }

    // TODO: Look into using malloc_deviceT as I don't see why we need shared (managed in cuda)
    // If user want host access just get a hostPointer no?
    _ext = popsift::sycl_common::malloc_sharedT<Feature>(
      num_ext, __FILE__, __LINE__, "Could not allocate shared memory Feautre for clone ", _device_queue);
    _ori = popsift::sycl_common::malloc_sharedT<Descriptor>(
      num_ori, __FILE__, __LINE__, "Could not allocate shared memory Descriptor for clone ", _device_queue);
    _rev = popsift::sycl_common::malloc_sharedT<int>(
      num_ori, __FILE__, __LINE__, "Could not allocate shared memory int array for clone ", _device_queue);

    setFeatureCount(num_ext);
    setDescriptorCount(num_ori);
}

// inline float l2_in_t0(const float4* lptr, const float4* rptr)

template<typename GroupType>
inline float l2_in_t0(const sycl::vec<float, 4>* lptr,
                      const sycl::vec<float, 4>* rptr,
                      GroupType& group,
                      sycl::nd_item<1>& it)
{
    // const float4 lval = lptr[it.get_local_id(0)];
    // const float4 rval = rptr[it.get_local_id(0)];l
    const sycl::vec<float, 4> lval = lptr[it.get_local_id(0)];
    const sycl::vec<float, 4> rval = rptr[it.get_local_id(0)];

    // Could be done as one minus the first one
    const sycl::vec<float, 4> mval =
      sycl::vec<float, 4>(lval.x() - rval.x(), lval.y() - rval.y(), lval.z() - rval.z(), lval.w() - rval.w());

    // Is probably a vec functon for this aswell sycl::dot mby
    float res = mval.x() * mval.x() + mval.y() * mval.y() + mval.z() * mval.z() + mval.w() * mval.w();

    // Sum of squared differences of complete 128 descriptors
    return sycl::reduce_over_group(group, res, sycl::plus<float>());
}

template<bool useSubGroup>
class Compute_distance
{
  private:
    sycl::vec<int, 3>* match_matrix;
    Descriptor* l;
    int l_len;
    Descriptor* r;
    int r_len;

  public:
    Compute_distance(sycl::vec<int, 3>* match_matrix, Descriptor* l, int l_len, Descriptor* r, int r_len)
      : match_matrix(match_matrix)
      , l(l)
      , l_len(l_len)
      , r(r)
      , r_len(r_len) {};

    inline void operator()(sycl::nd_item<1> it) const
    {
        // Could remove this statement when using global l_len * 32 and local 32 so one per
        // Hence no group could be superflous
        if(it.get_group(0) >= l_len) // Should be impossible (considering l_len is setting the dimension of global
            return;
        const int idx = it.get_group(0);

        float match_1st_val = std::numeric_limits<float>::infinity();
        float match_2nd_val = std::numeric_limits<float>::infinity();
        int match_1st_idx = 0;
        int match_2nd_idx = 0;

        auto group = [&]() {
            if constexpr(useSubGroup)
                return it.get_sub_group();
            else
                return it.get_group();
        }();

        // const float4* lptr = (const float4*)(&l[idx]);
        // Should use reinterpret_cast isntead mby?
        // const sycl::vec<float, 4>* lptr = (const sycl::vec<float, 4>*)(&l[idx]);
        const sycl::vec<float, 4>* lptr = reinterpret_cast<const sycl::vec<float, 4>*>(&l[idx]);

        for(int i = 0; i < r_len; i++)
        {
            // const float4* rptr = (const float4*)(&r[i]);
            // const sycl::vec<float, 4>* rptr = (const sycl::vec<float, 4>*)(&r[i]);
            const sycl::vec<float, 4>* rptr = reinterpret_cast<const sycl::vec<float, 4>*>(&r[i]);

            const float res = l2_in_t0(lptr, rptr, group, it);

            // if(threadIdx.x == 0)
            if(it.get_local_id(0) == 0) // Could use group.leader() for sub_group version
            {
                if(res < match_1st_val)
                {
                    match_2nd_val = match_1st_val;
                    match_2nd_idx = match_1st_idx;
                    match_1st_val = res;
                    match_1st_idx = i;
                }
                else if(res < match_2nd_val)
                {
                    match_2nd_val = res;
                    match_2nd_idx = i;
                }
            }
            // __syncthreads();
            sycl::group_barrier(group); // not sure if this is needed for sub_group
        }

        // if(threadIdx.x == 0)
        if(it.get_local_id(0) == 0)
        {
            bool accept = ((match_1st_val / match_2nd_val) < 0.8f);
            // if(accept)
            // {
            // sycl::ext::oneapi::experimental::printf(
            //   "idx = %d ---- match_1st_idx = %d (val %f) -- match_2nd_idx = %d (val %f) --> accept = %d\n",
            //   idx,
            //   match_1st_idx,
            //   match_1st_val,
            //   match_2nd_idx,
            //   match_2nd_val,
            //   accept);
            // }
            // match_matrix[blockIdx.x] = make_int3(match_1st_idx, match_2nd_idx, accept);
            match_matrix[it.get_group(0)] = sycl::vec<int, 3>(match_1st_idx, match_2nd_idx, accept);
        }
    }
};

// __global__ void show_distance(int3* match_matrix,
//                               Feature* l_ext,
//                               Descriptor* l_ori,
//                               int* l_fem,
//                               int l_len,
//                               Feature* r_ext,
//                               Descriptor* r_ori,
//                               int* r_fem,
//                               int r_len)
// {
//     for(int i = 0; i < l_len; i++)
//     {
//         const float4* lptr = (const float4*)(&l_ori[i]);
//         const float4* rptr1 = (const float4*)(&r_ori[match_matrix[i].x]);
//         const float4* rptr2 = (const float4*)(&r_ori[match_matrix[i].y]);
//         float d1 = l2_in_t0(lptr, rptr1);
//         float d2 = l2_in_t0(lptr, rptr2);
//         if(threadIdx.x == 0)
//         {
//             if(match_matrix[i].z)
//             {
//                 Feature* lx = &l_ext[l_fem[i]];
//                 Feature* rx = &r_ext[r_fem[match_matrix[i].x]];
//                 printf("accept feat %4d [%4d] matches feat %4d [%4d] ( 2nd feat %4d [%4d] ) dist %.3f vs %.3f"
//                        " (%.1f,%.1f)-(%.1f,%.1f)\n",
//                        l_fem[i],
//                        i,
//                        r_fem[match_matrix[i].x],
//                        match_matrix[i].x,
//                        r_fem[match_matrix[i].y],
//                        match_matrix[i].y,
//                        d1,
//                        d2,
//                        lx->xpos,
//                        lx->ypos,
//                        rx->xpos,
//                        rx->ypos);
//             }
//             else
//             {
//                 printf("reject feat %4d [%4d] matches feat %4d [%4d] ( 2nd feat %4d [%4d] ) dist %.3f vs %.3f\n",
//                        l_fem[i],
//                        i,
//                        r_fem[match_matrix[i].x],
//                        match_matrix[i].x,
//                        r_fem[match_matrix[i].y],
//                        match_matrix[i].y,
//                        d1,
//                        d2);
//             }
//         }
//         __syncthreads();
//     }
// }

// What good are you?? -- for demo purpose I think print outs the distance in ^

// class compute_distance_sub_group_match;
// void FeaturesDev::match(FeaturesDev* other)
// {
//     int l_len = getDescriptorCount();
//     int r_len = other->getDescriptorCount();
//
//     int3* match_matrix = popsift::cuda::malloc_devT<int3>(l_len, __FILE__, __LINE__);
//
//     dim3 grid;
//     grid.x = l_len;
//     grid.y = 1;
//     grid.z = 1;
//     dim3 block;
//     block.x = 32;
//     block.y = 1;
//     block.z = 1;
//
//     compute_distance<<<grid, block>>>(match_matrix, getDescriptors(), l_len, other->getDescriptors(), r_len);
//
//     POP_SYNC_CHK;
//
//     show_distance<<<1, 32>>>(match_matrix,
//                              getFeatures(),
//                              getDescriptors(),
//                              getReverseMap(),
//                              l_len,
//                              other->getFeatures(),
//                              other->getDescriptors(),
//                              other->getReverseMap(),
//                              r_len);
//
//     POP_SYNC_CHK;
//
//     cudaFree(match_matrix);
// }

class compute_distance_sub_group;

// Passes the pointer and a callback to wait for kernel to finish and callback to free the pointer
// NOTE: Might not work too well to use sycl::vec for portability's sake
std::tuple<sycl::vec<int, 3>*, std::function<void()>, std::function<void()>> FeaturesDev::matchAndReturn(
  FeaturesDev* other)
{
    int l_len = getDescriptorCount();
    int r_len = other->getDescriptorCount();

    // int3* match_matrix = popsift::cuda::malloc_mgdT<int3>(l_len, __FILE__, __LINE__);

    // Consider using device memory and explicit copy to host memory
    sycl::vec<int, 3>* match_matrix =
      popsift::sycl_common::malloc_sharedT<sycl::vec<int, 3>>(l_len, __FILE__, __LINE__, "", _device_queue);

    int size = get_kernel_subgroup_size<compute_distance_sub_group>(_device_queue);
    bool useSubGroup = size >= 32;

    sycl::range global{static_cast<size_t>(l_len * 32)}; // one 32 wide group per descriptor
    sycl::range local{32};                               // Could channge to width of sub group mby

    sycl::event matchEvent;
    if(useSubGroup)
    {
        fprintf(stderr, "Using sub group\n");
        matchEvent = _device_queue.parallel_for(
          sycl::nd_range{global, local},
          Compute_distance<true>(match_matrix, getDescriptors(), l_len, other->getDescriptors(), r_len));
    }
    else
    {
        fprintf(stderr, "Using work group\n");
        matchEvent = _device_queue.parallel_for<compute_distance_sub_group>(
          sycl::nd_range{global, local},
          Compute_distance<true>(match_matrix, getDescriptors(), l_len, other->getDescriptors(), r_len));
    }

    // auto wait_for_matrix = [&matchEvent, Q = _device_queue, l_len]() {
    // auto wait_for_matrix = [event = matchEvent, &Q = _device_queue]() {
    //     fprintf(stderr, "Called WAIT CALLBAKC\n");
    //     event.wait();
    //     Q.wait();
    // };

    auto wait_for_matrix = [event = std::make_shared<sycl::event>(matchEvent), &Q = _device_queue]() {
        fprintf(stderr, "Called WAIT CALLBACK\n");
        event->wait();
        Q.wait();
    };

    // If you wait  you most likely want to get the data
    // Not sure if this is better than not having it...
    // Q.prefetch(match_matrix, sizeof(sycl::vec<int, 3>) * l_len);

    auto free_matrix = [match_matrix, ctx = _device_queue.get_context()]() {
        if(sycl::get_pointer_type(match_matrix, ctx) != sycl::usm::alloc::unknown)
        {
            // Not sure if this throws and should be caught (could not see it in docs)
            fprintf(stderr, "FREEING match_matrix with callback :D\n");
            sycl::free(match_matrix, ctx);
        }
        else
        {
            fprintf(stderr, "Pointer already freed or invalid\n");
        }
    };

    return std::make_tuple(match_matrix, wait_for_matrix, free_matrix);
}

// void FeaturesDev::freeMatches(int3* match_matrix) { popsift::cuda::free_mgd(match_matrix); }

Descriptor* FeaturesDev::getDescriptor(int descIndex) { return &_ori[descIndex]; }

const Descriptor* FeaturesDev::getDescriptor(int descIndex) const { return &_ori[descIndex]; }

Feature* FeaturesDev::getFeatureForDescriptor(int descIndex) { return &_ext[_rev[descIndex]]; }

const Feature* FeaturesDev::getFeatureForDescriptor(int descIndex) const { return &_ext[_rev[descIndex]]; }

/*************************************************************
 * Feature
 *************************************************************/

void Feature::print(std::ostream& ostr, bool write_as_uchar) const
{
    float sigval = 1.0f / (sigma * sigma);

    for(int ori = 0; ori < num_ori; ori++)
    {
        ostr << xpos << " " << ypos << " " << sigval << " 0 " << sigval << " ";
        if(write_as_uchar)
        {
            for(int i = 0; i < 128; i++)
            {
                // ostr << roundf(desc[ori]->features[i]) << " ";
                ostr << sycl::round(desc[ori]->features[i]) << " ";
            }
        }
        else
        {
            ostr << std::setprecision(3);
            for(int i = 0; i < 128; i++)
            {
                ostr << desc[ori]->features[i] << " ";
            }
            ostr << std::setprecision(6);
        }
        ostr << std::endl;
    }
}

std::ostream& operator<<(std::ostream& ostr, const Feature& feature)
{
    feature.print(ostr, false);
    return ostr;
}

} // namespace popsift
