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
#include "sycl_popsift/popsift.hpp"

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

// For Joint matrix matching
namespace syclexp = sycl::ext::oneapi::experimental;

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
    const sycl::vec<float, 4> lval = lptr[it.get_local_id(0)];
    const sycl::vec<float, 4> rval = rptr[it.get_local_id(0)];

#if 0
    // Verbose write out of SSD
    const sycl::vec<float, 4> mval =
      sycl::vec<float, 4>(lval.x() - rval.x(), lval.y() - rval.y(), lval.z() - rval.z(), lval.w() - rval.w());

    float res = mval.x() * mval.x() + mval.y() * mval.y() + mval.z() * mval.z() + mval.w() * mval.w();
#else
    // Using sycl functions for potentially better performance
    const sycl::vec<float, 4> mval = lval - rval;
    float res = sycl::dot(mval, mval);
#endif

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

        const sycl::vec<float, 4>* lptr = reinterpret_cast<const sycl::vec<float, 4>*>(&l[idx]);

        for(int i = 0; i < r_len; i++)
        {
            const sycl::vec<float, 4>* rptr = reinterpret_cast<const sycl::vec<float, 4>*>(&r[i]);

            const float res = l2_in_t0(lptr, rptr, group, it);

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
            sycl::group_barrier(group); // not sure if this is needed for sub_group
        }

        if(it.get_local_id(0) == 0)
        {
            bool accept = ((match_1st_val / match_2nd_val) < 0.8f);
            match_matrix[it.get_group(0)] = sycl::vec<int, 3>(match_1st_idx, match_2nd_idx, accept);
        }
    }
};

// Probs need to be quite many for this to be advantageous
template<bool Swap> // Wheter l is the objects descriptrs or if it was swaped due to l_len < r_len
class Compute_distance_matrix
{
  private:
    sycl::vec<int, 3>* match_matrix;
    DescriptorHalf* l;
    int l_len;
    DescriptorHalf* r;
    int r_len;

  public:
    Compute_distance_matrix(sycl::vec<int, 3>* match_matrix, DescriptorHalf* l, int l_len, DescriptorHalf* r, int r_len)
      : match_matrix(match_matrix)
      , l(l)
      , l_len(l_len)
      , r(r)
      , r_len(r_len) {};

    inline void operator()(sycl::nd_item<1> it) const
    {
        // Load the matrix B that we are responsible for into shared memory and transpose it

        // First compute B^2 --> Save the 16 values --> Then transpose the matrix

        // Padded to 17 to avoid bank conflicts during transpose
        sycl::multi_ptr<sycl::half[128][17], sycl::access::address_space::local_space> ptr =
          sycl::ext::oneapi::group_local_memory<sycl::half[128][17]>(it.get_group());

        // The descriptors sub_group is responsible for transposed (after getting descT ^ 2)
        auto& descT = *ptr;

        sycl::sub_group group = it.get_sub_group();

        sycl::half b_norms[16]; // hopefully fine to store in regsters (consider moving to local_memory)
        for(int i = 0; i < 16; ++i)
        {
            // Compute the 16 norms && transpose:

            // Coaleced memory reads from global mem
            sycl::vec<sycl::half, 4> item_loads{l[it.get_global_id(0) * 16 + i].features[it.get_local_id(1)],
                                                l[it.get_global_id(0) * 16 + i].features[it.get_local_id(1) + 32],
                                                l[it.get_global_id(0) * 16 + i].features[it.get_local_id(1) + 32 * 2],
                                                l[it.get_global_id(0) * 16 + i].features[it.get_local_id(1) + 32 * 3]};

            b_norms[i] = sycl::dot(item_loads, item_loads);
            b_norms[i] = sycl::reduce_over_group(group, b_norms[i], sycl::plus<sycl::half>());

            // Store transposed to local memoroy
            // Should not be bank conflicty due to stride being 17 (padded with one column)
            descT[it.get_local_id(1)][i] = item_loads.x();
            descT[it.get_local_id(1) + 32][i] = item_loads.y();
            descT[it.get_local_id(1) + 32 * 2][i] = item_loads.z();
            descT[it.get_local_id(1) + 32 * 3][i] = item_loads.w();
        }

        // Compute a^2 here for each of the 16 desc we are responsible for so storing 16 floats in registers(mby local
        // mem) Not sure if we wan't to reaload the 8 A's or keep them as A0 A1 ... A7
        syclexp::matrix::
          joint_matrix<sycl::sub_group, sycl::half, syclexp::matrix::use::a, 16, 16, syclexp::matrix::layout::row_major>
            A; // Loaded in on the fly (colums is desc from other set(smaller one)) (this one is cheaper due to not
               // having to transpose)

        syclexp::matrix::
          joint_matrix<sycl::sub_group, sycl::half, syclexp::matrix::use::b, 16, 16, syclexp::matrix::layout::col_major>
            B; // Subgroup keep the 8 B for the whole kernel it is responsible for those 16 descriptors

        syclexp::matrix::joint_matrix<sycl::sub_group,
                                      sycl::half,
                                      syclexp::matrix::use::accumulator,
                                      16,
                                      16>
          accumulator; // Accumulate the term for the vector pairs

        // auto local_ptr =
        //   sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::yes>(
        //     &descT[0][0]);
        //
        // // template<typename Group, typename T1, typename T2, size_t Rows, size_t Cols, typename PropertyListT>
        // syclexp::matrix::joint_matrix_load(
        //   group, B, local_ptr, 17, syclexp::matrix::layout::row_major); // Load from local into B

        // auto local_ptr =
        //   sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::yes>(
        //     &descT[0][0]);
        //
        // syclexp::matrix::joint_matrix_load(group,
        //                                    B,
        //                                    local_ptr,
        //                                    128, // Must be 128 for column-major in [128][17]
        //                                    syclexp::matrix::layout::col_major);

        // auto local_ptr =
        //   sycl::multi_ptr<sycl::half, sycl::access::address_space::local_space, sycl::access::decorated::yes>(
        //     &descT[0][0]);

        auto local_ptr_alt = sycl::address_space_cast<sycl::access::address_space::local_space,
                                                      sycl::access::decorated::yes,
                                                      sycl::half>( // Explicit template parameter added
          &descT[0][0]);

        // syclexp::matrix::joint_matrix_load(group,
        //                                    B,
        //                                    local_ptr_alt, // or local_ptr_alt
        //                                    128,           // Leading dimension for column-major
        //                                    syclexp::matrix::layout::col_major);

        // load B (our responsibility)

        //

        //

        // DOING OLD VERSISON TO BE REMOVED

        // Could remove this statement when using global l_len * 32 and local 32 so one per
        // Hence no group could be superflous
        //     if(it.get_group(0) >= l_len) // Should be impossible (considering l_len is setting the dimension of
        //     global
        //         return;
        //     const int idx = it.get_group(0);
        //
        //     float match_1st_val = std::numeric_limits<float>::infinity();
        //     float match_2nd_val = std::numeric_limits<float>::infinity();
        //     int match_1st_idx = 0;
        //     int match_2nd_idx = 0;
        //
        //     // auto group = [&]() {
        //     //     if constexpr(useSubGroup)
        //     //         return it.get_sub_group();
        //     //     else
        //     //         return it.get_group();
        //     // }();
        //     // auto group = it.get_sub_group();
        //
        //     const sycl::vec<float, 4>* lptr = reinterpret_cast<const sycl::vec<float, 4>*>(&l[idx]);
        //
        //     for(int i = 0; i < r_len; i++)
        //     {
        //         const sycl::vec<float, 4>* rptr = reinterpret_cast<const sycl::vec<float, 4>*>(&r[i]);
        //
        //         const float res = l2_in_t0(lptr, rptr, group, it);
        //
        //         if(it.get_local_id(0) == 0) // Could use group.leader() for sub_group version
        //         {
        //             if(res < match_1st_val)
        //             {
        //                 match_2nd_val = match_1st_val;
        //                 match_2nd_idx = match_1st_idx;
        //                 match_1st_val = res;
        //                 match_1st_idx = i;
        //             }
        //             else if(res < match_2nd_val)
        //             {
        //                 match_2nd_val = res;
        //                 match_2nd_idx = i;
        //             }
        //         }
        //         sycl::group_barrier(group); // not sure if this is needed for sub_group
        //     }
        //
        //     if(it.get_local_id(0) == 0)
        //     {
        //         bool accept = ((match_1st_val / match_2nd_val) < 0.8f);
        //         match_matrix[it.get_group(0)] = sycl::vec<int, 3>(match_1st_idx, match_2nd_idx, accept);
        //     }
    }
};

class compute_distance_sub_group;

// Passes the pointer and a callback to wait for kernel to finish and callback to free the pointer
// NOTE: Might not work too well to use sycl::vec for portability's sake
std::tuple<sycl::vec<int, 3>*, std::function<void()>, std::function<void()>> FeaturesDev::matchAndReturn(
  FeaturesDev* other)
{
    int l_len = getDescriptorCount();
    int r_len = other->getDescriptorCount();

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
        matchEvent = _device_queue.parallel_for<compute_distance_sub_group>(
          sycl::nd_range{global, local},
          Compute_distance<true>(match_matrix, getDescriptors(), l_len, other->getDescriptors(), r_len));
    }
    else
    {
        matchEvent = _device_queue.parallel_for(
          sycl::nd_range{global, local},
          Compute_distance<false>(match_matrix, getDescriptors(), l_len, other->getDescriptors(), r_len));
    }

    auto wait_for_matrix = [event = std::make_shared<sycl::event>(matchEvent), &Q = _device_queue]() {
        event->wait();
        Q.wait();
    };

    auto free_matrix = [match_matrix, ctx = _device_queue.get_context()]() {
        if(sycl::get_pointer_type(match_matrix, ctx) != sycl::usm::alloc::unknown)
        {
            // Not sure if this throws and should be caught (could not see it in docs)
            sycl::free(match_matrix, ctx);
        }
        else
        {
            fprintf(stderr, "Pointer already freed or invalid\n");
        }
    };

    return std::make_tuple(match_matrix, wait_for_matrix, free_matrix);
}

// Just for testing should make the descriptors half when in use early on in pipeline in case of using
// Tensor That might have to bee compile time dependent and I'm currently running matrix check in runtime...
void convert_float_to_half_usm(sycl::queue& Q, Descriptor* float_ptr, DescriptorHalf* half_ptr, size_t count)
{
    Q.parallel_for(sycl::range<2>{count, 128},
                   [=](sycl::id<2> idx) {
                       half_ptr[idx[0]].features[idx[1]] = static_cast<sycl::half>(float_ptr[idx[0]].features[idx[1]]);
                   })
      .wait();
}

// class compute_distance_matrix_sub_group; // Don't think that is needed as if you support matrix you support 32
// wide sub_groups (Is my assumption :D) could be wrong
class compute_distance_fallback;

// Uses tensor cores to compute the SSD (should be better performance per SM but might not saturate the whole GPU
// unless the GPU is weak or there is alot of descriptors in one of the images largest dimension decides how many
// subgroups are used. Each subgroup computes matcches for 16 descriptros (size of matrix operation)
std::tuple<sycl::vec<int, 3>*, std::function<void()>, std::function<void()>> FeaturesDev::matrixMatchAndReturn(
  FeaturesDev* other)
{
    int l_len = getDescriptorCount();
    int r_len = other->getDescriptorCount();

    // Consider using device memory and explicit copy to host memory
    sycl::vec<int, 3>* match_matrix =
      popsift::sycl_common::malloc_sharedT<sycl::vec<int, 3>>(l_len, __FILE__, __LINE__, "", _device_queue);

    sycl::event matchEvent;

#if USE_JOINT_MATRIX
    if(PopSift::matrixSupported)
    {
        // Matrix -- Assumes that if you support matrix 16x16 you support 32 wide sub groups

        DescriptorHalf* l_half = sycl_common::malloc_devT<DescriptorHalf>(l_len, __FILE__, __LINE__, "", _device_queue);
        DescriptorHalf* r_half = sycl_common::malloc_devT<DescriptorHalf>(r_len, __FILE__, __LINE__, "", _device_queue);

        convert_float_to_half_usm(_device_queue, getDescriptors(), l_half, l_len);
        convert_float_to_half_usm(_device_queue, other->getDescriptors(), r_half, r_len);

        if(l_len < r_len) // We want most of the descriptrs on left side
        {
            sycl::range global{static_cast<size_t>(l_len * 32)}; // one 32 wide group per descriptor
            sycl::range local{32};
            // SWAP
            matchEvent =
              _device_queue.parallel_for(sycl::nd_range{global, local},
                                         Compute_distance_matrix<false>(match_matrix, l_half, l_len, r_half, r_len));
        }
        {
            sycl::range global{static_cast<size_t>(l_len * 32)}; // one 32 wide group per descriptor
            sycl::range local{32};
            // Normal order of returned matrix
            matchEvent = _device_queue.parallel_for(
              sycl::nd_range{global, local}, Compute_distance_matrix<true>(match_matrix, l_half, l_len, r_half, r_len));
            // Compute_distance_matrix<true>(match_matrix, getDescriptors(), l_len, other->getDescriptors(),
            // r_len));
        }

        // TMP
        matchEvent.wait();
        sycl::free(l_half, _device_queue);
        sycl::free(r_half, _device_queue);
    }
    else
#endif
    {
        sycl::range global{static_cast<size_t>(l_len * 32)}; // one 32 wide group per descriptor
        sycl::range local{32};

        int size = get_kernel_subgroup_size<compute_distance_fallback>(_device_queue);

        if(size >= 32)
        {
            // Fallback subgroup
            matchEvent = _device_queue.parallel_for<compute_distance_fallback>(
              sycl::nd_range{global, local},
              Compute_distance<true>(match_matrix, getDescriptors(), l_len, other->getDescriptors(), r_len));
        }
        else
        {
            //  Fallback work group
            matchEvent = _device_queue.parallel_for(
              sycl::nd_range{global, local},
              Compute_distance<false>(match_matrix, getDescriptors(), l_len, other->getDescriptors(), r_len));
        }
    }

    auto wait_for_matrix = [event = std::make_shared<sycl::event>(matchEvent), &Q = _device_queue]() {
        event->wait();
        Q.wait();
    };

    auto free_matrix = [match_matrix, ctx = _device_queue.get_context()]() {
        if(sycl::get_pointer_type(match_matrix, ctx) != sycl::usm::alloc::unknown)
        {
            // Not sure if this throws and should be caught (could not see it in docs)
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

// Missing show_distance and match funtions usued to printout matches (not too usefull)

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
