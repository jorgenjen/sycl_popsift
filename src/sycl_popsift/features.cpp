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
#include <cstdio>
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
    sycl::half* l;
    int l_len;
    sycl::half* r;
    int r_len;
    sycl::local_accessor<sycl::half, 1> test;
    sycl::local_accessor<float, 1> b_norm;
    sycl::local_accessor<sycl::half, 1> a_staging_tile;
    float* write_back;

  public:
    Compute_distance_matrix(sycl::vec<int, 3>* match_matrix,
                            sycl::half* l,
                            int l_len,
                            sycl::half* r,
                            int r_len,
                            sycl::local_accessor<sycl::half, 1> test,
                            sycl::local_accessor<float, 1> b_norm,
                            sycl::local_accessor<sycl::half, 1> a_staging_tile,
                            float* write_back) // tmp testing
      : match_matrix(match_matrix)
      , l(l)
      , l_len(l_len)
      , r(r)
      , r_len(r_len)
      , test(test)
      , b_norm(b_norm)
      , a_staging_tile(a_staging_tile)
      , write_back(write_back) {};

    inline void operator()(sycl::nd_item<1> it) const
    {
        // Should move this to the command group like the example docs
        sycl::global_ptr<sycl::half> l_ptr(l);
        sycl::global_ptr<float> backy(write_back);

        auto my_matrix = test.get_multi_ptr<sycl::access::decorated::no>();
        // sycl::global_ptr<sycl::half> r_ptr(r);

        // auto r_ptr =
        //   sycl::get_multi_ptr<sycl::half, sycl::access::address_space::global_space,
        //   sycl::access::decorated::yes>(r);

        auto r_ptr =
          sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::yes>(r);

        auto a_tile = a_staging_tile.get_multi_ptr<sycl::access::decorated::yes>();

        int x = it.get_local_id(0);
        int desc_start = it.get_group(0); // only for global reads of B

        sycl::sub_group group = it.get_sub_group();

        // Could load it all in but that would take wayy to much shared memory I think
        // Aka loading in 16x128. currently just loading 16x16 but needs 16 events for that...

        // std::array<sycl::device_event, 16> events;

        // Both needs to be decorated pointers...
        // Needed to be loaded like this If I want to store in array (can't use vector) and default constructor is
        // deleted for device event
        sycl::device_event events[16] = {it.get_group().async_work_group_copy(a_tile, r_ptr, 16),
                                         it.get_group().async_work_group_copy(a_tile + 16, r_ptr + 128, 16),
                                         it.get_group().async_work_group_copy(a_tile + 16 * 2, r_ptr + 128 * 2, 16),
                                         it.get_group().async_work_group_copy(a_tile + 16 * 3, r_ptr + 128 * 3, 16),
                                         it.get_group().async_work_group_copy(a_tile + 16 * 4, r_ptr + 128 * 4, 16),
                                         it.get_group().async_work_group_copy(a_tile + 16 * 5, r_ptr + 128 * 5, 16),
                                         it.get_group().async_work_group_copy(a_tile + 16 * 6, r_ptr + 128 * 6, 16),
                                         it.get_group().async_work_group_copy(a_tile + 16 * 7, r_ptr + 128 * 7, 16),
                                         it.get_group().async_work_group_copy(a_tile + 16 * 8, r_ptr + 128 * 8, 16),
                                         it.get_group().async_work_group_copy(a_tile + 16 * 9, r_ptr + 128 * 9, 16),
                                         it.get_group().async_work_group_copy(a_tile + 16 * 10, r_ptr + 128 * 10, 16),
                                         it.get_group().async_work_group_copy(a_tile + 16 * 11, r_ptr + 128 * 11, 16),
                                         it.get_group().async_work_group_copy(a_tile + 16 * 12, r_ptr + 128 * 12, 16),
                                         it.get_group().async_work_group_copy(a_tile + 16 * 13, r_ptr + 128 * 13, 16),
                                         it.get_group().async_work_group_copy(a_tile + 16 * 14, r_ptr + 128 * 14, 16),
                                         it.get_group().async_work_group_copy(a_tile + 16 * 15, r_ptr + 128 * 15, 16)};

        // Both needs to be decorated pointers...
        // tile_events[0] = it.get_group().async_work_group_copy(b_tile, r_ptr, 16);

        // auto tile_r0 = it.get_group().async_work_group_copy(b_tile, r_ptr, 16, 128);
        // auto tile_r1 = sycl::async_work_group_copy(b_tile + 16, r_ptr * 128, 16); // next row

#define use_register 0
#if use_register
        // This takes up 16 registers per work_item so it might bee to much more than a tile actually
        float my_norms[16]; // mby use shared mem  verssion b_norm (think you need to chnage reduce to joint_reduce then

#endif
        for(int i = 0; i < 16; ++i)
        {
            sycl::vec<sycl::half, 4> item_loads{l[(desc_start + i) * 128 + x],
                                                l[(desc_start + i) * 128 + x + 32],
                                                l[(desc_start + i) * 128 + x + 64],
                                                l[(desc_start + i) * 128 + x + 96]};

#if use_register
            my_norms[i] = sycl::dot(item_loads, item_loads);
            my_norms[i] = sycl::reduce_over_group(group, my_norms[i], sycl::plus<float>());
#else
            // Store norm in shared memory to reduce register usage
            float item_dot = sycl::dot(item_loads, item_loads);
            b_norm[i] = sycl::reduce_over_group(group, item_dot, sycl::plus<float>());
#endif

            // Store B to shred memory - without transposing need to load as column major
            my_matrix[(i * 128) + x] = item_loads.x();
            my_matrix[(i * 128) + x + 32] = item_loads.y();
            my_matrix[(i * 128) + x + 64] = item_loads.z();
            my_matrix[(i * 128) + x + 96] = item_loads.w();

            // Working load
            // my_matrix[(i * 128) + x] = l[(desc_start + i) * 128 + x];
            // my_matrix[(i * 128) + x + 32] = l[(desc_start + i) * 128 + x + 32];
            // my_matrix[(i * 128) + x + 64] = l[(desc_start + i) * 128 + x + 64];
            // my_matrix[(i * 128) + x + 96] = l[(desc_start + i) * 128 + x + 96];

            // NOTE: Could try to transpose and load as row_major
            // This would require to use stride 17 and larger shared memory to avoid bank conflicts
        }

        // my_matrix[128] = 0.1;

        sycl::group_barrier(it.get_group()); // Ensure shared memory is done

        syclexp::matrix::joint_matrix<sycl::sub_group,
                                      sycl::half,
                                      syclexp::matrix::use::a,
                                      16,
                                      16,
                                      syclexp::matrix::layout::row_major>
          A; // Loaded in on the fly (colums is desc from other set(smaller one)) (this one is cheaper due to not
             // having to transpose)

        // syclexp::matrix::joint_matrix<sycl::sub_group,
        //                               sycl::half,
        //                               syclexp::matrix::use::a,
        //                               16,
        //                               16,
        //                               syclexp::matrix::layout::row_major>
        //   A_squared; // Stores the sum of squares of A (used to compute A norm on the fly)

        // syclexp::matrix::use::a,
        // syclexp::matrix::joint_matrix<sycl::sub_group, float, syclexp::matrix::use::accumulator, 16, 16>
        //   // syclexp::matrix::layout::row_major>
        //   A_squared; // Stores the sum of squares of A (used to compute A norm on the fly)

        syclexp::matrix::joint_matrix<sycl::sub_group,
                                      sycl::half,
                                      syclexp::matrix::use::b,
                                      16,
                                      16,
                                      syclexp::matrix::layout::col_major>
          B; // Subgroup keep the 8 B for the whole kernel it is responsible for those 16 descriptors

        syclexp::matrix::joint_matrix<sycl::sub_group,
                                      float,
                                      syclexp::matrix::use::accumulator,
                                      16,
                                      16>
          C; // Accumulate the term for the vector pairs

        // syclexp::matrix::joint_matrix<sycl::sub_group,
        //                               float,
        //                               syclexp::matrix::use::accumulator,
        //                               16,
        //                               16>
        //   for_copy; // Accumulate the term for the vector pairs

        joint_matrix_fill(it.get_sub_group(), C, 0);

        // Do first iteration out of loop as we are writing to A_squared in loop we add to it
        // Avoids the need to fil it with zeros

        syclexp::matrix::joint_matrix_load(it.get_sub_group(), B, my_matrix, 128);

        // Wait for A tile to be loaded
        // Not sure if I could only wait on the last one but don'T think that's guaranteed to work
        for(auto& e : events)
            e.wait();

        syclexp::matrix::joint_matrix_load(it.get_sub_group(), A, a_tile, 16);
        // less precise due to only using fp16 but only option while using loaded A
        // As matrecies need to be same use and float type for use::a is not suported
        // syclexp::matrix::joint_matrix_apply(
        //   group, A, A_squared, [=](sycl::half& a, sycl::half& a_sum) {
        //     a_sum = a * a; });

        syclexp::matrix::joint_matrix_mad(it.get_sub_group(), C, A, B, C);

        // Inner loop computing for 16 descriptors of r_ptr
        for(int i = 1; i < 8; ++i)
        {
            // Compute the whole descriptor ab so 256 total in the end 16 x 16 descriptor pairs

            // Load from global should prefetch
            syclexp::matrix::joint_matrix_load(it.get_sub_group(), A, r_ptr + i * 16, 128);

            // Load from shared_memory
            syclexp::matrix::joint_matrix_load(it.get_sub_group(), B, my_matrix + i * 16, 128);

            // Computes A^2
            // syclexp::matrix::joint_matrix_apply(
            //   group, A, A_squared, [=](sycl::half& a, sycl::half& a_sum) { a_sum += a * a; });

            syclexp::matrix::joint_matrix_mad(it.get_sub_group(), C, A, B, C);
        }

        // copy A_squared to a accumulator then we can copy it to shared mem
        // then we can do the stuff we need on that
        // syclexp::matrix::joint_matrix_copy(group, A_squared, for_copy);
        // syclexp::matrix::joint_matrix_copy(group, C, for_copy);

        //

        //

        // syclexp::matrix::joint_matrix_load(it.get_sub_group(), A, l_ptr, 128); // WORKS WORKS WORKS
        //
        // syclexp::matrix::joint_matrix_load(it.get_sub_group(), B, my_matrix, 128); // WORKS ASWELL
        //
        // syclexp::matrix::joint_matrix_mad(it.get_sub_group(), C, A, B, C);
        //
        syclexp::matrix::joint_matrix_store(it.get_sub_group(), C, backy, 16, syclexp::matrix::layout::row_major);

        //
        //

        //

        //
        // syclexp::matrix::joint_matrix_load(it.get_sub_group(), B, local_ptr, 16); // WORKS ASWELL
        // syclexp::matrix::joint_matrix_load(it.get_sub_group(), B, me_loc_ptr, 16); // NO worky...

        // C = syclexp::matrix::joint_matrix_mad(it.get_sub_group(), A, B, C); // Not working

        // sycl::global_ptr<sycl::half> r_ptr(r);
        // syclexp::matrix::joint_matrix_load(it.get_sub_group(), A, r_ptr + 256, 128); // WORKS WORKS WORKS
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
void convert_float_to_half_usm(sycl::queue& Q, Descriptor* float_ptr, sycl::half* half_ptr, size_t count)
{
    Q.parallel_for(sycl::range<2>{count, 128},
                   [=](sycl::id<2> idx) {
                       half_ptr[idx[0] * 128 + idx[1]] = static_cast<sycl::half>(float_ptr[idx[0]].features[idx[1]]);
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

        sycl::half* l_half = sycl_common::malloc_devT<sycl::half>(l_len * 128, __FILE__, __LINE__, "", _device_queue);
        sycl::half* r_half = sycl_common::malloc_devT<sycl::half>(r_len * 128, __FILE__, __LINE__, "", _device_queue);
        float* res = sycl_common::malloc_sharedT<float>(600, __FILE__, __LINE__, "", _device_queue);

        convert_float_to_half_usm(_device_queue, getDescriptors(), l_half, l_len);
        convert_float_to_half_usm(_device_queue, other->getDescriptors(), r_half, r_len);

        // if(l_len < r_len) // We want most of the descriptrs on left side
        if(true) // We want most of the descriptrs on left side
        {
            fprintf(stderr, "Trying to do matrix stuff yayayayayay\n");
            // sycl::range global{static_cast<size_t>(l_len * 32)}; // one 32 wide group per descriptor
            sycl::range global{static_cast<size_t>(1 * 32)}; // Running for one as a test
            sycl::range local{32};
            // SWAP

            _device_queue.submit([&](sycl::handler& cgh) {
                // need 7 for storing the older result values final is stored in current work range idx 7
                auto local_test = sycl::local_accessor<sycl::half, 1>(128 * 16, cgh); // one per row in work-group
                auto b_norm = sycl::local_accessor<float, 1>(16, cgh);                // one per row in work-group
                // auto local_a_square = sycl::local_accessor<sycl::half, 1>(16 * 16, cgh); // one per row in work-group
                auto b_staging_tile = sycl::local_accessor<sycl::half, 1>(16 * 16, cgh); // one per row in work-group

                cgh.parallel_for(
                  sycl::nd_range{global, local},
                  Compute_distance_matrix<false>(
                    match_matrix, l_half, l_len, r_half, r_len, local_test, b_norm, b_staging_tile, res));
            });

            _device_queue.wait();

            fprintf(stderr, "\n ");
            fprintf(stderr, "\n ");
            for(int i = 0; i < 16; ++i)
            {
                for(int j = 0; j < 16; ++j)
                {
                    fprintf(stderr, " %f ", res[i * 16 + j]);
                }

                fprintf(stderr, "\n ");
            }

            // auto sum = sycl::local_accessor<float, 1>((local[2] + 7) * 16, cgh); // one per row in work-group
            // matchEvent =
            //   _device_queue.parallel_for(sycl::nd_range{global, local},
            //                              Compute_distance_matrix<false>(match_matrix, l_half, l_len, r_half,
            //                              r_len));
        }
        {
            // sycl::range global{static_cast<size_t>(l_len * 32)}; // one 32 wide group per descriptor
            // sycl::range local{32};
            // // Normal order of returned matrix
            // matchEvent = _device_queue.parallel_for(
            //   sycl::nd_range{global, local}, Compute_distance_matrix<true>(match_matrix, l_half, l_len, r_half,
            //   r_len));
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

// Failed as it made the result of matrix compute 0 for all locations when it shoudl not be
// Veri sensetive these matrecies

// hopefully fine to store in regsters (consider moving to local_memory)
// for(int i = 0; i < 16; ++i)
// for(int i = 0; i < 1; ++i)
// {
//     // Compute the 16 norms && transpose:
//
//     // Coaleced memory reads from global mem
//     // sycl::vec<sycl::half, 4> item_loads{l[it.get_global_id(0) * 16 + i].features[it.get_local_id(1)],
//     //                                     l[it.get_global_id(0) * 16 + i].features[it.get_local_id(1) + 32],
//     //                                     l[it.get_global_id(0) * 16 + i].features[it.get_local_id(1) + 32 *
//     //                                     2], l[it.get_global_id(0) * 16 + i].features[it.get_local_id(1) +
//     // 32
//     //                                     * 3]};
//
//     // sycl::vec<sycl::half, 4> item_loads{l[(it.get_global_id(0) * 16 + i) * 128 + it.get_local_id(1)],
//     //                                     l[(it.get_global_id(0) * 16 + i) * 128 + it.get_local_id(1) + 32],
//     //                                     l[(it.get_global_id(0) * 16 + i) * 128 + it.get_local_id(1) + 64],
//     //                                     l[(it.get_global_id(0) * 16 + i) * 128 + it.get_local_id(1) +
//     96]};
//     //
//     // b_norms[i] = sycl::dot(item_loads, item_loads);
//     // b_norms[i] = sycl::reduce_over_group(group, b_norms[i], sycl::plus<sycl::half>());
//
//     // Store transposed to local memoroy
//     // Should not be bank conflicty due to stride being 17 (padded with one column)
//
//     // Skipping to teset
//     // my_matrix[i * 128 + it.get_local_id(1)] = item_loads.x();
//     // my_matrix[i * 128 + it.get_local_id(1) + 32] = item_loads.y();
//     // my_matrix[i * 128 + it.get_local_id(1) + 64] = item_loads.z();
//     // my_matrix[i * 128 + it.get_local_id(1) + 96] = item_loads.w();
//
//     my_matrix[i * 128 + it.get_local_id(1)] = 5;
//     my_matrix[i * 128 + it.get_local_id(1) + 32] = 10;
//     my_matrix[i * 128 + it.get_local_id(1) + 64] = 15;
//     my_matrix[i * 128 + it.get_local_id(1) + 96] = 21;
// }

// my_matrix[1 * 128 + it.get_local_id(1)] = 5.5;
// my_matrix[1 * 128 + it.get_local_id(1) + 32] = 10.2;
// my_matrix[1 * 128 + it.get_local_id(1) + 64] = 15.2;
// my_matrix[1 * 128 + it.get_local_id(1) + 96] = 21.1;
