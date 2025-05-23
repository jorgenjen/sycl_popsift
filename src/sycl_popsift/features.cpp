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
#include "sycl_popsift/sift_desc_config.hpp" // For FeatureType

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
#if USE_JOINT_MATRIX
  , _squared_norms(nullptr)
#endif
{}

FeaturesDev::FeaturesDev(sycl::queue Q, int num_ext, int num_ori)
  : _device_queue(Q)
  , _ext(nullptr)
  , _ori(nullptr)
  , _rev(nullptr)
#if USE_JOINT_MATRIX
  , _squared_norms(nullptr)
#endif
{
    reset(num_ext, num_ori);
}

FeaturesDev::~FeaturesDev()
{
    sycl::free(_ext, _device_queue);
    sycl::free(_ori, _device_queue);
    sycl::free(_rev, _device_queue);

#if USE_JOINT_MATRIX
    sycl::free(_squared_norms, _device_queue);
#endif
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
#if USE_JOINT_MATRIX
    if(_squared_norms != nullptr)
    {
        sycl::free(_squared_norms, _device_queue);
        _squared_norms = nullptr;
    }
    _squared_norms = popsift::sycl_common::malloc_devT<float>(
      num_ori, __FILE__, __LINE__, "Failed to allocate squared norms array", _device_queue);
#endif

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
inline FeatureType l2_in_t0(const sycl::vec<FeatureType, 4>* lptr,
                            const sycl::vec<FeatureType, 4>* rptr,
                            GroupType& group,
                            sycl::nd_item<1>& it,
                            float l_norm, // DELETE
                            float r_norm  // DELETE
)
{
    const sycl::vec<FeatureType, 4> lval = lptr[it.get_local_id(0)];
    const sycl::vec<FeatureType, 4> rval = rptr[it.get_local_id(0)];

#if 0
    // Verbose write out of SSD
    const sycl::vec<float, 4> mval =
      sycl::vec<float, 4>(lval.x() - rval.x(), lval.y() - rval.y(), lval.z() - rval.z(), lval.w() - rval.w());

    float res = mval.x() * mval.x() + mval.y() * mval.y() + mval.z() * mval.z() + mval.w() * mval.w();
#else
#define COMPUTE_MATRIX_LIKE 1

#if COMPUTE_MATRIX_LIKE
    // JUST FOR VERIFICATION OF USING THE NORMS

    FeatureType res = sycl::dot(lval, rval);                              // Compute AB
    res = sycl::reduce_over_group(group, res, sycl::plus<FeatureType>()); // Compute FINAL AB value

    return static_cast<FeatureType>(l_norm) + static_cast<FeatureType>(r_norm) - 2 * res; // A^2 + B^2 - 2 * AB

    // ####################################
    // ############## VERIFIED TO DO SAME #
    // ####################################

#else
    // NORMAL FASTER METHOD
    // Using sycl functions for potentially better performance
    const sycl::vec<FeatureType, 4> mval = lval - rval;
    FeatureType res = sycl::dot(mval, mval); // might be better to juse do mval = mval * mval and then explicitly sum

    // MOVE OUT OF IFDEF AS IT's the same for both outer IF DEFS
    // Sum of squared differences of complete 128 descriptors
    return sycl::reduce_over_group(group, res, sycl::plus<FeatureType>());

#endif

#endif
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
    float* l_norm;
    float* r_norm;

  public:
    // Compute_distance(sycl::vec<int, 3>* match_matrix, Descriptor* l, int l_len, Descriptor* r, int r_len)
    Compute_distance(
      sycl::vec<int, 3>* match_matrix, Descriptor* l, int l_len, Descriptor* r, int r_len, float* l_norm, float* r_norm)
      : match_matrix(match_matrix)
      , l(l)
      , l_len(l_len)
      , r(r)
      , r_len(r_len)
      , l_norm(l_norm) // REMOVE
      , r_norm(r_norm) // REMOVE jUST FOR TESTING
    {};

    inline void operator()(sycl::nd_item<1> it) const
    {
        // Could remove this statement when using global l_len * 32 and local 32 so one per
        // Hence no group could be superflous
        if(it.get_group(0) >= l_len) // Should be impossible (considering l_len is setting the dimension of global
            return;
        const int idx = it.get_group(0);

        float match_1st_val = std::numeric_limits<FeatureType>::infinity();
        float match_2nd_val = std::numeric_limits<FeatureType>::infinity();
        int match_1st_idx = 0;
        int match_2nd_idx = 0;

        auto group = [&]() {
            if constexpr(useSubGroup)
                return it.get_sub_group();
            else
                return it.get_group();
        }();

        const sycl::vec<FeatureType, 4>* lptr = reinterpret_cast<const sycl::vec<FeatureType, 4>*>(&l[idx]);

        for(int i = 0; i < r_len; i++)
        {
            const sycl::vec<FeatureType, 4>* rptr = reinterpret_cast<const sycl::vec<FeatureType, 4>*>(&r[i]);

            const float res = l2_in_t0(lptr, rptr, group, it, l_norm[idx], r_norm[i]);

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

            if(accept)
            {
                syclexp::printf("match_matrix[%d] = (%d, %d) --> (%f, %f)\n",
                                idx,
                                match_1st_idx,
                                match_2nd_idx,
                                match_1st_val,
                                match_2nd_val);
            }
        }
    }
};

// Bitonic sort helper (same as in bitonic sort class)
// inline int shiftit(const int my_index, const int shift, const int direction, const bool increasing)
// {
//     const T my_val = _array[my_index];
//
//     const T other_val = sycl::permute_group_by_xor(_group, my_val, 1 << shift);
//
//     const bool reverse = (_it.get_local_id(1) & (1 << direction));
//
//     const bool id_less = ((_it.get_local_id(1) & (1 << shift)) == 0);
//
//     // If it thread get other_val from a thread with higher id it will be true if it's value is higher than
//     // other otherwise if it gets other_val from lower thread id it will be true if my_val is smaler than
//     // other_val if equal it's always false
//     const bool my_more = id_less ? (my_val > other_val) : (my_val < other_val);
//
//     // xor my_more with reverse and then xor that with increasing ^ is bitwise xor but onely one bit for bool
//     const bool must_swap = !(my_more ^ reverse ^ increasing);
//
//     // If we must swap we pass the mask so we swap with the assigned lane
//     // otherwise we pass 0 and we don't (not sure if using different masks is alowed in sycl for a permute)
//     int lane = must_swap ? (1 << shift) : 0;
//     // Should not be allowed according to docs but seem to work...
//     return sycl::permute_group_by_xor(_group, my_index, lane);
// }

class Compute_squared_norm
{
  private:
    Descriptor* desc;
    float* squared_norms; // Is float as it's used with addtition to a float in matching kernel

  public:
    Compute_squared_norm(Descriptor* desc, float* squared_norms)
      : desc(desc)
      , squared_norms(squared_norms) {};

    inline void operator()(sycl::nd_item<1> it) const
    {
        const int idx = it.get_group(0);

        const sycl::vec<sycl::half, 4>* desc_ptr = reinterpret_cast<const sycl::vec<sycl::half, 4>*>(&desc[idx]);

        sycl::vec<sycl::half, 4> desc_val = desc_ptr[it.get_local_id(0)];

        desc_val = desc_val * desc_val;
        sycl::half sum = desc_val.x() + desc_val.y() + desc_val.z() + desc_val.w();

        sum = sycl::reduce_over_group(it.get_sub_group(), sum, sycl::plus<sycl::half>());
        if(it.get_local_id(0) == 0)
        {
            // Only leader writes
            squared_norms[idx] = static_cast<float>(sum); // to avoid cast in matching kernel
        }
    }
};

// Only int is used so could probably just make it always use int and drop the unsigned char idea
template<typename T>
struct scan_state_old
{
    float v1;
    float v2;
    sycl::vec<T, 2> idx;
};

struct scan_state
{
    sycl::vec<float, 2> value;
    sycl::vec<int, 2> idx;
};
// ####################################################################################################
// ############################################ HERE HERE HERE ########################################
// ####################################################################################################

// ####################################################################################################
// ############################################ HERE HERE HERE ########################################
// ####################################################################################################

// ####################################################################################################
// ############################################ HERE HERE HERE ########################################
// ####################################################################################################

#if false
inline void operator()(sycl::nd_item<1> it) const
{
    // compute
    // We have norms
    // need to compute ab -> then do norm + norm - 2ab; then find best along columns if presistent is B

    // Could transpose and store in shared memory but that would cause the shared memory usage for that to be
    // 128*16*2 = 4096 per sub_group which is too much even for ada lovlace with 48 warps per SM resuling in memory
    // usage of 4096*48 = 196608 which is more than the 128KB available L1 cache (shared memory) Hence occupancy
    // would be lower (31 would be max) and we already need more shared memory for the column compute hence it would
    // be too much (so would have to rely on fast transpose by the hardware by doing the load transposed)

    sycl::sub_group sg = it.get_sub_group();

    // Prefeth first tiles as early as possible
    syclexp::matrix::joint_matrix_prefetch<16, 16>(sg, l, 128, syclexp::matrix::layout::col_major);
    syclexp::matrix::joint_matrix_prefetch<16, 16>(sg, r, 128, syclexp::matrix::layout::row_major);

    // Store the l_norms to shared memory as they are reused a lot
    // NOTE: could try to store in registers but I'm afraid that would use too much registers unless we can
    // partition work-items to only need to read one or two different norms then it could work...

    auto l_ptr = sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::yes>(l);

    auto r_ptr = sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::yes>(r);

    auto l_norm_ptr =
      sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::yes>(l_norm);

    auto compute = compute_tile.get_multi_ptr<sycl::access::decorated::yes>();

    auto my_norm_ptr = my_norm.get_multi_ptr<sycl::access::decorated::yes>();

    // Did not work..
    // auto compute =
    //   compute_tile.get_multi_ptr<sycl::access::address_space::local_space, sycl::access::decorated::yes>();

    sycl::device_event load_my_norms = it.get_group().async_work_group_copy(my_norm_ptr, l_norm_ptr, 16);

    syclexp::matrix::joint_matrix<sycl::sub_group,
                                  sycl::half,
                                  syclexp::matrix::use::a,
                                  16,
                                  16,
                                  syclexp::matrix::layout::row_major>
      A; // Loaded in on the fly (colums is desc from other set(smaller one)) (this one is cheaper due to not
         // having to transpose)

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

    // NEED LOOP HERE TO LOOP OVER ALL r descriptors that are not the tale (non divisible by 16 tale is done one by
    // one below the loop)
    // Loading from global memory

    joint_matrix_fill(sg, C, 0); // reset
#pragma unroll
    for(int i = 0; i < 8; ++i)
    {
        // i * 16 == i << 4
        syclexp::matrix::joint_matrix_load(sg, B, l_ptr + (i << 4), 128); // our tile - const in outer
        syclexp::matrix::joint_matrix_load(sg, A, r_ptr + (i << 4), 128); // Changes in outer

        // Hopefully removed the if by unroll
        if(i < 7) // Could test with the prefetch being two ahead of the running instead of just one (more mem used)
        {
            // prefetches next to L1 which is the default (can be set to l2 in passed properties)
            syclexp::matrix::joint_matrix_prefetch<16, 16>(
              sg, l + ((i + 1) << 4), 128, syclexp::matrix::layout::col_major);
            syclexp::matrix::joint_matrix_prefetch<16, 16>(
              sg, r + ((i + 1) << 4), 128, syclexp::matrix::layout::row_major);
        }

        syclexp::matrix::joint_matrix_mad(sg, C, A, B, C);
    }

    syclexp::matrix::joint_matrix_store(sg, C, compute, 16, syclexp::matrix::layout::row_major);

    // Compute l_norm + r_norm - 2C
}
#endif

// template<bool SWAP> // Wheter l is the objects descriptrs or if it was swaped due to l_len < r_len
class Compute_distance_matrix_pre_norm
{
    sycl::vec<int, 3>* match_matrix;
    // Descriptor* l;
    sycl::half* l;
    float* l_norm;
    // int l_len; // NOTE: Don't think I need l_len
    // Descriptor* r;
    sycl::half* r; // for clear intent and one cast
    float* r_norm;
    int r_len;

    sycl::local_accessor<float, 1> compute_tile;

  public:
    // sycl::vec<int, 3>* match_matrix, Descriptor* l, float* l_norm, int l_len, Descriptor* r, float* r_norm, int
    // r_len)
    Compute_distance_matrix_pre_norm(sycl::vec<int, 3>* match_matrix,
                                     sycl::half* l,
                                     float* l_norm,
                                     // int l_len,
                                     sycl::half* r,
                                     float* r_norm,
                                     int r_len,
                                     sycl::local_accessor<float, 1> compute_tile)
      : match_matrix(match_matrix)
      , l(l)
      , l_norm(l_norm)
      // , l_len(l_len)
      , r(r)
      , r_norm(r_norm)
      , r_len(r_len)
      , compute_tile(compute_tile) {};

    inline void operator()(sycl::nd_item<1> it) const
    {
        // compute
        // We have norms
        // need to compute ab -> then do norm + norm - 2ab; then find best along columns if presistent is B

        // Could transpose and store in shared memory but that would cause the shared memory usage for that to be
        // 128*16*2 = 4096 per sub_group which is too much even for ada lovlace with 48 warps per SM resuling in memory
        // usage of 4096*48 = 196608 which is more than the 128KB available L1 cache (shared memory) Hence occupancy
        // would be lower (31 would be max) and we already need more shared memory for the column compute hence it would
        // be too much (so would have to rely on fast transpose by the hardware by doing the load transposed)

        sycl::sub_group sg = it.get_sub_group();
        // const int desc_start = it.get_group(0);

        // Store start for this sub_group/work_group
        // sycl::half* l_start = l + (it.get_group(0) << 7); // ERROR( need to offset for 16 vectors not just one)

        sycl::half* l_start = l + (it.get_group(0) << 11); // * 2048 == 128 * 16 (so 16 descriptors)

        // Move outer loop invariant pointers to start position for this sub_group/work_group
        // l += (desc_start << 7);
        // match_matrix += (desc_start << 7);

        // Prefeth first tiles as early as possible
        // syclexp::matrix::joint_matrix_prefetch<16, 16>(sg, l_start, 128, syclexp::matrix::layout::col_major);
        // syclexp::matrix::joint_matrix_prefetch<16, 16>(sg, r, 128, syclexp::matrix::layout::row_major);

        // syclexp::matrix::joint_matrix_prefetch<16, 16>(sg, l, 128, syclexp::matrix::layout::col_major);
        // syclexp::matrix::joint_matrix_prefetch<16, 16>(sg, r, 128, syclexp::matrix::layout::row_major);

        // Store the l_norms to shared memory as they are reused a lot
        // NOTE: could try to store in registers but I'm afraid that would use too much registers unless we can
        // partition work-items to only need to read one or two different norms then it could work...

        auto l_ptr =
          sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::yes>(l_start);

        auto r_ptr =
          sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::yes>(r);

        auto l_norm_ptr =
          sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::yes>(l_norm);

        auto compute = compute_tile.get_multi_ptr<sycl::access::decorated::yes>();

        // auto my_norm_ptr = my_norm.get_multi_ptr<sycl::access::decorated::yes>();

        // Did not work..
        // auto compute =
        //   compute_tile.get_multi_ptr<sycl::access::address_space::local_space, sycl::access::decorated::yes>();

        // sycl::device_event load_my_norms = it.get_group().async_work_group_copy(my_norm_ptr, l_norm_ptr, 16); // can
        // just store in register

        syclexp::matrix::joint_matrix<sycl::sub_group,
                                      sycl::half,
                                      syclexp::matrix::use::a,
                                      16,
                                      16,
                                      syclexp::matrix::layout::row_major>
          A; // Loaded in on the fly (colums is desc from other set(smaller one)) (this one is cheaper due to not
             // having to transpose)

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

        const int x = it.get_local_id(0);
        const float my_norm = l_norm[(it.get_group(0) << 4) + (x % 16)];
        const unsigned char second_row = it.get_local_id(0) / 16; // 0-15 -> 0 -- 16-31 -> 1

        // init global to large value so it will be replaced instantly by lower values
        // could have an explicit first iterations out of the loop to avoid this but don't think that's worth the zero
        // to minescule performance bump
        scan_state global_leader; // x > 15 don't really need this one (not used in their case)
        global_leader.value.x() = std::numeric_limits<float>::infinity();
        global_leader.value.y() = std::numeric_limits<float>::infinity();

        for(int outer = 0; outer < (r_len - 15); outer += 16) // Only full 16's remainder are in loop below this one
        // for(int outer = 0; outer < 16; outer += 16) // one iteration for test
        {
            joint_matrix_fill(sg, C, 0); // reset
#pragma unroll
            for(int i = 0; i < 8; ++i)
            {
                // i * 16 == i << 4
                syclexp::matrix::joint_matrix_load(sg, B, l_ptr + (i << 4), 128); // our tile - const in outer
                syclexp::matrix::joint_matrix_load(sg, A, r_ptr + (i << 4) + (outer << 7), 128); // Changes in outer

                // Hopefully removed the if by unroll
                if(i < 7)
                { // Could test with the prefetch being two ahead of the running instead of just one (more mem used)
                  // prefetches next to L1 which is the default (can be set to l2 in passed properties)
                  // syclexp::matrix::joint_matrix_prefetch<16, 16>(
                  //   sg, l_start + ((i + 1) << 4), 128, syclexp::matrix::layout::col_major);
                  // syclexp::matrix::joint_matrix_prefetch<16, 16>(
                  //   sg, r + ((i + 1) << 4) + (outer << 7), 128, syclexp::matrix::layout::row_major);
                }

                syclexp::matrix::joint_matrix_mad(sg, C, A, B, C);
            }

            syclexp::matrix::joint_matrix_store(sg, C, compute, 16, syclexp::matrix::layout::row_major);

            // Compute l_norm + r_norm - 2C

            sycl::group_barrier(it.get_group());

#define P_OUT false
#if P_OUT
            if(x == 0)
            {
                syclexp::printf("\n\n");
                for(int i = 0; i < 16; ++i)
                {
                    for(int j = 0; j < 16; ++j)
                    {
                        // Access the element - exact access method may depend on your specific implementation
                        float element = compute[i * 16 + j]; // For row-major layout

                        // Print with 2 decimal places, aligned
                        syclexp::printf("%6.2f ", element);
                    }
                    syclexp::printf("\n");
                }
                syclexp::printf("\nOUR AB[0] = %f -- A^2 = %f -- B^2 = %f --> SSD = %f",
                                compute[0],
                                r_norm[0],
                                l_norm[0],
                                r_norm[0] + l_norm[0] - 2 * compute[0]);

                syclexp::printf("\nMY norm = %f \n", my_norm);
            }
#endif

            // compute AB of 0, 0 in compute matrix as verifictaion

            // Should unroll all 8 iterations
#pragma unroll
            for(unsigned char i = 0; i < 16; i += 2)
            {
                const float other_norm = r_norm[outer + (i + second_row)];
                const unsigned char pos = (i << 4) + x; // 0 - 255 (16x16 tile positions)
#if P_OUT
                {
                    // syclexp::printf("\nOther norm = %f -- pos = %d \n", other_norm, pos);
                    syclexp::printf("\n POS %d i=%d sr=%d: %f + %f - 2 * %f  = %f \n",
                                    pos,
                                    i,
                                    second_row,
                                    other_norm,
                                    my_norm,
                                    compute[pos],
                                    other_norm + my_norm - (2 * compute[pos]));
                }
#endif

                compute[pos] = my_norm + other_norm - (2 * compute[pos]);
            }

            sycl::group_barrier(it.get_group()); // Think it needs to be group and not subgroup for mem consistency
            // Might not be needed as updating work-item is the one reading here so no communication between
            // wrok-items

#if false 
            if(x == 0 && outer == 0 && it.get_group(0) == 0)
            {
                syclexp::printf("\n\nCOMPUTED A + B - 2AB\n");
                for(int i = 0; i < 16; ++i)
                {
                    for(int j = 0; j < 16; ++j)
                    {
                        // Access the element - exact access method may depend on your specific implementation
                        float element = compute[i * 16 + j]; // For row-major layout

                        // Print with 2 decimal places, aligned
                        syclexp::printf("%6.2f ", element);
                    }
                    syclexp::printf("\n");
                }
            }
#endif

#if P_OUT
            {
                const sycl::vec<sycl::half, 4>* desc_ptr = reinterpret_cast<const sycl::vec<sycl::half, 4>*>(l_start);
                sycl::vec<sycl::half, 4> l_val = desc_ptr[it.get_local_id(0)];

                const sycl::vec<sycl::half, 4>* desc_ptr_r = reinterpret_cast<const sycl::vec<sycl::half, 4>*>(r);
                sycl::vec<sycl::half, 4> r_val = desc_ptr_r[it.get_local_id(0)];

                const sycl::vec<sycl::half, 4> mval = l_val - r_val;
                sycl::half res = sycl::dot(mval, mval);

                res = sycl::reduce_over_group(sg, res, sycl::plus<sycl::half>());

                if(x == 0)
                {
                    syclexp::printf("Verification val = %f -- our val = %f \n", static_cast<float>(res), compute[0]);
                    // syclexp::printf("Verification val = %f ", static_cast<float>(res));
                }
            }
#endif

#if P_OUT
            {
                // Compute A^2 and B^2 and A*B and then do SSD from that
                const sycl::vec<sycl::half, 4>* desc_ptr = reinterpret_cast<const sycl::vec<sycl::half, 4>*>(l_start);
                sycl::vec<sycl::half, 4> l_val = desc_ptr[it.get_local_id(0)];

                const sycl::vec<sycl::half, 4>* desc_ptr_r = reinterpret_cast<const sycl::vec<sycl::half, 4>*>(r);
                sycl::vec<sycl::half, 4> r_val = desc_ptr_r[it.get_local_id(0)];

                // COMpute AB

                sycl::half AB_val = sycl::dot(l_val, r_val);

                AB_val = sycl::reduce_over_group(sg, AB_val, sycl::plus<sycl::half>());

                const sycl::vec<sycl::half, 4> mval = l_val - r_val;
                sycl::half res = sycl::dot(mval, mval);

                res = sycl::reduce_over_group(sg, res, sycl::plus<sycl::half>());

                if(x == 0)
                {
                    syclexp::printf("Verification SSD = %f AB = %f -- our val = %f \n",
                                    static_cast<float>(res),
                                    static_cast<float>(AB_val),
                                    compute[0]);
                    // syclexp::printf("Verification val = %f ", static_cast<float>(res));
                }
            }

#endif

            // Compare the values in the column and find the best
            scan_state lead;

            // no bank conflicts
            lead.value.x() = compute[x];
            lead.value.y() = compute[x + 32];

            if(lead.value.y() < lead.value.x())
            {
                // Need to swap as x should have smallest
                float tmp = lead.value.x();
                lead.value.x() = lead.value.y();
                lead.value.y() = tmp;

                // Mistake made in long version here fogot to add second row and use 0 and 2 as base to get correct
                // row_idx: (fixed in this version)
                lead.idx.x() = 2 + second_row;
                lead.idx.y() = 0 + second_row;
            }
            else
            {
                // Can keep value as is
                lead.idx.x() = 0 + second_row;
                lead.idx.y() = 2 + second_row;
            }

#if false 
            if(it.get_group(0) == 0 && outer == 0)
            {
                syclexp::printf("PRE: x = %d, best(%d, %f), second(%d, %f)\n",
                                x,
                                lead.idx.x(),
                                lead.value.x(),
                                lead.idx.y(),
                                lead.value.y());
            }

#endif

#pragma unroll
            for(unsigned char i = 4; i < 16; i += 2) // 6 iterations
            {
                const float local_val = compute[(i << 4) + x];
                if(local_val < lead.value.x())
                {
                    // new leader delete second
                    lead.value.y() = lead.value.x();
                    lead.idx.y() = lead.idx.x();

                    // Set new leader
                    lead.value.x() = local_val;
                    lead.idx.x() = i + second_row;
                }
                else if(local_val < lead.value.y())
                {
                    // new second
                    lead.value.y() = local_val;
                    lead.idx.y() = i + second_row;
                }

                // if(it.get_group(0) == 0 && outer == 0)
                // {
                //     syclexp::printf("i = %d: x = %d, best(%d, %f), second(%d, %f) -- compute[%d] = %f\n",
                //                     i,
                //                     x,
                //                     lead.idx.x(),
                //                     lead.value.x(),
                //                     lead.idx.y(),
                //                     lead.value.y(),
                //                     (i << 4) + x,
                //                     local_val);
                // }
            }

            // Now we have the four best and we need to find the best of the 16 per column
            // as two work_items work on one column

            // 0-16 work on same column and so does 1-17 and so on

            // Need to move IDX to the global iteration space
            lead.idx += outer;

#if 0
            if(it.get_group(0) == 0 && outer < 16 * 1 + 1)
            {
                syclexp::printf("x = %d, best(%d, %f), second(%d, %f)\n",
                                x,
                                lead.idx.x(),
                                lead.value.x(),
                                lead.idx.y(),
                                lead.value.y());
            }

#endif

            // Compare best
            float other_val = sycl::permute_group_by_xor(sg, lead.value.x(), 16);
            int other_idx = sycl::permute_group_by_xor(sg, lead.idx.x(), 16);

            bool swap = second_row ? lead.value.x() < other_val : lead.value.x() > other_val;

            if(swap)
            {
                lead.value.x() = other_val;
                lead.idx.x() = other_idx;
            }

            // compare seconds
            other_val = sycl::permute_group_by_xor(sg, lead.value.y(), 16);
            other_idx = sycl::permute_group_by_xor(sg, lead.idx.y(), 16);

            swap = second_row ? lead.value.y() < other_val : lead.value.y() > other_val;

            if(swap)
            {
                lead.value.y() = other_val;
                lead.idx.y() = other_idx;
            }

            // compare middle
            other_val = sycl::permute_group_by_xor(sg, second_row ? lead.value.x() : lead.value.y(), 16);
            other_idx = sycl::permute_group_by_xor(sg, second_row ? lead.idx.x() : lead.idx.y(), 16);

            // Only care what happens to non second_row (id < 16) the others are discarded
            swap = other_val < lead.value.y();

            if(swap)
            {
                lead.value.y() = other_val;
                lead.idx.y() = other_idx;
            }

#if false
            if(it.get_group(0) == 0 && outer < 16 * 1 + 1 && x < 16) // only these have important vals
            {
                syclexp::printf("AFTER SORT x = %d, best(%d, %f), second(%d, %f)\n",
                                x,
                                lead.idx.x(),
                                lead.value.x(),
                                lead.idx.y(),
                                lead.value.y());
            }

#endif

            // Lower 16 has correct best two for the column
            // Compare ours to global state of the columns

            if(!second_row) // x < 16
            {
                // compare seconds -- store best in globals y
                if(lead.value.y() < global_leader.value.y())
                {
                    global_leader.value.y() = lead.value.y();
                    global_leader.idx.y() = lead.idx.y();

                    // lead .y() is discarded
                }

                // // compare best
                // if(lead.value.x() < global_leader.value.x())
                // {
                //     // uses lead .y as tmp as it's discarded (not sure if better than using a local tmp variable)
                //     lead.value.y() = global_leader.value.x();
                //     lead.idx.y() = global_leader.idx.x();
                //
                //     global_leader.value.x() = lead.value.x();
                //     global_leader.idx.x() = lead.idx.x();
                //
                //     // Store global from tmp value to local
                //     lead.value.x() = lead.value.y();
                //     lead.idx.x() = lead.idx.y();
                // }

                // compare best
                if(lead.value.x() < global_leader.value.x())
                {
#define USE_TMP 0
#if USE_TMP
                    // uses lead .y as tmp as it's discarded (not sure if better than using a local tmp variable)
                    float tmp_value = global_leader.value.x();
                    int tmp_idx = global_leader.idx.x();

#else
                    // uses lead .y as tmp as it's discarded (not sure if better than using a local tmp variable)
                    lead.value.y() = global_leader.value.x();
                    lead.idx.y() = global_leader.idx.x();

#endif

                    // Set local to new global leader
                    global_leader.value.x() = lead.value.x();
                    global_leader.idx.x() = lead.idx.x();

#if USE_TMP
                    // Store global (from tmp value) to local leader
                    lead.value.x() = tmp_value;
                    lead.idx.x() = tmp_idx;

#else
                    // Store global (from tmp value) to local leader
                    lead.value.x() = lead.value.y();
                    lead.idx.x() = lead.idx.y();

#endif
                    // swap complete
                }

                // compare middle // depends on the two above
                if(lead.value.x() < global_leader.value.y())
                {
                    global_leader.value.y() = lead.value.x();
                    global_leader.idx.y() = lead.idx.x();
                }
            }
            // Global updated with the two smallest values and  correpsonding idx

            // ####################################################################################
            // ################################ SEEMS TO BE CORRECT UNTIL THIS POINT ##############
            // ####################################################################################
#if false
            if(it.get_group(0) == 0 && outer < 16 * 1 + 1 && x < 16) // only these have important vals
            {
                syclexp::printf("Global: x = %d, best(%d, %f), second(%d, %f)\n",
                                x,
                                global_leader.idx.x(),
                                global_leader.value.x(),
                                global_leader.idx.y(),
                                global_leader.value.y());
            }

#endif
        }

        // Need to loop over the remainder and compare those to global_leader (be carefull to use correct work_item for
        // that)

        // Loop over the tail features of r to match with this set of 16 l features (not sure where the limit goes for
        // when it's better to do zero padding

        for(int outer = (r_len - 15); outer < r_len; outer++) // Remainder - done one by one
        {
            // Compute SSD and compare to global

            // Compute the (0-16) SSD's

            // r_ptr // Decorated

            // Compare for x < 16 if it's value is less than global for it's column and replace accordingly
        }

        // Then need to do the 80 percent of nearest neigtour and write back the match matrix
        if(!second_row) // x < 16
        {
            const bool accept = ((global_leader.value.x() / global_leader.value.y()) < 0.8f);
            match_matrix[(it.get_group(0) << 4) + x] =
              sycl::vec<int, 3>(global_leader.idx.x(), global_leader.idx.y(), accept);

            if(accept)
            {
                // ######################################################
                // Something wrong here as we have duplicates in printout
                // ######################################################

                syclexp::printf("idx = %d + %d --> match_matrix[%d] = (%d, %d) --> (%f, %f) -- MATRIX\n",
                                static_cast<int>(it.get_group(0)) << 4,
                                x,
                                static_cast<int>(it.get_group(0) << 4) + x,
                                global_leader.idx.x(),
                                global_leader.idx.y(),
                                global_leader.value.x(),
                                global_leader.value.y());
            }
        }
    }
};

// const sycl::vec<FeatureType, 4>* lptr = reinterpret_cast<const sycl::vec<FeatureType, 4>*>(&desc[it.get]);

// const sycl::vec<FeatureType, 4> lval = lptr[it.get_local_id(0)];

// const sycl::vec<FeatureType, 4> desc_val =
//   *reinterpret_cast<const sycl::vec<FeatureType, 4>*>(&desc[it.get_group(0) * 128 + it.get_local_id(0) * 4]);

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
    sycl::local_accessor<float, 1> a_norm;
    sycl::local_accessor<sycl::half, 1> a_staging_tile;
    sycl::local_accessor<float, 1> compute_tile;
    sycl::local_accessor<scan_state_old<int>, 1> global_leader;
    float* write_back;

  public:
    Compute_distance_matrix(sycl::vec<int, 3>* match_matrix,
                            sycl::half* l,
                            int l_len,
                            sycl::half* r,
                            int r_len,
                            sycl::local_accessor<sycl::half, 1> test,
                            sycl::local_accessor<float, 1> b_norm,
                            sycl::local_accessor<float, 1> a_norm,
                            sycl::local_accessor<sycl::half, 1> a_staging_tile,
                            sycl::local_accessor<float, 1> compute_tile,
                            sycl::local_accessor<scan_state_old<int>, 1> global_leader,
                            float* write_back) // tmp testing
      : match_matrix(match_matrix)
      , l(l)
      , l_len(l_len)
      , r(r)
      , r_len(r_len)
      , test(test)
      , b_norm(b_norm)
      , a_norm(a_norm)
      , a_staging_tile(a_staging_tile)
      , compute_tile(compute_tile)
      , global_leader(global_leader)
      , write_back(write_back) {};

    inline void operator()(sycl::nd_item<1> it) const
    {
        // Should move this to the command group like the example docs
        // sycl::global_ptr<sycl::half> l_ptr(l); // not in use ...
        sycl::global_ptr<float> backy(write_back);

        auto my_matrix = test.get_multi_ptr<sycl::access::decorated::no>();
        // sycl::global_ptr<sycl::half> r_ptr(r);

        // auto r_ptr =
        //   sycl::get_multi_ptr<sycl::half, sycl::access::address_space::global_space,
        //   sycl::access::decorated::yes>(r);

        auto r_ptr =
          sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::yes>(r);

        auto a_tile = a_staging_tile.get_multi_ptr<sycl::access::decorated::yes>();
        auto compute = compute_tile.get_multi_ptr<sycl::access::decorated::yes>();
        auto col_leaders = global_leader.get_multi_ptr<sycl::access::decorated::no>();

        const int x = it.get_local_id(0);
        const int desc_start = it.get_group(0); // only for global reads of B (our descriptors (constant for sub_group))

        sycl::sub_group group = it.get_sub_group();

        // Could load it all in but that would take wayy to much shared memory I think
        // Aka loading in 16x128. currently just loading 16x16 but needs 16 events for that...

        // Would like to get this dimension to 32 (or rather the other one but this load as that would be faster as
        // 32 wide I think) Want to keep the other 16 to have better occupancy
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

        // std::array<sycl::device_event, 16> events;

        // Both needs to be decorated pointers...
        // Needed to be loaded like this If I want to store in array (can't use vector) and default constructor is
        // deleted for device event

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

            // NOTE: Could try to transpose and load as row_major
            // This would require to use stride 17 and larger shared memory to avoid bank conflicts
            //
            // NOTE: Would probably be better to store with stride 16 it will be bankconflits but it would allow
            // loads to not be bank conflicty. As currently loads are most likely having a solid bank conflict when
            // loading as colum_major. If stored as transposed we could load as row_major and in that case it's best
            // to not have it padded which would allow for no bank conflicsts for loads and we are oding way more
            // loads than store so initial bank conflicts are worth it
        }

        // my_matrix[128] = 0.1;

        syclexp::matrix::joint_matrix<sycl::sub_group,
                                      sycl::half,
                                      syclexp::matrix::use::a,
                                      16,
                                      16,
                                      syclexp::matrix::layout::row_major>
          A; // Loaded in on the fly (colums is desc from other set(smaller one)) (this one is cheaper due to not
             // having to transpose)

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

        sycl::group_barrier(it.get_group());
        // set to zero shlud be a way to do this in two steps with pointers
        if(x < 16)
        {
            // but this is probably just as good
            col_leaders[x].v1 = 30000.0; // easy to beat
            col_leaders[x].v2 = 30000.0;
            // col_leaders[x].idx.x() = 0; // Should not really be required to initialize
            // col_leaders[x].idx.y() = 0;
        }

        // Should move around the async copies mby use double buffering
        // for(int outer = 0; outer < r_len; outer += 16)

        // for(int iter = 0; iter < r_len / 16; ++iter)
        // for(int outer = 0; outer + 16 <= r_len; outer += 16) // only runs for full 16's
        for(int outer = 0; outer < (r_len - 15); outer += 16) // should be same as above (only for full 16's)
        // for(int outer = 0; outer + 16 <= 16; outer += 16)
        {
            joint_matrix_fill(it.get_sub_group(), C, 0); // reset
            // Do first iteration out of loop as we are writing to A_squared in loop we add to it
            // Avoids the need to fil it with zeros

            syclexp::matrix::joint_matrix_load(it.get_sub_group(), B, my_matrix, 128); // first tile of our

            // Wait for A tile to be loaded
            // Not sure if I could only wait on the last one but don'T think that's guaranteed to work
            for(auto& e : events)
                e.wait();
            // Not sure if the barrier makes this wait call redundant?

            syclexp::matrix::joint_matrix_load(it.get_sub_group(), A, a_tile, 16); // first tile of new

            syclexp::matrix::joint_matrix_mad(it.get_sub_group(), C, A, B, C);

            // NOTE: Should stsart load here into another tile for a double buffer?

            // Done with the a_tile so we compute A^2 inplace for tile
            for(int i = 0; i < 8; ++i)
            {
                a_tile[i * 32 + x] *= a_tile[i * 32 + x];
            }

            sycl::group_barrier(it.get_group());

            int row = x / 8; //  0 1 2 3 -- spliting task

            // We do 256 wide reduction but within the rows that are 16 to get 16 values total one per row

            // take upper 8 from a row of 16 and add with lower 8 in row and store to lower 8 upper index read is 63
            a_tile[x + (row << 3)] += a_tile[x + ((row + 1) << 3)];
            a_tile[x + (row << 3) + 64] += a_tile[x + ((row + 1) << 3) + 64];
            a_tile[x + (row << 3) + 128] += a_tile[x + ((row + 1) << 3) + 128];
            a_tile[x + (row << 3) + 192] += a_tile[x + ((row + 1) << 3) + 192];
            // No overlapping regions of these first steps

            row = x / 4; // 0 1 2 3 4 5 6 7
            sycl::group_barrier(it.get_group());

            // The bank conflicts are getting worse for evely level... (sestructure if possible)
            a_tile[x + (row * 12)] += a_tile[x + (row * 12) + 4];
            a_tile[x + (row * 12) + 128] += a_tile[x + (row * 12) + 4 + 128];

            row = x / 2; // [0 - 15]
            sycl::group_barrier(it.get_group());

            // full width 256
            a_tile[x + (row * 14)] += a_tile[x + (row * 14) + 2];

            sycl::group_barrier(it.get_group());

            if(x < 16)
            {
                a_norm[x] = a_tile[x * 16] + a_tile[x * 16 + 1]; // extreme bank conflicts...
            }

            sycl::group_barrier(it.get_group());

            // Inner loop computing for 16 descriptors of r_ptr
            for(int i = 1; i < 8; ++i)
            {
                // Should prefetch next here and double buffer this
                // Look at simplifying and optimizing this math
                sycl::device_event events_inner[16] = {
                  it.get_group().async_work_group_copy(a_tile, (r_ptr + outer * 128) + 16 * i, 16),
                  it.get_group().async_work_group_copy(a_tile + 16, (r_ptr + outer * 128) + 16 * i + 128, 16),
                  it.get_group().async_work_group_copy(a_tile + 16 * 2, (r_ptr + outer * 128) + 16 * i + 128 * 2, 16),
                  it.get_group().async_work_group_copy(a_tile + 16 * 3, (r_ptr + outer * 128) + 16 * i + 128 * 3, 16),
                  it.get_group().async_work_group_copy(a_tile + 16 * 4, (r_ptr + outer * 128) + 16 * i + 128 * 4, 16),
                  it.get_group().async_work_group_copy(a_tile + 16 * 5, (r_ptr + outer * 128) + 16 * i + 128 * 5, 16),
                  it.get_group().async_work_group_copy(a_tile + 16 * 6, (r_ptr + outer * 128) + 16 * i + 128 * 6, 16),
                  it.get_group().async_work_group_copy(a_tile + 16 * 7, (r_ptr + outer * 128) + 16 * i + 128 * 7, 16),
                  it.get_group().async_work_group_copy(a_tile + 16 * 8, (r_ptr + outer * 128) + 16 * i + 128 * 8, 16),
                  it.get_group().async_work_group_copy(a_tile + 16 * 9, (r_ptr + outer * 128) + 16 * i + 128 * 9, 16),
                  it.get_group().async_work_group_copy(a_tile + 16 * 10, (r_ptr + outer * 128) + 16 * i + 128 * 10, 16),
                  it.get_group().async_work_group_copy(a_tile + 16 * 11, (r_ptr + outer * 128) + 16 * i + 128 * 11, 16),
                  it.get_group().async_work_group_copy(a_tile + 16 * 12, (r_ptr + outer * 128) + 16 * i + 128 * 12, 16),
                  it.get_group().async_work_group_copy(a_tile + 16 * 13, (r_ptr + outer * 128) + 16 * i + 128 * 13, 16),
                  it.get_group().async_work_group_copy(a_tile + 16 * 14, (r_ptr + outer * 128) + 16 * i + 128 * 14, 16),
                  it.get_group().async_work_group_copy(a_tile + 16 * 15, (r_ptr + outer * 128) + 16 * i + 128 * 15, 16)

                };

                // Compute the whole descriptor ab so 256 total in the end 16 x 16 descriptor pairs

                // Load from global should prefetch
                // syclexp::matrix::joint_matrix_load(it.get_sub_group(), A, r_ptr + i * 16, 128);

                // Load from shared_memory
                syclexp::matrix::joint_matrix_load(it.get_sub_group(), B, my_matrix + i * 16, 128);

                for(auto& e : events_inner)
                    e.wait();

                // syclexp::matrix::joint_matrix_load(it.get_sub_group(), A, r_ptr + i * 16, 128);
                syclexp::matrix::joint_matrix_load(it.get_sub_group(), A, a_tile, 16);

                syclexp::matrix::joint_matrix_mad(it.get_sub_group(), C, A, B, C);

                // Compute a^2 for tile

                for(int i = 0; i < 8; ++i)
                {
                    a_tile[i * 32 + x] *= a_tile[i * 32 + x];
                }

                // Reduction along rows to go from 16 values per row donw to one sum of them
                sycl::group_barrier(it.get_group());
                // sum rows and store += a_norm

                int row = x / 8; //  0 1 2 3 -- spliting task

                // We do 256 wide reduction but within the rows that are 16 to get 16 values total one per row

                // take upper 8 from a row of 16 and add with lower 8 in row and store to lower 8 upper index read
                // is 63
                a_tile[x + (row << 3)] += a_tile[x + ((row + 1) << 3)];
                a_tile[x + (row << 3) + 64] += a_tile[x + ((row + 1) << 3) + 64];
                a_tile[x + (row << 3) + 128] += a_tile[x + ((row + 1) << 3) + 128];
                a_tile[x + (row << 3) + 192] += a_tile[x + ((row + 1) << 3) + 192];
                // No overlapping regions of these first steps

                row = x / 4; // 0 1 2 3 4 5 6 7
                sycl::group_barrier(it.get_group());

                // The bank conflicts are getting worse for evely level... (sestructure if possible)
                a_tile[x + (row * 12)] += a_tile[x + (row * 12) + 4];
                a_tile[x + (row * 12) + 128] += a_tile[x + (row * 12) + 4 + 128];

                row = x / 2; // [0 - 15]
                sycl::group_barrier(it.get_group());

                // full width 256
                a_tile[x + (row * 14)] += a_tile[x + (row * 14) + 2];

                sycl::group_barrier(it.get_group());

                if(x < 16)
                {
                    // Add final sum for row to it's row result
                    a_norm[x] += a_tile[x * 16] + a_tile[x * 16 + 1]; // extreme bank conflicts...
                }
            }

            sycl::group_barrier(it.get_group()); // not sure if needed

            // Compute the actual SSD

            // Need to store to shred mem
            // syclexp::matrix::joint_matrix_store(it.get_sub_group(), C, compute, 17,
            // syclexp::matrix::layout::row_major);

            // remove padded row
            syclexp::matrix::joint_matrix_store(it.get_sub_group(), C, compute, 16, syclexp::matrix::layout::row_major);

            sycl::group_barrier(it.get_group()); // not sure if required

            int pos = x / 16; // split first 16 -> 0 final 16 -> 1
            for(int i = 0; i < 8; ++i)
            {
                // a_norm is along the rows -- b_norm is along columns (no bank conflicts due to pad-column removal)
                // compute[x + i * 32] = a_norm[pos + i * 2] + b_norm[x % 16] - 2 compute[x + i * 32];
                compute[x + (i << 5)] = a_norm[pos + (i << 1)] + b_norm[x % 16] - 2 * compute[x + (i << 5)];
            }
            // Start fetching the data for a_tile here (should be enough time as we have some steps left :D)

            if(outer + 16 <= r_len) // next is valid we prefetch
            {
                // prefetch next tile
                events[0] = it.get_group().async_work_group_copy(a_tile, (r_ptr + (outer + 16) * 128), 16);
                events[1] = it.get_group().async_work_group_copy(a_tile + 16, (r_ptr + (outer + 16) * 128) + 128, 16);
                events[2] =
                  it.get_group().async_work_group_copy(a_tile + 16 * 2, (r_ptr + (outer + 16) * 128) + 128 * 2, 16);
                events[3] =
                  it.get_group().async_work_group_copy(a_tile + 16 * 3, (r_ptr + (outer + 16) * 128) + 128 * 3, 16);
                events[4] =
                  it.get_group().async_work_group_copy(a_tile + 16 * 4, (r_ptr + (outer + 16) * 128) + 128 * 4, 16);
                events[5] =
                  it.get_group().async_work_group_copy(a_tile + 16 * 5, (r_ptr + (outer + 16) * 128) + 128 * 5, 16);
                events[6] =
                  it.get_group().async_work_group_copy(a_tile + 16 * 6, (r_ptr + (outer + 16) * 128) + 128 * 6, 16);
                events[7] =
                  it.get_group().async_work_group_copy(a_tile + 16 * 7, (r_ptr + (outer + 16) * 128) + 128 * 7, 16);
                events[8] =
                  it.get_group().async_work_group_copy(a_tile + 16 * 8, (r_ptr + (outer + 16) * 128) + 128 * 8, 16);
                events[9] =
                  it.get_group().async_work_group_copy(a_tile + 16 * 9, (r_ptr + (outer + 16) * 128) + 128 * 9, 16);
                events[10] =
                  it.get_group().async_work_group_copy(a_tile + 16 * 10, (r_ptr + (outer + 16) * 128) + 128 * 10, 16);
                events[11] =
                  it.get_group().async_work_group_copy(a_tile + 16 * 11, (r_ptr + (outer + 16) * 128) + 128 * 11, 16);
                events[12] =
                  it.get_group().async_work_group_copy(a_tile + 16 * 12, (r_ptr + (outer + 16) * 128) + 128 * 12, 16);
                events[13] =
                  it.get_group().async_work_group_copy(a_tile + 16 * 13, (r_ptr + (outer + 16) * 128) + 128 * 13, 16);
                events[14] =
                  it.get_group().async_work_group_copy(a_tile + 16 * 14, (r_ptr + (outer + 16) * 128) + 128 * 14, 16);
                events[15] =
                  it.get_group().async_work_group_copy(a_tile + 16 * 15, (r_ptr + (outer + 16) * 128) + 128 * 15, 16);
            }

            // Now we have SSD computed in a 16x16 matrix we need to find the best two along the rows

            // Scan for the best two elements in the 16 rows

            // scan_state<unsigned char> lead;
            // Need int for global swaping after all...
            scan_state_old<int> lead;

            // scan_state<unsigned char> lead;
            // #else
            // COLUMN SCAN Correct version (no bank conflicts witout padding)
            // also simpler code than row scan hence better

            sycl::group_barrier(it.get_group());
            // NOTE: Bitinic sort might also be better than doing this scan not sure
            // but I think this is lesss operations. Also way less comunication between work-items

            // Doing two 16 at a time  0 - 15 (even rows) 16-31 odd rows
            lead.v1 = compute[x];
            lead.v2 = compute[x + 32]; // Next rows

            // Might be bette to compare to global lead and only store if better?
            // Keep default value to very large val.
            // That would alow for not comparing with global later if values are -1
            // But logic would be more complext needing more checks so might be slower
            if(lead.v2 < lead.v1)
            {
                // swap (smalest in v1)
                float tmp = lead.v1;
                lead.v1 = lead.v2;
                lead.v2 = tmp;

                // Need to store row_idx and not linear idx
                lead.idx.x() = 1;
                lead.idx.y() = 0;
            }
            else
            {
                // store row_idx
                lead.idx.x() = 0;
                lead.idx.y() = 1;
            }

            // syclexp::printf(
            //   "PRE LOOP: lead state vec %d - (%f, %d) (%f, %d)\n", x, lead.v1, lead.idx.x(), lead.v2,
            //   lead.idx.y());

            // loop over remainig 6 (double rows) and compare to lead

            int upper_split = x / 16; // 1 for x >= 16 and 0 for x < 16
            // for(unsigned char i = 2; i < 8; ++i)
            for(unsigned char i = 2; i < 15; i += 2) // two rows per iteration is done
            {
                // unsigned char cur_idx = x + (i << 5); // x + i * 32 (max idx is 255 perfect byte :D)
                float cur = compute[x + (i << 5)];
                int cur_row_idx = (upper_split) ? i + 1 : i;

                // if(outer < 16)
                // {
                //     syclexp::printf("i = %d - cur_row_idx = %d ---- x = %d\n", i, cur_row_idx, x);
                // }

                if(cur < lead.v1)
                {
                    // Move down leader to second
                    lead.v2 = lead.v1;
                    lead.idx.y() = lead.idx.x();
                    // place cur as new leader
                    lead.v1 = cur;
                    lead.idx.x() = cur_row_idx;
                }
                else if(cur < lead.v2)
                {
                    // place as second
                    lead.v2 = cur;
                    lead.idx.y() = cur_row_idx;
                }
            }

            // Now we have the two best of the 8 need to combine the columns(16)
            // We need to track both value and idx

            // Print all lead before merge
            // syclexp::printf("lead state vec %d - (%f, %d) (%f, %d)\n", x, lead.v1, lead.idx.x(), lead.v2,
            // lead.idx.y());

            // Swap with same column so 0-16 1-17 2-18 ... 15-31 (xor 16)

            const bool id_more = x > 15;

            float other = sycl::permute_group_by_xor(group, lead.v1, 16);                  // compare firsts
            unsigned char other_idx = sycl::permute_group_by_xor(group, lead.idx.x(), 16); // compare firsts

            // compare smallest (store in smallest idx the smallest value)
            bool swap = id_more ? (lead.v1 < other) : (lead.v1 > other); // First vs first
            if(swap)
            {
                lead.v1 = other;
                lead.idx.x() = other_idx;
            }

            other = sycl::permute_group_by_xor(group, lead.v2, 16);
            other_idx = sycl::permute_group_by_xor(group, lead.idx.y(), 16);

            // second vs second (store best(smallest) in smallest idx)
            bool swap_second = id_more ? (lead.v2 < other) : (lead.v2 > other);
            if(swap)
            {
                lead.v2 = other;
                lead.idx.y() = other_idx;
            }

            // Compare the middle values of the four -- share first for higher and second for lower
            other = sycl::permute_group_by_xor(group, id_more ? lead.v1 : lead.v2, 16);
            other_idx = sycl::permute_group_by_xor(group, id_more ? lead.idx.x() : lead.idx.y(), 16);

            // Don't care what happens to the ID more case here as that value is now don't care anyways
            // swap = id_more ? (lead.v2 < other) : (lead.v2 > other);
            bool swap_middle = lead.v2 > other; // if true lower is worse(larger) and need to swap
            if(swap_middle)
            { // only care what even ID's do odd don't care as the values in it's lead is not used anymore
                // only care what x < 16 do as they store the smallest 16-31 don't matter they store the larger
                // values
                lead.v2 = other;
                lead.idx.y() = other_idx;
            }

            // #####################################
            // COMPARE TO GLOBAL STATE
            // #####################################

            // if(!id_more) // x < 16
            if(x < 16) // x < 16
            {
                // should be element wise increment
                lead.idx += outer;

                // Compare seconds store smallest in v2 as that is register and we don't need shared mem consistency
                // for that (so opposite direction of what we want for register usage)
                if(col_leaders[x].v2 < lead.v2)
                {
                    // lead.v2 can't be top 2 (hence written over)
                    lead.v2 = col_leaders[x].v2;
                    lead.idx.y() = col_leaders[x].idx.y();

                    // Storing the best in the lead which is register so we don't need to do a barrier after a write
                    // to local memory (which should not really be required anyways (due to only one work_item
                    // modifiying and reading a value, but register more speed I guess)
                }

                // Winner of seconds stored in lead.v2

                // Compare best
                if(col_leaders[x].v1 > lead.v1)
                {
                    // Swap
                    float tmp_v1 = col_leaders[x].v1;
                    int tmp_idx = col_leaders[x].idx.x();
                    col_leaders[x].v1 = lead.v1;
                    col_leaders[x].idx.x() = lead.idx.x();
                    lead.v1 = tmp_v1;
                    lead.idx.x() = tmp_idx;
                }
                // Winner of the firsts/best are stored directly to shared memory as it's position is safe and won't
                // be read before next loop iterations

                // Compare middle
                // seconds winner in v2 and firsts loser in v1 of lead (registers so no need for shared memory sync)

                // could also be done as turnary operator
                if(lead.v1 < lead.v2)
                {
                    col_leaders[x].v2 = lead.v1;
                    col_leaders[x].idx.y() = lead.idx.x();
                }
                else
                {
                    col_leaders[x].v2 = lead.v2;
                    col_leaders[x].idx.y() = lead.idx.y();
                }
            }
            // Global is now up to date

            // #endif
        }

        // Might do this
        // Load linearly into a tile (fits the whole descriptor as it's 256 elemens wide (desc is 128))
        // sycl::device_event desc_load_evt =
        //     (r_len % 16 > 1) ? it.get_group().async_work_group_copy(a_tile, r_ptr + remainder_idx + 1, 128) :
        //     events[0];
        int remainder_base = r_len - (r_len % 16);

        // might need a barrier here

        // #####################################
        // DO THE REMAINDER
        // #####################################

        // for(int i = 0; i < r_len % 16; ++i)
        // {
        //     // Basically same as sthe simpler matching kernel
        //
        //     // Could reinterpret cast but don't think it's different could be wrong...
        //     // might be able to load two in one go for sycl::half?
        //     // const sycl::vec<float, 4>* lptr = reinterpret_cast<const sycl::vec<float, 4>*>(&l[idx]);
        //
        //     sycl::vec<sycl::half, 4> remainder{r_ptr[(remainder_base + i) * 128],
        //                                        r_ptr[(remainder_base + i) * 128 + 32],
        //                                        r_ptr[(remainder_base + i) * 128 + 64],
        //                                        r_ptr[(remainder_base + i) * 128 + 96]};
        //
        //     // Might need this barrier due to share memeory but only one work_item reads and writes to one
        //     location
        //     // No inbetween so should be fine without?
        //     sycl::group_barrier(it.get_group());
        //     sycl::half res[16];
        //     for(int j = 0; j < 16; ++j) // need to loop over all 16 of my_matrix
        //     {
        //         // 16 of my matrix -- NOTE: If my_matix is stored transposed this need's to be changed
        //         sycl::vec<sycl::half, 4> my_vals{my_matrix[x], my_matrix[x + 32], my_matrix[x + 64], my_matrix[x
        //         + 96]};
        //
        //         const sycl::vec<sycl::half, 4> mval = remainder - my_vals;
        //         res[j] = sycl::dot(mval, mval);
        //
        //         // Sum of squared differences of complete 128 descriptors
        //         res[j] = sycl::reduce_over_group(group, res[j], sycl::plus<sycl::half>());
        //     }
        //     // compare to the result to their global leader
        //     if(x < 16)
        //     {
        //         if(col_leaders[x].v1 > res[x])
        //         {
        //             // move leader to second
        //             col_leaders[x].v2 = col_leaders[x].v1;
        //             col_leaders[x].idx.y() = col_leaders[x].idx.x();
        //             // Take first
        //             col_leaders[x].v1 = res[x];
        //             col_leaders[x].idx.x() = remainder_base + i;
        //         }
        //         else if(col_leaders[x].v2 > res[x])
        //         {
        //             // take first second
        //             col_leaders[x].v2 = res[x];
        //             col_leaders[x].idx.y() = remainder_base + i;
        //         }
        //     }
        // }

        // Write back the results to their respective locations 16 results in total

        // Again not sure if needed as work_item that wrote to location is the one reading so no communication
        // between threads here
        sycl::group_barrier(it.get_group());
        if(x < 16)
        {
            bool accept = ((col_leaders[x].v1 / col_leaders[x].v2) < 0.8f);
            match_matrix[desc_start + x] = sycl::vec<int, 3>(col_leaders[x].idx.x(), col_leaders[x].idx.y(), accept);
            // syclexp::printf("\n(%d, %d, %d) ", col_leaders[x].idx.x(), col_leaders[x].idx.y(), accept);
        }

        syclexp::matrix::joint_matrix_store(it.get_sub_group(), C, backy, 16, syclexp::matrix::layout::row_major);
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
        matchEvent =
          _device_queue.parallel_for<compute_distance_sub_group>(sycl::nd_range{global, local},
                                                                 Compute_distance<true>(match_matrix,
                                                                                        getDescriptors(),
                                                                                        l_len,
                                                                                        other->getDescriptors(),
                                                                                        r_len,
                                                                                        getSquaredNorms(),
                                                                                        other->getSquaredNorms()));
    }
    else
    {
        matchEvent = _device_queue.parallel_for(sycl::nd_range{global, local},
                                                Compute_distance<false>(match_matrix,
                                                                        getDescriptors(),
                                                                        l_len,
                                                                        other->getDescriptors(),
                                                                        r_len,
                                                                        getSquaredNorms(),
                                                                        other->getSquaredNorms()));
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

void FeaturesDev::compute_squared_norms()
{
    // const int l_len = getDescriptorCount();

    sycl::range global{static_cast<size_t>(getDescriptorCount() * 32)};
    sycl::range local{32};

    _norms_computed_event = _device_queue.parallel_for(sycl::nd_range{global, local},
                                                       Compute_squared_norm(getDescriptors(), getSquaredNorms()));

    // Compute_distance(match_matrix, getDescriptors(), l_len, other->getDescriptors(), r_len));
}

std::tuple<sycl::vec<int, 3>*, std::function<void()>, std::function<void()>> FeaturesDev::preNormMatrixMatchAndReturn(
  FeaturesDev* other)
{
    int l_len = getDescriptorCount();
    int r_len = other->getDescriptorCount();

    // should swap around so that we use l as the longest for better occupancy

    sycl::vec<int, 3>* match_matrix =
      popsift::sycl_common::malloc_sharedT<sycl::vec<int, 3>>(l_len, __FILE__, __LINE__, "", _device_queue);

    sycl::event matchEvent;

    // sycl::range global{static_cast<size_t>(((l_len) - (l_len % 16)) * 32)}; // WRONG
    sycl::range global{static_cast<size_t>((l_len >> 4) * 32)}; // floor division by 16 then multiply by 32
    // Need to deal with remainder in normal loop (different kernel)

    // sycl::range global{32}; // just one for testing
    sycl::range local{32};

    matchEvent = _device_queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on({getNormsEvent(), other->getNormsEvent()});

        auto compute_tile = sycl::local_accessor<float, 1>(16 * 16, cgh);
        cgh.parallel_for(sycl::nd_range{global, local},
                         Compute_distance_matrix_pre_norm(match_matrix,
                                                          reinterpret_cast<sycl::half*>(getDescriptors()),
                                                          getSquaredNorms(),
                                                          reinterpret_cast<sycl::half*>(other->getDescriptors()),
                                                          other->getSquaredNorms(),
                                                          r_len,
                                                          compute_tile));
    });

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
        float* res = sycl_common::malloc_sharedT<float>(600, __FILE__, __LINE__, "", _device_queue); // delete

        convert_float_to_half_usm(_device_queue, getDescriptors(), l_half, l_len);
        convert_float_to_half_usm(_device_queue, other->getDescriptors(), r_half, r_len);

        // if(l_len < r_len) // We want most of the descriptrs on left side
        if(true) // We want most of the descriptrs on left side
        {
            fprintf(stderr, "Trying to do matrix stuff yayayayayay\n");
            sycl::range global{static_cast<size_t>(l_len * 32)}; // one 32 wide group per descriptor
            // sycl::range global{static_cast<size_t>((l_len - (l_len % 16)) * 32)}; // only full 16's
            // sycl::range global{32}; // just one for test
            sycl::range local{32};
            // SWAP

            _device_queue.submit([&](sycl::handler& cgh) {
                // need 7 for storing the older result values final is stored in current work range idx 7
                auto b_matrix = sycl::local_accessor<sycl::half, 1>(128 * 16, cgh);
                auto b_norm = sycl::local_accessor<float, 1>(16, cgh);
                auto a_norm = sycl::local_accessor<float, 1>(16, cgh);
                // auto local_a_square = sycl::local_accessor<sycl::half, 1>(16 * 16, cgh);
                auto a_staging_tile = sycl::local_accessor<sycl::half, 1>(16 * 16, cgh);

                // Added one column for stagger avoiding bank conflits for row reduce
                auto compute_tile =
                  sycl::local_accessor<float, 1>(16 * 17, cgh); // larger due to old padding (no longer used)
                auto global_leader = sycl::local_accessor<scan_state_old<int>, 1>(16, cgh);

                cgh.parallel_for(sycl::nd_range{global, local},
                                 Compute_distance_matrix<false>(match_matrix,
                                                                l_half,
                                                                l_len,
                                                                r_half,
                                                                r_len,
                                                                b_matrix,
                                                                b_norm,
                                                                a_norm,
                                                                a_staging_tile,
                                                                compute_tile,
                                                                global_leader,
                                                                res));
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

            fprintf(stderr, "\n DONE PRINT ( l_len = %d AND r_len = %d\n\n", l_len, r_len);

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
            matchEvent =
              _device_queue.parallel_for<compute_distance_fallback>(sycl::nd_range{global, local},
                                                                    Compute_distance<true>(match_matrix,
                                                                                           getDescriptors(),
                                                                                           l_len,
                                                                                           other->getDescriptors(),
                                                                                           r_len,
                                                                                           getSquaredNorms(),
                                                                                           other->getSquaredNorms()));
        }
        else
        {
            //  Fallback work group
            matchEvent = _device_queue.parallel_for(sycl::nd_range{global, local},
                                                    Compute_distance<false>(match_matrix,
                                                                            getDescriptors(),
                                                                            l_len,
                                                                            other->getDescriptors(),
                                                                            r_len,
                                                                            getSquaredNorms(),
                                                                            other->getSquaredNorms()));
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

#define row_scan 0
#if row_scan
  // NEEd to have a padded column to avoid bank conflicts

// int scan_start = x*8 + x/2;
unsigned char scan_start = (x << 3) + (x >> 1);

// v1 smallest v2 second smallest)
lead.v1 = compute[scan_start];
lead.v2 = compute[scan_start + 1];

// Storing 0 - 16 idx as we want the idx of the best matching descriptor for the whole
// Final store of idx need to be offset by outer most loop * 16
// unsigned char idx_base = ((x % 2) << 3); // x % 2 * 8
unsigned char idx_base = ((x & 1) << 3); // x % 2 * 8 (Should be the same as mod 2)
if(lead.v2 < lead.v1)
{
    // swap
    float tmp = lead.v1;
    lead.v1 = lead.v2;
    lead.v2 = tmp;

    lead.idx.x() = idx_base + 1;
    lead.idx.y() = idx_base;
}
else
{
    // store idx
    lead.idx.x() = idx_base;
    lead.idx.y() = idx_base + 1;
}

// Now loop over remaining 6 values

for(unsigned char i = 2; i < 8; ++i)
{
    float cur = compute[scan_start + i];
    if(cur < lead.v1)
    {
        // Move down leader to second
        lead.v2 = lead.v1;
        lead.idx.y() = lead.idx.x();
        // place cur as new leader
        lead.v1 = cur;
        lead.idx.x() = idx_base + i;
    }
    else if(cur < lead.v2)
    {
        lead.v2 = cur;
        lead.idx.y() = idx_base + i;
    }
}
// Now we have the best two from the 8 values that we scaned over
// Need to sort the 4 that belongs to one row
// -> then compare to the global state when outer loop exists

// Swap with neigour
// const bool id_more = !(_it.get_local_id(1) & 1 == 0);
const bool id_more = x & 1; // even is false odd is true same as x % 2

float other = sycl::permute_group_by_xor(group, lead.v1, 1); // compare smallest

// Smallest to smallest idx
bool swap = id_more ? (lead.v1 < other) : (lead.v1 > other);
if(swap)
{
    lead.v1 = other;
}

// Compare the two second samllest
other = sycl::permute_group_by_xor(group, lead.v2, 1); // compare smallest

// second vs second
swap = id_more ? (lead.v2 < other) : (lead.v2 > other);
if(swap)
{
    lead.v1 = other;
}

// Compare the middle values of the four
other = sycl::permute_group_by_xor(group, id_more ? lead.v1 : lead.v2, 1); // middle

// Don't care what happens to the ID more case here as that value is now don't care anyways
// swap = id_more ? (lead.v2 < other) : (lead.v2 > other);
if(lead.v2 < other)
{ // only care what even ID's do odd don't care as the values in it's lead is not used anymore
    lead.v2 = other;
}

// Now the even work_items have their respective rows two best stored in it's leader struct
// This now needs to be compared to global leader and if updated need to store it's index in the global
// context
}

sycl::group_barrier(it.get_group()); // not sure if required
#endif
