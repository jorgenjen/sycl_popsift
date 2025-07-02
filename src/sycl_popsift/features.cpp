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
#include <memory>
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
                            sycl::nd_item<1>& it)
{
    const sycl::vec<FeatureType, 4> lval = lptr[it.get_local_id(0)];
    const sycl::vec<FeatureType, 4> rval = rptr[it.get_local_id(0)];

#if 0
    // Verbose write out of SSD
    const sycl::vec<float, 4> mval =
      sycl::vec<float, 4>(lval.x() - rval.x(), lval.y() - rval.y(), lval.z() - rval.z(), lval.w() - rval.w());

    float res = mval.x() * mval.x() + mval.y() * mval.y() + mval.z() * mval.z() + mval.w() * mval.w();
#else
#define COMPUTE_MATRIX_LIKE 0

#if COMPUTE_MATRIX_LIKE
    // JUST FOR VERIFICATION OF USING THE NORMS

#if false
    FeatureType res = sycl::dot(lval, rval);                              // Compute AB
    res = sycl::reduce_over_group(group, res, sycl::plus<FeatureType>()); // Compute FINAL AB value

    return static_cast<FeatureType>(l_norm) + static_cast<FeatureType>(r_norm) - 2 * res; // A^2 + B^2 - 2 * AB

#else
    // Closer to tensor

    float res = static_cast<float>(sycl::dot(lval, rval));          // Compute AB
    res = sycl::reduce_over_group(group, res, sycl::plus<float>()); // Compute FINAL AB value

    return static_cast<FeatureType>(l_norm + r_norm - 2 * res); // A^2 + B^2 - 2 * AB

#endif

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

  public:
    Compute_distance(sycl::vec<int, 3>* match_matrix, Descriptor* l, int l_len, Descriptor* r, int r_len)
      // Compute_distance(
      //   sycl::vec<int, 3>* match_matrix, Descriptor* l, int l_len, Descriptor* r, int r_len, float* l_norm, float*
      //   r_norm)
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

            // if(accept)
            // {
            //     syclexp::printf("match_matrix[%d] = (%d, %d) --> (%f, %f)\n",
            //                     idx,
            //                     match_1st_idx,
            //                     match_2nd_idx,
            //                     match_1st_val,
            //                     match_2nd_val);
            // }
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

struct scan_state
{
    sycl::vec<float, 2> value;
    sycl::vec<int, 2> idx;
};

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

            // compute AB of 0, 0 in compute matrix as verifictaion

            // Should unroll all 8 iterations
#pragma unroll
            for(unsigned char i = 0; i < 16; i += 2)
            {
                const float other_norm = r_norm[outer + (i + second_row)];
                const unsigned char pos = (i << 4) + x; // 0 - 255 (16x16 tile positions)

                compute[pos] = my_norm + other_norm - (2 * compute[pos]);
            }

            sycl::group_barrier(it.get_group()); // Think it needs to be group and not subgroup for mem consistency
            // Might not be needed as updating work-item is the one reading here so no communication between
            // wrok-items

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
            }

            // Now we have the four best and we need to find the best of the 16 per column
            // as two work_items work on one column

            // 0-16 work on same column and so does 1-17 and so on

            // Need to move IDX to the global iteration space
            lead.idx += outer;

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
        }

        // Loop over the tail features of r to match with this set of 16 l features (not sure where the limit goes for
        // when it's better to do zero padding now just doing one by one for all)

        const sycl::vec<sycl::half, 4>* lptr = reinterpret_cast<const sycl::vec<sycl::half, 4>*>(l_start);
        const sycl::vec<sycl::half, 4>* rptr = reinterpret_cast<const sycl::vec<sycl::half, 4>*>(r);

        for(int outer = r_len - (r_len % 16); outer < r_len; outer++) // Remainder - done one by one
        {
            // Compute SSD and compare to global
            float local_val = std::numeric_limits<float>::infinity(); // So it will lose to global if not replaced

            // Compute the (0-16) SSD's
            const sycl::vec<sycl::half, 4> rval = rptr[(outer << 5) + it.get_local_id(0)];
#pragma unroll
            for(int i = 0; i < 16; ++i) // loop over the 16 "our vectors / left"
            {
                // Compute AB

                // each index is now 4 values so desc is 32 wide
                const sycl::vec<sycl::half, 4> lval = lptr[(i << 5) + it.get_local_id(0)]; // load a quad per work-item
                float res = static_cast<float>(sycl::dot(lval, rval));
                res = sycl::reduce_over_group(sg, res, sycl::plus<float>());
                if(x == i)
                {
                    local_val = res; // Store for compare with coresponding global_leader
                }
            }

            if(x < 16) // could do !second_row but would keep in in scope for longer
            {
                // Compute A^2 + B^2 - 2 * AB
                local_val = my_norm + r_norm[outer] - 2 * local_val;

                // Compare the 16 values with their respective global_leader
                if(local_val < global_leader.value.x())
                {
                    // New leader push first to second
                    global_leader.value.y() = global_leader.value.x();
                    global_leader.idx.y() = global_leader.idx.x();

                    // Place new leader
                    global_leader.value.x() = local_val;
                    global_leader.idx.x() = outer;
                }
                else if(local_val < global_leader.value.y())
                {
                    // New second
                    global_leader.value.y() = local_val;
                    global_leader.idx.y() = outer;
                }
            }
        }

        // Then need to do the 80 percent of nearest neigtour and write back the match matrix
        if(!second_row) // x < 16
        {
            const bool accept = ((global_leader.value.x() / global_leader.value.y()) < 0.8f);
            match_matrix[(it.get_group(0) << 4) + x] =
              sycl::vec<int, 3>(global_leader.idx.x(), global_leader.idx.y(), accept);
        }
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

#if USE_JOINT_MATRIX
void FeaturesDev::compute_squared_norms()
{
    // const int l_len = getDescriptorCount();

    sycl::range global{static_cast<size_t>(getDescriptorCount() * 32)};
    sycl::range local{32};

    _norms_computed_event = _device_queue.parallel_for(sycl::nd_range{global, local},
                                                       Compute_squared_norm(getDescriptors(), getSquaredNorms()));

    // Compute_distance(match_matrix, getDescriptors(), l_len, other->getDescriptors(), r_len));
}
#endif

std::tuple<sycl::vec<int, 3>*, std::function<void()>, std::function<void()>> FeaturesDev::preNormMatrixMatchAndReturn(
  FeaturesDev* other)
{
#if !USE_JOINT_MATRIX
    return matchAndReturn(other);
#else
    if(!PopSift::matrixSupported)
    {
        // Could have gotten here before we have found supported matrix layout or matrix layout not supported
        // This falg is set in multithreaded setting and has potentiall race conflict but worse case it just uses backup
        // instead of tensor. But should in most cases have determined that before getting to this point

        fprintf(stderr,
                "Using backup for Matching (NOT JOINT MATRIX) could be due to dimensions not being suported or race "
                "condition of setting the flag\n");
        return matchAndReturn(other);
    }
    int l_len = getDescriptorCount();
    int r_len = other->getDescriptorCount();

    printf("l_len = %d -- r_len = %d\n", l_len, r_len);

    sycl::vec<int, 3>* match_matrix =
      popsift::sycl_common::malloc_sharedT<sycl::vec<int, 3>>(l_len, __FILE__, __LINE__, "", _device_queue);

    // sycl::event matchEvent;

    // sycl::range global{static_cast<size_t>(((l_len) - (l_len % 16)) * 32)}; // WRONG
    sycl::range global{static_cast<size_t>((l_len >> 4) * 32)}; // floor division by 16 then multiply by 32
    // Need to deal with remainder in normal loop (different kernel)

    // sycl::range global{32}; // just one for testing
    sycl::range local{32};

    sycl::event matchEvent = _device_queue.submit([&](sycl::handler& cgh) {
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

    sycl::range remainderGlobal{static_cast<size_t>((l_len % 16) * 32)};

// FROM INITIAL TEST NOT USING NORM IS FASTER (not huge (1.894ms vs 1.839))

// DATA COLLECTION TO PERFORM:
// SHOULD HAVE AN EXPERIMENT WHERE WE COMPARE USING PRECOMPUTED NORMS VS NOT FOR PERFORMANCE BOTH INCLUDING THE NORM
// COMPUTE COST AND WITHOUT Could have all three in a plot and from that justify design decision
#define USE_NORMS 0
#if USE_NORMS
    sycl::event remainderMatchEvent = _device_queue.parallel_for(
      sycl::nd_range{remainderGlobal, local},
      [=,
       l = reinterpret_cast<const sycl::vec<sycl::half, 4>*>(getDescriptors() + (l_len - (l_len % 16))),
       l_norm = getSquaredNorms(),
       r = reinterpret_cast<const sycl::vec<sycl::half, 4>*>(other->getDescriptors()),
       r_norm = other->getSquaredNorms(),
       l_base = (l_len - (l_len % 16))](sycl::nd_item<1> it) {
          scan_state leader;
          leader.value = std::numeric_limits<float>::infinity(); // set to max so the are replaced. Could do same
                                                                 //  as in matrix version having it out of the loop
          float my_norm = l_norm[l_base + it.get_group(0)];
          const sycl::vec<sycl::half, 4> lval = l[(it.get_group(0) << 5) + it.get_local_id(0)];
          for(int i = 0; i < r_len; ++i) // looping over r
          {
              // each index is now 4 values so desc is 32 wide
              const sycl::vec<sycl::half, 4> rval = r[(i << 5) + it.get_local_id(0)];

              float res = static_cast<float>(sycl::dot(lval, rval)); // similar in precision to tensor (not fully)
              res = sycl::reduce_over_group(it.get_sub_group(), res, sycl::plus<float>());
              res = my_norm + r_norm[i] - 2 * res; // A^2 + B^2 - 2 * AB

              // Compare res to leader
              if(res < leader.value.x())
              {
                  // New leader
                  leader.value.y() = leader.value.x();
                  leader.idx.y() = leader.idx.x();

                  leader.value.x() = res;
                  leader.idx.x() = i;
              }
              else if(res < leader.value.y())
              {
                  // New second
                  leader.value.y() = res;
                  leader.idx.y() = i;
              }
          }

          const bool accept = ((leader.value.x() / leader.value.y()) < 0.8f);
          match_matrix[l_base + it.get_group(0)] = sycl::vec<int, 3>(leader.idx.x(), leader.idx.y(), accept);
      });

#else

    sycl::event remainderMatchEvent = _device_queue.parallel_for(
      sycl::nd_range{remainderGlobal, local},
      {getNormsEvent(), other->getNormsEvent()},
      [=,
       l = reinterpret_cast<const sycl::vec<sycl::half, 4>*>(getDescriptors() + (l_len - (l_len % 16))),
       r = reinterpret_cast<const sycl::vec<sycl::half, 4>*>(other->getDescriptors()),
       l_base = (l_len - (l_len % 16))](sycl::nd_item<1> it) {
          scan_state leader;
          leader.value = std::numeric_limits<float>::infinity(); // set to max so the are replaced. Could do same
                                                                 //  as in matrix version having it out of the loop
          const sycl::vec<sycl::half, 4> lval = l[(it.get_group(0) << 5) + it.get_local_id(0)];
          for(int i = 0; i < r_len; ++i) // looping over r
          {
              // each index is now 4 values so desc is 32 wide
              const sycl::vec<sycl::half, 4> rval = r[(i << 5) + it.get_local_id(0)];
              const sycl::vec<sycl::half, 4> mval = rval - lval;

              float res = static_cast<float>(sycl::dot(mval, mval)); // Could also just use sycl::half all the way
              res = sycl::reduce_over_group(it.get_sub_group(), res, sycl::plus<float>());

              // Compare res to leader
              if(res < leader.value.x())
              {
                  // New leader
                  leader.value.y() = leader.value.x();
                  leader.idx.y() = leader.idx.x();

                  leader.value.x() = res;
                  leader.idx.x() = i;
              }
              else if(res < leader.value.y())
              {
                  // New second
                  leader.value.y() = res;
                  leader.idx.y() = i;
              }
          }

          const bool accept = ((leader.value.x() / leader.value.y()) < 0.8f);
          match_matrix[l_base + it.get_group(0)] = sycl::vec<int, 3>(leader.idx.x(), leader.idx.y(), accept);
      });
#endif

#if PERF_TESTING_FUNCTIONS
    matrix_match_event = matchEvent;
    matrix_remainder_event = remainderMatchEvent;
#endif

    auto wait_for_matrix = [event = std::make_shared<sycl::event>(matchEvent),
                            remainderEvent = std::make_shared<sycl::event>(remainderMatchEvent),
                            &Q = _device_queue]() {
        event->wait();
        remainderEvent->wait();
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
#endif
}

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
