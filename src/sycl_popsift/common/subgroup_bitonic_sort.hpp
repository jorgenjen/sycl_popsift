/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once

// #include "assist.h"
#include "sycl/sycl.hpp"
#include "sycl_popsift/sift_constants.hpp"

#include <cmath>

namespace popsift {
namespace BitonicSort {

// Might move away from being a  class and in common as it's not as generic as the one used in the cuda code

// Works only when the work_group is 1, 32
template<typename T, typename GroupType>
class WorkGroup32
{
    sycl::local_accessor<T, 1> _array;
    sycl::nd_item<2> _it;
    GroupType _group;

  public:
    inline WorkGroup32(sycl::local_accessor<T, 1> array, sycl::nd_item<2> it, GroupType group)
      : _array(array)
      , _it(it)
      , _group(group)
    {}

    // Not in use
    inline int sort32(int my_index)
    {
        for(int outer = 0; outer < 5; outer++)
        {
            for(int inner = outer; inner >= 0; inner--)
            {
                my_index = shiftit(my_index, inner, outer + 1, false);
            }
        }
        return my_index;
    }

    // Only for sub_groups of size 32
    inline void minimal_sort64(int& my_index) // Modifies given my_index
    {
        // sycl::ext::oneapi::sub_group_mask is the type

        // Could use sycl::isinf to check if the value is infinity negative or poseetive true returned if it is so
        // opposite of what we want however, Could use sycl::isfinite
        auto mask_lower = sycl::ext::oneapi::group_ballot(_group, _array[_it.get_local_id(1)] != -INFINITY);
        auto mask_upper = sycl::ext::oneapi::group_ballot(_group, _array[_it.get_local_id(1) + 32] != -INFINITY);

        auto lower_count = mask_lower.count();
        auto upper_count = mask_upper.count();
        auto total_count = lower_count + upper_count;

        if(total_count >= 32)
        {
            // Running backup mode -- Seems highly improbable to happen in my testing
            // Max was 5 for 947 images
            sort64(my_index);
        }
        else
        {
            // Could merge the masks and use the mask directly so we can remove if upper and lower
            // Assign indecies to work-items 0 take lsb and 1 second lsb and so on

            // Could also do it based on the pure mask by extract_bits and using sycl::ctz to count trailing 0 bits
            if(_it.get_local_id(1) < lower_count)
            {
                // Assigned in lower
                for(int i = 0; i < lower_count; ++i)
                {
                    sycl::id<1> idx = mask_lower.find_low();
                    mask_lower.reset(idx);
                    if(_it.get_local_id(1) == i)
                    {
                        my_index = idx[0];
                        // break; // Could break here but not sure if we want to
                    }
                }
            }
            else
            {
                // Assigned in upper
                for(int i = lower_count; i < total_count; ++i)
                {
                    // if upper_count is 0 this will never run
                    sycl::id<1> idx = mask_upper.find_low();
                    mask_upper.reset(idx);
                    if(_it.get_local_id(1) == i)
                    {
                        my_index = idx[0] + 32;
                        // break; // Could break here but not sure if we want to
                    }
                }
            }

            // This ^ could be done similarly with __ffsll in cuda or potentially faster with
            // __fns(unsigned mask, unsigned base, int offset)
            // Find the position of the n-th set to 1 bit in a 32-bit integer.

            // Now we sort the non -inf values
            int max_outer = (total_count <= 2)    ? 1 // Selects how wide the sort need to be (2, 4, 8, 16, 32)
                            : (total_count <= 4)  ? 2
                            : (total_count <= 8)  ? 3
                            : (total_count <= 16) ? 4
                                                  : 5;

            sycl::group_barrier(_group); // To ensure they are not divierged -- not sure if needed
            for(int outer = 0; outer < max_outer; outer++)
            {
                for(int inner = outer; inner >= 0; inner--)
                {
                    // Could consider masking out the threads in the permute_by_xor
                    my_index = shiftit(my_index, inner, outer + 1, false);
                }
            }
        }
    }

    // Consider adding mask to check who is not -inf
    // and if 32 or less are not -inf we can do the sort in one
    // 32 group and don't need the whole 64. Could also do in even less
    // if it is very few ( could make this conditional logic) should be faseter
    // in average case if -inf is quite common (as it seems to be ) need to test this
    // just do maks and popcount and printout the count of non -inf and run on many imags
    // to get a picture

    // inline void sort64(sycl::vec<int, 2>& my_indecies)
    inline void sort64(int& lower)
    {
        // Should not add overhead for sub_group version  I think...
        const auto& ref = [&]() {
            if constexpr(std::is_same_v<GroupType, sycl::sub_group>)
            {
                return nullptr;
            }
            else // std::is_same_v<GroupType, sycl::group<2>> // could have else if
            {
                // Only need this shared memory if we use work group
                sycl::multi_ptr<int[64], sycl::access::address_space::local_space> ptr =
                  sycl::ext::oneapi::group_local_memory<int[64]>(_it.get_group());
                return *ptr;
            }
        }();

        lower = _it.get_local_id(1);
        int upper = _it.get_local_id(1) + 32;
        for(int outer = 0; outer < 5; outer++)
        {
            for(int inner = outer; inner >= 0; inner--)
            {
                if constexpr(std::is_same_v<GroupType, sycl::sub_group>)
                {
                    // my_indecies.x() = shiftit(my_indecies.x(), inner, outer + 1, false);
                    // my_indecies.y() = shiftit(my_indecies.y(), inner, outer + 1, true);

                    lower = shiftit(lower, inner, outer + 1, false);
                    upper = shiftit(upper, inner, outer + 1, true);
                }
                else
                {
                    // my_indecies.x() = shiftit_local_mem(my_indecies.x(), inner, outer + 1, false, ref);
                    // my_indecies.y() = shiftit_local_mem(my_indecies.y(), inner, outer + 1, true, ref);

                    lower = shiftit_local_mem(lower, inner, outer + 1, false, ref);
                    upper = shiftit_local_mem(upper, inner, outer + 1, true, ref);
                }
            }
        }

        // We want the largest values in the lower half
        if(_array[lower] < _array[upper])
            swap(lower, upper);

        // Sort the lower half (upper is now discarded -- Was kept in cuda for generic reasons I guess)
        for(int outer = 0; outer < 5; outer++)
        {
            for(int inner = outer; inner >= 0; inner--)
            {
                if constexpr(std::is_same_v<GroupType, sycl::sub_group>)
                {
                    // my_indecies.x() = shiftit(my_indecies.x(), inner, outer + 1, false);
                    lower = shiftit(lower, inner, outer + 1, false);
                }
                else
                {
                    lower = shiftit_local_mem(lower, inner, outer + 1, false, ref);
                }
            }
        }
    }

  private:
    inline int shiftit(const int my_index, const int shift, const int direction, const bool increasing)
    {
        const T my_val = _array[my_index];

        const T other_val = sycl::permute_group_by_xor(_group, my_val, 1 << shift);

        const bool reverse = (_it.get_local_id(1) & (1 << direction));

        const bool id_less = ((_it.get_local_id(1) & (1 << shift)) == 0);

        // If it thread get other_val from a thread with higher id it will be true if it's value is higher than
        // other otherwise if it gets other_val from lower thread id it will be true if my_val is smaler than
        // other_val if equal it's always false
        const bool my_more = id_less ? (my_val > other_val) : (my_val < other_val);

        // xor my_more with reverse and then xor that with increasing ^ is bitwise xor but onely one bit for bool
        const bool must_swap = !(my_more ^ reverse ^ increasing);

        // If we must swap we pass the mask so we swap with the assigned lane
        // otherwise we pass 0 and we don't (not sure if using different masks is alowed in sycl for a permute)
        int lane = must_swap ? (1 << shift) : 0;
        // Should not be allowed according to docs but seem to work...
        return sycl::permute_group_by_xor(_group, my_index, lane);
    }

    inline int shiftit_local_mem(
      const int my_index, const int shift, const int direction, const bool increasing, int* local_mem)
    {
        const T my_val = _array[my_index]; // Store my val
        // local_mem[_it.get_local_id(1) + (increasing ? 32 : 0)] = my_index; // Store my index in my work-item location
        local_mem[_it.get_local_id(1)] = my_index; // Store my index in my work-item location
        sycl::group_barrier(_group);

        const int remote_id = _it.get_local_id(1) ^ (1 << shift);
        const int idx_other_val = local_mem[remote_id];

        sycl::group_barrier(_group); // need this to ensure when running small sub_groups that they don't contunue to
                                     // next iteration and function call and updates the local_mem before everyone
                                     // has stored idx_other_val This barrier could also be before local_mem update
                                     // Don't really matter where it isaslong as it blocks from going around to update

        const T other_val = _array[idx_other_val];

        const bool reverse = (_it.get_local_id(1) & (1 << direction));

        const bool id_less = ((_it.get_local_id(1) & (1 << shift)) == 0);

        // If it thread get other_val from a thread with higher id it will be true if it's value is higher than
        // other otherwise if it gets other_val from lower thread id it will be true if my_val is smaler than
        // other_val if equal it's always false
        const bool my_more = id_less ? (my_val > other_val) : (my_val < other_val);

        // xor my_more with reverse and then xor that with increasing ^ is bitwise xor but onely one bit for bool
        const bool must_swap = !(my_more ^ reverse ^ increasing);

        return must_swap ? idx_other_val : my_index;
    }

    inline void swap(int& l, int& r)
    {
        int m = r;
        r = l;
        l = m;
    }
};
} // namespace popsift
} // namespace BitonicSort

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
//     const bool my_more = id_less ? (my_val > other_val) : (my_val < other_val);
//
//     const bool must_swap = !(my_more ^ reverse ^ increasing);
//
//     int lane = must_swap ? (1 << shift) : 0;
//     return sycl::permute_group_by_xor(_group, my_index, lane);
// }
//
// inline int shiftit_local_mem(
//   const int my_index, const int shift, const int direction, const bool increasing, int* local_mem)
// {
//     const T my_val = _array[my_index];         // Store my val
//     local_mem[_it.get_local_id(1)] = my_index; // Store my index in my work-item location
//     sycl::group_barrier(_group);
//
//     const int remote_id = _it.get_local_id(1) ^ (1 << shift);
//     const int idx_other_val = local_mem[remote_id];
//
//     const T other_val = _array[idx_other_val];
//
//     const bool reverse = (_it.get_local_id(1) & (1 << direction));
//
//     const bool id_less = ((_it.get_local_id(1) & (1 << shift)) == 0);
//
//     const bool my_more = id_less ? (my_val > other_val) : (my_val < other_val);
//
//     const bool must_swap = !(my_more ^ reverse ^ increasing);
//
//     return must_swap ? idx_other_val : my_index;
// }
