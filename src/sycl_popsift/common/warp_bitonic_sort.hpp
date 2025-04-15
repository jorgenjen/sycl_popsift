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

namespace popsift {
namespace BitonicSort {

// Might move away from being a  class and in common as it's not as generic as the one used in the cuda code

// Currently working for work-groups should make a versio for sub_groups for better performance Might be hard to make
// generic so might need to switch between implemntation based on sub_group size or the multiplier it gives
// template<typename T>
template<typename T, typename GroupType>
class Warp32
{
    sycl::local_accessor<T, 1> _array;
    sycl::nd_item<2> _it;
    GroupType _group;

  public:
    inline Warp32(sycl::local_accessor<T, 1> array, sycl::nd_item<2> it, GroupType group)
      : _array(array)
      , _it(it)
      , _group(group)
    {}

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

    inline void sort64(sycl::vec<int, 2>& my_indecies)
    {
        // Consider adding mask to check who is not -inf
        // and if 32 or less are not -inf we can do the sort in one
        // 32 group and don't need the whole 64. Could also do in even less
        // if it is very few ( could make this conditional logic) should be faseter
        // in average case if -inf is quite common (as it seems to be ) need to test this
        // just do maks and popcount and printout the count of non -inf and run on many imags
        // to get a picture

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

        // if(_it.get_global_linear_id() == 0)
        //     sycl::ext::oneapi::experimental::printf("WE GOING BOYYYYYY");

        for(int outer = 0; outer < 5; outer++)
        {
            for(int inner = outer; inner >= 0; inner--)
            {
                if constexpr(std::is_same_v<GroupType, sycl::sub_group>)
                {
                    my_indecies.x() = shiftit(my_indecies.x(), inner, outer + 1, false);
                    my_indecies.y() = shiftit(my_indecies.y(), inner, outer + 1, true);
                }
                else
                {
                    my_indecies.x() = shiftit_local_mem(my_indecies.x(), inner, outer + 1, false, ref);
                    my_indecies.y() = shiftit_local_mem(my_indecies.y(), inner, outer + 1, true, ref);
                }
            }
        }

        if(_array[my_indecies.x()] < _array[my_indecies.y()])
            swap(my_indecies.x(), my_indecies.y());

        for(int outer = 0; outer < 5; outer++)
        {
            for(int inner = outer; inner >= 0; inner--)
            {
                if constexpr(std::is_same_v<GroupType, sycl::sub_group>)
                {
                    my_indecies.x() = shiftit(my_indecies.x(), inner, outer + 1, false);
                    // my_indecies.y() = shiftit(my_indecies.y(), inner, outer + 1, false);
                    // why are we sorting Y here it can never be selected as ther are no swaps between x and y after
                    // this point and we are sorting it just to sort it?? the best y can be is 32...
                }
                else
                {
                    my_indecies.x() = shiftit_local_mem(my_indecies.x(), inner, outer + 1, false, ref);
                    // my_indecies.y() = shiftit(my_indecies.y(), inner, outer + 1, false);
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
