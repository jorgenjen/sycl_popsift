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
        if(_it.get_global_linear_id() == 0)
            sycl::ext::oneapi::experimental::printf("WE GOING BOYYYYYY");

        for(int outer = 0; outer < 5; outer++)
        {
            for(int inner = outer; inner >= 0; inner--)
            {
                my_indecies.x() = shiftit(my_indecies.x(), inner, outer + 1, false);
                my_indecies.y() = shiftit(my_indecies.y(), inner, outer + 1, true);
            }
        }

        if(_array[my_indecies.x()] < _array[my_indecies.y()])
            swap(my_indecies.x(), my_indecies.y());

        for(int outer = 0; outer < 5; outer++)
        {
            for(int inner = outer; inner >= 0; inner--)
            {
                my_indecies.x() = shiftit(my_indecies.x(), inner, outer + 1, false);
                my_indecies.y() = shiftit(my_indecies.y(), inner, outer + 1, false);
            }
        }
    }

  private:
    inline int shiftit(const int my_index, const int shift, const int direction, const bool increasing)
    {
        const T my_val = _array[my_index];
        // const T other_val = popsift::shuffle_xor(my_val, 1 << shift);
        // Work group
        // const T other_val = sycl::select_from_group(_it.get_group(), my_val, _it.get_local_id(1) ^ (1 << shift));
        // const T other_val = sycl::select_from_group(_group, my_val, _it.get_local_id(1) ^ (1 << shift));

        // takes the work-item id in the group and xor with the mask (1 << shift)
        // This decides which work_items exchange data (butterfly pattern)

        const T other_val = [&]() {
            if constexpr(std::is_same_v<GroupType, sycl::sub_group>)
                return sycl::permute_group_by_xor(_group, my_val, 1 << shift);
            else // std::is_same_v<GroupType, sycl::group<2>> // could have else if
            {
                // Could be better to use handcrafted shared memory version
                int remote_id = _it.get_local_id(1) ^ (1 << shift);
                return sycl::select_from_group(_group, my_val, remote_id);
            }
            // Could add else if and else and say it's unsuported group
        }();
        // const T other_val = sycl::permute_group_by_xor(_group, my_val, 1 << shift);

        const bool reverse = (_it.get_local_id(1) & (1 << direction));

        const bool id_less = ((_it.get_local_id(1) & (1 << shift)) == 0);

        // If it thread get other_val from a thread with higher id it will be true if it's value is higher than
        // other otherwise if it gets other_val from lower thread id it will be true if my_val is smaler than
        // other_val if equal it's always false
        const bool my_more = id_less ? (my_val > other_val) : (my_val < other_val);

        // xor my_more with reverse and then xor that with increasing ^ is bitwise xor but onely one bit for bool
        const bool must_swap = !(my_more ^ reverse ^ increasing);

        if constexpr(std::is_same_v<GroupType, sycl::sub_group>)
        {
            // If we must swap we pass the mask so we swap with the assigned lane
            // otherwise we pass 0 and we don't (not sure if using different masks is alowed in sycl for a permute)
            int lane = must_swap ? (1 << shift) : 0;
            // Should not be allowed according to docs but seem to work...
            return sycl::permute_group_by_xor(_group, my_index, lane);
        }
        else
        {
            // the threads that exchanged and got other_val must have same must_swap and hence this
            // should work as well as using the select_from_subgroup version below

            int remote_id = must_swap ? (_it.get_local_id(1) ^ (1 << shift)) : _it.get_local_id(1);
            return sycl::select_from_group(_group, my_index, remote_id);
        }
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
