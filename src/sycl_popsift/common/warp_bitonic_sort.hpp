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
template<class T>
class Warp32
{
    sycl::local_accessor<T, 1> _array;
    sycl::nd_item<2> _it;

  public:
    inline Warp32(sycl::local_accessor<T, 1> array, sycl::nd_item<2> it)
      : _array(array)
      , _it(it)
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
        const T other_val = sycl::select_from_group(_it.get_sub_group(), my_val, _it.get_local_id(1) ^ (1 << shift));

        const bool reverse = (_it.get_local_id(1) & (1 << direction));

        const bool id_less = ((_it.get_local_id(1) & (1 << shift)) == 0);

        // If it thread get other_val from a thread with higher id it will be true if it's value is higher than other
        // otherwise if it gets other_val from lower thread id it will be true if my_val is smaler than other_val
        // if equal it's always false
        const bool my_more = id_less ? (my_val > other_val) : (my_val < other_val);

        // xor my_more with reverse and then xor that with increasing ^ is bitwise xor but onely one bit for bool
        const bool must_swap = !(my_more ^ reverse ^ increasing);

        // int lane = must_swap ? (1 << shift) : 0;
        int remote_id = must_swap ? (_it.get_local_id(1) ^ (1 << shift)) : _it.get_local_id(1);

        // Returns wether or not the index need to swap if lane == 0 it will keep it's value
        // return popsift::shuffle_xor(my_index, lane);
        // return sycl::select_from_group(_it.get_group(), my_index, remote_id);
        return sycl::select_from_group(_it.get_sub_group(), my_index, remote_id);
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
