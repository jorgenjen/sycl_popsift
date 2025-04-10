/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once

#include "assist.h"

// #include <cuda_runtime.h>
#include <typeinfo>

namespace ExclusivePrefixSum {
class IgnoreTotal
{
    inline void set(int, int) {}
};

class IgnoreWriteMapping
{
  public:
    inline void set(int, int, int) {}
};

template<typename GroupType,
         class Reader,
         class Writer,
         class Total = IgnoreTotal,
         class WriteMapping = IgnoreWriteMapping>
class Block
{
    const sycl::nd_item<2>& _it;
    const Reader& _reader;
    Writer& _writer;
    Total& _total_writer;
    WriteMapping& _mapping_writer;
    const int _num;
    sycl::local_accessor<int, 1> _sum;
    sycl::local_accessor<int, 1> _loop_total;
    GroupType _group;

  public:
    /* Instantiate an object of this type.
     * This ExclusivePrefixSum works correctly exclusively in a configuration with
     * block=(32,32,1) and grid=(1,1,1).
     * The parameter num can be any number, but only small numbers up to a few
     * thousand make sense. The template is not intended for large sets.
     * The template classes Reader, Writer and Total must provide ()-operators.
     *   inline __device__ is strongly recommended.
     * Reader must provide operator()(int n) that returns the input int at pos n.
     * Writer must provide operator()(int n) that returns int& for writing at pos n.
     * Total  must provide operator()() that returns int& for writing the total sum.
     */
    Block(sycl::nd_item<2>& it,
          int num,
          const Reader& reader,
          Writer& writer,
          Total& total_writer,
          WriteMapping& mapping_writer,
          sycl::local_accessor<int, 1> sum_arr,
          sycl::local_accessor<int, 1> loop_total,
          GroupType group)

      : _it(it)
      , _num(num)
      , _reader(reader)
      , _writer(writer)
      , _total_writer(total_writer)
      , _mapping_writer(mapping_writer)
      , _sum(sum_arr)
      , _loop_total(loop_total)
      , _group(group)
    {
        sum();
    }

  private:
    /* This function computes the actual exclusive prefix summation
     */
    void sum()
    {
        // __shared__ int sum[32];
        // __shared__ int loop_total;

        // if(threadIdx.x == 0 && threadIdx.y == 0)
        if(_it.get_local_id(1) == 0 && _it.get_local_id(0) == 0)
        {
            _loop_total[0] = 0;
        }

        // __syncthreads();
        sycl::group_barrier(_group);

        // const int start = threadIdx.y * blockDim.x + threadIdx.x;
        // const int wrap = blockDim.x * blockDim.y;

        const int start = _it.get_global_linear_id();
        const int wrap = _it.get_local_range(1) * _it.get_local_range(0);
        const int end = (_num & (wrap - 1)) ? (_num & ~(wrap - 1)) + wrap : _num;

        for(int x = start; x < end; x += wrap)
        {
            // __syncthreads();
            sycl::group_barrier(_group);

            const bool valid = (x < _num);
            // const int cell = min(x, _num - 1);
            const int cell = sycl::min(x, _num - 1);

            int ews = 0; // exclusive warp prefix sum
            int self = (valid) ? _reader.get(cell) : 0;

            // This loop is an exclusive prefix sum for one warp
            for(int s = 0; s < 5; s++)
            {
                // const int add = popsift::shuffle_up(ews + self, 1 << s);
                const int add = sycl::shift_group_right(_group, ews + self, 1 << s);
                // ews += threadIdx.x < (1 << s) ? 0 : add;
                ews += _it.get_local_id(1) < (1 << s) ? 0 : add;
            }

            // if(threadIdx.x == 31)
            if(_it.get_local_id(1) == 31)
            {
                // store inclusive warp prefix sum in shared mem
                // to be summed up in next phase
                // sum[threadIdx.y] = ews + self;
                _sum[_it.get_local_id(0)] = ews + self;
            }
            // __syncthreads();
            sycl::group_barrier(_group);

            int ibs; // inclusive block prefix sum
            // if(threadIdx.y == 0)
            if(_it.get_local_id(0) == 0)
            {
                int ebs = 0; // exclusive block prefix sum
                // int self = sum[threadIdx.x];
                int self = _sum[_it.get_local_id(1)];

                for(int s = 0; s < 5; s++)
                {
                    // const int add = popsift::shuffle_up(ebs + self, 1 << s);
                    const int add = sycl::shift_group_right(_group, ebs + self, 1 << s);
                    // ebs += threadIdx.x < (1 << s) ? 0 : add;
                    ebs += _it.get_local_id(1) < (1 << s) ? 0 : add;
                }

                _sum[_it.get_local_id(1)] = ebs;
                _sum[_it.get_local_id(1)] = ebs;
                ibs = ebs + self;
            }
            // __syncthreads();
            sycl::group_barrier(_group);

            if(valid)
            {
                // const int ebs = loop_total + sum[threadIdx.y] + ews;
                const int ebs = _loop_total[0] + _sum[_it.get_local_id(1)] + ews;

                /* Conceptually: at index cell of the _writer,
                 * store the exclusive prefix sum ebs.
                 */
                _writer.set(cell, ebs);

                /* Conceptually: at index ebs of the _mapping_writer,
                 * and the self-1 indices after it, store the position
                 * cell within the original array, _reader.
                 */
                _mapping_writer.set(ebs, self, cell);
            }
            // __syncthreads();
            sycl::group_barrier(_group);

            // if(threadIdx.y == 0 && threadIdx.x == 31)
            if(_it.get_local_id(0) == 0 && _it.get_local_id(1) == 31)
            {
                _loop_total[0] += ibs;
            }
            // __syncthreads();
            sycl::group_barrier(_group);
        }

        _total_writer.set(_loop_total[0]);
    }
};

} // namespace ExclusivePrefixSum
