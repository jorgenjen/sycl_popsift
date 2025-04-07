/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "common/assist.h"
#include "s_solve.h"
#include "sift_extremum.h"
// #include "common/clamp.h"
#include "sycl_popsift/common/debug_macros.hpp"

// #include "s_solve.h" # Need this one later on
#include "sift_constants.hpp"
#include "sift_pyramid.hpp"
#include "sycl/access/access.hpp"
#include "sycl/atomic_ref.hpp"
#include "sycl/group_algorithm.hpp"
#include "sycl/kernel_bundle_enums.hpp"
#include "sycl/memory_enums.hpp"
#include "sycl/nd_item.hpp"
#include "sycl/sub_group.hpp"
#include "sycl/usm.hpp"
#include "sycl/vector.hpp"
#include "sycl_popsift/non_sycl/sift_conf.hpp"

#include <cmath>
#include <cstdio>
#include <sstream>
#include <vector>

namespace popsift {
#define LOCAL_X 32

#define SUB_GROUP_COUNTING 0

#if SUB_GROUP_COUNTING
// template<int HEIGHT> // did not do anything
// Must take care here as sub-group will not be 32 in all cases like a warp in cuda
// should also make the blocks based on sub-group multiplier(idk what the name was) preference of the device used
// Not sure why indicator was not bool ??
static inline uint32_t extrema_count(bool indicator, int* extrema_counter, sycl::nd_item<3>& it)
{
    // sub_grousp is undergoing change and not recomended to use but seems most fitting in this case
    sycl::sub_group sub_group = it.get_sub_group();

#define USE_MASK 1
#if USE_MASK == 1
    // Should be the same as ballot_sync and __popc
    // Will work as long as sub-group is not larger than 32
    uint32_t mask = sycl::reduce_over_group(
      sub_group,
      indicator ? (1u << sub_group.get_local_id()[0]) : 0u,
      sycl::ext::oneapi::bit_or<uint32_t>()); // better to get_local_id from sub group or it? should be same
    int ct = sycl::popcount(mask);
#else
    // basic reduce to sum indicators being true
    // int ct = sycl::inclusive_scan_over_group(sub_group, indicator ? 1 : 0, sycl::plus<>());
    int ct = sycl::reduce_over_group(sub_group, indicator ? 1 : 0, sycl::plus<>());
#endif
    int write_index;
    if(sub_group.leader()) // is always work-item with local_id 0 in the sub_group
    {
        // SHould probably query first to ensure the memory scope and order is supported by the device
        // see page 540 (560 in pdf) // using memory_order_relaxed which any device should support

        // BUG: Seems to run for all work-items in sub-group and not only one as required hence resulting in adding the
        // ct value 8 times causing the end value to be 8 times greater than it should be
        //      Can see that this is the case as it prints for all sub_group.get_group_linear_id so [0-8] in my

        // The atomic add returns the old value in extrema_coutner before the addition which is considered the base
        // As each thread uses this and adds to it's own counter (write_index) how many of the threads in the
        // sub-group before it had it's indicator to true
        if(sub_group.get_local_linear_id() == 0) // should be same as leader()
        {
            write_index = sycl::atomic_ref<int,
                                           sycl::memory_order_relaxed,
                                           sycl::memory_scope_device,
                                           sycl::access::address_space::global_space>(*extrema_counter) += ct;

            // if(sub_group.get_group_linear_id() == 89 &&
            if(it.get_group_linear_id() == 20000 && it.get_global_range(2) == 1280)
            {
                // Why in tha lordy lordy does this print when it prints out that sub_group.get_local_linear_id == [1-7]
                // the value is printed corectly but this code should only run when it is 0... must be something wrong
                // with my system  I guess I don't understand this
                sycl::ext::oneapi::experimental::printf(
                  "\n\t HOYY:: Sub_group.get_local_id()[0] = %d = %d -- ct = %d, write_index = %d",
                  sub_group.get_local_linear_id(),
                  sub_group.get_local_id()[0],
                  ct,
                  write_index);
            }
        }
    }

    // work-item 0 broadcassts to all other same as leader work-item
    // everyone now get's the base value that they can add to
    write_index = sycl::group_broadcast(sub_group, write_index, 0);

    // Adds the sum of set bits in mask that has sub_grop local id lower than the current (exclusive)
    //  this provides the 0 result and every result up to ct
    write_index += sycl::popcount(mask & ((1 << sub_group.get_local_id()[0]) - 1)); // breaks if USE_MASK != 1

    return write_index;
}
#else
// Do the counting for the whole work-group will be less efficient but hopefully work correctly
// eventhoug I  believe that it is my system causing the sub-group not to work
#define every_body_add 1
#if every_body_add

static inline uint32_t extrema_count(bool indicator, int* extrema_counter, sycl::nd_item<3>& it)
{
    sycl::group<3> work_group = it.get_group();
    int local_linear = it.get_local_linear_id();

    // int write_index = sycl::
    //   atomic_ref<int, sycl::memory_order_relaxed, sycl::memory_scope_device,
    //   sycl::access::address_space::global_space>(
    //     *extrema_counter) += (indicator ? 1 : 0);
    int write_index = sycl::atomic_ref<int,
                                       sycl::memory_order_seq_cst,
                                       sycl::memory_scope_device,
                                       sycl::access::address_space::global_space>(*extrema_counter)
                        .fetch_add(indicator ? 1 : 0);
    return write_index;
}
#else
static inline uint32_t extrema_count(bool indicator, int* extrema_counter, sycl::nd_item<3>& it)
{
    sycl::group<3> work_group = it.get_group();
    int local_linear = it.get_local_linear_id();
    // reduce over the whole work group
    // int ct = sycl::reduce_over_group(work_group, indicator ? 1 : 0, sycl::plus<>());
    // Final result is the count but we don't use it as write-index since it is zero based -- note to self
    int ct = sycl::exclusive_scan_over_group(work_group, indicator ? 1 : 0, sycl::plus<>());

    int last_work_item = it.get_local_range(0) * it.get_local_range(1) * it.get_local_range(2) - 1;
    int write_index;

    // I would assume that there is no need to have a barrier here
    if(local_linear == last_work_item) // only last has the complete value
    {
        // Need to add it's own value to the exclusive result making it inclusive for the atmoic add
        // Returns the base which is the value before tha this add was done(old value)
        write_index = sycl::atomic_ref<int,
                                       sycl::memory_order_relaxed,
                                       sycl::memory_scope_device,
                                       sycl::access::address_space::global_space>(*extrema_counter) +=
          (ct + (indicator ? 1 : 0));

        if(it.get_group_linear_id() == 13558)
        // if(ct > 1)
        {
            // sycl::ext::oneapi::experimental::printf("group_linear = %d", it.get_group_linear_id());
            sycl::ext::oneapi::experimental::printf(
              "work-item id in work-group %d == %d -- ct = %d  --- indicator = %d\n",
              local_linear,
              last_work_item,
              ct,
              indicator ? 1 : 0);
        }
    }

    write_index = sycl::group_broadcast(work_group, write_index, last_work_item); // last broadcasts
    return write_index + ct;
}

#endif
#endif

static inline void extremum_cmp(float val, float f, uint32_t& gt, uint32_t& lt, uint32_t mask)
{
    gt |= ((val > f) ? mask : 0);
    lt |= ((val < f) ? mask : 0);
}

// #define TX(dx, dy, dz) readTex(obj, x + dx, y + dy, z + dz)
// #define DOG(dx, dy, dz) dog[z + dz][x + dx + (y + dy) * width]

// different clamping I think I only need for bottom and right -- z should always be safe
// REFINE makes the requirement go up to full clamp mode
#define CLAMP_MODE 3
#if CLAMP_MODE == 0
// no clamping
#define DOG(dx, dy, dz) dog[z + dz][x + dx + (y + dy) * width]
#elif CLAMP_MODE == 1
// Clamping for bottom and right
#define DOG(dx, dy, dz)                                                                                                \
    (x + dx >= width && y + dy >= height) ? dog[z + dz][width - 1 + (height - 1) * width]                              \
    : (y + dy >= height)                  ? dog[z + dz][x + dx + (height - 1) * width]                                 \
    : (x + dx >= width)                   ? dog[z + dz][width - 1 + (y + dy) * width]                                  \
                                          : dog[z + dz][x + dx + (y + dy) * width]

#elif CLAMP_MODE == 2
#define CLAMP(val, max) ((val) > (max) ? (max) : (val)) // the parenthesies avoid operator precedence problems(?)
#define DOG(dx, dy, dz) dog[z + dz][CLAMP(x + dx, width - 1) + CLAMP(y + dy, height - 1) * width]

#else
// Full clamping // Probs better of using sycl::clamp here but I as not able to when trying
// will probably be replaced by bindless image in best version anyays :D
#define CLAMP(val, min, max) ((val) < (min) ? (min) : ((val) > (max) ? (max) : (val)))
#define DOG(dx, dy, dz) dog[z + dz][CLAMP(x + dx, 0, width - 1) + CLAMP(y + dy, 0, height - 1) * width]

#endif

static inline bool is_extremum(float** dog, int x, int y, int z, int width, int height)
{
    uint32_t gt = 0;
    uint32_t lt = 0;

    const float val0 = DOG(0, 1, 1);
    const float val2 = DOG(2, 1, 1);
    const float val = DOG(1, 1, 1);

    // bit indeces for neighbours:
    //     7 0 1    0x80 0x01 0x02
    //     6   2 -> 0x40      0x04
    //     5 4 3    0x20 0x10 0x08
    // upper layer << 24 ; own layer << 16 ; lower layer << 8
    // 1st group: left and right neigbhour
    extremum_cmp(val, val0, gt, lt, 0x00400000); // ( 0x01<<6 ) << 16
    extremum_cmp(val, val2, gt, lt, 0x00040000); // ( 0x01<<2 ) << 16

    if((gt != 0x00440000) && (lt != 0x00440000))
        return false;

    // 2nd group: requires a total of 8 128-byte reads
    extremum_cmp(val, DOG(0, 0, 1), gt, lt, 0x00800000); // ( 0x01<<7 ) << 16
    extremum_cmp(val, DOG(0, 2, 1), gt, lt, 0x00200000); // ( 0x01<<5 ) << 16
    extremum_cmp(val, DOG(0, 0, 0), gt, lt, 0x80000000); // ( 0x01<<6 ) << 24
    extremum_cmp(val, DOG(0, 2, 0), gt, lt, 0x40000000); // ( 0x01<<6 ) << 24
    extremum_cmp(val, DOG(0, 1, 0), gt, lt, 0x20000000); // ( 0x01<<6 ) << 24
    extremum_cmp(val, DOG(0, 0, 2), gt, lt, 0x00008000); // ( 0x01<<6 ) <<  8
    extremum_cmp(val, DOG(0, 1, 2), gt, lt, 0x00004000); // ( 0x01<<6 ) <<  8
    extremum_cmp(val, DOG(0, 2, 2), gt, lt, 0x00002000); // ( 0x01<<6 ) <<  8

    if((gt != 0xe0e4e000) && (lt != 0xe0e4e000))
        return false;

    // 3rd group: remaining 2 cache misses in own layer
    extremum_cmp(val, DOG(1, 0, 1), gt, lt, 0x00010000); // ( 0x01<<0 ) << 16
    extremum_cmp(val, DOG(2, 0, 1), gt, lt, 0x00020000); // ( 0x01<<1 ) << 16
    extremum_cmp(val, DOG(1, 2, 1), gt, lt, 0x00100000); // ( 0x01<<4 ) << 16
    extremum_cmp(val, DOG(2, 2, 1), gt, lt, 0x00080000); // ( 0x01<<3 ) << 16

    if((gt != 0xe0ffe000) && (lt != 0xe0ffe000))
        return false;

    // 4th group: 3 cache misses higher layer
    extremum_cmp(val, DOG(1, 0, 0), gt, lt, 0x01000000); // ( 0x01<<0 ) << 24
    extremum_cmp(val, DOG(2, 0, 0), gt, lt, 0x02000000); // ( 0x01<<1 ) << 24
    extremum_cmp(val, DOG(1, 1, 0), gt, lt, 0x00000004); // ( 0x01<<2 )
    extremum_cmp(val, DOG(2, 1, 0), gt, lt, 0x04000000); // ( 0x01<<2 ) << 24
    extremum_cmp(val, DOG(1, 2, 0), gt, lt, 0x10000000); // ( 0x01<<4 ) << 24
    extremum_cmp(val, DOG(2, 2, 0), gt, lt, 0x08000000); // ( 0x01<<3 ) << 24

    if((gt != 0xffffe004) && (lt != 0xffffe004))
        return false;

    // 5th group: 3 cache misss lower layer
    extremum_cmp(val, DOG(1, 0, 2), gt, lt, 0x00000100); // ( 0x01<<0 ) <<  8
    extremum_cmp(val, DOG(2, 0, 2), gt, lt, 0x00000200); // ( 0x01<<1 ) <<  8
    extremum_cmp(val, DOG(1, 1, 2), gt, lt, 0x00000001); // ( 0x01<<0 )
    extremum_cmp(val, DOG(2, 1, 2), gt, lt, 0x00000400); // ( 0x01<<2 ) <<  8
    extremum_cmp(val, DOG(1, 2, 2), gt, lt, 0x00001000); // ( 0x01<<4 ) <<  8
    extremum_cmp(val, DOG(2, 2, 2), gt, lt, 0x00000800); // ( 0x01<<3 ) <<  8

    if((gt != 0xffffff05) && (lt != 0xffffff05))
        return false;

    return true;
}

template<int sift_mode>
class ModeFunctions
{
  public:
    /* refine
     * returns 0 : continue looping
     *         1 : break loop and succeed
     */
    inline int refine(sycl::vec<float, 3>& d,
                      sycl::vec<int, 3>& n,
                      int width,
                      int height,
                      int maxlevel,
                      bool last_it,
                      size_t global_id);
};

template<>
class ModeFunctions<Config::RefineInLevel>
{
  public:
    inline int refine(sycl::vec<float, 3>& d,
                      sycl::vec<int, 3>& n,
                      int width,
                      int height,
                      int maxlevel,
                      bool last_it,
                      size_t global_id) const
    {
        if(last_it)
            return 0;

        // int2 t;
        sycl::vec<int, 2> t;

        t.x() = ((d.x() >= 0.6f && n.x() < width - 2) ? 1 : 0) + ((d.x() <= -0.6f && n.x() > 1) ? -1 : 0);

        t.y() = ((d.y() >= 0.6f && n.y() < height - 2) ? 1 : 0) + ((d.y() <= -0.6f && n.y() > 1) ? -1 : 0);

        if(t.x() == 0 && t.y() == 0)
        {
            // no more changes
            return 1;
        }

        n.x() += t.x();
        n.y() += t.y();
        // n.z += t.z; - VLFeat is not changing levels !!!

        return 0;
    }
};

template<>
class ModeFunctions<Config::RefineInOctave>
{
  public:
    // inline int refine(float3& d, int3& n, int width, int height, int maxlevel, bool last_it) const
    inline int refine(sycl::vec<float, 3>& d,
                      sycl::vec<int, 3>& n,
                      int width,
                      int height,
                      int maxlevel,
                      bool last_it,
                      size_t global_id) const
    {
        if(last_it)
            return 0;

        sycl::vec<int, 3> t;

        t.x() = ((d.x() >= 0.6f && n.x() < width - 2) ? 1 : 0) + ((d.x() <= -0.6f && n.x() > 1) ? -1 : 0);

        t.y() = ((d.y() >= 0.6f && n.y() < height - 2) ? 1 : 0) + ((d.y() <= -0.6f && n.y() > 1) ? -1 : 0);

        t.z() = ((d.z() >= 0.6f && n.z() < maxlevel - 1) ? 1 : 0) + ((d.z() <= -0.6f && n.z() > 1) ? -1 : 0);

        if(t.x() == 0 && t.y() == 0 && t.z() == 0)
        {
            // no more changes
            return 1;
        }

        n.x() += t.x();
        n.y() += t.y();
        n.z() += t.z();

        return 0;
    }
};

// Not sure if I should use a function or just add the code but doing it to be close to cuda version
inline static bool first_contrast_ok(const float val, const popsift::ConstInfo* d_consts)
{
    // fabs should be equivalent to fabsf as it's overloaded with support for float in sycl
    return (sycl::fabs(val) >= 1.6f * d_consts->threshold);
}

/** verify() checks whether a refine position is outside the image boundaries or
 *  outside the DoG boundaries.
 *  returns true  : values after refine make sense
 *          false : they do not
 */
inline static bool verify(float xn, float yn, float sn, int width, int height, int maxlevel)
{
    // reject if outside of image bounds or far outside DoG bounds
    return ((xn < 0.0f || xn > width - 1.0f || yn < 0.0f || yn > height - 1.0f || sn < -0.0f || sn > maxlevel) ? false
                                                                                                               : true);
}

template<int sift_mode>
inline bool find_extrema_in_dog_sub(float** dog,
                                    int debug_octave,
                                    int width,
                                    int height,
                                    uint32_t maxlevel,
                                    float w_grid_divider,
                                    float h_grid_divider,
                                    int grid_width,
                                    InitialExtremum* ec,
                                    sycl::nd_item<3>& it,
                                    const popsift::ConstInfo* d_consts)
{
    ec->xpos = 0.0f;
    ec->ypos = 0.0f;
    ec->lpos = 0;
    ec->sigma = 0.0f;

    /*
     * First consideration: extrema cannot be found on any outermost edge,
     * one pixel on the left, right, upper, lower edge will never qualify.
     * Also, the upper and lower DoG layer will never qualify. So there is
     * no reason for selecting any of those pixel for the center of a 3x3x3
     * region.
     * Instead, I use groups of 32xHEIGHT threads that read from a 34x34x3 area,
     * but implicitly, they fetch * 64xHEIGHT+2x3 floats (bad luck).
     * To find maxima, compare first on the left edge of the 3x3x3 cube, ie.
     * a 1x3x3 area. If the rightmost 2 threads of a warp (x==30 and 3==31)
     * are not extreme w.r.t. to the left slice, 8 fetch operations.
     */

    // sub-group(warp in cuda) is in 3D space along the nd_range[2] dimension and hence the problem needs to be
    // reorganized from the cuda equivalent for it to be the similar in hardware

    // value in variables correspond to the values in the cuda version making the rest of the function almost identical
    const int block_x = it.get_group(2) * it.get_local_range(2); // local[2] == 32
    const int block_y = it.get_group(1) * it.get_local_range(1); // local[1] == 4
    const int block_z = it.get_group(0);
    const int x = it.get_global_id(2) + 1;
    const int y = it.get_global_id(1) + 1;
    const int level = it.get_global_id(0) + 1;

#define CLAMP_READ_DOG 1

#if CLAMP_READ_DOG == 0
#define READ_DOG(x, y, z) dog[z][x + y * width]
#else
// Full clamping // Should probably use sycl::clamp (but i cant get sycl::clamp to work)
#define CLAMP(val, min, max) ((val) < (min) ? (min) : ((val) > (max) ? (max) : (val)))
#define READ_DOG(x, y, z)                                                                                              \
    dog[CLAMP(z, 0, it.get_global_range(0) + 1)][CLAMP(x, 0, width - 1) + CLAMP(y, 0, height - 1) * width]

// DOES NOT WORK...
// #define READ_DOG(x, y, z) \
//     dog[sycl::clamp<int>(z, 0, it.get_global_range(0) + 1)] \
//        [sycl::clamp<int>(x, 0, width - 1) + sycl::clamp<int>(y, 0, height - 1) * width]
#endif

    // const float val = dog[level][x + y * width];
    const float val = READ_DOG(x, y, level); // not sure if this one need's clamping (dont think so)

    ModeFunctions<sift_mode> f;
    if(!first_contrast_ok(val, d_consts))
        return false;

    if(!is_extremum(dog, x - 1, y - 1, level - 1, width, height))
    {
        return false;
    }

    sycl::vec<float, 3> D;  // Dx Dy Ds
    sycl::vec<float, 3> DD; // Dxx Dyy Dss
    sycl::vec<float, 3> DX; // Dxy Dxs Dys
    sycl::vec<float, 3> d;  // dx dy ds

    float v = val;

    sycl::vec<int, 3> n(x, y, level); // nj ni ns

    int32_t iter = 0;

#define MAX_ITERATIONS 5

    do
    {
        iter++;

        // const int z = level - 1;

        // to ensure we are withing boundary (-1 is no problem) z is safe
        // const int x_add = (n.x() + 1 >= width) ? 0 : 1;
        // const int y_add = (n.y() + 1 >= height) ? 0 : 1;

        /* compute gradient */
        const float x2y1z1 = READ_DOG(n.x() + 1, n.y(), n.z());
        const float x0y1z1 = READ_DOG(n.x() - 1, n.y(), n.z());
        const float x1y2z1 = READ_DOG(n.x(), n.y() + 1, n.z());
        const float x1y0z1 = READ_DOG(n.x(), n.y() - 1, n.z());
        const float x1y1z2 = READ_DOG(n.x(), n.y(), n.z() + 1);
        const float x1y1z0 = READ_DOG(n.x(), n.y(), n.z() - 1);

        // TODO: Compare scalbnf vs straight up doing the computation
#define USE_SCALBNF 1

#if USE_SCALBNF
        // Uses the cmath implementatino of scalbnf which is should be the same result as the cuda version
        // does scalbnf(x, n) = x * 2^n -- not sure if this is faster than the below in the case of SYCL
        D.x() = scalbnf(x2y1z1 - x0y1z1, -1);
        D.y() = scalbnf(x1y2z1 - x1y0z1, -1);
        D.z() = scalbnf(x1y1z2 - x1y1z0, -1);
#else

        D.x() = 0.5f * (x2y1z1 - x0y1z1);
        D.y() = 0.5f * (x1y2z1 - x1y0z1);
        D.z() = 0.5f * (x1y1z2 - x1y1z0);

#endif

        /* compute Hessian */
        const float x1y1z1 = READ_DOG(n.x(), n.y(), n.z());

#if USE_SCALBNF
        DD.x() = x2y1z1 + x0y1z1 - scalbnf(x1y1z1, 1);
        DD.y() = x1y2z1 + x1y0z1 - scalbnf(x1y1z1, 1);
        DD.z() = x1y1z2 + x1y1z0 - scalbnf(x1y1z1, 1);
#else
        DD.x() = x2y1z1 + x0y1z1 - 2.0f * x1y1z1;
        DD.y() = x1y2z1 + x1y0z1 - 2.0f * x1y1z1;
        DD.z() = x1y1z2 + x1y1z0 - 2.0f * x1y1z1;
#endif

        const float x0y0z1 = READ_DOG(n.x() - 1, n.y() - 1, n.z());
        const float x0y1z0 = READ_DOG(n.x() - 1, n.y(), n.z() - 1);
        const float x0y1z2 = READ_DOG(n.x() - 1, n.y(), n.z() + 1);
        const float x0y2z1 = READ_DOG(n.x() - 1, n.y() + 1, n.z());
        const float x1y0z0 = READ_DOG(n.x(), n.y() - 1, n.z() - 1);
        const float x1y0z2 = READ_DOG(n.x(), n.y() - 1, n.z() + 1);
        const float x1y2z0 = READ_DOG(n.x(), n.y() + 1, n.z() - 1);
        const float x1y2z2 = READ_DOG(n.x(), n.y() + 1, n.z() + 1);
        const float x2y0z1 = READ_DOG(n.x() + 1, n.y() - 1, n.z());
        const float x2y1z0 = READ_DOG(n.x() + 1, n.y(), n.z() - 1);
        const float x2y1z2 = READ_DOG(n.x() + 1, n.y(), n.z() + 1);
        const float x2y2z1 = READ_DOG(n.x() + 1, n.y() + 1, n.z());

#if USE_SCALBNF
        DX.x() = scalbnf(x2y2z1 + x0y0z1 - x0y2z1 - x2y0z1, -2);
        DX.y() = scalbnf(x2y1z2 + x0y1z0 - x0y1z2 - x2y1z0, -2);
        DX.z() = scalbnf(x1y2z2 + x1y0z0 - x1y2z0 - x1y0z2, -2);
#else
        DX.x() = 0.25f * (x2y2z1 + x0y0z1 - x0y2z1 - x2y0z1);
        DX.y() = 0.25f * (x2y1z2 + x0y1z0 - x0y1z2 - x2y1z0);
        DX.z() = 0.25f * (x1y2z2 + x1y0z0 - x1y2z0 - x1y0z2);
#endif

        sycl::vec<float, 3> b;
        float A[3][3];

        /* Solve linear system. */
        A[0][0] = DD.x();
        A[1][1] = DD.y();
        A[2][2] = DD.z();
        A[1][0] = A[0][1] = DX.x();
        A[2][0] = A[0][2] = DX.y();
        A[2][1] = A[1][2] = DX.z();

        b.x() = -D.x();
        b.y() = -D.y();
        b.z() = -D.z();

        if(!solve(A, b))
        {
            d.x() = 0;
            d.y() = 0;
            d.z() = 0;
            break;
        }

        d = b;

        /* If the translation of the keypoint is big, move the keypoint
         * and re-iterate the computation. Otherwise we are all set.
         */
        const int retval = f.refine(d, n, width, height, maxlevel, iter == MAX_ITERATIONS, it.get_global_linear_id());

        if(retval == 1)
        {
            break;
        }
    } while(iter < MAX_ITERATIONS); /* go to next iter */

    if(d.x() >= 1.5f || d.y() >= 1.5f || d.z() >= 1.5f)
    {
        // excessive pixel movement in at least dimension, reject
        return false;
    }

    const float xn = n.x() + d.x();
    const float yn = n.y() + d.y();
    const float sn = n.z() + d.z();

    if(!verify(xn, yn, sn, width, height, maxlevel))
    {
        return false;
    }

#if USE_SCALBNF
    const float contr = v + scalbnf(D.x() * d.x() + D.y() * d.y() + D.z() * d.z(), -1);
#else
    float contr = v + 0.5f * (D.x() * d.x() + D.y() * d.y() + D.z() * d.z());
#endif
    const float tr = DD.x() + DD.y();
    const float det = DD.x() * DD.y() - DX.x() * DX.x();
    const float edgeval = tr * tr / det;

    /* negative determinant => curvatures have different signs -> reject it */
    if(det <= 0.0f)
    {
        return false;
    }

    /* accept-reject extremum */
#if USE_SCALBNF
    if(fabsf(contr) < scalbnf(d_consts->threshold, 1))
#else
    if(fabsf(contr) < (d_consts.threshold * 2.0f))
#endif
    {
        return false;
    }

    /* reject condition: tr(H)^2/det(H) < (r+1)^2/r */
    if(edgeval >= (d_consts->edge_limit + 1.0f) * (d_consts->edge_limit + 1.0f) / d_consts->edge_limit)
    {
        return false;
    }

    ec->xpos = xn;
    ec->ypos = yn;
    ec->lpos = (int)roundf(sn);
    ec->sigma = d_consts->sigma0 * pow(d_consts->sigma_k, sn); // * 2;
    ec->cell = floorf(yn / h_grid_divider) * grid_width + floorf(xn / w_grid_divider);
    // const float sigma_k = powf(2.0f, 1.0f / levels );

    return true;
}

template<int HEIGHT, int sift_mode>
class find_extrema_in_dog
{
  private:
    float** dog;
    int octave;
    int width;
    int height;
    const size_t max_level;
    int* d_number_of_blocks;
    int number_of_blocks;
    const float w_grid_divider;
    const float h_grid_divider;
    const int grid_width;
    const popsift::ConstInfo* d_consts;
    ExtremaCounters* dct;
    DevBuffers* dobuf;
    // const int max_extrema;

  public:
    find_extrema_in_dog(float** dog,
                        int octave,
                        int width,
                        int height,
                        const size_t max_level,
                        int* d_number_of_blocks,
                        int number_of_blocks,
                        const float w_grid_divider,
                        const float h_grid_divider,
                        const int grid_width,
                        const popsift::ConstInfo* d_consts,
                        ExtremaCounters* dct,
                        DevBuffers* dobuf)
      // const int max_extrema)
      : dog(dog)
      , octave(octave)
      , width(width)
      , height(height)
      , max_level(max_level)
      , d_number_of_blocks(d_number_of_blocks)
      , number_of_blocks(number_of_blocks)
      , w_grid_divider(w_grid_divider)
      , h_grid_divider(h_grid_divider)
      , grid_width(grid_width)
      , d_consts(d_consts)
      , dct(dct)
      , dobuf(dobuf)
    // , max_extrema(max_extrema)
    {}

    inline void operator()(sycl::nd_item<3> it) const
    {
        InitialExtremum ec;
        ec.ignore = false;
        const int max_extrema = d_consts->max_extrema;

        if(it.get_global_linear_id() == 0)
        {
            sycl::sub_group sub_group = it.get_sub_group();
            sycl::ext::oneapi::experimental::printf(
              "\n\nNUMBER OF WORK GROUPS %zu -- IN OCTAVE %d -- sub_group size %zu -- max_sub_group_size %zu \n\n ",
              it.get_group_range().size(),
              octave,
              sub_group.get_local_range()[0],
              sub_group.get_max_local_range()[0]);
        }

        bool indicator = find_extrema_in_dog_sub<sift_mode>(
          dog, octave, width, height, max_level, w_grid_divider, h_grid_divider, grid_width, &ec, it, d_consts);

        // if(indicator)
        // {
        //     sycl::ext::oneapi::experimental::printf("\n\t xpos = %f ypos = %f -- lpos = %d -- sigma = %f  -- cell =
        //     %d",
        //                                             ec.xpos,
        //                                             ec.ypos,
        //                                             ec.lpos,
        //                                             ec.sigma,
        //                                             ec.cell);
        // }

        // if (indicator

        // Don't think the tamplate argument does anything
        // uint32_t write_index = extrema_count<HEIGHT>(indicator, &dct.ext_ct[octave]);
        uint32_t write_index = extrema_count(indicator, &dct->ext_ct[octave], it);

        InitialExtremum* d_extrema = dobuf->i_ext_dat[octave];
        int* d_ext_off = dobuf->i_ext_off[octave];

        if(indicator && write_index < max_extrema)
        {
            // sycl::ext::oneapi::experimental::printf(
            //   "\n\t\t xpos = %f ypos = %f -- lpos = %d -- sigma = %f  -- cell = % d ",
            //   ec.xpos,
            //   ec.ypos,
            //   ec.lpos,
            //   ec.sigma,
            //   ec.cell);
            // sycl::ext::oneapi::experimental::printf("indicator = %d -- write_index = %d\n", indicator,
            // write_index);
            ec.write_index = write_index;
            // store the initial extremum in an array
            d_extrema[write_index] = ec;
            // if(write_index == 0)
            //     sycl::ext::oneapi::experimental::printf("ec --> xpos = %f  ypos = %f", ec.xpos, ec.ypos);

            // index for indirect access to d_extrema, to enable
            // access after filtering some initial extrema
            d_ext_off[write_index] = write_index; // not sure how this is usefull... yet...
        }

        // without syncthreads, (0,0) threads may precede some calls to extrema_count()
        // in non-(0,0) threads and increase barrier count too early
        // work-group barrier
        sycl::group_barrier(it.get_group()); // from book -- barrier on the group same as __syncthreads();
        // can also be done for sub-groups by passing that group like __syncwarp

        // We only want one of the threads in a work-group to execute this code

        /// TESTING
        if(it.get_local_linear_id() == 0) // work-item 0 in work-group
        {
            // TOdo make ct and d_number_of_blocks unsigned int
            int ct = sycl::atomic_ref<int,
                                      sycl::memory_order_relaxed,
                                      sycl::memory_scope_device,
                                      sycl::access::address_space::global_space>(*d_number_of_blocks)++;
            // consider using size_t for d_number_of_blocks but 64 bit is more than needed...
            // if(ct >= static_cast<int>(it.get_group_range().size() - 1))
            if(ct >= (number_of_blocks - 1))
            {
                // Final 0 work-item that executes this code, so num_extrema count is finished computing and we ensure
                // it is not larger than the max if it is, it's set to the max value
                sycl::atomic_ref<int,
                                 sycl::memory_order_relaxed,
                                 sycl::memory_scope_device,
                                 sycl::access::address_space::global_space>(dct->ext_ct[octave])
                  .fetch_min(max_extrema);

                sycl::ext::oneapi::experimental::printf(
                  "\n\t Octave: %d extrema_count = %d --> ct = %d && number_of_blocks - 1 = %d  \n",
                  octave,
                  dct->ext_ct[octave],
                  ct,
                  static_cast<int>(it.get_group_range().size()) - 1);
            }
        }
    }
};

// void Pyramid::find_extrema(const Config& conf, sycl::event d_consts_write)
void Pyramid::find_extrema(const Config& conf, sycl::event d_consts_write)
{
    static const int HEIGHT = 4;

    for(int octave = 0; octave < _num_octaves; octave++)
    {
        Octave& oct_obj = _octaves[octave];

        int* num_blocks = getNumberOfBlocks(octave);

        int width = oct_obj.getWidth();
        int height = oct_obj.getHeight();

        fprintf(stderr, "\tWidht=%d, height=%d", width, height);

        // Based on the fact that sub-group is along nd_range[2]:
        // NOTE: should probably change this to be based on the device prefered sub-group multiplier
        // currently same as cuda
        sycl::range local{1, HEIGHT, LOCAL_X};
        sycl::range global{
          (size_t)_levels - 3, (size_t)grid_divide(height, local[1]), (size_t)grid_divide(width, local[2])};

        int work_group_count = grid_divide_cuda(height, local[1]) * grid_divide_cuda(width, local[2]) * (_levels - 3);
        sycl::event dog_done = oct_obj._dog_done_event;

        printf("\nFIND EXTREMA octave %d: Local(%zu, %zu, %zu) --- --- Global(%zu, %zu, %zu) work_group(%d, %d, %d) "
               "Work_group_count = %d\n\n",
               octave,
               local[0],
               local[1],
               local[2],
               global[0],
               global[1],
               global[2],
               (_levels - 3),
               grid_divide_cuda(height, local[1]),
               grid_divide_cuda(width, local[2]),
               work_group_count);

        // Buffer for debugging
        switch(conf.getSiftMode())
        {
            case Config::RefineInLevel:
                printf("RefineInLevel type VLfeat, NOT IMPLEMENTED AS OF NOW");
                // find_extrema_in_dog<HEIGHT, Config::RefineInLevel>
                //   <<<grid, block, 0, oct_str>>>(oct_obj.getDogTexturePoint(),
                //                                 octave,
                //                                 cols,
                //                                 rows,
                //                                 _levels - 1,
                //
                //                                 grid.x * grid.y,
                //                                 oct_obj.getWGridDivider(),
                //                                 oct_obj.getHGridDivider(),
                //                                 conf.getFilterGridSize());
                // POP_SYNC_CHK;
                break;
            default:
                printf("RefineInOctave type popsift default\n");
                oct_obj._extrema_done_event = _device_queue.submit([&](sycl::handler& cgh) {
                    cgh.depends_on({dog_done, d_consts_write, _dobuf_write, _zero_dct, _zero_extrema_num_blocks});
                    cgh.parallel_for(sycl::nd_range{global, local},
                                     find_extrema_in_dog<HEIGHT, Config::RefineInOctave>(oct_obj.getDogArray(),
                                                                                         octave,
                                                                                         width,
                                                                                         height,
                                                                                         _levels - 1,
                                                                                         num_blocks,
                                                                                         work_group_count,
                                                                                         oct_obj.getWGridDivider(),
                                                                                         oct_obj.getHGridDivider(),
                                                                                         conf.getFilterGridSize(),
                                                                                         _d_consts,
                                                                                         _dct,
                                                                                         _dobuf));
                    // _d_consts->max_extrema));
                });
                break;
        }

        _device_queue.wait();

#if false // seems to print similar value (float differences) to the cuda version from the few samples I've compared and
         // the number of extrema is exactly the same for each octave
        _device_queue.single_task([=, dct = _dct, dobuf = _dobuf, max_extrema = _d_consts->max_extrema]() {
            sycl::ext::oneapi::experimental::printf("dct->ext_ct[%d] = %d\n", octave, dct->ext_ct[octave]);
            // For all octaves dct->ext_ct[octave] is 8 times what it should be for sub-group of 8 hance every thread
            // in sub-group must be doing the atomic add but I don't know how to make it stop doing that

            if(octave == 0)
            {
                for(int i = 0; i < 600; ++i)
                {
                    auto dat = &dobuf->i_ext_dat[octave][i];
                    sycl::ext::oneapi::experimental::printf(
                      "\n\t write_index = %d == %d  ---- xpos = %f ypos = %f -- lpos = %d -- sigma = %f  -- cell = %d "
                      "ignore = %d write_index = %d",
                      dobuf->i_ext_off[octave][i],
                      i,
                      dat->xpos,
                      dat->ypos,
                      dat->lpos,
                      dat->sigma,
                      dat->cell,
                      dat->ignore,
                      dat->write_index);
                }
            }
        });
#endif
    }

    _device_queue.wait();
    // fflush(stderr);
    // fflush(stdout);
    // fprintf(stderr, "\n\n\t\tHello there mate how we doin");
}

} // namespace popsift
