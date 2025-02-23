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

// template<int HEIGHT> // did not do anything
// Must take care here as sub-group will not be 32 in all cases like a warp in cuda
// should also make the blocks based on sub-group multiplier(idk what the name was) preference of the device used
// Not sure why indicator was not bool ??
static inline uint32_t extrema_count(bool indicator, int* extrema_counter, sycl::nd_item<3>& it)
{
    // sub_group is undergoing change and not recomended to use but seems most fitting in this case
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

        // The atomic add returns the old value in extrema_coutner before the addition which is considered the base
        // As each thread uses this and adds to it's own counter (write_index) how many of the threads in the
        // sub-group before it had it's indicator to true
        write_index = sycl::atomic_ref<int,
                                       sycl::memory_order_relaxed,
                                       sycl::memory_scope_device,
                                       sycl::access::address_space::global_space>(*extrema_counter) += ct;
    }

    // work-item 0 broadcassts to all other same as leader work-item
    // everyone now get's the base value that they can add to
    write_index = sycl::group_broadcast(sub_group, write_index, 0);

    // Adds the sum of set bits in mask that has sub_grop local id lower than the current (exclusive)
    //  this provides the 0 result and every result up to ct
    write_index += sycl::popcount(mask & ((1 << sub_group.get_local_id()[0]) - 1)); // breaks if USE_MASK != 1

    return write_index;
}

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
// Full clamping // Probs better of using sycl::clamp here
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
                                    InitialExtremum& ec,
                                    sycl::nd_item<3>& it,
                                    const popsift::ConstInfo* d_consts)
{
    ec.xpos = 0.0f;
    ec.ypos = 0.0f;
    ec.lpos = 0;
    ec.sigma = 0.0f;

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

    ec.xpos = xn;
    ec.ypos = yn;
    ec.lpos = (int)roundf(sn);
    ec.sigma = d_consts->sigma0 * pow(d_consts->sigma_k, sn); // * 2;
    ec.cell = floorf(yn / h_grid_divider) * grid_width + floorf(xn / w_grid_divider);
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
                        ExtremaCounters* dct)
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
    {}

    inline void operator()(sycl::nd_item<3> it) const
    {
        InitialExtremum ec;
        ec.ignore = false;

        bool indicator = find_extrema_in_dog_sub<sift_mode>(
          dog, octave, width, height, max_level, w_grid_divider, h_grid_divider, grid_width, ec, it, d_consts);

        // Don't think the tamplate argument does anything
        // uint32_t write_index = extrema_count<HEIGHT>(indicator, &dct.ext_ct[octave]);
        uint32_t write_index = extrema_count(indicator, &dct->ext_ct[octave], it);

        //     InitialExtremum* d_extrema = dobuf.i_ext_dat[octave];
        //     int* d_ext_off = dobuf.i_ext_off[octave];
        //
        //     if(indicator && write_index < d_consts.max_extrema)
        //     {
        //         ec.write_index = write_index;
        //         // store the initial extremum in an array
        //         d_extrema[write_index] = ec;
        //
        //         // index for indirect access to d_extrema, to enable
        //         // access after filtering some initial extrema
        //         d_ext_off[write_index] = write_index;
        //     }
        //
        //     // without syncthreads, (0,0) threads may precede some calls to extrema_count()
        //     // in non-(0,0) threads and increase barrier count too early
        //     // __syncthreads();
        //     sycl::group_barrier(it.get_group()); // from book -- barrier on the group same as __syncthreads();
        //                                          // can also be done for sub-groups by passing that group like
        //                                          __syncwarp()
        //
        //     if(threadIdx.x == 0 && threadIdx.y == 0)
        //     {
        //         int ct = atomicAdd(d_number_of_blocks, 1);
        //         if(ct >= number_of_blocks - 1)
        //         {
        //             int num_ext = atomicMin(&dct.ext_ct[octave], d_consts.max_extrema);
        //             // printf( "Block %d,%d,%d num ext %d\n", blockIdx.x, blockIdx.y, blockIdx.z,
        //             dct.ext_ct[octave]
        //             );
        //         }
        //     }
    }
};

void Pyramid::find_extrema(const Config& conf, std::vector<sycl::event> dependencies, sycl::event d_consts_write)
{
    static const int HEIGHT = 4;

    for(int octave = 0; octave < _num_octaves; octave++)
    {
        if(octave > 0)
            break;
        Octave& oct_obj = _octaves[octave];

        // int* extrema_num_blocks = getNumberOfBlocks(octave); // not ready for this :C

        // dim3 block(32, HEIGHT);
        // dim3 grid;
        // grid.x = grid_divide(cols, block.x);
        // grid.y = grid_divide(rows, block.y);
        // grid.z = _levels - 3;

        // cudaStream_t oct_str = oct_obj.getStream();

        int* num_blocks = getNumberOfBlocks(octave);

        int width = oct_obj.getWidth();
        int height = oct_obj.getHeight();

        // Think z needs to be same for global and local (local needs to divide global perfectly)
        // z == 1 does also compile... strange that
        // sycl::range local{32, HEIGHT, 1};
        // sycl::range local{32, HEIGHT, (size_t)_levels - 3};
        // sycl::range local{LOCAL_X, HEIGHT, (size_t)_levels - 3};
        // // sycl::range local{LOCAL_X, HEIGHT, 1};
        // sycl::range global{
        //   (size_t)grid_divide(width, local.get(0)), (size_t)grid_divide(height, local.get(1)), (size_t)_levels -
        //   3};

        // Based on the fact that sub-group is along nd_range[2]:
        sycl::range local{1, HEIGHT, LOCAL_X};
        sycl::range global{
          (size_t)_levels - 3, (size_t)grid_divide(height, local.get(1)), (size_t)grid_divide(width, local.get(0))};

        printf("\nFIND EXTREMA: Local(%zu, %zu, %zu) --- --- Global(%zu, %zu, %zu)\n\n",
               local[0],
               local[1],
               local[2],
               global[0],
               global[1],
               global[2]);

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
                _device_queue.submit([&](sycl::handler& cgh) {
                    cgh.depends_on({dependencies[octave], d_consts_write});
                    cgh.parallel_for(sycl::nd_range{global, local},
                                     find_extrema_in_dog<HEIGHT, Config::RefineInOctave>(oct_obj.getDogArray(),
                                                                                         octave,
                                                                                         width,
                                                                                         height,
                                                                                         _levels - 1,
                                                                                         num_blocks,
                                                                                         global.get(0) * global.get(1),
                                                                                         oct_obj.getWGridDivider(),
                                                                                         oct_obj.getHGridDivider(),
                                                                                         conf.getFilterGridSize(),
                                                                                         _d_consts,
                                                                                         _dct));
                });
                break;
        }

        // cuda::event_record(oct_obj.getEventExtremaDone(), oct_str, __FILE__, __LINE__);
    }
    _device_queue.wait();
    // fflush(stderr);
    // fflush(stdout);
    // fprintf(stderr, "\n\n\t\tHello there mate how we doin");
}

} // namespace popsift
