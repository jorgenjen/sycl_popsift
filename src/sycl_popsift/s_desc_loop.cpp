// /*
//  * Copyright 2016-2017, Simula Research Laboratory
//  *
//  * This Source Code Form is subject to the terms of the Mozilla Public
//  * License, v. 2.0. If a copy of the MPL was not distributed with this
//  * file, You can obtain one at http://mozilla.org/MPL/2.0/.
//  */
// #include "sycl_popsift/s_desc_loop.hpp"
//
// #include "sycl_popsift/sift_pyramid.hpp"
//
// // #include "s_gradiant.h"
// // #include "sycl_popsift/sift_constants.h" // do I need this one?
//
// #include <sycl/sycl.hpp>
//
// #include <cstdio> // do  I need ?
//
// #undef BLOCK_3_DIMS
//
// // Is used in cuda to say no aliasing (aka this pointer is the only way to access the underlying data in this scope)
// // float* __restrict__ features,
// // __restrict__ is supported by codeplay nvidia extension but not standards c++
// // can also use [[intel::kernel_args_restrict]]
//
// namespace popsift {
//
// static inline void ext_desc_loop_sub(const float ang,
//                                      const Extremum* ext,
//                                      float* __restrict__ features, // should work for codeplay
//                                      float** data,
//                                      const int width,
//                                      const int height,
//                                      sycl::nd_item<3> it)
// {
// #ifndef BLOCK_3_DIMS
//     // const int ix = threadIdx.y;
//     // const int iy = threadIdx.z;
//     // const int tile = (((iy << 2) + ix) << 3); // base of the 8 floats written by this group of 16 threads
//
//     const int ix = it.get_local_id(1);
//     const int iy = it.get_local_id(0);
//     const int tile = (((iy << 2) + ix) << 3); // base of the 8 floats written by this group of 16 threads
// #else
//     // const int ix = (threadIdx.z & 0x3);
//     // const int iy = (threadIdx.z >> 2);
//     // const int tile = (threadIdx.z << 3);
//
//     const int ix = (it.get_local_id(0) & 0x3);
//     const int iy = (it.get_local_id(0) >> 2);
//     const int tile = (it.get_local_id(0) << 3);
//
// #endif
//
//     const float x = ext->xpos;
//     const float y = ext->ypos;
//     const int level = ext->lpos; // old_level;
//     const float sig = ext->sigma;
//     const float SBP = sycl::fabs(DESC_MAGNIFY * sig);
//
//     if(SBP == 0)
//     {
//         return;
//     }
//
//     // const float cos_t = cosf(ang);
//     // const float sin_t = sinf(ang);
//     // float cos_t;
//     // float sin_t;
//     // __sincosf(ang, &sin_t, &cos_t);
//
// #define use_sincos true
// #if use_sincos
//     float cos_t;
//     sycl::multi_ptr<float, sycl::access::address_space::private_space> cos_ptr(&cos_t);
//     float sin_t = sycl::sincos(ang, cos_ptr); // Need to be a multi_ptr (for some reason)
// #else
//     float sin_t = sycl::sin(ang);
//     float cos_t = sycl::cos(ang);
// #endif
//
//     const float csbp = cos_t * SBP;
//     const float ssbp = sin_t * SBP;
//     const float crsbp = cos_t / SBP;
//     const float srsbp = sin_t / SBP;
//
//     // const float2 offsetpt = make_float2(ix - 1.5f, iy - 1.5f);
//     sycl::vec<float, 2> offsetpt(ix - 1.5, iy - 1.5f);
//
//     // The following 2 lines were the primary bottleneck of this kernel
//     // const float ptx = csbp * offsetptx - ssbp * offsetpty + x;
//     // const float pty = csbp * offsetpty + ssbp * offsetptx + y;
//     // const float ptx = ::fmaf(csbp, offsetpt.x(), ::fmaf(-ssbp, offsetpt.y(), x));
//     // const float pty = ::fmaf(csbp, offsetpt.y(), ::fmaf(ssbp, offsetpt.x(), y));
//
//     const float ptx = sycl::fma(csbp, offsetpt.x(), sycl::fma(-ssbp, offsetpt.y(), x));
//     const float pty = sycl::fma(csbp, offsetpt.y(), sycl::fma(-ssbp, offsetpt.x(), y));
//
//     // Less precise version (of ^) BUT FASTER!!
//     // const float ptx = sycl::mad(csbp, offsetpt.x(), sycl::mad(-ssbp, offsetpt.y(), x));
//     // const float pty = sycl::mad(csbp, offsetpt.y(), sycl::mad(-ssbp, offsetpt.x(), y));
//
//     // CURRENT LOCAITON OF CONVERSION
//     const float bsz = sycl::fabs(csbp) + sycl::fabs(ssbp);
//     const int xmin = sycl::max(1, (int)sycl::floor(ptx - bsz));
//     const int ymin = sycl::max(1, (int)sycl::floor(pty - bsz));
//     const int xmax = sycl::min(width - 2, (int)sycl::floor(ptx + bsz));
//     const int ymax = sycl::min(height - 2, (int)sycl::floor(pty + bsz));
//
//     const int wx = xmax - xmin + 1;
//     const int hy = ymax - ymin + 1;
//     const int loops = wx * hy;
//
//     float dpt[9] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
//
//     // This code does not work for both sub_groups and work_group so need a spearate version for work_groups for
//     devices
//     // that has max sub_group for kernel smaller than 32
//     sycl::sub_group sub_group = it.get_sub_group();
//     // for(int i = threadIdx.x; popsift::any(i < loops); i += blockDim.x)
//     for(int i = it.get_local_id(2); sycl::any_of_group(sub_group, i < loops); i += it.get_local_range(2))
//     {
//         if(i >= loops)
//             continue;
//
//         const int ii = i / wx + ymin;
//         const int jj = i % wx + xmin;
//
//         // const float2 d = make_float2(jj - ptx, ii - pty);
//
//         sycl::vec<float, 2> d(jj - ptx, ii - pty);
//
//         // const float nx = crsbp * dx + srsbp * dy;
//         // const float ny = crsbp * dy - srsbp * dx;
//         // const float2 n = make_float2(::fmaf(crsbp, d.x, srsbp * d.y), ::fmaf(crsbp, d.y, -srsbp * d.x));
//         sycl::vec<float, 2> n(sycl::fma(crsbp, d.x(), srsbp * d.y()), sycl::fma(crsbp, d.y(), -srsbp * d.x()));
//         // sycl::vec<2> n(sycl::mad(crsbp, d.x(), srsbp * d.y()), sycl::mad(crsbp, d.y(), -srsbp * d.x())); // faster
//         // version
//         // const float2 nn = abs(n);
//         sycl::vec<float, 2> nn = sycl::fabs(n); // does element wise absolute of n vector
//
//         // #############################################################################
//         // ###########################  CURRENTLY HERE  ################################
//         // #############################################################################
//         //     if(nn.x < 1.0f && nn.y() < 1.0f)
//         //     {
//         //         float mod;
//         //         float th;
//         //         get_gradiant(mod, th, jj, ii, layer_tex, level);
//         //
//         //         const float2 dn = n + offsetpt;
//         //         const float ww = __expf(-scalbnf(dn.x * dn.x + dn.y * dn.y, -3));
//         //         // const float ww  = __expf(-0.125f * (dnx*dnx + dny*dny)); // speedup !
//         //         const float2 w = make_float2(1.0f - nn.x, 1.0f - nn.y);
//         //         const float wgt = ww * w.x * w.y * mod;
//         //
//         //         th -= ang;
//         //         th += (th < 0.0f ? M_PI2 : 0.0f);   //  if (th <  0.0f ) th += M_PI2;
//         //         th -= (th >= M_PI2 ? M_PI2 : 0.0f); //  if (th >= M_PI2) th -= M_PI2;
//         //
//         //         const float tth = __fmul_ru(th, M_4RPI); // th * M_4RPI;
//         //         const int fo0 = (int)floorf(tth);
//         //         const float do0 = tth - fo0;
//         //         const float wgt1 = 1.0f - do0;
//         //         const float wgt2 = do0;
//         //
//         //         int fo = fo0 % DESC_BINS;
//         //
//         //         // maf: multiply-add
//         //         // _ru - round to positive infinity equiv to froundf since always >=0
//         //         dpt[fo] = __fmaf_ru(wgt1, wgt, dpt[fo]);         // dpt[fo]   += (wgt1*wgt);
//         //         dpt[fo + 1] = __fmaf_ru(wgt2, wgt, dpt[fo + 1]); // dpt[fo+1] += (wgt2*wgt);
//         //     }
//         // }
//         // __syncthreads();
//         //
//         // dpt[0] += dpt[8];
//         //
//         // /* reduction here */
//         // for(int i = 0; i < 8; i++)
//         // {
//         //     dpt[i] += popsift::shuffle_down(dpt[i], 16);
//         //     dpt[i] += popsift::shuffle_down(dpt[i], 8);
//         //     dpt[i] += popsift::shuffle_down(dpt[i], 4);
//         //     dpt[i] += popsift::shuffle_down(dpt[i], 2);
//         //     dpt[i] += popsift::shuffle_down(dpt[i], 1);
//         //     dpt[i] = popsift::shuffle(dpt[i], 0);
//         // }
//         //
//         // if(threadIdx.x < 8)
//         // {
//         //     features[tile + threadIdx.x] = dpt[threadIdx.x];
//         // }
//     }
// } //
//
// // Uses the blured pyramid (not the DoG pyramid)
//
// class Ext_desc_loop
// {
//   private:
//     ExtremaCounters* dct;
//     ExtremaBuffers* dbuf;
//     DevBuffers* dobuf;
//     float** data;
//     const int octave;
//     const int width;
//     const int height;
//
//   public:
//     Ext_desc_loop(ExtremaCounters* dct,
//                   ExtremaBuffers* dbuf,
//                   DevBuffers* dobuf,
//                   float** data,
//                   const int octave,
//                   const int width,
//                   const int height)
//       : dct(dct)
//       , dbuf(dbuf)
//       , dobuf(dobuf)
//       , octave(octave)
//       , width(width)
//       , height(height)
//     {}
//
//     inline void operator()(sycl::nd_item<3> it) const
//     {
//         const int o_offset = dct->ori_ps[octave] + it.get_local_id(2);
//         Descriptor* desc = &dbuf->desc[o_offset];
//         const int ext_idx = dobuf->feat_to_ext_map[o_offset];
//         Extremum* ext = dobuf->extrema + ext_idx;
//
//         const int ext_base = ext->idx_ori;
//         const int ori_num = o_offset - ext_base;
//         const float ang = ext->orientation[ori_num];
//
//         ext_desc_loop_sub(ang, ext, desc->features, data, width, height, it);
//     }
// };
//
// inline void Pyramid::start_ext_desc_loop(const int octave, Octave& oct_obj)
// {
//     // dim3 block;
//     // dim3 grid;
//     // grid.x = _hct.ori_ct[octave]; // orientation count for the octave
//     // grid.y = 1;
//     // grid.z = 1;
//
//     if(_hct.ori_ct[octave] == 0)
//         return;
//
// #ifndef BLOCK_3_DIMS
//     // block.x = 32;
//     // block.y = 4;
//     // block.z = 4;
//
//     sycl::range global{4, 4, static_cast<size_t>(_hct.ori_ct[octave])};
//     sycl::range local{4, 4, 32};
// #else
//     // block.x = 32;
//     // block.y = 1;
//     // block.z = 16;
//
//     sycl::range global{16, 1, _hct.ori_ct[octave]};
//     sycl::range local{16, 1, 32};
// #endif
//
//     // ext_desc_loop<<<grid, block, 0, oct_obj.getStream()>>>(
//     //   octave, oct_obj.getDataTexPoint(), oct_obj.getWidth(), oct_obj.getHeight());
//
//     _device_queue.parallel_for(
//       sycl::nd_range{global, local},
//       Ext_desc_loop(_dct, _dbuf, _dobuf, oct_obj.getDataArray(), octave, oct_obj.getWidth(), oct_obj.getHeight()));
//
//     // octave, oct_obj.getDataArray(), oct_obj.getWidth(), oct_obj.getHeight()));
//
//     // POP_SYNC_CHK;
//
//     // return true;
// }
//
// } // namespace popsift
