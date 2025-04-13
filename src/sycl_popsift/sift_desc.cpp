#include "sycl_popsift/common/assist.h"
#include "sycl_popsift/common/debug_macros.hpp"
#include "sycl_popsift/non_sycl/sift_conf.hpp"
#include "sycl_popsift/s_desc_loop.hpp"
#include "sycl_popsift/s_desc_norm_rs.h"
#include "sycl_popsift/s_desc_normalize.h"
#include "sycl_popsift/s_gradient.hpp"
#include "sycl_popsift/sift_constants.hpp"
#include "sycl_popsift/sift_pyramid.hpp"

// #include <sycl/ext/intel/math.hpp>

#include <sycl/sycl.hpp>

// #include <cmath> // including cmath did not work due to conflict so could not use INFINITY
#include <cstdio>
#include <limits>

#undef BLOCK_3_DIMS

// Is used in cuda to say no aliasing (aka this pointer is the only way to access the underlying data in this scope)
// float* __restrict__ features,
// __restrict__ is supported by codeplay nvidia extension but not standards c++
// can also use [[intel::kernel_args_restrict]]

namespace popsift {

static inline void ext_desc_loop_sub(const float ang,
                                     const Extremum* ext,
                                     float* __restrict__ features, // should work for codeplay
                                     float** data,
                                     const int width,
                                     const int height,
                                     sycl::nd_item<3> it)
{
#ifndef BLOCK_3_DIMS
    // const int ix = threadIdx.y;
    // const int iy = threadIdx.z;
    // const int tile = (((iy << 2) + ix) << 3); // base of the 8 floats written by this group of 16 threads

    const int ix = it.get_local_id(1);
    const int iy = it.get_local_id(0);
    const int tile = (((iy << 2) + ix) << 3); // base of the 8 floats written by this group of 16 threads
#else
    // const int ix = (threadIdx.z & 0x3);
    // const int iy = (threadIdx.z >> 2);
    // const int tile = (threadIdx.z << 3);

    const int ix = (it.get_local_id(0) & 0x3);
    const int iy = (it.get_local_id(0) >> 2);
    const int tile = (it.get_local_id(0) << 3);

#endif

    const float x = ext->xpos;
    const float y = ext->ypos;
    const int level = ext->lpos; // old_level;
    const float sig = ext->sigma;
    const float SBP = sycl::fabs(DESC_MAGNIFY * sig);

// #define XPOS 26.643719f
// #define YPOS 185.853836f
#define XPOS 90.565437f
#define YPOS 137.517151f

    // if(x == 451.221741f && y == 305.580322f)
    // if(x == XPOS)
    // if(x == XPOS && y == YPOS)
    // {
    //     // sycl::ext::oneapi::experimental::printf("Tile = %d ", tile);
    //     sycl::ext::oneapi::experimental::printf("idx (%d, %d, %d) -- x = %f y = %f \n",
    //                                             (int)it.get_local_id(2),
    //                                             (int)it.get_local_id(1),
    //                                             (int)it.get_local_id(0),
    //                                             x,
    //                                             y);
    // }

    if(SBP == 0)
    {
        return;
    }

    // const float cos_t = cosf(ang);
    // const float sin_t = sinf(ang);
    // float cos_t;
    // float sin_t;
    // __sincosf(ang, &sin_t, &cos_t);

#define use_sincos true
#if use_sincos
    float cos_t;
    sycl::multi_ptr<float, sycl::access::address_space::private_space> cos_ptr(&cos_t);
    float sin_t = sycl::sincos(ang, cos_ptr); // Need to be a multi_ptr (for some reason)
#else
    float sin_t = sycl::sin(ang);
    float cos_t = sycl::cos(ang);
#endif

    const float csbp = cos_t * SBP;
    const float ssbp = sin_t * SBP;
    const float crsbp = cos_t / SBP;
    const float srsbp = sin_t / SBP;

    // const float2 offsetpt = make_float2(ix - 1.5f, iy - 1.5f);

    // NOTE: This should mby be set to -1 and -1 when not using textures
    // or is it -2 and -2??
    const sycl::vec<float, 2> offsetpt(ix - 1.5, iy - 1.5f);
    // const sycl::vec<float, 2> offsetpt(ix - 2, iy + 1);

    // The following 2 lines were the primary bottleneck of this kernel
    // const float ptx = csbp * offsetptx - ssbp * offsetpty + x;
    // const float pty = csbp * offsetpty + ssbp * offsetptx + y;
    // const float ptx = ::fmaf(csbp, offsetpt.x(), ::fmaf(-ssbp, offsetpt.y(), x));
    // const float pty = ::fmaf(csbp, offsetpt.y(), ::fmaf(ssbp, offsetpt.x(), y));

    const float ptx = sycl::fma(csbp, offsetpt.x(), sycl::fma(-ssbp, offsetpt.y(), x));
    const float pty = sycl::fma(csbp, offsetpt.y(), sycl::fma(ssbp, offsetpt.x(), y));

    // Less precise version (of ^) BUT FASTER!!
    // const float ptx = sycl::mad(csbp, offsetpt.x(), sycl::mad(-ssbp, offsetpt.y(), x));
    // const float pty = sycl::mad(csbp, offsetpt.y(), sycl::mad(-ssbp, offsetpt.x(), y));

    // CURRENT LOCAITON OF CONVERSION
    const float bsz = sycl::fabs(csbp) + sycl::fabs(ssbp);
    const int xmin = sycl::max(1, (int)sycl::floor(ptx - bsz));
    const int ymin = sycl::max(1, (int)sycl::floor(pty - bsz));
    const int xmax = sycl::min(width - 2, (int)sycl::floor(ptx + bsz));
    const int ymax = sycl::min(height - 2, (int)sycl::floor(pty + bsz));

    const int wx = xmax - xmin + 1;
    const int hy = ymax - ymin + 1;
    const int loops = wx * hy;

    // if(x == XPOS && y == YPOS)
    // {
    //     sycl::ext::oneapi::experimental::printf(
    //       "ang=%.3f | cos=%.3f sin=%.3f | csbp=%.3f ssbp=%.3f | crsbp=%.3f srsbp=%.3f | offset=(%.3f,%.3f) | "
    //       "pt=(%.3f,%.3f) | bsz=%.3f | \nx=[%d,%d] y=[%d,%d] | wx=%d hy=%d | loops=%d\n\n",
    //       ang,
    //       cos_t,
    //       sin_t,
    //       csbp,
    //       ssbp,
    //       crsbp,
    //       srsbp,
    //       offsetpt.x(),
    //       offsetpt.y(),
    //       ptx,
    //       pty,
    //       bsz,
    //       xmin,
    //       xmax,
    //       ymin,
    //       ymax,
    //       wx,
    //       hy,
    //       loops);
    // }

    float dpt[9] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    // TODO: This code does not work for both sub_groups and work_group so need a spearate version for work_groups for
    // devices that has max sub_group for kernel smaller than 32

    // for(int i = threadIdx.x; popsift::any(i < loops); i += blockDim.x)
    for(int i = it.get_local_id(2); sycl::any_of_group(it.get_sub_group(), i < loops); i += it.get_local_range(2))
    {
        if(i >= loops)
            continue;

        const int ii = i / wx + ymin;
        const int jj = i % wx + xmin;

        // if(x == XPOS && y == YPOS)
        //     sycl::ext::oneapi::experimental::printf(
        //       "ii = %d / %d + %d = %d\n jj = %d / %d + %d = %d\n\n", i, wx, ymin, ii, i, wx, xmin, jj);

        // const float2 d = make_float2(jj - ptx, ii - pty);

        const sycl::vec<float, 2> d(jj - ptx, ii - pty);

        // const float nx = crsbp * dx + srsbp * dy;
        // const float ny = crsbp * dy - srsbp * dx;
        // const float2 n = make_float2(::fmaf(crsbp, d.x, srsbp * d.y), ::fmaf(crsbp, d.y, -srsbp * d.x));
        const sycl::vec<float, 2> n(sycl::fma(crsbp, d.x(), srsbp * d.y()), sycl::fma(crsbp, d.y(), -srsbp * d.x()));
        // sycl::vec<2> n(sycl::mad(crsbp, d.x(), srsbp * d.y()), sycl::mad(crsbp, d.y(), -srsbp * d.x())); // faster
        // version
        // const float2 nn = abs(n);
        const sycl::vec<float, 2> nn = sycl::fabs(n); // does element wise absolute of n vector

        // #############################################################################
        // ###########################  CURRENTLY HERE  ################################
        // #############################################################################
        if(nn.x() < 1.0f && nn.y() < 1.0f)
        {
            float mod;
            float th;

            // if(x == XPOS && y == YPOS)
            // {
            //     // if(it.get_local_linear_id() == 40)
            //     if(jj == 99 && ii == 135 && it.get_local_linear_id() == 40)
            //         sycl::ext::oneapi::experimental::printf(
            //           "block = %d --- radient center read (%d, %d) w=%d - h=%d - level = %d\n",
            //           (int)it.get_global_linear_id(),
            //           jj,
            //           ii,
            //           width,
            //           height,
            //           level);
            //     // if(it.get_local_linear_id() == 40)
            //     if(jj == 99 && ii == 135 && it.get_local_linear_id() == 40)
            //         get_gradient(mod, th, jj, ii, width, height, data, level);
            // }
            get_gradient(mod, th, jj, ii, width, height, data, level);

            const sycl::vec<float, 2> dn = n + offsetpt;
            // const float ww = __expf(-scalbnf(dn.x * dn.x + dn.y * dn.y, -3));
            // // const float ww  = __expf(-0.125f * (dnx*dnx + dny*dny)); // speedup !
            // const float2 w = make_float2(1.0f - nn.x, 1.0f - nn.y);
            // const float wgt = ww * w.x * w.y * mod;

            const float ww = sycl::exp(sycl::ldexp(dn.x() * dn.x() + dn.y() * dn.y(), -3));
            // const float ww = sycl::exp(sycl::ldexp(sycl::dot(dn, dn), -3)); // Should be same as above
            const sycl::vec<float, 2> w(1.0f - nn.x(), 1.0f - nn.y());
            const float wgt = ww * w.x() * w.y() * mod;

            th -= ang;
            th += (th < 0.0f ? M_PI2 : 0.0f);   //  if (th <  0.0f ) th += M_PI2;
            th -= (th >= M_PI2 ? M_PI2 : 0.0f); //  if (th >= M_PI2) th -= M_PI2;

            // const float tth = __fmul_ru(th, M_4RPI); // th * M_4RPI;
            // const int fo0 = (int)floorf(tth);

            // Could not get the extnsion to work due to conflict with sycl math library
            // const float tth = sycl::ext::intel::math::fmul_ru(th, M_4RPI); // intel extension
            const float tth = sycl::nextafter(
              th * M_4RPI, std::numeric_limits<float>::infinity()); // not exactly the same as it always rounds

            // // Think this inline assembly would work as well but only for cuda backend so would need to template
            // // funtion and only use this version if cuda inside a constexpr if statements
            // // Does not work as it stand now however
            // float tth = 0.0f;
            // // asm volatile("fmul.ru.f32 %0, %1, %2;" : "=f"(tth) : "f"(th), "f"(M_4RPI));

            // Might be possible to set rouding mode (global) but does not seem to be a good way
            // seems like vecotr has some options as they mention rouding mode but I can figure out how to set rounding
            // mode for the vecor or the operation on the vector

            const int fo0 = static_cast<int>(sycl::floor(tth));

            const float do0 = tth - fo0;
            const float wgt1 = 1.0f - do0;
            const float wgt2 = do0;

            int fo = fo0 % DESC_BINS;

            // maf: multiply-add
            // _ru - round to positive infinity equiv to froundf since always >=0
            // dpt[fo] = __fmaf_ru(wgt1, wgt, dpt[fo]);         // dpt[fo]   += (wgt1*wgt);
            // dpt[fo + 1] = __fmaf_ru(wgt2, wgt, dpt[fo + 1]); // dpt[fo+1] += (wgt2*wgt);

            // Could not get these to not conflict with math in sycl
            // dpt[fo] = sycl::ext::intel::math::fma_ru(wgt1, wgt, dpt[fo]); // dpt[fo]   += (wgt1*wgt); // rounded up
            // dpt[fo + 1] =
            //   sycl::ext::intel::math::fma_ru(wgt2, wgt, dpt[fo + 1]); // dpt[fo+1] += (wgt2*wgt); // rounded up

            // Attempt to use inline assembly but it did not work...
            // asm volatile("fma.ru.f32 %0, %1, %2, %3;" : "=f"(dpt[fo]) : "f"(wgt1), "f"(wgt), "f"(dpt[fo]));
            // asm volatile("fma.ru.f32 %0, %1, %2, %3;" : "=f"(dpt[fo + 1]) : "f"(wgt2), "f"(wgt), "f"(dpt[fo + 1]));

            // Precise version
            // dpt[fo] = sycl::nextafter(sycl::fma(wgt1, wgt, dpt[fo]), INFINITY);
            // dpt[fo + 1] = sycl::nextafter(sycl::fma(wgt2, wgt, dpt[fo + 1]), INFINITY);

            // Not sure if we need nextafter to try to do rounding to wards postiive infinity
            // it does however always round...
            dpt[fo] = sycl::nextafter(sycl::mad(wgt1, wgt, dpt[fo]), std::numeric_limits<float>::infinity());
            dpt[fo + 1] = sycl::nextafter(sycl::mad(wgt2, wgt, dpt[fo + 1]), std::numeric_limits<float>::infinity());
        }
    }
    // __syncthreads();
    sycl::group_barrier(it.get_group());

    dpt[0] += dpt[8];

    /* reduction here */
    for(int i = 0; i < 8; i++)
    {
        // dpt[i] += popsift::shuffle_down(dpt[i], 16);
        // dpt[i] += popsift::shuffle_down(dpt[i], 8);
        // dpt[i] += popsift::shuffle_down(dpt[i], 4);
        // dpt[i] += popsift::shuffle_down(dpt[i], 2);
        // dpt[i] += popsift::shuffle_down(dpt[i], 1);
        // dpt[i] = popsift::shuffle(dpt[i], 0);

        dpt[i] = sycl::reduce_over_group(it.get_sub_group(), dpt[i], sycl::plus<float>());
    }

    // if(threadIdx.x < 8)
    // {
    //     features[tile + threadIdx.x] = dpt[threadIdx.x];
    // }

    // Write the 8 results asigning one work-item to do the job 24 does nothing here
    if(it.get_local_id(2) < 8)
    {
        features[tile + it.get_local_id(2)] = dpt[it.get_local_id(2)];
    }
}

// Uses the blured pyramid (not the DoG pyramid)

class Ext_desc_loop
{
  private:
    ExtremaCounters* dct;
    ExtremaBuffers* dbuf;
    DevBuffers* dobuf;
    float** data;
    const int octave;
    const int width;
    const int height;

  public:
    Ext_desc_loop(ExtremaCounters* dct,
                  ExtremaBuffers* dbuf,
                  DevBuffers* dobuf,
                  float** data,
                  const int octave,
                  const int width,
                  const int height)
      : dct(dct)
      , dbuf(dbuf)
      , dobuf(dobuf)
      , data(data)
      , octave(octave)
      , width(width)
      , height(height)
    {}

    inline void operator()(sycl::nd_item<3> it) const
    {
        const int o_offset = dct->ori_ps[octave] + it.get_group(2);
        // if(octave == 1 && it.get_local_id(2) == 6)
        // {
        //     sycl::ext::oneapi::experimental::printf("Data value ja %f", data);
        // }
        Descriptor* desc = &dbuf->desc[o_offset];
        const int ext_idx = dobuf->feat_to_ext_map[o_offset];
        Extremum* ext = dobuf->extrema + ext_idx;

        const int ext_base = ext->idx_ori;
        const int ori_num = o_offset - ext_base;
        const float ang = ext->orientation[ori_num];

        ext_desc_loop_sub(ang, ext, desc->features, data, width, height, it);
    }
};

inline void Pyramid::start_ext_desc_loop(const int octave, Octave& oct_obj)
{
    // dim3 block;
    // dim3 grid;
    // grid.x = _hct.ori_ct[octave]; // orientation count for the octave
    // grid.y = 1;
    // grid.z = 1;

    if(_hct.ori_ct[octave] == 0)
        return;

#ifndef BLOCK_3_DIMS
    // block.x = 32;
    // block.y = 4;
    // block.z = 4;

    sycl::range global{4, 4, static_cast<size_t>(_hct.ori_ct[octave] * 32)};
    sycl::range local{4, 4, 32};
#else
    // block.x = 32;
    // block.y = 1;
    // block.z = 16;

    sycl::range global{16, 1, _hct.ori_ct[octave]};
    sycl::range local{16, 1, 32};
#endif

    // ext_desc_loop<<<grid, block, 0, oct_obj.getStream()>>>(
    //   octave, oct_obj.getDataTexPoint(), oct_obj.getWidth(), oct_obj.getHeight());

    fprintf(stderr,
            "global(%zu, %zu, %zu) -- local(%zu, %zu, %zu)\n\n",
            global[0],
            global[1],
            global[2],
            local[0],
            local[1],
            local[2]);

    _device_queue.parallel_for(
      sycl::nd_range{global, local},
      Ext_desc_loop(_dct, _dbuf, _dobuf, oct_obj.getDataArray(), octave, oct_obj.getWidth(), oct_obj.getHeight()));

    // octave, oct_obj.getDataArray(), oct_obj.getWidth(), oct_obj.getHeight()));

    // POP_SYNC_CHK;

    // return true;
}

} // namespace popsift

void popsift::Pyramid::descriptors(const Config& conf)
{
    readDescCountersFromDevice().wait();

    for(int octave = _num_octaves - 1; octave >= 0; octave--)
    // for( int octave=0; octave<_num_octaves; octave++ )
    {
        if(_hct.ori_ct[octave] != 0)
        {
            Octave& oct_obj = _octaves[octave];

            if(conf.getDescMode() == Config::Loop)
            {
                // Default
                start_ext_desc_loop(octave, oct_obj);
            }
            else if(conf.getDescMode() == Config::VLFeat_Desc)
            {
                // start_ext_desc_vlfeat(octave, oct_obj);
            }
            else
            {
                POP_FATAL("not yet");
            }
            // cuda::event_record(oct_obj.getEventDescDone(), oct_obj.getStream(), __FILE__, __LINE__);
            // cuda::event_wait(oct_obj.getEventDescDone(), _download_stream, __FILE__, __LINE__);
        }
    }

    if(_hct.ori_total == 0)
    {
        // cerr << "Warning: no descriptors extracted" << endl;
        fprintf(stderr, "Warning: no descriptors extracted\n");
        return;
    }
    // dim3 block;
    // dim3 grid;
    // grid.x = popsift::grid_divide(_hct.ori_total, 32);
    // block.x = 32;
    // block.y = 32;
    // block.z = 1;

    sycl::range global{32, static_cast<size_t>(popsift::grid_divide(_hct.ori_total, 32))};
    sycl::range local{32, 32};

    if(conf.getUseRootSift())
    {
        // DEFAULT

        _device_queue.wait(); // should use events instead

        _device_queue.parallel_for(sycl::nd_range{global, local},
                                   Normalize_histogram<NormalizeRootSift>(_dbuf_host.desc, _d_consts, _hct.ori_total));

        // normalize_histogram<NormalizeRootSift><<<grid, block, 0, _download_stream>>>();
        // POP_SYNC_CHK;
    }
    else
    {
        // normalize_histogram<NormalizeL2><<<grid, block, 0, _download_stream>>>();
        // POP_SYNC_CHK;
    }

    _device_queue.wait();

    // // START OF PRINT OUT THAT I DID NOT BOTHER TO FINISH
    // _device_queue.single_task([=, dct = _dct]() {
    //     sycl::ext::oneapi::experimental::printf("\n\n");
    //
    //     for(int i = 0; i < dct->ori_total; ++i)
    //     {
    //         for(int j = 0; j < 128; ++j)
    //         {
    //             sycl::ext::oneapi::experimental::printf();
    //         }
    //     }
    // })
}
