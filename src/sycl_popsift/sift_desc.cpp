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
#include <fstream> // FOR file append
#include <iostream>
#include <limits>

#undef BLOCK_3_DIMS

// Is used in cuda to say no aliasing (aka this pointer is the only way to access the underlying data in this scope)
// float* __restrict__ features,
// __restrict__ is supported by codeplay nvidia extension but not standard c++
// can also use [[intel::kernel_args_restrict]]

namespace popsift {

template<bool UseLocalAccessor = false, typename... Args>
static inline void ext_desc_loop_sub(const float ang,
                                     const Extremum* ext,
                                     float* __restrict__ features, // should work for codeplay
                                     float** data,
                                     const int width,
                                     const int height,
                                     sycl::nd_item<3> it,
                                     Args&&... args) // Optional: local_accessor (only one or none works)
{
#ifndef BLOCK_3_DIMS
    const int ix = it.get_local_id(1);
    const int iy = it.get_local_id(0);
    const int tile = (((iy << 2) + ix) << 3); // base of the 8 floats written by this group of 16 threads
#else
    const int ix = (it.get_local_id(0) & 0x3);
    const int iy = (it.get_local_id(0) >> 2);
    const int tile = (it.get_local_id(0) << 3);

#endif

    const float x = ext->xpos;
    const float y = ext->ypos;
    const int level = ext->lpos; // old_level;
    const float sig = ext->sigma;
    const float SBP = sycl::fabs(DESC_MAGNIFY * sig);

#define XPOS 90.565437f
#define YPOS 137.517151f

    if(SBP == 0)
    {
        return;
    }

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

    const sycl::vec<float, 2> offsetpt(ix - 1.5, iy - 1.5f);

// Not sure if using sycl::mad is precise enough but seems to be significantly faster than fma
#define USE_MAD false
#if USE_MAD
    // Less precise version (of fma) BUT FASTER!!
    const float ptx = sycl::mad(csbp, offsetpt.x(), sycl::mad(-ssbp, offsetpt.y(), x));
    const float pty = sycl::mad(csbp, offsetpt.y(), sycl::mad(ssbp, offsetpt.x(), y));
#else
    const float ptx = sycl::fma(csbp, offsetpt.x(), sycl::fma(-ssbp, offsetpt.y(), x));
    const float pty = sycl::fma(csbp, offsetpt.y(), sycl::fma(ssbp, offsetpt.x(), y));
#endif

    const float bsz = sycl::fabs(csbp) + sycl::fabs(ssbp);
    const int xmin = sycl::max(1, (int)sycl::floor(ptx - bsz));
    const int ymin = sycl::max(1, (int)sycl::floor(pty - bsz));
    const int xmax = sycl::min(width - 2, (int)sycl::floor(ptx + bsz));
    const int ymax = sycl::min(height - 2, (int)sycl::floor(pty + bsz));

    const int wx = xmax - xmin + 1;
    const int hy = ymax - ymin + 1;
    const int loops = wx * hy;

    float dpt[9] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    // TODO: This code does not work for both sub_groups and work_group so need a spearate version for work_groups
    // for devices that has max sub_group for kernel smaller than 32

    // for(int i = threadIdx.x; popsift::any(i < loops); i += blockDim.x)
    for(int i = it.get_local_id(2); sycl::any_of_group(it.get_sub_group(), i < loops); i += it.get_local_range(2))
    {
        if(i >= loops)
            continue;

        const int ii = i / wx + ymin;
        const int jj = i % wx + xmin;

        const sycl::vec<float, 2> d(jj - ptx, ii - pty);

#if USE_MAD
        const sycl::vec<float, 2> n(sycl::mad(crsbp, d.x(), srsbp * d.y()), sycl::mad(crsbp, d.y(), -srsbp * d.x()));
#else
        const sycl::vec<float, 2> n(sycl::fma(crsbp, d.x(), srsbp * d.y()), sycl::fma(crsbp, d.y(), -srsbp * d.x()));
#endif
        const sycl::vec<float, 2> nn = sycl::fabs(n); // does element wise absolute of n vector

        if(nn.x() < 1.0f && nn.y() < 1.0f)
        {
            float mod;
            float th;

            get_gradient(mod, th, jj, ii, width, height, data, level);
            // mod = 2.1;
            // th = 1.1;

            // SEEMS TO BE CORRECT UP UNTIL THIS POINT JUST SMALL DEVIATIONS DUE TO FLOAT DIFFERENCES

            const sycl::vec<float, 2> dn = n + offsetpt;
            // const float ww = __expf(-scalbnf(dn.x * dn.x + dn.y * dn.y, -3));
            // // const float ww  = __expf(-0.125f * (dnx*dnx + dny*dny)); // speedup !
            // const float2 w = make_float2(1.0f - nn.x, 1.0f - nn.y);
            // const float wgt = ww * w.x * w.y * mod;

            // Cant include cmath makes nextafter ambigous...
            // const float ww = sycl::exp(-scalbnf(dn.x() * dn.x() + dn.y() * dn.y(), -3));
            // const float ww = sycl::exp(-sycl::ldexp(dn.x() * dn.x() + dn.y() * dn.y(), -3));

            // Using sycl::dot seems to be slightly faster than the version above (on average)
            const float ww = sycl::exp(-sycl::ldexp(sycl::dot(dn, dn), -3)); // Verified Identical as the one above
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
            // seems like vecotr has some options as they mention rouding mode but I can figure out how to set
            // rounding mode for the vecor or the operation on the vector

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
            // dpt[fo] = sycl::ext::intel::math::fma_ru(wgt1, wgt, dpt[fo]); // dpt[fo]   += (wgt1*wgt); // rounded
            // up dpt[fo + 1] =
            //   sycl::ext::intel::math::fma_ru(wgt2, wgt, dpt[fo + 1]); // dpt[fo+1] += (wgt2*wgt); // rounded up

            // Attempt to use inline assembly but it did not work...
            // asm volatile("fma.ru.f32 %0, %1, %2, %3;" : "=f"(dpt[fo]) : "f"(wgt1), "f"(wgt), "f"(dpt[fo]));
            // asm volatile("fma.ru.f32 %0, %1, %2, %3;" : "=f"(dpt[fo + 1]) : "f"(wgt2), "f"(wgt), "f"(dpt[fo +
            // 1]));

#if USE_MAD

            // Not sure if we need nextafter to try to do rounding to wards postiive infinity
            // it does however always round...
            dpt[fo] = sycl::nextafter(sycl::mad(wgt1, wgt, dpt[fo]), std::numeric_limits<float>::infinity());
            dpt[fo + 1] = sycl::nextafter(sycl::mad(wgt2, wgt, dpt[fo + 1]), std::numeric_limits<float>::infinity());
#else
            // Precise version
            dpt[fo] = sycl::nextafter(sycl::fma(wgt1, wgt, dpt[fo]), std::numeric_limits<float>::infinity());
            dpt[fo + 1] = sycl::nextafter(sycl::fma(wgt2, wgt, dpt[fo + 1]), std::numeric_limits<float>::infinity());
#endif
        }
    }
    sycl::group_barrier(it.get_group());

    dpt[0] += dpt[8];

    if constexpr(!UseLocalAccessor)
    {
        // Default case using sub_group
        for(int i = 0; i < 8; i++)
        {
            dpt[i] = sycl::reduce_over_group(it.get_sub_group(), dpt[i], sycl::plus<float>());
        }

        // Write the 8 results asigning one work-item to do the job 24 does nothing here
        if(it.get_local_id(2) < 8)
        {
            features[tile + it.get_local_id(2)] = dpt[it.get_local_id(2)];
        }
    }
    else
    {
        // Local memory accessor of type sycl::local_accessor<float, 1>
        auto& sum = std::get<0>(std::forward_as_tuple(args...));

        // Where each row starts each use 39 --> 32 for compute and 7 for storing prev values
        const int base = (it.get_local_linear_id() >> 5) * 39;
        for(int i = 0; i < 8; i++)
        {
            // shifted to have result in 0 - 7 pos in shared memory and write in one go
            sum[base + it.get_local_id(2) + i] = dpt[i];
            sycl::group_barrier(it.get_group());

            // Stride is 16, 8, 4, 2, 1
            for(int stride = it.get_local_range(2) / 2; stride > 0; stride >>= 1)
            {
                if(it.get_local_id(2) < stride)
                    sum[base + it.get_local_id(2) + i] += sum[base + it.get_local_id(2) + i + stride];

                sycl::group_barrier(it.get_group());
            }
        }

        // Write the 8 results asigning one work-item to do the job 24 does nothing here
        if(it.get_local_id(2) < 8)
        {
            features[tile + it.get_local_id(2)] = sum[base + it.get_local_id(2)];
        }
    }
}
// can have different versions based on Local_mem size to do mix of sub_group and shared mem
// Say sub_group is 8 so we can do 3 sub_group reductions and use shared memory to sum result
// Then repeat for the 8 different values

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

    // inline void operator()(sycl::nd_item<3> it, Local_mem&&... sum) const
    inline void operator()(sycl::nd_item<3> it) const
    {
        const int o_offset = dct->ori_ps[octave] + it.get_group(2);
        // if(it.get_local_linear_id() == 0)
        //     sycl::ext::oneapi::experimental::printf(
        //       "\n\to_offset = %d -- = %d + %d\n", o_offset, dct->ori_ps[octave], it.get_group(2));

        Descriptor* desc = &dbuf->desc[o_offset];
        const int ext_idx = dobuf->feat_to_ext_map[o_offset];
        Extremum* ext = dobuf->extrema + ext_idx;

        const int ext_base = ext->idx_ori;
        const int ori_num = o_offset - ext_base;
        const float ang = ext->orientation[ori_num];

        // Default case
        ext_desc_loop_sub(ang, ext, desc->features, data, width, height, it);
    }
};

// Same as above but using local memory
// Could not find a way to template it to one class without having shared memroy as an attribute and passed
class Ext_desc_loop_local_mem
{
  private:
    sycl::local_accessor<float, 1> sum;
    ExtremaCounters* dct;
    ExtremaBuffers* dbuf;
    DevBuffers* dobuf;
    float** data;
    const int octave;
    const int width;
    const int height;

  public:
    Ext_desc_loop_local_mem(sycl::local_accessor<float, 1> sum,
                            ExtremaCounters* dct,
                            ExtremaBuffers* dbuf,
                            DevBuffers* dobuf,
                            float** data,
                            const int octave,
                            const int width,
                            const int height)
      : sum(sum)
      , dct(dct)
      , dbuf(dbuf)
      , dobuf(dobuf)
      , data(data)
      , octave(octave)
      , width(width)
      , height(height)
    {}

    inline void operator()(sycl::nd_item<3> it) const

    {
        // Seems like something is wrong with o_offset
        const int o_offset = dct->ori_ps[octave] + it.get_group(2);

        Descriptor* desc = &dbuf->desc[o_offset];
        const int ext_idx = dobuf->feat_to_ext_map[o_offset];
        Extremum* ext = dobuf->extrema + ext_idx;

        const int ext_base = ext->idx_ori;
        const int ori_num = o_offset - ext_base;
        const float ang = ext->orientation[ori_num];

        ext_desc_loop_sub<true>(ang, ext, desc->features, data, width, height, it, sum);
    }
};

class sub_group_desc_loop; // To check sub_group size for kernel

inline void Pyramid::start_ext_desc_loop(const int octave, Octave& oct_obj, bool use_sub_group)
{
    if(_hct.ori_ct[octave] == 0)
        return;

    fprintf(stderr,
            "Orientation count for octave(%d) = %d -- prefix sum for octave = %d\n",
            octave,
            _hct.ori_ct[octave],
            _hct.ori_ps[octave]);

#ifndef BLOCK_3_DIMS
    sycl::range global{4, 4, static_cast<size_t>(_hct.ori_ct[octave] * 32)};
    sycl::range local{4, 4, 32};
#else

    // Unverified
    sycl::range global{16, 1, _hct.ori_ct[octave]};
    sycl::range local{16, 1, 32};
#endif

    // fprintf(stderr, "using global(%)
    // Print global range dimensions
    // fprintf(stderr, "Global range: [%zu, %zu, %zu]\n", global[0], global[1], global[2]);

    // Print local range dimensions
    // fprintf(stderr, "Local range: [%zu, %zu, %zu]\n", local[0], local[1], local[2]);

    if(use_sub_group)
    {
        // fprintf(stderr, "I'M RUNNING SUB GROUP MODE\n");
        _device_queue.parallel_for<sub_group_desc_loop>(
          sycl::nd_range{global, local},
          Ext_desc_loop(_dct, _dbuf, _dobuf, oct_obj.getDataArray(), octave, oct_obj.getWidth(), oct_obj.getHeight()));
    }
    else
    {
        // fprintf(stderr, "I'M RUNNING THE LOCLAL ACCESSOR STUFS\n");
        _device_queue.submit([&](sycl::handler& cgh) {
            // need 7 for storing the older result values final is stored in current work range idx 7
            auto sum = sycl::local_accessor<float, 1>((local[2] + 7) * 16, cgh); // one per row in work-group

            cgh.parallel_for(
              sycl::nd_range{global, local},
              Ext_desc_loop_local_mem(
                sum, _dct, _dbuf, _dobuf, oct_obj.getDataArray(), octave, oct_obj.getWidth(), oct_obj.getHeight()));
        });

        // fprintf(stderr, "\n\tCOMPLETE WITH EXT DESC LOOP MEM\n");
    }
}

void popsift::Pyramid::descriptors(const Config& conf)
{
    sycl::event readDesc = readDescCountersFromDevice();
    // readDescCountersFromDevice().wait();

    auto group_size = get_kernel_subgroup_size<sub_group_desc_loop>(_device_queue);
    bool use_sub_group = group_size >= 32;

    // Was not able to do this (probs due to it being in different files?? IDK)
    // auto normalize_size = get_kernel_subgroup_size<sub_group_normalize>(_device_queue);
    // const bool sub_group_normalize = normalize_size >= 32;

    readDesc.wait();

    // Now we got my counters on host so we can write them to a file
    fprintf(stderr,
            "\n\t\tBitonic numbers [%d, %d, %d, %d, %d]\n",
            _hct.bitonic_val_counter[0],
            _hct.bitonic_val_counter[1],
            _hct.bitonic_val_counter[2],
            _hct.bitonic_val_counter[3],
            _hct.bitonic_val_counter[4]);

    std::ofstream outfile("bitonic_count.txt", std::ios::app);

    if(!outfile)
    {
        std::cerr << "Error opening file!" << std::endl;
    }
    else
    {
        for(int i = 0; i < 5; ++i)
        {
            outfile << _hct.bitonic_val_counter[i] << " ";
        }
        outfile << "\n";
    }

    for(int o = 0; o < _num_octaves; o++)
    {
        fprintf(stderr,
                "\n\toctave = %d | num_extrema %d | num_ori = %d | extrema_prefixsum = %d | ori_prefix_sum = %d\n",
                o,
                _hct.ext_ct[o],
                _hct.ori_ct[o],
                _hct.ext_ps[o],
                _hct.ori_ps[o]);
    }
    fprintf(stderr, "\n\t Extrema total = %d | ori_total = %d", _hct.ext_total, _hct.ori_total);

    for(int octave = _num_octaves - 1; octave >= 0; octave--)
    // for( int octave=0; octave<_num_octaves; octave++ )
    {
        if(_hct.ori_ct[octave] != 0)
        {
            Octave& oct_obj = _octaves[octave];

            if(conf.getDescMode() == Config::Loop)
            {
                // Default

                fprintf(stderr,
                        "\nRunning start_ext_desc_loop for octave %d, with num ori = %d\n",
                        octave,
                        _hct.ori_ct[octave]);
                start_ext_desc_loop(octave, oct_obj, use_sub_group);
            }
            else if(conf.getDescMode() == Config::VLFeat_Desc)
            {
                // start_ext_desc_vlfeat(octave, oct_obj);
            }
            else
            {
                POP_FATAL("not yet");
            }
        }
        // _device_queue.wait(); // just for testing
    }

    if(_hct.ori_total == 0)
    {
        fprintf(stderr, "Warning: no descriptors extracted\n");
        return;
    }
    // _device_queue.wait(); // just for testing

    // fprintf(stderr, "\n\tPAST start_ext_loop loop\n");

    sycl::range global{32, static_cast<size_t>(popsift::grid_divide(_hct.ori_total, 32))};
    sycl::range local{32, 32};

    if(conf.getUseRootSift())
    {
        // DEFAULT

        _device_queue.wait(); // should use events instead

        if(use_sub_group) // basing decision on Ext_desc_loop (not the one I'm launching
                          // as I was not able to pass struct/class to parallel_for and use it as in last one
        {
            _device_queue.parallel_for(
              sycl::nd_range{global, local},
              Normalize_histogram<NormalizeRootSift, false>(_dbuf_host.desc, _d_consts, _hct.ori_total));
        }
        else
        {
            // Template param is work_gropu scan
            _device_queue.parallel_for(
              sycl::nd_range{global, local},
              Normalize_histogram<NormalizeRootSift, true>(_dbuf_host.desc, _d_consts, _hct.ori_total));
        }
    }
    else
    {
        // normalize_histogram<NormalizeL2><<<grid, block, 0, _download_stream>>>();
        // POP_SYNC_CHK;
    }
    _device_queue.wait();
}

} // namespace popsift
