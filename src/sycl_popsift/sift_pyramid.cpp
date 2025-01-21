
#include "sycl_popsift/sift_pyramid.hpp"

#include "sycl_popsift/s_image.hpp" // not sure if needed to include here aswell clean up #includes at some point

namespace popsift {

Pyramid::Pyramid(const Config& config, int width, int height)
  : _num_octaves(config.octaves)
  , _levels(config.levels + 3)
  , _assume_initial_blur(config.hasInitialBlur())
  , _initial_blur(config.getInitialBlur())
{
    _octaves = new Octave[_num_octaves];

    int w = width;
    int h = height;

    // memset(&hct, 0, sizeof(ExtremaCounters));
    // cudaMemcpyToSymbol(dct, &hct, sizeof(ExtremaCounters), 0,
    // cudaMemcpyHostToDevice);
    //
    // memset(&hbuf, 0, sizeof(ExtremaBuffers));
    // memset(&dbuf_shadow, 0, sizeof(ExtremaBuffers));
    //
    // _d_extrema_num_blocks = popsift::cuda::malloc_devT<int>(_num_octaves,
    // __FILE__, __LINE__);
    //
    // for(int o = 0; o < _num_octaves; o++)
    // {
    //     _octaves[o].debugSetOctave(o);
    //     _octaves[o].alloc(config, w, h, _levels, _gauss_group);
    //     w = ceilf(w / 2.0f);
    //     h = ceilf(h / 2.0f);
    // }
    //
    // int sz = _num_octaves * h_consts.max_extrema;
    // dobuf_shadow.i_ext_dat[0] =
    // popsift::cuda::malloc_devT<InitialExtremum>(sz, __FILE__, __LINE__);
    // dobuf_shadow.i_ext_off[0] = popsift::cuda::malloc_devT<int>(sz, __FILE__,
    // __LINE__); for(int o = 1; o < _num_octaves; o++)
    // {
    //     dobuf_shadow.i_ext_dat[o] = dobuf_shadow.i_ext_dat[0] + (o *
    //     h_consts.max_extrema); dobuf_shadow.i_ext_off[o] =
    //     dobuf_shadow.i_ext_off[0] + (o * h_consts.max_extrema);
    // }
    // for(int o = _num_octaves; o < MAX_OCTAVES; o++)
    // {
    //     dobuf_shadow.i_ext_dat[o] = nullptr;
    //     dobuf_shadow.i_ext_off[o] = nullptr;
    // }
    //
    // sz = h_consts.max_extrema;
    // dobuf_shadow.extrema = popsift::cuda::malloc_devT<Extremum>(sz, __FILE__,
    // __LINE__); dobuf_shadow.features =
    // popsift::cuda::malloc_devT<Feature>(sz, __FILE__, __LINE__);
    // hbuf.ext_allocated = sz;
    // dbuf_shadow.ext_allocated = sz;
    //
    // sz = max(2 * h_consts.max_extrema, h_consts.max_orientations);
    // hbuf.desc = popsift::cuda::malloc_hstT<Descriptor>(sz, __FILE__,
    // __LINE__); dbuf_shadow.desc = popsift::cuda::malloc_devT<Descriptor>(sz,
    // __FILE__, __LINE__); dobuf_shadow.feat_to_ext_map =
    // popsift::cuda::malloc_devT<int>(sz, __FILE__, __LINE__);
    // hbuf.ori_allocated = sz;
    // dbuf_shadow.ori_allocated = sz;
    //
    // cudaMemcpyToSymbol(dbuf, &dbuf_shadow, sizeof(ExtremaBuffers), 0,
    // cudaMemcpyHostToDevice); cudaMemcpyToSymbol(dobuf, &dobuf_shadow,
    // sizeof(DevBuffers), 0, cudaMemcpyHostToDevice);
    //
    // cudaStreamCreate(&_download_stream);
}

void Pyramid::step1(const Config& conf, popsift::Image* img)
{
    // reset_extrema_mgmt();
    build_pyramid(conf, img);
}

// void Pyramid::reset_extrema_mgmt()
// {
//     memset(&hct, 0, sizeof(ExtremaCounters));
//     cudaMemcpyToSymbol(dct, &hct, sizeof(ExtremaCounters), 0,
//     cudaMemcpyHostToDevice);
//
//     popcuda_memset_sync(_d_extrema_num_blocks, 0, _num_octaves *
//     sizeof(int));
// }

} // namespace popsift
