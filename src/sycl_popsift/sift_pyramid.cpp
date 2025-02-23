#include "sycl_popsift/sift_pyramid.hpp"

#include "sycl/usm.hpp"
#include "sycl_popsift/common/debug_macros.hpp"
#include "sycl_popsift/gauss_filter.hpp"
#include "sycl_popsift/malloc_devt.hpp"
#include "sycl_popsift/s_image.hpp" // not sure if needed to include here aswell clean up #includes at some point
#include "sycl_popsift/sift_constants.hpp"

#include <cmath>
#include <sstream>
#include <vector>

namespace popsift {

// T* malloc_devT(int num, sycl::queue& Q)
// template<class T>
// T* malloc_devT(int num, const char* file, int line, sycl::queue Q)
// {
//     T* ptr;
//     try
//     {
//         ptr = sycl::malloc_device<T>(num, Q);
//     }
//     catch(const sycl::exception& e)
//     {
//         std::stringstream ss;
//         ss << "Memory allocation failed" << e.what();
//         POP_FATAL_FL(ss.str(), file, line);
//     }
//     return ptr;
// }

Pyramid::Pyramid(const Config& config,
                 int width,
                 int height,
                 sycl::queue& Q,
                 popsift::GaussInfo* d_gauss,
                 popsift::ConstInfo* d_consts)
  : _num_octaves(config.octaves)
  , _levels(config.levels + 3)
  , _assume_initial_blur(config.hasInitialBlur())
  , _initial_blur(config.getInitialBlur())
  , _device_queue(Q)
  , _d_gauss(d_gauss)
  , _d_consts(d_consts)
{
    // _octaves = new Octave[_num_octaves];
    // Could not find a way to use C array so using vector

    _octaves.reserve(_num_octaves);
    for(int i = 0; i < _num_octaves; ++i)
    {
        _octaves.emplace_back(Q);
    }

    _d_extrema_num_blocks = popsift::common_sycl::malloc_devT<int>(
      _num_octaves, __FILE__, __LINE__, "Global octave barrier allocation failed", Q);

    memset(&_hct, 0, sizeof(ExtremaCounters));

    _dct = popsift::common_sycl::malloc_devT<ExtremaCounters>(
      1, __FILE__, __LINE__, "Device Extrema counter allocation failed", Q);
    Q.memset(_dct, 0, sizeof(ExtremaCounters));

    // memset(&hbuf, 0, sizeof(ExtremaBuffers));
    // memset(&dbuf_shadow, 0, sizeof(ExtremaBuffers));
    //
    // _d_extrema_num_blocks = popsift::cuda::malloc_devT<int>(_num_octaves,
    // __FILE__, __LINE__);
    //

    int w = width;
    int h = height;
    for(int o = 0; o < _num_octaves; o++)
    {
        _octaves[o].debugSetOctave(o);
        _octaves[o].alloc(config, w, h, _levels);
        w = ceilf(w / 2.0f);
        h = ceilf(h / 2.0f);
    }

    // int sz = _num_octaves * h_consts.max_extrema;
    // _dobuf = popsift::sycl_helpers::malloc_devT<DevBuffers>(
    //   sz, __FILE__, __LINE__, "Devcie Orientation buffer allocation failed", _device_queue);

    // _dobuf = cuda::malloc_devT<DevBuffers>(
    //   sz, __FILE__, __LINE__, "Devcie Orientation buffer allocation failed", _device_queue);
    // DOBUF

    // int sz = _num_octaves * h_consts.max_extrema;
    // try
    // {
    //     _dobuf = sycl::malloc_device<DevBuffers>(1, Q);
    // }
    // catch(const sycl::exception& e)
    // {
    //     std::stringstream ss;
    //     ss << "Devcie Orientation buffer allocation failed" << e.what();
    //     POP_FATAL(ss.str());
    // }
    //
    // _dobuf->i_ext_dat[0] = popsift::cuda::malloc_devT<InitialExtremum>(sz, __FILE__, __LINE__, ");
    // _dobuf->i_ext_off[0] = popsift::cuda::malloc_devT<int>(sz, __FILE__, __LINE__);
    // for(int o = 1; o < _num_octaves; o++)
    // {
    //     _dobuf.i_ext_dat[o] = dobuf_shadow.i_ext_dat[0] + (o * h_consts.max_extrema);
    //     dobuf.i_ext_off[o] = dobuf_shadow.i_ext_off[0] + (o * h_consts.max_extrema);
    // }

    //

    // int sz = _num_octaves * h_consts.max_extrema;
    // dobuf_shadow.i_ext_dat[0] = popsift::cuda::malloc_devT<InitialExtremum>(sz, __FILE__, __LINE__);
    // dobuf_shadow.i_ext_off[0] = popsift::cuda::malloc_devT<int>(sz, __FILE__, __LINE__);
    // for(int o = 1; o < _num_octaves; o++)
    // {
    //     dobuf_shadow.i_ext_dat[o] = dobuf_shadow.i_ext_dat[0] + (o * h_consts.max_extrema);
    //     dobuf_shadow.i_ext_off[o] = dobuf_shadow.i_ext_off[0] + (o * h_consts.max_extrema);
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

Pyramid::~Pyramid()
{
    // Octaves stored in vector so they will be destroyed/deleted by this object being destroyed
    sycl::free(_d_extrema_num_blocks, _device_queue);
    sycl::free(_dct, _device_queue);
    sycl::free(_dobuf, _device_queue);
}

std::vector<sycl::event> Pyramid::step1(const Config& conf,
                                        popsift::Image* img,
                                        sycl::event d_gauss_write,
                                        sycl::event img_transfer)
{
    // TODO: Implement the reset -- far down the line need to find extrema first
    // reset_extrema_mgmt();
    return build_pyramid(conf, img, d_gauss_write, img_transfer);
}

// could probably pass the dependencies as a reference...
void Pyramid::step2(const Config& conf, std::vector<sycl::event> dependencies, sycl::event d_consts_write)
{
    find_extrema(conf, dependencies, d_consts_write);

    // orientation( conf );
    //
    // descriptors( conf );
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

void Pyramid::resetDimensions(const Config& conf, int width, int height)
{
    int w = width;
    int h = height;

    for(int o = 0; o < _num_octaves; o++)
    {
        _octaves[o].resetDimensions(conf, w, h);
        w = ceilf(w / 2.0f);
        h = ceilf(h / 2.0f);
    }
}

int* Pyramid::getNumberOfBlocks(int octave) { return &_d_extrema_num_blocks[octave]; }

} // namespace popsift
