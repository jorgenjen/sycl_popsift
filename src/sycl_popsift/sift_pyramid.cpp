
#include "sycl_popsift/sift_pyramid.hpp"

#include "sycl/usm.hpp"
#include "sycl_popsift/common/debug_macros.hpp"
#include "sycl_popsift/gauss_filter.hpp"
#include "sycl_popsift/s_image.hpp" // not sure if needed to include here aswell clean up #includes at some point
#include "sycl_popsift/sift_constants.hpp"

#include <cmath>
#include <sstream>
#include <vector>

namespace popsift {

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

    // _octaves = &(new Octave(Q))[_num_octaves];

    int w = width;
    int h = height;

    // Don't understand how this works as a octave global barrier as of yet
    try
    {
        _d_extrema_num_blocks = sycl::malloc_device<int>(_num_octaves, Q);
    }
    catch(const sycl::exception& e)
    {
        std::stringstream ss;
        ss << "Octave memory allocation failed" << e.what();
        POP_FATAL(ss.str());
        // std::cerr << "Memory allocation failed: " << e.what() << std::endl;
    }

    // IMPORTANT: do next!!
    // TODO: IMPORTANT -> Implement the constructor to pyramid

    // NOT SURE IF I want them to be global like in POPSIFT or if
    // I want them to be an attribute of the pyramid calss
    // memset(&hct, 0, sizeof(ExtremaCounters));
    // cudaMemcpyToSymbol(dct, &hct, sizeof(ExtremaCounters), 0,
    // cudaMemcpyHostToDevice);

    memset(&_hct, 0, sizeof(ExtremaCounters));

    try
    {
        _dct = sycl::malloc_device<ExtremaCounters>(1, Q);
    }
    catch(const sycl::exception& e)
    {
        std::stringstream ss;
        ss << "Octave memory allocation failed" << e.what();
        POP_FATAL(ss.str());
        // std::cerr << "Memory allocation failed: " << e.what() << std::endl;
    }
    Q.memset(_dct, 0, sizeof(ExtremaCounters));

    // memset(&hbuf, 0, sizeof(ExtremaBuffers));
    // memset(&dbuf_shadow, 0, sizeof(ExtremaBuffers));
    //
    // _d_extrema_num_blocks = popsift::cuda::malloc_devT<int>(_num_octaves,
    // __FILE__, __LINE__);
    //
    for(int o = 0; o < _num_octaves; o++)
    {
        _octaves[o].debugSetOctave(o);
        _octaves[o].alloc(config, w, h, _levels);
        w = ceilf(w / 2.0f);
        h = ceilf(h / 2.0f);
    }

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

Pyramid::~Pyramid()
{
    // Octaves stored in vector so they will be destroyed/deleted by this object being destroyed
    sycl::free(_d_extrema_num_blocks, _device_queue);
    sycl::free(_dct, _device_queue);
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
