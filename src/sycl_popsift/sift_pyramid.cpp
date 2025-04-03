#include "sycl_popsift/sift_pyramid.hpp"

#include "sycl/usm.hpp"
#include "sycl_popsift/common/debug_macros.hpp"
#include "sycl_popsift/gauss_filter.hpp"
// #include "sycl_popsift/malloc_devt.hpp"
#include "sycl_popsift/s_image.hpp" // not sure if needed to include here aswell clean up #includes at some point
#include "sycl_popsift/sift_constants.hpp"

#include <cmath>
#include <cstdio>
#include <sstream>
#include <vector>

namespace popsift {

Pyramid::Pyramid(const Config& config,
                 int width,
                 int height,
                 sycl::queue Q,
                 popsift::GaussInfo* d_gauss,
                 popsift::ConstInfo* d_consts,
                 popsift::ConstInfo& h_consts)
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

    fprintf(stderr, "Before emplace back and reserve\n");
    _octaves.reserve(_num_octaves);
    for(int i = 0; i < _num_octaves; ++i)
    {
        // _octaves.emplace_back(_device_queue.get_context(), _device_queue.get_device());
        _octaves.emplace_back(_device_queue);
    }
    fprintf(stderr, "After emplace back and reserve\n");

    int w = width;
    int h = height;
    for(int o = 0; o < _num_octaves; o++)
    {
        _octaves[o].debugSetOctave(o);

        _octaves[o].alloc(config, w, h, _levels);
        w = ceilf(w / 2.0f);
        h = ceilf(h / 2.0f);
    }

    memset(&_hct, 0, sizeof(ExtremaCounters)); // might have a puprpose on the host side

    // NOTE: Not sure if we want to have the custom error message for each malloc not sure if it gives any usefull
    // information and it might slightly slow us down...

    _dct = popsift::sycl_common::malloc_devT<ExtremaCounters>(
      1, __FILE__, __LINE__, "Device Extrema counter allocation failed", _device_queue);
    fprintf(stderr, "After using malloc_devT\n");
    _device_queue.memset(_dct, 0, sizeof(ExtremaCounters));

    _d_extrema_num_blocks = popsift::sycl_common::malloc_devT<int>(
      _num_octaves, __FILE__, __LINE__, "Global octave barrier allocation failed", _device_queue);

    // don't see the purpose of dobuf_shadow, so not using it untill I see a purpose for it (I probs will in future :D)
    // I think it's mainly for some memory optimizations in cuda but might be wrong!

    int sz = _num_octaves * h_consts.max_extrema; // h_consts.max_extrema is 100 000 by default
    // _dobuf = popsift::sycl_common::malloc_devT<DevBuffers>(
    //   1, __FILE__, __LINE__, "Allocating device DevBuffers struct failed", Q);

    // TODO: Rethink structure of memory not sure if we actually want to use shared here as it results in a large
    // penalty in terms of performance (potentially)
    _dobuf = popsift::sycl_common::malloc_sharedT<DevBuffers>(
      1, __FILE__, __LINE__, "Allocating device DevBuffers struct failed", _device_queue);

    fprintf(stderr, "Before i_ext_dat\n");
    // For 7 octaves case the total memory useage for this array is 196MB
    _dobuf->i_ext_dat[0] = popsift::sycl_common::malloc_devT<InitialExtremum>(
      sz, __FILE__, __LINE__, "Device InitialExtremum array allocation failed", _device_queue);

    fprintf(stderr, "after i_ext_dat\n");
    // For 7 octaves case the total memory useage for this array is 2.8MB
    _dobuf->i_ext_off[0] = popsift::sycl_common::malloc_devT<int>(
      sz, __FILE__, __LINE__, "Device extremum offset array allocation failed", _device_queue);

    // All octaves in one contigous memory segment that has 100k each in default case
    // loop sets poitners to appropriate 0 positions per octave
    for(int o = 1; o < _num_octaves; o++)
    {
        _dobuf->i_ext_dat[o] = _dobuf->i_ext_dat[0] + (o * h_consts.max_extrema);
        _dobuf->i_ext_off[o] = _dobuf->i_ext_off[0] + (o * h_consts.max_extrema);
    }
    // set remaining to nullptr
    for(int o = _num_octaves; o < MAX_OCTAVES; o++)
    {
        _dobuf->i_ext_dat[o] = nullptr;
        _dobuf->i_ext_off[o] = nullptr;
    }

    sz = h_consts.max_extrema; // setting to 100 000 in default case octave num invariant
    _dobuf->extrema = popsift::sycl_common::malloc_devT<Extremum>(
      sz, __FILE__, __LINE__, "Device Extremum array allocation failed", _device_queue);
    _dobuf->features = popsift::sycl_common::malloc_devT<Feature>(
      sz, __FILE__, __LINE__, "Device Feature array allocation failed", _device_queue);
    // hbuf.ext_allocated = sz; // don't know the purpose of this boy yet
    // dbuf_shadow.ext_allocated = sz; // don't know puppose of this boy yet either

    sz = std::max(2 * h_consts.max_extrema, h_consts.max_orientations);
    // hbuf.desc = popsift::cuda::malloc_hstT<Descriptor>(sz, __FILE__, __LINE__);
    // dbuf_shadow.desc = popsift::cuda::malloc_devT<Descriptor>(sz, __FILE__, __LINE__);
    _dobuf->feat_to_ext_map = popsift::sycl_common::malloc_devT<int>(
      sz, __FILE__, __LINE__, "Device feat_to_ext_map allocation failed", _device_queue);
    // hbuf.ori_allocated = sz;
    // dbuf_shadow.ori_allocated = sz;
}

Pyramid::~Pyramid()
{
    // Octaves stored in vector so they will be destroyed/deleted by this object being destroyed
    fprintf(stderr, "Destroying the Pyramid!\n");
    sycl::free(_d_extrema_num_blocks, _device_queue);
    sycl::free(_dct, _device_queue);
    sycl::free(_dobuf->i_ext_dat[0], _device_queue);
    sycl::free(_dobuf->i_ext_off[0], _device_queue);
    sycl::free(_dobuf->features, _device_queue);
    sycl::free(_dobuf->extrema, _device_queue);
    sycl::free(_dobuf->feat_to_ext_map, _device_queue);
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
    // return {sycl::event()};
}

// could probably pass the dependencies as a reference...
void Pyramid::step2(const Config& conf, std::vector<sycl::event> dependencies, sycl::event d_consts_write)
{
    find_extrema(conf, dependencies, d_consts_write);

    // orientation(conf);

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

// Fine to use on device memory as it is just pointer arithmetic
int* Pyramid::getNumberOfBlocks(int octave) { return &_d_extrema_num_blocks[octave]; }

sycl::event Pyramid::readDescCountersFromDevice() { return _device_queue.memcpy(&_hct, _dct, sizeof(ExtremaCounters)); }

} // namespace popsift
